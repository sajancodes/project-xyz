#!/usr/bin/env python3
"""
SEMANTIC V2 TRAINING: Clean per-row mixture, masked loss, LR schedule
====================================================================
Created: 2026-08-12

What is different from the old mixed training (train_mixed.py)?
1.  OLD: fineweb and semantic tokens were interleaved INSIDE one row -
    the model almost never saw a clean "User:...Assistant:..." pattern.
    NEW: each row is EITHER a full fineweb window OR a row of packed
    complete semantic examples.
2.  OLD: loss on every token (including the user question).
    NEW: loss masking - for semantic rows only Assistant spans contribute
    to the loss; the model must learn to RESPOND, not to echo.
3.  OLD: constant learning rate (caused oscillation past ~2k steps).
    NEW: warmup + cosine decay to 10% of peak.
4.  OLD: 41 fixed examples repeated endlessly.
    NEW: procedurally generated unique examples + a subset of a real
    instruction dataset (yahma/alpaca-cleaned).
5.  NEW: fineweb held-out validation loss monitor (catastrophic-forgetting
    check) and periodic generation logging.

Mixture (per batch): SEMANTIC_RATIO of the 16 rows are semantic/instruction,
the rest are FineWeb windows.

Resume: checkpoint stores step/optimizer/scheduler/scaler/RNG states.

Usage:
  python src/training/train_semantic_v2.py                  # defaults
  python src/training/train_semantic_v2.py --steps 200 --smoke
  python src/training/train_semantic_v2.py --lr 2e-4 --semantic-ratio 0.2
"""

import os
import sys
import time
import json
import argparse
import random
import math
import torch

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))

from paths import (
    CHECKPOINT_106K,
    TOKENIZER_PATH as PATHS_TOKENIZER,
    SEMANTIC_V2_DIR,
    LATEST_CHECKPOINT,
)
from model import SmallEnglishLLM
from model_config import ModelConfig
from semantic_data import SemanticGenerator, load_instruction_subset

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BATCH_SIZE = 16
SEQ_LEN = 512
MAX_STEPS = 5000
WARMUP_STEPS = 150
PEAK_LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
SEMANTIC_RATIO = 0.15          # fraction of ROWS that are semantic
FINEWEB_RATIO = 1.0 - SEMANTIC_RATIO
SEMANTIC_EXAMPLES_N = 20000    # procedural examples to generate
INSTRUCTION_SUBSET_N = 8000    # alpaca examples to load
LOG_EVERY = 10
CHECKPOINT_EVERY = 500
GEN_LOG_EVERY = 500
VAL_EVERY = 500
BASE_CHECKPOINT = CHECKPOINT_106K
DEVICE = torch.device("cuda")
USE_BF16 = True

ASSISTANT_TOKEN = 11326        # " Assistant" (byte-level token id)


def find_assistant_start(ids, tokenizer):
    """Index of the first token of the 'Assistant:' marker (robust to
    tokenization variants: [11326,29] mid-sentence or [Ass][istant][:]
    after a newline). Returns None if not found."""
    for i, tok in enumerate(ids):
        if i + 2 < len(ids) and ids[i + 2] == 29:  # ':'
            if tokenizer.decode([tok]).lower().startswith("ass"):
                return i
    return None


def cosine_schedule(step, max_steps, warmup, peak):
    """Warmup + cosine decay to 10% of peak."""
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, (max_steps - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))


def main():
    global MAX_STEPS, SEMANTIC_RATIO, FINEWEB_RATIO, SEMANTIC_EXAMPLES_N, INSTRUCTION_SUBSET_N
    global CHECKPOINT_EVERY

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--lr", type=float, default=PEAK_LR)
    parser.add_argument("--semantic-ratio", type=float, default=SEMANTIC_RATIO)
    parser.add_argument("--base", default=BASE_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to verify the pipeline works")
    parser.add_argument("--name", default="semv2",
                        help="checkpoint file prefix")
    args = parser.parse_args()

    MAX_STEPS = args.steps
    if args.smoke:
        MAX_STEPS = 20
        CHECKPOINT_EVERY = 10
        SEMANTIC_EXAMPLES_N = 200
        INSTRUCTION_SUBSET_N = 50
    SEMANTIC_RATIO = args.semantic_ratio
    FINEWEB_RATIO = 1.0 - SEMANTIC_RATIO

    print("=" * 70)
    print("SEMANTIC V2 TRAINING (clean rows, masked loss, LR schedule)")
    print("=" * 70)
    print(f"Base checkpoint : {args.base}")
    print(f"Steps           : {MAX_STEPS}")
    print(f"Peak LR         : {args.lr}  (warmup {WARMUP_STEPS} + cosine to 10%)")
    print(f"Batch           : {BATCH_SIZE} x {SEQ_LEN}")
    print(f"Semantic rows   : {SEMANTIC_RATIO*100:.1f}%  FineWeb rows: {FINEWEB_RATIO*100:.1f}%")
    print()

    os.makedirs(SEMANTIC_V2_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LATEST_CHECKPOINT), exist_ok=True)
    OUTPUT_CHECKPOINT = LATEST_CHECKPOINT

    # ------------------------------------------------------------
    # TOKENIZER + MODEL
    # ------------------------------------------------------------
    tokenizer = Tokenizer.from_file(PATHS_TOKENIZER)
    config = ModelConfig()
    model = SmallEnglishLLM(config).to(DEVICE)

    if not os.path.exists(args.base):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base}")
    ckpt = torch.load(args.base, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    base_step = ckpt.get("step", "unknown")
    print(f"Loaded base checkpoint (step {base_step})")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # ------------------------------------------------------------
    # SEMANTIC + INSTRUCTION POOL (tokenized with masks)
    # ------------------------------------------------------------
    print("Generating semantic examples...")
    gen = SemanticGenerator(seed=args.seed)
    proc_examples = gen.generate(SEMANTIC_EXAMPLES_N, seed=args.seed)
    print(f"  procedural: {len(proc_examples)}")

    alpaca_examples = load_instruction_subset(n=INSTRUCTION_SUBSET_N)
    all_examples = proc_examples + alpaca_examples
    print(f"  total semantic+instruction: {len(all_examples)}")

    # Tokenize each example into (ids, mask). The assistant answer is
    # terminated with <eos> (id 3) so the model learns to STOP after
    # answering instead of looping into the next packed example's "User:".
    EOS_ID = 3
    semantic_pool = []
    skipped = 0
    for text in all_examples:
        ids = tokenizer.encode(text).ids
        if len(ids) < 3 or len(ids) > 300:
            skipped += 1
            continue
        # find "Assistant:" marker; mask from there to end (include marker)
        start = find_assistant_start(ids, tokenizer)
        if start is None:
            skipped += 1
            continue
        mask = [0] * len(ids)
        for i in range(start, len(ids)):
            mask[i] = 1
        if len(ids) + 1 > 301:  # keep window small enough to still pack
            continue
        ids = ids + [EOS_ID]
        mask = mask + [1]
        semantic_pool.append((ids, mask))
    print(f"  usable tokenized examples: {len(semantic_pool)} (skipped {skipped})")

    # ------------------------------------------------------------
    # FINEWEB STREAM
    # ------------------------------------------------------------
    print("Opening FineWeb stream...")
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2025-26",
                           split="train", streaming=True)
    fw_iter = iter(dataset)
    fw_buffer = []

    def fill_fw(min_tokens):
        while len(fw_buffer) < min_tokens:
            doc = next(fw_iter)
            text = doc.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            if len(ids) >= 2:
                fw_buffer.extend(ids)

    # Held-out validation buffer: first docs of the stream, never trained on.
    VAL_BUFFER = []
    while len(VAL_BUFFER) < 40000:
        doc = next(fw_iter)
        text = doc.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text).ids
        if len(ids) >= 100:
            VAL_BUFFER.extend(ids[:4000])

    def val_loss():
        """FineWeb held-out loss - catastrophic forgetting monitor."""
        model.eval()
        losses = []
        for _ in range(8):
            start = random.randint(0, len(VAL_BUFFER) - SEQ_LEN - 1)
            window = VAL_BUFFER[start:start + SEQ_LEN + 1]
            x = torch.tensor([window[:-1]], dtype=torch.long, device=DEVICE)
            y = torch.tensor([window[1:]], dtype=torch.long, device=DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    # ------------------------------------------------------------
    # BATCH CONSTRUCTION
    # ------------------------------------------------------------
    semantic_rows = max(1, round(BATCH_SIZE * SEMANTIC_RATIO))
    fineweb_rows = BATCH_SIZE - semantic_rows

    def make_semantic_row():
        """Pack complete examples into one 512+1 window with mask."""
        ids = []
        mask = []
        while len(ids) <= SEQ_LEN:
            ex_ids, ex_mask = semantic_pool[random.randrange(len(semantic_pool))]
            if len(ids) + len(ex_ids) > SEQ_LEN + 1:
                break
            ids.extend(ex_ids)
            mask.extend(ex_mask)
        if len(ids) < SEQ_LEN + 1:
            pad = SEQ_LEN + 1 - len(ids)
            ids.extend([0] * pad)
            mask.extend([0] * pad)
        return ids, mask

    def make_fineweb_row():
        fill_fw(SEQ_LEN + 1)
        window = fw_buffer[:SEQ_LEN + 1]
        del fw_buffer[:SEQ_LEN]
        return window

    def get_batch():
        xs, ys, masks = [], [], []
        for _ in range(fineweb_rows):
            window = make_fineweb_row()
            xs.append(window[:-1])
            ys.append(window[1:])
            masks.append([1] * SEQ_LEN)
        for _ in range(semantic_rows):
            ids, mask = make_semantic_row()
            xs.append(ids[:-1])
            ys.append(ids[1:])
            masks.append(mask[:-1])
        x = torch.tensor(xs, dtype=torch.long, device=DEVICE)
        y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
        m = torch.tensor(masks, dtype=torch.float32, device=DEVICE)
        return x, y, m

    # ------------------------------------------------------------
    # OPTIMIZER + SCHEDULER + SCALER (fresh by default)
    # ------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lambda s: cosine_schedule(s, MAX_STEPS, WARMUP_STEPS, args.lr))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_BF16)

    # Resume support
    start_step = 0
    if os.path.exists(LATEST_CHECKPOINT) and not args.smoke:
        print(f"Resuming from {LATEST_CHECKPOINT}...")
        old = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(old["model_state_dict"])
        optimizer.load_state_dict(old["optimizer_state_dict"])
        scheduler.load_state_dict(old["scheduler_state_dict"])
        scaler.load_state_dict(old["scaler_state_dict"])
        start_step = old.get("step", 0)
        torch.set_rng_state(old.get("torch_rng").cpu())
        torch.cuda.set_rng_state(old.get("cuda_rng").cpu())
        random.setstate(old.get("py_rng"))
        print(f"Resumed at step {start_step}")

    torch.manual_seed(args.seed + start_step)
    torch.cuda.manual_seed(args.seed + start_step)

    # ------------------------------------------------------------
    # CHECKPOINT + LOGGING
    # ------------------------------------------------------------
    def save_checkpoint(step, train_loss, val_loss_value):
        data = {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "py_rng": random.getstate(),
            "config": config.__dict__,
            "loss": float(train_loss),
            "val_loss": None if val_loss_value is None else float(val_loss_value),
            "base_checkpoint": args.base,
            "training_type": "semantic_v2_mixed",
            "semantic_ratio": SEMANTIC_RATIO,
            "fineweb_ratio": FINEWEB_RATIO,
            "peak_lr": args.lr,
            "scheduler": f"warmup{WARMUP_STEPS}+cosine",
            "procedural_examples": len(proc_examples),
            "instruction_examples": len(alpaca_examples),
        }
        torch.save(data, LATEST_CHECKPOINT)
        # Also save milestone archive
        archive = os.path.join(SEMANTIC_V2_DIR, f"checkpoint_{args.name}-{step}.pt")
        torch.save(data, archive)
        print(f"  -> checkpoint saved: {LATEST_CHECKPOINT} (archive: {archive})")

    def log_generations(step):
        model.eval()
        prompts = ["User: The sun rises in the east. Where does the sun set?\nAssistant:",
                   "the barking dog",
                   "User: Emma handed Olivia a hat. Who got the hat?\nAssistant:",
                   "the students are"]
        log_path = os.path.join(SEMANTIC_V2_DIR, f"generations_{args.name}.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== step {step} ===\n")
            for p in prompts:
                encoded = tokenizer.encode(p)
                input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=DEVICE)
                for _ in range(40):
                    x = input_ids[:, -SEQ_LEN:]
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                        logits, _ = model(x)
                    probs = torch.softmax(logits[:, -1, :] / 0.7, dim=-1)
                    topk = torch.topk(probs, 40)
                    nxt = torch.multinomial(topk.values, 1)
                    input_ids = torch.cat([input_ids, topk.indices[0][nxt].view(1, 1)], dim=1)
                    if input_ids[0, -1].item() == 3:
                        break
                out = tokenizer.decode(input_ids[0, len(encoded.ids):].tolist())
                f.write(f"[{p[:40]}] -> {out}\n")
        model.train()

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    model.train()
    print("=" * 70)
    print("STARTING SEMANTIC V2 TRAINING")
    print("=" * 70)
    start_time = time.time()
    val_loss_value = None

    for step in range(start_step + 1, MAX_STEPS + 1):
        step_start = time.time()
        x, y, m = get_batch()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits, _ = model(x, None)
        logits_flat = logits.view(-1, config.vocab_size)
        m_flat = m.view(-1)
        loss_per = F.cross_entropy(logits_flat, y.view(-1), reduction="none")
        loss = (loss_per * m_flat).sum() / m_flat.sum().clamp(min=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % LOG_EVERY == 0 or step == 1:
            torch.cuda.synchronize()
            elapsed = time.time() - step_start
            tok_s = (BATCH_SIZE * SEQ_LEN) / elapsed
            vram = torch.cuda.max_memory_allocated() / 1024**3
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Step {step:5d}/{MAX_STEPS} | loss {loss.item():.4f} | "
                  f"lr {lr_now:.2e} | {tok_s:,.0f} tok/s | VRAM {vram:.2f} GB",
                  flush=True)

        if step % VAL_EVERY == 0:
            val_loss_value = val_loss()
            print(f"  [val] fineweb held-out loss: {val_loss_value:.4f}", flush=True)

        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(step, loss.item(), val_loss_value)

        if step % GEN_LOG_EVERY == 0:
            log_generations(step)

    save_checkpoint(MAX_STEPS, loss.item(), val_loss_value)

    total_time = time.time() - start_time
    total_tokens = (MAX_STEPS - start_step) * BATCH_SIZE * SEQ_LEN
    print()
    print("=" * 70)
    print("SEMANTIC V2 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Base: {args.base} (step {base_step})")
    print(f"Steps this run: {MAX_STEPS - start_step}")
    print(f"Final loss: {loss.item():.4f} | val loss: {val_loss_value}")
    print(f"Tokens: {total_tokens:,} in {total_time/60:.2f} min")
    print(f"Checkpoint: {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
