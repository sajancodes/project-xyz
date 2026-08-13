#!/usr/bin/env python3
"""
SEMV2E TRAINING - CONTROLLED SEMANTIC BINDING EXPERIMENT
==========================================================
Continues from checkpoint_best.pt (semv2c-3000, the frozen baseline).
Do NOT overwrite checkpoint_best.pt - every SEMV2E checkpoint is saved
separately under checkpoints/semv2e/.

Data mixture (per batch of 16 rows) - mission section 10:
   55% semantic binding (new SEMV2E procedural, leakage-safe names)
   15% instruction/natural language (Alpaca subset + existing instruction)
   15% existing semantic/reasoning (existing SemanticGenerator)
   10% FineWeb preservation (real language)
    5% arithmetic (controlled secondary experiment)

Loss masking: for semantic/instruction rows, loss only on the Assistant
span (preserved semantic-v2 principle, mission section 11). FineWeb rows
are fully masked (all tokens contribute).

Schedule: warmup 150 + cosine decay to 10% of peak.
Milestones saved at 500/1000/2000/3000/5000 (+ final).

Reproducibility: seed, parent checkpoint, ratios, lr, batch, seq_len,
optimizer, scheduler, token counts all recorded in each checkpoint.
"""

import os
import sys
import time
import json
import argparse
import random
import math

uv_base = r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none"
# NOTE: The venv provides its own torch (2.11.0+cu128, CUDA-enabled).
# Do NOT prepend the uv CPU torch - use the venv's CUDA build.
if os.environ.get("UV_FORCE"):
    if uv_base not in sys.path:
        sys.path.insert(0, uv_base)
    if uv_base + r"\Lib\site-packages" not in sys.path:
        sys.path.insert(0, uv_base + r"\Lib\site-packages")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJ_ROOT, "src")
for p in (SRC, os.path.join(SRC, "training"), os.path.join(SRC, "utils"),
          os.path.join(SRC, "models"), os.path.join(SRC, "config")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paths import CHECKPOINT_BEST, TOKENIZER_PATH as PATHS_TOKENIZER
from model import SmallEnglishLLM
from model_config import ModelConfig
from semantic_data import SemanticGenerator, load_instruction_subset
from semv2e_data import Semv2eDataGen

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
BATCH_SIZE = 16
SEQ_LEN = 512
DEFAULT_STEPS = 5000
WARMUP_STEPS = 150
PEAK_LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Mission section 10 mixture
SEMANTIC_BINDING_RATIO = 0.55
INSTRUCTION_RATIO = 0.15
EXISTING_SEMANTIC_RATIO = 0.15
FINEWEB_RATIO = 0.10
ARITHMETIC_RATIO = 0.05

ASSISTANT_TOKEN = 11326
EOS_ID = 3

MILESTONES = [500, 1000, 2000, 3000, 5000]
SEMV2E_DIR = os.path.join(PROJ_ROOT, "checkpoints", "semv2e")

# FineWeb preset text for smoke tests (no network).
SMOKE_PRESET = [
    "The cat sat on the mat and watched the birds outside the window.",
    "Sajan went to the store to buy some bread and milk for breakfast.",
    "Ravi read a book about the ocean and learned many new facts.",
    "Priya sang a song while cooking dinner in the kitchen.",
    "The dog ran fast across the field chasing the red ball.",
    "Nina baked a cake for her friend's birthday celebration.",
    "The sun is bright and warm on a clear summer afternoon.",
    "Leo painted a picture of the mountains at sunset.",
    "Zoe wrote a letter to her grandmother about her new school.",
    "The students are learning about plants and how they grow.",
]


def cosine_schedule(step, max_steps, warmup, peak):
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, (max_steps - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))


def find_assistant_start(ids, tokenizer):
    """Index of the first token of the 'Assistant:' marker."""
    for i, tok in enumerate(ids):
        if i + 2 < len(ids) and ids[i + 2] == 29:
            if tokenizer.decode([tok]).lower().startswith("ass"):
                return i
    return None


def tokenize_pool(examples, tokenizer, min_len=3, max_len=220):
    """Return list of (ids, mask) with assistant-only loss mask + EOS."""
    pool = []
    skipped = 0
    for text in examples:
        ids = tokenizer.encode(text).ids
        if len(ids) < min_len or len(ids) > max_len:
            skipped += 1
            continue
        start = find_assistant_start(ids, tokenizer)
        if start is None:
            skipped += 1
            continue
        mask = [0] * len(ids)
        for i in range(start, len(ids)):
            mask[i] = 1
        ids = ids + [EOS_ID]
        mask = mask + [1]
        pool.append((ids, mask))
    return pool, skipped


def make_row(pool, seq_len, rng):
    """Pack complete examples into one window (id+1 for shift)."""
    ids, mask = [], []
    while len(ids) <= seq_len:
        ex_ids, ex_mask = pool[rng.randrange(len(pool))]
        if len(ids) + len(ex_ids) > seq_len + 1:
            break
        ids.extend(ex_ids)
        mask.extend(ex_mask)
    if len(ids) < seq_len + 1:
        ids.extend([0] * (seq_len + 1 - len(ids)))
        mask.extend([0] * (seq_len + 1 - len(mask)))
    return ids, mask


def make_fineweb_row(fw_buffer, seq_len):
    window = fw_buffer[:seq_len + 1]
    del fw_buffer[:seq_len]
    return window


def build_fineweb_buffer(tokenizer, smoke):
    if smoke:
        ids = []
        for t in SMOKE_PRESET:
            ids.extend(tokenizer.encode(t).ids)
        return ids
    from datasets import load_dataset
    print("  Opening FineWeb stream (hold-out window reserved)...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2025-26",
                      split="train", streaming=True)
    it = iter(ds)
    # Reserve first ~40k tokens for validation (never trained)
    val = []
    while len(val) < 40000:
        doc = next(it)
        t = doc.get("text", "")
        if not t:
            continue
        ids = tokenizer.encode(t).ids
        if len(ids) >= 100:
            val.extend(ids[:4000])
    # fill training buffer
    buffer = []
    while len(buffer) < 60000:
        doc = next(it)
        t = doc.get("text", "")
        if not t:
            continue
        ids = tokenizer.encode(t).ids
        if len(ids) >= 2:
            buffer.extend(ids)
    print(f"  FineWeb buffer: {len(buffer)} tokens (val: {len(val)})")
    return buffer, val


def val_loss(model, val_buffer, device, seq_len, config):
    model.eval()
    losses = []
    for _ in range(8):
        start = random.randint(0, len(val_buffer) - seq_len - 1)
        window = val_buffer[start:start + seq_len + 1]
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([window[1:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    global PEAK_LR
    ap = argparse.ArgumentParser(description="SEMV2E training")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--lr", type=float, default=PEAK_LR)
    ap.add_argument("--base", default=CHECKPOINT_BEST)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default="semv2e")
    ap.add_argument("--sb", type=float, default=SEMANTIC_BINDING_RATIO)
    ap.add_argument("--inst", type=float, default=INSTRUCTION_RATIO)
    ap.add_argument("--sem", type=float, default=EXISTING_SEMANTIC_RATIO)
    ap.add_argument("--fw", type=float, default=FINEWEB_RATIO)
    ap.add_argument("--arith", type=float, default=ARITHMETIC_RATIO)
    args = ap.parse_args()
    PEAK_LR = args.lr

    if args.smoke:
        args.steps = 10
    n_examples = 1000 if args.smoke else 20000

    rng = random.Random(args.seed)

    print("=" * 70)
    print("SEMV2E TRAINING - semantic binding experiment")
    print("=" * 70)
    print(f"Base checkpoint : {args.base}")
    print(f"Steps           : {args.steps}")
    print(f"Peak LR         : {args.lr} (warmup {WARMUP_STEPS} + cosine to 10%)")
    print(f"Mixture         : {args.sb*100:.0f}% binding | {args.inst*100:.0f}% instruction | "
          f"{args.sem*100:.0f}% existing semantic | {args.fw*100:.0f}% FineWeb | {args.arith*100:.0f}% arithmetic")
    print(f"Output dir      : {SEMV2E_DIR}")
    print("=" * 70)

    os.makedirs(SEMV2E_DIR, exist_ok=True)

    tokenizer = Tokenizer.from_file(PATHS_TOKENIZER)
    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallEnglishLLM(config).to(device)

    if not os.path.exists(args.base):
        raise FileNotFoundError(args.base)
    ckpt = torch.load(args.base, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    base_step = ckpt.get("step", "unknown")
    print(f"Loaded base checkpoint (step {base_step}, loss {ckpt.get('loss')})")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # ---------------- DATA POOLS ----------------
    print("Generating SEMV2E semantic-binding data (leakage-safe names)...")
    gen = Semv2eDataGen(seed=args.seed)
    binding_examples = gen.generate(n_examples, seed=args.seed)
    print(f"  {len(binding_examples)} binding examples")

    print("Generating existing semantic/reasoning data...")
    old_gen = SemanticGenerator(seed=args.seed)
    existing_examples = old_gen.generate(n_examples, seed=args.seed)
    print(f"  {len(existing_examples)} existing semantic examples")

    print("Loading instruction subset (Alpaca)...")
    instruction_examples = load_instruction_subset(n=2000 if args.smoke else 8000)
    if not instruction_examples:
        print("  WARNING: no instruction data loaded; will repeat binding examples")
        instruction_examples = binding_examples[:100]
    print(f"  {len(instruction_examples)} instruction examples")

    print("Generating arithmetic examples...")
    arith_examples = []
    for i in range(500 if not args.smoke else 50):
        a = rng.randint(10, 89)
        b = rng.randint(10, 89)
        arith_examples.append(
            f"User: What is {a} + {b}?\nAssistant: {a + b}")
        a2, b2 = rng.randint(20, 98), rng.randint(10, 19)
        arith_examples.append(
            f"User: What is {a2} - {b2}?\nAssistant: {a2 - b2}")
    print(f"  {len(arith_examples)} arithmetic examples")

    print("Building FineWeb buffer...")
    if args.smoke:
        fw_buffer = build_fineweb_buffer(tokenizer, smoke=True)
        val_buffer = fw_buffer[:200]
        fw_buffer = fw_buffer[200:]
    else:
        fw_buffer, val_buffer = build_fineweb_buffer(tokenizer, smoke=False)

    binding_pool, s1 = tokenize_pool(binding_examples, tokenizer)
    existing_pool, s2 = tokenize_pool(existing_examples, tokenizer)
    instruction_pool, s3 = tokenize_pool(instruction_examples, tokenizer)
    arith_pool, s4 = tokenize_pool(arith_examples, tokenizer)
    print(f"  usable: binding {len(binding_pool)} (skip {s1}), existing {len(existing_pool)} (skip {s2}), "
          f"instruction {len(instruction_pool)} (skip {s3}), arithmetic {len(arith_pool)} (skip {s4})")

    # ---------------- ROW ALLOCATION ----------------
    def rows_for(ratio):
        return max(1, round(BATCH_SIZE * ratio))

    n_binding = rows_for(args.sb)
    n_inst = rows_for(args.inst)
    n_sem = rows_for(args.sem)
    n_fw = rows_for(args.fw)
    n_arith = rows_for(args.arith)
    # normalize so total == BATCH_SIZE
    total = n_binding + n_inst + n_sem + n_fw + n_arith
    scale = BATCH_SIZE / total
    n_binding = max(1, round(n_binding * scale))
    n_inst = max(1, round(n_inst * scale))
    n_sem = max(1, round(n_sem * scale))
    n_fw = max(1, round(n_fw * scale))
    n_arith = max(0, round(n_arith * scale))
    while n_binding + n_inst + n_sem + n_fw + n_arith > BATCH_SIZE:
        n_arith = max(0, n_arith - 1)
        n_fw = max(1, n_fw - 1) if n_fw > 1 else n_fw
        n_inst = max(1, n_inst - 1) if n_inst > 1 else n_inst
    while n_binding + n_inst + n_sem + n_fw + n_arith < BATCH_SIZE:
        n_binding += 1
    print(f"Rows per batch: binding {n_binding} | instruction {n_inst} | "
          f"existing {n_sem} | FineWeb {n_fw} | arithmetic {n_arith}")

    def get_batch():
        xs, ys, masks = [], [], []
        pools = [(n_binding, binding_pool), (n_inst, instruction_pool),
                 (n_sem, existing_pool), (n_arith, arith_pool)]
        for count, pool in pools:
            for _ in range(count):
                ids, mask = make_row(pool, SEQ_LEN, rng)
                xs.append(ids[:-1]); ys.append(ids[1:]); masks.append(mask[:-1])
        for _ in range(n_fw):
            window = make_fineweb_row(fw_buffer, SEQ_LEN)
            if window is None or len(window) < SEQ_LEN + 1:
                ids, mask = make_row(binding_pool, SEQ_LEN, rng)
                xs.append(ids[:-1]); ys.append(ids[1:]); masks.append(mask[:-1])
            else:
                xs.append(window[:-1]); ys.append(window[1:]); masks.append([1] * SEQ_LEN)
        return (torch.tensor(xs, dtype=torch.long, device=device),
                torch.tensor(ys, dtype=torch.long, device=device),
                torch.tensor(masks, dtype=torch.float32, device=device))

    # ---------------- OPTIMIZER ----------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer,
                         lambda s: cosine_schedule(s, args.steps, WARMUP_STEPS, args.lr))
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    def save_ckpt(step, loss_val, val_loss_value):
        data = {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "py_rng": random.getstate(),
            "config": config.__dict__,
            "loss": float(loss_val),
            "val_loss": None if val_loss_value is None else float(val_loss_value),
            "base_checkpoint": args.base,
            "training_type": "semv2e",
            "mixture": {
                "semantic_binding": args.sb,
                "instruction": args.inst,
                "existing_semantic": args.sem,
                "fineweb": args.fw,
                "arithmetic": args.arith,
            },
            "peak_lr": args.lr,
            "scheduler": f"warmup{WARMUP_STEPS}+cosine",
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "seed": args.seed,
            "params": params,
        }
        torch.save(data, os.path.join(SEMV2E_DIR, f"checkpoint_{args.name}-{step}.pt"))
        torch.save(data, os.path.join(SEMV2E_DIR, f"checkpoint_{args.name}-latest.pt"))

    # ---------------- TRAIN ----------------
    model.train()
    start = time.time()
    val_loss_value = None
    print("=" * 70)
    print("STARTING SEMV2E TRAINING")
    print("=" * 70)

    for step in range(1, args.steps + 1):
        t0 = time.time()
        x, y, m = get_batch()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss_per = F.cross_entropy(logits.view(-1, config.vocab_size),
                                   y.view(-1), reduction="none")
        loss = (loss_per * m.view(-1)).sum() / m.view(-1).sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            print(f"Step {step:5d}/{args.steps} | loss {loss.item():.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {BATCH_SIZE*SEQ_LEN/dt:,.0f} tok/s",
                  flush=True)

        if step % 250 == 0:
            val_loss_value = val_loss(model, val_buffer, device, SEQ_LEN, config)
            print(f"  [val] fineweb held-out: {val_loss_value:.4f}", flush=True)

        if step in MILESTONES or step == args.steps:
            save_ckpt(step, loss.item(), val_loss_value)
            print(f"  -> saved milestone step {step}")

    if args.steps not in MILESTONES:
        save_ckpt(args.steps, loss.item(), val_loss_value)

    total_tokens = args.steps * BATCH_SIZE * SEQ_LEN
    print()
    print("=" * 70)
    print("SEMV2E TRAINING COMPLETE")
    print("=" * 70)
    print(f"Base     : {args.base} (step {base_step})")
    print(f"Steps    : {args.steps}")
    print(f"Final loss: {loss.item():.4f} | val {val_loss_value}")
    print(f"Tokens   : {total_tokens:,} in {time.time()-start:.1f}s")
    print(f"Checkpoints in {SEMV2E_DIR}")


if __name__ == "__main__":
    main()