#!/usr/bin/env python3
"""
SEMANTIC V2D TRAINING: Controlled improvement experiment
================================================================

Continues from the current best checkpoint and adds targeted data types
to improve previously weak categories while preserving strong ones.

Data mixture (per batch of 16 rows):
  50% general semantic/instruction/context (preserves relations, state,
  transitive, negation, context retention)
  20% arithmetic (new: addition, subtraction, multiplication, word problems)
  15% sun rise/set (new: balanced east/west paraphrases)
  15% existing FineWeb preservation data

Primary goals:
  1. Investigate whether the model can learn arithmetic.
  2. Fix sunrise/sunset directional knowledge (especially sun SET → west).
  3. Improve plain-QA behavior (reduce chat-format dependency).
  4. Preserve: relations, context retention, transitive reasoning,
     instruction following, negation, paraphrase/topic relevance.

The objective is NOT merely to increase training loss performance.
The objective is to produce a model that performs better on genuinely
held-out semantic tests.
"""

import os
import sys
import time
import json
import argparse
import random
import math

# Ensure uv Python packages are findable (torch etc.)
uv_base = r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none"
if uv_base not in sys.path:
    sys.path.insert(0, uv_base)
if uv_base + r"\Lib\site-packages" not in sys.path:
    sys.path.insert(0, uv_base + r"\Lib\site-packages")

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import (
    CHECKPOINT_BEST,
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
DEFAULT_MAX_STEPS = 5000
WARMUP_STEPS = 150
PEAK_LR = 1e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Mixture percentages (used when not in smoke mode)
SEMANTIC_RATIO = 0.50   # general semantic/context/relations
ARITHMETIC_RATIO = 0.20  # arithmetic (addition, sub, multiplication, word problems)
SUN_RATIO = 0.15        # sun rise/set balanced
FINEWEB_RATIO = 0.15    # real-language preservation

ASSISTANT_TOKEN = 11326        # " Assistant" byte-level token id
EOS_ID = 3


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def cosine_schedule(step, max_steps, warmup, peak):
    """Warmup + cosine decay to 10% of peak."""
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


# ------------------------------------------------------------
# ARITHMETIC DATA GENERATION
# ------------------------------------------------------------
def generate_arithmetic_examples(n):
    """Generate unique arithmetic examples (not repeated from training)."""
    tests = []
    name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
    obj_opts = ["apples", "books", "balls", "marbles", "coins"]
    seen_pairs = set()
    seen_questions = set()

    while len(tests) < n:
        # Direct addition
        if len(tests) < n and len(seen_pairs) < n * 2:
            a = random.randint(10, 89)
            b = random.randint(10, 89)
            key = (min(a, b), max(a, b))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            name = random.choice(name_opts)
            question = f"What is {a} + {b}?\nAssistant:"
            answer = str(a + b)
            if question not in seen_questions:
                seen_questions.add(question)
                tests.append((question, answer, "addition"))

        # Direct subtraction
        if len(tests) < n:
            a = random.randint(20, 98)
            b = random.randint(10, a - 1)
            key = (b, a)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            name = random.choice(name_opts)
            question = f"What is {a} - {b}?\nAssistant:"
            answer = str(a - b)
            if question not in seen_questions:
                seen_questions.add(question)
                tests.append((question, answer, "subtraction"))

        # Direct multiplication
        if len(tests) < n:
            a = random.randint(2, 9)
            b = random.randint(2, 9)
            key = (a, b)
            prod = a * b
            if prod > 100 or key in seen_pairs:
                continue
            seen_pairs.add(key)
            name = random.choice(name_opts)
            question = f"What is {a} * {b}?\nAssistant:"
            answer = str(prod)
            if question not in seen_questions:
                seen_questions.add(question)
                tests.append((question, answer, "multiplication"))

        # Word problem addition
        if len(tests) < n:
            a = random.randint(2, 20)
            b = random.randint(2, 20)
            name = random.choice(name_opts)
            obj = random.choice(obj_opts)
            question = f"{name} has {a} {obj}. They receive {b} more {obj}. How many {obj} does {name} have now?\nAssistant:"
            answer = str(a + b)
            if question not in seen_questions:
                seen_questions.add(question)
                tests.append((question, answer, "word_problem_addition"))

        # Word problem subtraction
        if len(tests) < n:
            a = random.randint(5, 30)
            b = random.randint(1, a - 1)
            name = random.choice(name_opts)
            obj = random.choice(obj_opts)
            question = f"{name} had {a} {obj}. They gave away {b} {obj}. How many {obj} does {name} have left?\nAssistant:"
            answer = str(a - b)
            if question not in seen_questions:
                seen_questions.add(question)
                tests.append((question, answer, "word_problem_subtraction"))

    return tests


# ------------------------------------------------------------
# SUN DATA GENERATION
# ------------------------------------------------------------
def generate_sun_examples(n):
    """Generate balanced sun rise/set examples."""
    tests = []
    rise_expected = "east"
    set_expected = "west"

    rise_paraphrases = [
        "The sun rises in the east.",
        "The sun comes up in the east.",
        "Sunrise occurs in the east.",
        "In the morning, the sun appears in the east.",
        "Dawn breaks in the east.",
    ]

    set_paraphrases = [
        "The sun sets in the west.",
        "The sun goes down in the west.",
        "Sunset occurs in the west.",
        "In the evening, the sun disappears in the west.",
        "Dusk falls in the west.",
    ]

    # Add rise examples
    while len(tests) < n and len([t for t in tests if t[2] == "rise"]) < n // 2:
        para = random.choice(rise_paraphrases)
        if para not in [t[0] for t in tests]:
            tests.append((para, rise_expected, "rise"))

    # Add set examples
    while len(tests) < n and len([t for t in tests if t[2] == "set"]) < n // 2:
        para = random.choice(set_paraphrases)
        if para not in [t[0] for t in tests]:
            tests.append((para, set_expected, "set"))

    # If we need more, add variety
    rise_variants = [
        "Where does the sun rise?",
        "Where does sunrise occur?",
        "In which direction does the sun come up?",
        "Where does the sun come up in the morning?",
        "Sunrise occurs in which direction?",
        "Where does the sun appear at dawn?",
        "What direction is sunrise?",
    ]
    set_variants = [
        "Where does the sun set?",
        "Where does sunset occur?",
        "In which direction does the sun go down?",
        "Where does the sun go down in the evening?",
        "Sunset occurs in which direction?",
        "Where does the sun appear at dusk?",
        "What direction is sunset?",
    ]

    while len(tests) < n:
        if len([t for t in tests if t[2] == "rise"]) < n:
            para = random.choice(rise_variants)
            if para not in [t[0] for t in tests]:
                tests.append((para, rise_expected, "rise"))
        if len([t for t in tests if t[2] == "set"]) < n:
            para = random.choice(set_variants)
            if para not in [t[0] for t in tests]:
                tests.append((para, set_expected, "set"))
        if len(tests) >= n:
            break

    return tests


# ------------------------------------------------------------
# SEMANTIC DATA (from existing generator)
# ------------------------------------------------------------
def load_semantic_data(n_examples):
    """Load procedural semantic + instruction examples."""
    gen = SemanticGenerator(seed=20260812)
    proc_examples = gen.generate(n_examples // 2, seed=20260812)
    alpaca_examples = load_instruction_subset(n=n_examples // 2)
    all_examples = proc_examples + alpaca_examples
    return all_examples[:n_examples]


# ------------------------------------------------------------
# FINEWEB STREAM
# ------------------------------------------------------------
def load_fineweb_stream(buffer_size=50000):
    """Load FineWeb stream for preservation data.
    
    For smoke test, uses a small preset buffer of synthetic text.
    """
    from tokenizers import Tokenizer

    # Use preset buffer for smoke test to avoid network dependency
    if buffer_size <= 1000:
        # Preset: short diverse text snippets
        preset_texts = [
            "The cat sat on the mat .",
            "Sajan went to the store .",
            "Ravi bought a blue book .",
            "Priya read a story .",
            "Tom threw the ball .",
            "Nina baked a cake .",
            "Leo painted a picture .",
            "Zoe wrote a letter .",
            "The dog ran fast .",
            "The sun is bright .",
        ]
        tok = Tokenizer.from_file(PATHS_TOKENIZER)
        ids = []
        for text in preset_texts:
            encoded = tok.encode(text)
            ids.extend(encoded.ids)
        # Truncate/pad to buffer_size
        if len(ids) < buffer_size:
            ids.extend([0] * (buffer_size - len(ids)))
        else:
            ids = ids[:buffer_size]
        return ids

    from datasets import load_dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="CC-MAIN-2025-26",
        split="train", streaming=True
    )
    fw_iter = iter(dataset)
    buffer = []

    def fill(min_tokens):
        while len(buffer) < min_tokens:
            doc = next(fw_iter)
            text = doc.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            if len(ids) >= 2:
                buffer.extend(ids)

    fill(buffer_size)
    return buffer


# ------------------------------------------------------------
# DATA MIXTURE: pack into batches
# ------------------------------------------------------------
def make_semantic_row(semantic_pool, tokenizer, SEQ_LEN):
    """Pack complete semantic examples with assistant masking."""
    ids = []
    mask = []
    while len(ids) <= SEQ_LEN:
        ex_idx = random.randrange(len(semantic_pool))
        ex_ids, ex_mask = semantic_pool[ex_idx]
        if len(ids) + len(ex_ids) > SEQ_LEN + 1:
            break
        ids.extend(ex_ids)
        mask.extend(ex_mask)
    if len(ids) < SEQ_LEN + 1:
        pad = SEQ_LEN + 1 - len(ids)
        ids.extend([0] * pad)
        mask.extend([0] * pad)
    # Assistant-only loss mask: find "Assistant:" and mask from there
    start = find_assistant_start(ids, tokenizer)
    m = [0] * len(ids)
    if start is not None:
        for i in range(start, len(ids)):
            m[i] = 1
    # Append EOS token
    ids = ids + [EOS_ID]
    m = m + [1]
    if len(ids) > SEQ_LEN + 1:
        ids = ids[:SEQ_LEN + 1]
        m = m[:SEQ_LEN + 1]
    return ids, m


def make_arithmetic_row(tokenizer, arithmetic_examples, SEQ_LEN):
    """Create a batch row from arithmetic examples."""
    question, answer, atype = random.choice(arithmetic_examples)
    ids = tokenizer.encode(question).ids
    # Truncate if too long
    if len(ids) > SEQ_LEN:
        ids = ids[:SEQ_LEN]
    # Create mask: 1s from some point onward (assistant-only)
    max_len = min(len(ids) + len(answer) + 2, SEQ_LEN + 1)
    ids.extend([0] * (max_len - len(ids)))
    # Question IDs length
    question_ids_len = len(tokenizer.encode(question).ids)
    mask = [0] * max_len
    for i in range(question_ids_len, max_len):
        mask[i] = 1
    # Append EOS
    ids.append(EOS_ID)
    mask.append(1)
    if len(ids) > SEQ_LEN + 1:
        ids = ids[:SEQ_LEN + 1]
        mask = mask[:SEQ_LEN + 1]
    return ids, mask


def make_sun_row(tokenizer, sun_examples, SEQ_LEN):
    """Create a batch row from sun examples."""
    text, expected_dir, sun_type = random.choice(sun_examples)
    ids = tokenizer.encode(text).ids
    max_len = min(len(ids) + 3, SEQ_LEN + 1)
    ids.extend([0] * (max_len - len(ids)))
    # Mask from middle onward
    split_point = max(1, max_len // 2)
    mask = [0] * split_point + [1] * (max_len - split_point)
    # Append EOS
    ids.append(EOS_ID)
    mask.append(1)
    if len(ids) > SEQ_LEN + 1:
        ids = ids[:SEQ_LEN + 1]
        mask = mask[:SEQ_LEN + 1]
    return ids, mask


def make_fineweb_row(fw_buffer, tokenizer, SEQ_LEN):
    """Create a FineWeb row from the buffer."""
    if len(fw_buffer) < SEQ_LEN + 1:
        return None
    window = fw_buffer[:SEQ_LEN + 1]
    del fw_buffer[:SEQ_LEN]
    return window


def get_batch_v2d(tokenizer, semantic_pool, arithmetic_examples,
                  sun_examples, fw_buffer, SEQ_LEN,
                  semantic_ratio=SEMANTIC_RATIO,
                  arithmetic_ratio=ARITHMETIC_RATIO,
                  sun_ratio=SUN_RATIO):
    """Construct a batch with the v2d mixture."""
    # Calculate rows per category
    semantic_rows = max(1, round(BATCH_SIZE * semantic_ratio))
    arithmetic_rows = max(1, round(BATCH_SIZE * arithmetic_ratio))
    sun_rows = max(1, round(BATCH_SIZE * sun_ratio))
    fineweb_rows = BATCH_SIZE - semantic_rows - arithmetic_rows - sun_rows

    xs, ys, masks = [], [], []

    # Semantic rows
    for _ in range(semantic_rows):
        ids, mask = make_semantic_row(semantic_pool, tokenizer, SEQ_LEN)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(mask[:-1])

    # Arithmetic rows
    for _ in range(arithmetic_rows):
        ids, mask = make_arithmetic_row(tokenizer, arithmetic_examples, SEQ_LEN)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(mask[:-1])

    # Sun rows
    for _ in range(sun_rows):
        ids, mask = make_sun_row(tokenizer, sun_examples, SEQ_LEN)
        xs.append(ids[:-1])
        ys.append(ids[1:])
        masks.append(mask[:-1])

    # FineWeb rows
    for _ in range(max(0, fineweb_rows)):
        window = make_fineweb_row(fw_buffer, tokenizer, SEQ_LEN)
        if window is not None:
            xs.append(window[:-1])
            ys.append(window[1:])
            masks.append([1] * SEQ_LEN)
        else:
            # fallback: repeat last valid row
            if xs:
                last_x = xs[-1]
                xs.append(last_x)
                ys.append(last_x[1:] + [last_x[-1]] if len(last_x) > 1 else [0])
                masks.append([1] * SEQ_LEN)

    x = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    m = torch.tensor(masks, dtype=torch.float32, device=DEVICE)
    return x, y, m


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def main():
    DEVICE = torch.device("cpu")  # CPU for training consistency
    parser = argparse.ArgumentParser(description="SEMANTIC V2D training")
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=PEAK_LR,
                        help="Peak learning rate")
    parser.add_argument("--base", default=CHECKPOINT_BEST,
                        help="Parent checkpoint to continue from")
    parser.add_argument("--seed", type=int, default=20260812,
                        help="Random seed")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (10 steps) to verify the pipeline")
    parser.add_argument("--name", default="semv2d",
                        help="Experiment name / checkpoint prefix")
    parser.add_argument("--semantic-ratio", type=float, default=SEMANTIC_RATIO,
                        help="Fraction of rows that are general semantic")
    parser.add_argument("--arithmetic-ratio", type=float, default=ARITHMETIC_RATIO,
                        help="Fraction of rows that are arithmetic")
    parser.add_argument("--sun-ratio", type=float, default=SUN_RATIO,
                        help="Fraction of rows that are sun facts")
    args = parser.parse_args()

    # ------------------------------------------------------------
    # SMOKE TEST OVERRIDES (inside main, after arg parsing)
    # ------------------------------------------------------------
    if args.smoke:
        args.steps = 10
        args.semantic_ratio = 0.7
        args.arithmetic_ratio = 0.1
        args.sun_ratio = 0.1
        fineweb_buffer_size = 100  # tiny preset buffer
    else:
        fineweb_buffer_size = 50000

    # ------------------------------------------------------------
    # PRINT CONFIGURATION
    # ------------------------------------------------------------
    print("=" * 70)
    print(f"SEMANTIC V2D TRAINING: Experiment '{args.name}'")
    print("=" * 70)
    print(f"Base checkpoint : {args.base}")
    print(f"Steps           : {args.steps}")
    print(f"Peak LR         : {args.lr}  (warmup {WARMUP_STEPS} + cosine to 10%)")
    print(f"Mixture         : {args.semantic_ratio*100:.1f}% semantic, "
          f"{args.arithmetic_ratio*100:.1f}% arithmetic, "
          f"{args.sun_ratio*100:.1f}% sun, "
          f"{1.0 - args.semantic_ratio - args.arithmetic_ratio - args.sun_ratio:.1f}% FineWeb")
    print()

    # ------------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------------
    os.makedirs(SEMANTIC_V2_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LATEST_CHECKPOINT), exist_ok=True)

    tokenizer = Tokenizer.from_file(PATHS_TOKENIZER)
    config = ModelConfig()
    model = SmallEnglishLLM(config).to(DEVICE)

    # Load base checkpoint
    if not os.path.exists(args.base):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base}")
    ckpt = torch.load(args.base, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    base_step = ckpt.get("step", "unknown")
    print(f"Loaded base checkpoint (step {base_step})")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # ------------------------------------------------------------
    # GENERATE/LOAD DATA
    # ------------------------------------------------------------
    print("Generating arithmetic examples...")
    arithmetic_examples = generate_arithmetic_examples(500)
    print(f"  {len(arithmetic_examples)} arithmetic examples")

    print("Generating sun examples...")
    sun_examples = generate_sun_examples(200)
    print(f"  {len(sun_examples)} sun examples")

    print("Loading semantic+instruction data...")
    semantic_data = load_semantic_data(2000)
    print(f"  {len(semantic_data)} semantic examples")

    print("Loading FineWeb buffer...")
    fw_buffer = load_fineweb_stream(buffer_size=fineweb_buffer_size)
    print(f"  FineWeb buffer: {len(fw_buffer)} tokens")

    # ------------------------------------------------------------
    # OPTIMIZER + SCHEDULER + SCALER
    # ------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lambda s: cosine_schedule(s, args.steps, WARMUP_STEPS, args.lr))
    scaler = GradScaler("cuda", enabled=False)  # Disable for CPU

    # Resume from latest if available (but don't overwrite best)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + start_step)

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    model.train()
    start_time = time.time()
    val_loss_value = None

    print("=" * 70)
    print("STARTING SEMANTIC V2D TRAINING")
    print("=" * 70)

    for step in range(start_step + 1, args.steps + 1):
        step_start = time.time()

        x, y, m = get_batch_v2d(
            tokenizer, semantic_data, arithmetic_examples, sun_examples,
            fw_buffer, SEQ_LEN,
            semantic_ratio=args.semantic_ratio,
            arithmetic_ratio=args.arithmetic_ratio,
            sun_ratio=args.sun_ratio,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            logits, _ = model(x)
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

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - step_start
            tok_s = (BATCH_SIZE * SEQ_LEN) / elapsed
            vram = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Step {step:5d}/{args.steps} | loss {loss.item():.4f} | "
                  f"lr {lr_now:.2e} | {tok_s:,.0f} tok/s | VRAM {vram:.2f} GB",
                  flush=True)

        if step % 100 == 0:
            # Save milestone checkpoint
            ckpt_path = os.path.join(SEMANTIC_V2_DIR,
                                       f"checkpoint_{args.name}-{step}.pt")
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
                "loss": float(loss.item()),
                "base_checkpoint": args.base,
                "training_type": "semantic_v2d",
                "semantic_ratio": args.semantic_ratio,
                "arithmetic_ratio": args.arithmetic_ratio,
                "sun_ratio": args.sun_ratio,
                "peak_lr": args.lr,
                "scheduler": f"warmup{WARMUP_STEPS}+cosine",
                "procedural_examples": len(arithmetic_examples),
                "sun_examples": len(sun_examples),
            }
            torch.save(data, ckpt_path)
            print(f"  -> Milestone checkpoint: {ckpt_path}")

        # Simple validation: check loss is NaN
        if torch.isnan(loss):
            print(f"  WARNING: Loss became NaN at step {step}, stopping.")
            break

    # Save final checkpoint
    final_ckpt = os.path.join(SEMANTIC_V2_DIR, f"checkpoint_{args.name}-final.pt")
    data = {
        "step": int(args.steps),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "py_rng": random.getstate(),
        "config": config.__dict__,
        "loss": float(loss.item()),
        "base_checkpoint": args.base,
        "training_type": "semantic_v2d",
        "semantic_ratio": args.semantic_ratio,
        "arithmetic_ratio": args.arithmetic_ratio,
        "sun_ratio": args.sun_ratio,
        "peak_lr": args.lr,
        "scheduler": f"warmup{WARMUP_STEPS}+cosine",
        "procedural_examples": len(arithmetic_examples),
        "sun_examples": len(sun_examples),
    }
    torch.save(data, final_ckpt)
    print(f"  -> Final checkpoint: {final_ckpt}")

    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("SEMANTIC V2D TRAINING COMPLETE")
    print("=" * 70)
    print(f"Base: {args.base} (step {base_step})")
    print(f"Steps this run: {args.steps - start_step}")
    print(f"Final loss: {loss.item():.4f}")
    total_tokens = (args.steps - start_step) * BATCH_SIZE * SEQ_LEN
    smoke_str = "smoke test" if args.smoke else f"{total_time/60:.2f} min"
    print(f"Tokens: {total_tokens:,} in {smoke_str}")
    print(f"Checkpoint: {LATEST_CHECKPOINT}")


if __name__ == "__main__":
    main()