import os
import sys
import time
import torch
from torch.optim import AdamW

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import pretrain_checkpoint

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# EXPERIMENT 1 — TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16
SEQ_LEN = 512

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

SMOKE_TEST_STEPS = 100

LOG_EVERY = 10

CHECKPOINT_PATH = pretrain_checkpoint("smoke_test")


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Install the CUDA-enabled PyTorch build first."
    )

device = torch.device("cuda")

print("=" * 60)
print("EXPERIMENT 1 — TRAINING SMOKE TEST")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# MODEL
# ============================================================

config = ModelConfig()

model = SmallEnglishLLM(config).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Parameters: {parameter_count:,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Tokens per step: {BATCH_SIZE * SEQ_LEN:,}")
print()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TEMPORARY DATA SOURCE
# ============================================================
#
# IMPORTANT:
# This is ONLY for the smoke test.
#
# We use random token sequences here to verify that the
# training machinery works.
#
# After this succeeds, we replace this section with the
# FineWeb streaming + tokenizer pipeline you already built.
# ============================================================

def get_batch():

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (BATCH_SIZE, SEQ_LEN),
        dtype=torch.long
    )

    target_ids = torch.randint(
        0,
        config.vocab_size,
        (BATCH_SIZE, SEQ_LEN),
        dtype=torch.long
    )

    input_ids = input_ids.to(device, non_blocking=True)
    target_ids = target_ids.to(device, non_blocking=True)

    return input_ids, target_ids


# ============================================================
# TRAINING LOOP
# ============================================================

model.train()

start_time = time.time()

for step in range(1, SMOKE_TEST_STEPS + 1):

    step_start = time.time()

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    input_ids, target_ids = get_batch()

    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = model(
        input_ids,
        target_ids
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if step % LOG_EVERY == 0 or step == 1:

        torch.cuda.synchronize()

        elapsed = time.time() - step_start

        tokens_per_second = (
            BATCH_SIZE
            * SEQ_LEN
            / elapsed
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Step {step:4d}/{SMOKE_TEST_STEPS} | "
            f"Loss {loss.item():.4f} | "
            f"{tokens_per_second:,.0f} tok/s | "
            f"VRAM {peak_vram:.2f} GB"
        )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

checkpoint = {
    "step": SMOKE_TEST_STEPS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config.__dict__,
    "loss": loss.item(),
}

torch.save(
    checkpoint,
    CHECKPOINT_PATH
)

total_time = time.time() - start_time

print()
print("=" * 60)
print("SMOKE TEST COMPLETE")
print("=" * 60)

print(f"Final loss: {loss.item():.4f}")
print(f"Time: {total_time:.2f} seconds")
print(f"Checkpoint: {CHECKPOINT_PATH}")

print()
print("Next step:")
print("Replace the temporary random-data source with")
print("the FineWeb streaming + tokenizer pipeline.")
print("=" * 60)