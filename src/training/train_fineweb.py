# train_fineweb.py
#
# ============================================================
# EXPERIMENT 1 — REAL FINEWEB TRAINING
# ============================================================
#
# Pipeline:
#
# FineWeb stream
#      ↓
# tokenizer
#      ↓
# token buffer
#      ↓
# 513-token sequences
#      ↓
# input / target pairs of 512 tokens
#      ↓
# batch of 16
#      ↓
# 14.66M parameter model
#      ↓
# loss
#      ↓
# backward
#      ↓
# AdamW
#
# IMPORTANT:
# - FineWeb is streamed.
# - The complete dataset is NOT downloaded into RAM.
# - checkpoint_fineweb.pt is always the latest checkpoint.
# - Permanent checkpoint copies are never overwritten.
#
# Example:
#
# checkpoint_fineweb.pt
# checkpoint-5k.pt
# checkpoint-10k.pt
# checkpoint-15k.pt
# checkpoint-20k.pt
#
# If the same milestone is saved again:
#
# checkpoint-10k.pt
# checkpoint-10k-1.pt
# checkpoint-10k-2.pt
#
# ============================================================


import os
import time

import torch
from torch.optim import AdamW
from datasets import load_dataset
from tokenizers import Tokenizer

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

BATCH_SIZE = 16
SEQ_LEN = 512

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

# Change this for the experiment.
#
# Example:
# 5000  = 5k steps
# 10000 = 10k steps
# 20000 = 20k steps
#
MAX_STEPS = 50000

LOG_EVERY = 10

# Save checkpoint every N steps.
#
# 10000 means:
# 10000, 20000, 30000, 40000, 50000
CHECKPOINT_EVERY = 10000

# Latest/current checkpoint.
#
# This file is intentionally overwritten.
CHECKPOINT_PATH = "checkpoint_fineweb.pt"

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "CC-MAIN-2025-26"

TOKENIZER_PATH = "tokenizer.json"


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available.\n"
        "Make sure the CUDA-enabled PyTorch installation is active."
    )

device = torch.device("cuda")

print("=" * 70)
print("EXPERIMENT 1 — REAL FINEWEB TRAINING")
print("=" * 70)

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

print()


# ============================================================
# TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

tokenizer_vocab_size = tokenizer.get_vocab_size()

print(
    "Tokenizer vocabulary:",
    tokenizer_vocab_size
)

print()


# ============================================================
# MODEL
# ============================================================

config = ModelConfig()

model = SmallEnglishLLM(
    config
).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "Model parameters:",
    f"{parameter_count:,}"
)

print(
    "Model parameters:",
    f"{parameter_count / 1_000_000:.2f}M"
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Sequence length:",
    SEQ_LEN
)

print(
    "Tokens per optimizer step:",
    f"{BATCH_SIZE * SEQ_LEN:,}"
)

print()


# ============================================================
# TOKENIZER / MODEL VOCABULARY CHECK
# ============================================================

if tokenizer_vocab_size != config.vocab_size:

    raise RuntimeError(
        "\nTokenizer/model vocabulary mismatch!\n\n"
        f"Tokenizer vocabulary: {tokenizer_vocab_size:,}\n"
        f"Model vocabulary:     {config.vocab_size:,}\n\n"
        "These must be identical before training."
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# OPTIONAL RESUME
# ============================================================
#
# IMPORTANT:
#
# If checkpoint_fineweb.pt exists, this script CONTINUES
# training from that checkpoint instead of creating a new
# model from scratch.
#
# This is what makes:
#
# 5k → 10k → 20k
#
# one continuous training run.
#
# ============================================================

start_step = 0

if os.path.exists(CHECKPOINT_PATH):

    print("=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Loading:",
        CHECKPOINT_PATH
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    # --------------------------------------------------------
    # Load model weights
    # --------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Load optimizer state
    # --------------------------------------------------------

    if "optimizer_state_dict" in checkpoint:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # --------------------------------------------------------
    # Continue from previous step
    # --------------------------------------------------------

    start_step = int(
        checkpoint.get("step", 0)
    )

    previous_loss = checkpoint.get(
        "loss",
        None
    )

    print(
        "Previous training step:",
        start_step
    )

    if previous_loss is not None:

        print(
            "Previous loss:",
            f"{previous_loss:.4f}"
        )

    print(
        "Continuing training..."
    )

    print()

else:

    print("=" * 70)
    print("NO CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Starting from a randomly initialized model."
    )

    print()


# ============================================================
# OPEN FINEWEB STREAM
# ============================================================

print("=" * 70)
print("OPENING FINEWEB STREAM")
print("=" * 70)

print(
    "Dataset:",
    DATASET
)

print(
    "Configuration:",
    CONFIG
)

print()

dataset = load_dataset(
    DATASET,
    name=CONFIG,
    split="train",
    streaming=True
)

# IterableDataset is iterable but is not itself an iterator.
#
# Therefore:
#
# iter(dataset)
#
# creates the iterator that can be used with next().

dataset_iterator = iter(dataset)

print(
    "FineWeb stream opened successfully."
)

print()


# ============================================================
# TOKEN BUFFER
# ============================================================

# Documents are streamed one at a time.
#
# document 1 → tokenize → buffer
# document 2 → tokenize → buffer
# document 3 → tokenize → buffer
#
# Once enough tokens exist:
#
# 513 tokens
#
# become:
#
# input  = tokens[0:512]
# target = tokens[1:513]
#
# This is next-token prediction.

token_buffer = []


def fill_token_buffer(min_tokens):
    """
    Continue streaming documents until the token buffer
    contains at least min_tokens tokens.
    """

    while len(token_buffer) < min_tokens:

        document = next(
            dataset_iterator
        )

        text = document.get(
            "text",
            ""
        )

        if not text:
            continue

        # ----------------------------------------------------
        # Tokenize document
        # ----------------------------------------------------

        encoded = tokenizer.encode(
            text
        )

        ids = encoded.ids

        # Ignore extremely tiny documents
        if len(ids) < 2:
            continue

        # Add tokens to rolling buffer
        token_buffer.extend(ids)


def get_batch():
    """
    Create one batch of next-token prediction examples.

    Returns:

        input_ids:
            [BATCH_SIZE, SEQ_LEN]

        target_ids:
            [BATCH_SIZE, SEQ_LEN]
    """

    # Each training example requires:
    #
    # 512 input tokens
    # +
    # 1 token for shifted target

    tokens_per_example = (
        SEQ_LEN + 1
    )

    required_tokens = (
        BATCH_SIZE
        * tokens_per_example
    )

    fill_token_buffer(
        required_tokens
    )

    input_batch = []
    target_batch = []

    for _ in range(BATCH_SIZE):

        # Take 513 consecutive tokens
        sequence = token_buffer[
            :SEQ_LEN + 1
        ]

        # Remove them from rolling buffer
        del token_buffer[
            :SEQ_LEN + 1
        ]

        # Next-token prediction

        input_ids = sequence[
            :-1
        ]

        target_ids = sequence[
            1:
        ]

        input_batch.append(
            input_ids
        )

        target_batch.append(
            target_ids
        )

    # Convert Python lists → PyTorch tensors

    input_ids = torch.tensor(
        input_batch,
        dtype=torch.long,
        device=device
    )

    target_ids = torch.tensor(
        target_batch,
        dtype=torch.long,
        device=device
    )

    return (
        input_ids,
        target_ids
    )


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    step,
    loss_value,
    archive=True
):
    """
    Save the current training state.

    Always saves:
        checkpoint_fineweb.pt

    If archive=True, also creates a permanent copy ONLY at specific milestones:
        checkpoint-10k.pt
        checkpoint-20k.pt
        checkpoint-50k.pt
        checkpoint-100k.pt
        etc. (every 10k steps)

    Existing archive files are NEVER overwritten.
    """

    checkpoint = {
        "step":
            step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "config":
            config.__dict__,

        "loss":
            loss_value,
    }

    # --------------------------------------------------------
    # 1. Save latest checkpoint (overwrites each time)
    # --------------------------------------------------------

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print()
    print(
        f"Latest checkpoint saved: "
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # 2. Permanent archive ONLY at 10k intervals
    # --------------------------------------------------------

    if archive and step % 10000 == 0:

        step_k = step // 1000

        base_name = (
            f"checkpoint-{step_k}k"
        )

        archive_name = (
            f"{base_name}.pt"
        )

        counter = 0

        # If file already exists:
        #
        # checkpoint-10k.pt
        #
        # try:
        #
        # checkpoint-10k-1.pt
        # checkpoint-10k-2.pt
        # ...

        while os.path.exists(
            archive_name
        ):

            counter += 1

            archive_name = (
                f"{base_name}-{counter}.pt"
            )

        torch.save(
            checkpoint,
            archive_name
        )

        print(
            f"Permanent archive saved: "
            f"{archive_name}"
        )

    print()


# ============================================================
# TRAINING
# ============================================================

model.train()

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    f"Starting step: {start_step}"
)

print(
    f"Target step:   {MAX_STEPS}"
)

print(
    f"Remaining:     {max(0, MAX_STEPS - start_step)}"
)

print()


# If checkpoint is already at or beyond target,
# don't accidentally train backwards.

if start_step >= MAX_STEPS:

    print(
        "Checkpoint is already at or beyond "
        "MAX_STEPS."
    )

    print(
        "Increase MAX_STEPS if you want to "
        "continue training."
    )

    raise SystemExit


start_time = time.time()

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

for step in range(
    start_step + 1,
    MAX_STEPS + 1
):

    step_start = time.time()

    # --------------------------------------------------------
    # GET REAL ENGLISH BATCH
    # --------------------------------------------------------

    input_ids, target_ids = get_batch()

    # --------------------------------------------------------
    # CLEAR GRADIENTS
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    logits, loss = model(
        input_ids,
        target_ids
    )

    # --------------------------------------------------------
    # BACKWARD PASS
    # --------------------------------------------------------

    loss.backward()

    # --------------------------------------------------------
    # OPTIMIZER UPDATE
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if (
        step % LOG_EVERY == 0
        or step == start_step + 1
    ):

        torch.cuda.synchronize()

        step_time = (
            time.time()
            - step_start
        )

        tokens_per_second = (
            BATCH_SIZE
            * SEQ_LEN
            / step_time
        )

        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Step {step:5d}/{MAX_STEPS} | "
            f"Loss {loss.item():.4f} | "
            f"{tokens_per_second:,.0f} tok/s | "
            f"VRAM {peak_vram:.2f} GB"
        )

    # --------------------------------------------------------
    # PERIODIC CHECKPOINT
    # --------------------------------------------------------

    if (
        step % CHECKPOINT_EVERY == 0
    ):

        save_checkpoint(
            step=step,
            loss_value=loss.item(),
            archive=True
        )


# ============================================================
# FINAL CHECKPOINT
# ============================================================

# The periodic checkpoint may already have saved this exact
# step. We still save the final checkpoint explicitly.
#
# Because save_checkpoint() detects existing archive files,
# it will NOT overwrite an existing archive.

save_checkpoint(
    step=MAX_STEPS,
    loss_value=loss.item(),
    archive=True
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

total_time = (
    time.time()
    - start_time
)

# Tokens processed during THIS RUN
run_steps = (
    MAX_STEPS
    - start_step
)

run_tokens = (
    run_steps
    * BATCH_SIZE
    * SEQ_LEN
)

# Total tokens represented by the checkpoint
total_training_tokens = (
    MAX_STEPS
    * BATCH_SIZE
    * SEQ_LEN
)

average_tokens_per_second = (
    run_tokens
    / total_time
)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Starting step:",
    start_step
)

print(
    "Final step:",
    MAX_STEPS
)

print(
    "Steps this run:",
    run_steps
)

print(
    "Final loss:",
    f"{loss.item():.4f}"
)

print(
    "Tokens this run:",
    f"{run_tokens:,}"
)

print(
    "Total training tokens:",
    f"{total_training_tokens:,}"
)

print(
    "Training time:",
    f"{total_time / 60:.2f} minutes"
)

print(
    "Average throughput:",
    f"{average_tokens_per_second:,.0f} tok/s"
)

print(
    "Latest checkpoint:",
    CHECKPOINT_PATH
)

print("=" * 70)