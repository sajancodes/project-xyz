#!/usr/bin/env python3
"""
MIXED-DATA TRAINING: FineWeb + Semantic Data
=============================================
Combines general language pretraining with semantic/instruction data
to prevent catastrophic forgetting while teaching factual knowledge.

Strategy: Each batch = 90% FineWeb tokens + 10% semantic tokens
"""

import os
import time
import torch
from torch.optim import AdamW
from datasets import load_dataset
from tokenizers import Tokenizer

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
SEQ_LEN = 512

LEARNING_RATE = 1e-4          # Lower than pretraining
WEIGHT_DECAY = 0.01           # Lower weight decay

MAX_STEPS = 10000             # 10k mixed steps
LOG_EVERY = 10
CHECKPOINT_EVERY = 1000

# Mixture ratio
FINEWEB_RATIO = 0.9           # 90% FineWeb
SEMANTIC_RATIO = 0.1          # 10% semantic

# Checkpoints
BASE_CHECKPOINT = "checkpoint-106k.pt"
OUTPUT_CHECKPOINT = "checkpoint_mixed_10k.pt"
TOKENIZER_PATH = "tokenizer.json"

DEVICE = torch.device("cuda")
USE_BF16 = True


# ============================================================
# SEMANTIC TRAINING DATA (same as semantic fine-tuning)
# ============================================================

SEMANTIC_EXAMPLES = [
    # IDENTITY
    "User: My name is Sajan.\nAssistant: Your name is Sajan.",
    "User: I am Sajan.\nAssistant: You are Sajan.",
    "User: What is my name if I say 'I am Sajan'?\nAssistant: Your name is Sajan.",

    # SUN / EAST (multiple phrasings)
    "User: Where does the sun rise?\nAssistant: The sun rises in the east.",
    "User: What direction does the sun rise from?\nAssistant: The sun rises from the east.",
    "User: In which direction does sunrise occur?\nAssistant: Sunrise occurs in the east.",
    "User: The sun comes up in the ___\nAssistant: east.",
    "User: Every morning the sun appears in the eastern sky. What direction is that?\nAssistant: East.",

    # SIMPLE FACTS
    "User: What color is grass usually?\nAssistant: Grass is usually green.",
    "User: What color is the sky on a clear day?\nAssistant: The sky is usually blue.",
    "User: What do humans breathe?\nAssistant: Humans breathe air.",
    "User: What animal says meow?\nAssistant: A cat says meow.",
    "User: What animal says bark?\nAssistant: A dog says bark.",

    # SIMPLE RELATIONSHIPS
    "User: Sajan gave Ravi an apple. Who received the apple?\nAssistant: Ravi received the apple.",
    "User: Ravi gave Sajan a book. Who gave the book?\nAssistant: Ravi gave the book.",
    "User: The dog chased the cat. Who chased the cat?\nAssistant: The dog chased the cat.",
    "User: The cat chased the dog. What did the cat chase?\nAssistant: The cat chased the dog.",

    # PARAPHRASES
    "User: The boy is running. What is the boy doing?\nAssistant: The boy is running.",
    "User: The child is running. What action is the child doing?\nAssistant: The child is running.",
    "User: The dog is sleeping. What is the dog doing?\nAssistant: The dog is sleeping.",
    "User: The dog is asleep. What is happening to the dog?\nAssistant: The dog is sleeping.",

    # BASIC LOGIC
    "User: All birds have wings. A robin is a bird. Does a robin have wings?\nAssistant: Yes, a robin has wings.",
    "User: All cats are animals. Milo is a cat. Is Milo an animal?\nAssistant: Yes, Milo is an animal.",
    "User: If something is bigger than a box, and the box is bigger than a cup, what is bigger: the first thing or the cup?\nAssistant: The first thing is bigger than the cup.",

    # BASIC ARITHMETIC
    "User: What is 2 + 2?\nAssistant: 4.",
    "User: What is 5 + 3?\nAssistant: 8.",
    "User: What is 10 - 4?\nAssistant: 6.",
    "User: What is 3 multiplied by 4?\nAssistant: 12.",

    # INSTRUCTION FOLLOWING
    "User: Complete this sentence: I ___ Sajan.\nAssistant: am.",
    "User: Complete this sentence: He ___ happy.\nAssistant: is.",
    "User: Complete this sentence: They ___ happy.\nAssistant: are.",
    "User: Complete this sentence: We ___ students.\nAssistant: are.",
    "User: Complete this sentence: She ___ here.\nAssistant: is.",

    # QUESTION UNDERSTANDING
    "User: Who is running if I say 'The boy is running'?\nAssistant: The boy is running.",
    "User: What is happening in 'The dog is sleeping'?\nAssistant: The dog is sleeping.",
    "User: Where is the apple if Sajan puts the apple on the table?\nAssistant: The apple is on the table.",

    # CONVERSATION
    "User: Hello.\nAssistant: Hello! How can I help you?",
    "User: Good morning.\nAssistant: Good morning!",
    "User: How are you?\nAssistant: I am doing well. How are you?",
    "User: Thank you.\nAssistant: You're welcome.",
    "User: Good night.\nAssistant: Good night!",
]


def main():
    print("=" * 70)
    print("MIXED-DATA TRAINING: FineWeb + Semantic")
    print("=" * 70)
    print(f"FineWeb ratio: {FINEWEB_RATIO * 100:.0f}%")
    print(f"Semantic ratio: {SEMANTIC_RATIO * 100:.0f}%")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    # ------------------------------------------------------------------
    # TOKENIZER
    # ------------------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Tokenizer vocabulary: {tokenizer.get_vocab_size()}")
    print()

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    config = ModelConfig()
    model = SmallEnglishLLM(config).to(DEVICE)

    # ------------------------------------------------------------------
    # LOAD PRETRAINED CHECKPOINT
    # ------------------------------------------------------------------
    if not os.path.exists(BASE_CHECKPOINT):
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_CHECKPOINT}")

    print(f"Loading base checkpoint: {BASE_CHECKPOINT}")
    checkpoint = torch.load(BASE_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    base_step = checkpoint.get("step", "unknown")
    print(f"Base step: {base_step}")
    print()

    # ------------------------------------------------------------------
    # PARAMETER COUNT
    # ------------------------------------------------------------------
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {parameter_count:,} ({parameter_count / 1e6:.2f}M)")
    print()

    # ------------------------------------------------------------------
    # FINEWEB STREAM
    # ------------------------------------------------------------------
    print("Opening FineWeb stream...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2025-26",
        split="train",
        streaming=True
    )
    dataset_iterator = iter(dataset)
    print("FineWeb stream opened.")
    print()

    # ------------------------------------------------------------------
    # TOKENIZE SEMANTIC DATA
    # ------------------------------------------------------------------
    print("Tokenizing semantic data...")
    semantic_tokens = []
    for text in SEMANTIC_EXAMPLES:
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        if len(ids) >= 2:
            semantic_tokens.extend(ids)
    print(f"Semantic tokens: {len(semantic_tokens):,}")
    print()

    # ------------------------------------------------------------------
    # TOKEN BUFFERS
    # ------------------------------------------------------------------
    fineweb_buffer = []
    semantic_buffer = []

    def fill_fineweb_buffer(min_tokens):
        while len(fineweb_buffer) < min_tokens:
            document = next(dataset_iterator)
            text = document.get("text", "")
            if not text:
                continue
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            if len(ids) < 2:
                continue
            fineweb_buffer.extend(ids)

    def fill_semantic_buffer(min_tokens):
        while len(semantic_buffer) < min_tokens:
            semantic_buffer.extend(semantic_tokens)

    # ------------------------------------------------------------------
    # BATCH CREATION
    # ------------------------------------------------------------------
    def get_mixed_batch():
        # We need BATCH_SIZE * (SEQ_LEN + 1) tokens total
        tokens_per_example = SEQ_LEN + 1
        total_tokens_needed = BATCH_SIZE * tokens_per_example

        fineweb_needed = int(total_tokens_needed * FINEWEB_RATIO)
        semantic_needed = total_tokens_needed - fineweb_needed

        fill_fineweb_buffer(fineweb_needed)
        fill_semantic_buffer(semantic_needed)

        # Interleave: take tokens from both buffers
        all_tokens = []
        fw_idx = 0
        sem_idx = 0

        for _ in range(BATCH_SIZE):
            # For each example, mix FineWeb and semantic tokens
            example_tokens = []
            example_fw = int(tokens_per_example * FINEWEB_RATIO)
            example_sem = tokens_per_example - example_fw

            # Take from FineWeb buffer
            if fw_idx + example_fw <= len(fineweb_buffer):
                example_tokens.extend(fineweb_buffer[fw_idx:fw_idx + example_fw])
                fw_idx += example_fw
            else:
                # Wrap around
                example_tokens.extend(fineweb_buffer[fw_idx:])
                remaining = example_fw - (len(fineweb_buffer) - fw_idx)
                example_tokens.extend(fineweb_buffer[:remaining])
                fw_idx = remaining

            # Take from semantic buffer
            if sem_idx + example_sem <= len(semantic_buffer):
                example_tokens.extend(semantic_buffer[sem_idx:sem_idx + example_sem])
                sem_idx += example_sem
            else:
                example_tokens.extend(semantic_buffer[sem_idx:])
                remaining = example_sem - (len(semantic_buffer) - sem_idx)
                example_tokens.extend(semantic_buffer[:remaining])
                sem_idx = remaining

            all_tokens.append(example_tokens)

        # Remove consumed tokens from buffers
        del fineweb_buffer[:fw_idx]
        del semantic_buffer[:sem_idx]

        # Create input/target pairs
        input_batch = []
        target_batch = []
        for tokens in all_tokens:
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            input_batch.append(input_ids)
            target_batch.append(target_ids)

        input_ids = torch.tensor(input_batch, dtype=torch.long, device=DEVICE)
        target_ids = torch.tensor(target_batch, dtype=torch.long, device=DEVICE)
        return input_ids, target_ids

    # ------------------------------------------------------------------
    # OPTIMIZER (FRESH - not continued from pretraining)
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_BF16)

    # ------------------------------------------------------------------
    # CHECKPOINT SAVING
    # ------------------------------------------------------------------
    def save_checkpoint(step, loss_value):
        checkpoint_data = {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "loss": float(loss_value),
            "base_checkpoint": BASE_CHECKPOINT,
            "training_type": "mixed_fineweb_semantic",
            "fineweb_ratio": FINEWEB_RATIO,
            "semantic_ratio": SEMANTIC_RATIO,
            "semantic_examples": len(SEMANTIC_EXAMPLES),
        }

        torch.save(checkpoint_data, OUTPUT_CHECKPOINT)
        print(f"  -> Checkpoint saved: {OUTPUT_CHECKPOINT}")

        # Archive
        step_k = step // 1000
        archive_name = f"checkpoint-mixed-{step_k}k.pt"
        counter = 0
        while os.path.exists(archive_name):
            counter += 1
            archive_name = f"checkpoint-mixed-{step_k}k-{counter}.pt"
        torch.save(checkpoint_data, archive_name)
        print(f"  -> Archive saved: {archive_name}")
        print()

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    model.train()
    print("=" * 70)
    print("STARTING MIXED TRAINING")
    print("=" * 70)
    print()

    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        step_start = time.time()

        # Batch
        input_ids, target_ids = get_mixed_batch()

        # Forward
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits, loss = model(input_ids, target_ids)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % LOG_EVERY == 0 or step == 1:
            torch.cuda.synchronize()
            elapsed = time.time() - step_start
            tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / elapsed
            vram = torch.cuda.max_memory_allocated() / 1024**3

            print(
                f"Step {step:5d}/{MAX_STEPS} | "
                f"Loss {loss.item():.4f} | "
                f"{tokens_per_sec:,.0f} tok/s | "
                f"VRAM {vram:.2f} GB"
            )

        # Checkpoint
        if step % CHECKPOINT_EVERY == 0:
            save_checkpoint(step, loss.item())

    # Final checkpoint
    save_checkpoint(MAX_STEPS, loss.item())

    # Summary
    total_time = time.time() - start_time
    total_tokens = MAX_STEPS * BATCH_SIZE * SEQ_LEN

    print()
    print("=" * 70)
    print("MIXED TRAINING COMPLETE")
    print("=" * 70)
    print(f"Base checkpoint: {BASE_CHECKPOINT} (step {base_step})")
    print(f"Mixed steps: {MAX_STEPS}")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Tokens processed: {total_tokens:,}")
    print(f"Training time: {total_time / 60:.2f} minutes")
    print(f"Checkpoint: {OUTPUT_CHECKPOINT}")
    print("=" * 70)


if __name__ == "__main__":
    main()