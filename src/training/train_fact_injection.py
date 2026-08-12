# train_fact_injection.py
# Inject targeted factual knowledge: "the sun rises in the east"

import os
import time
import torch
from torch.optim import AdamW
from tokenizers import Tokenizer

from model import SmallEnglishLLM
from model_config import ModelConfig

# Speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

BATCH_SIZE = 24
SEQ_LEN = 512
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
WEIGHT_DECAY = 0.1
MAX_STEPS = 5000      # Quick injection
LOG_EVERY = 10
CHECKPOINT_EVERY = 1000

CHECKPOINT_PATH = "checkpoint_fineweb.pt"
TOKENIZER_PATH = "tokenizer.json"
DEVICE = torch.device("cuda")

# Targeted training data - repeat the fact many times
FACT_SENTENCES = [
    "the sun rises in the east.",
    "the sun rises in the east and sets in the west.",
    "every morning the sun rises in the east.",
    "the sun rises in the east each day.",
    "when the sun rises it appears in the east.",
    "the sun always rises in the east.",
    "in the east the sun rises.",
    "the sun comes up in the east.",
    "east is where the sun rises.",
    "the sun rises from the east.",
] * 1000  # 10,000 examples

def main():
    print("=" * 70)
    print("FACT INJECTION: 'the sun rises in the east'")
    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    print()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print("Tokenizer vocabulary:", tokenizer.get_vocab_size())

    # Load model from latest checkpoint
    config = ModelConfig()
    model = SmallEnglishLLM(config).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded checkpoint from step:", checkpoint.get("step", "unknown"))

    # Verify vocab
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise RuntimeError("Vocab mismatch")

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters:", f"{parameter_count:,} ({parameter_count/1e6:.2f}M)")
    print()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=False)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # Tokenize all fact sentences
    print("Tokenizing fact data...")
    all_tokens = []
    for sent in FACT_SENTENCES:
        encoded = tokenizer.encode(sent)
        ids = encoded.ids
        if len(ids) >= 2:
            all_tokens.extend(ids)
    print(f"Total fact tokens: {len(all_tokens):,}")
    print()

    # Training loop on fact data
    model.train()
    print("=" * 70)
    print("STARTING FACT INJECTION TRAINING")
    print("=" * 70)
    print(f"Steps: {MAX_STEPS}")
    print(f"Fact examples: {len(FACT_SENTENCES):,}")
    print()

    start_time = time.time()
    token_buffer = []

    def get_fact_batch():
        nonlocal token_buffer
        tokens_per_example = SEQ_LEN + 1
        required_tokens = BATCH_SIZE * tokens_per_example
        
        while len(token_buffer) < required_tokens:
            token_buffer.extend(all_tokens)
        
        input_batch = []
        target_batch = []
        for _ in range(BATCH_SIZE):
            sequence = token_buffer[:SEQ_LEN + 1]
            del token_buffer[:SEQ_LEN + 1]
            input_ids = sequence[:-1]
            target_ids = sequence[1:]
            input_batch.append(input_ids)
            target_batch.append(target_ids)

        input_ids = torch.tensor(input_batch, dtype=torch.long, device=DEVICE)
        target_ids = torch.tensor(target_batch, dtype=torch.long, device=DEVICE)
        return input_ids, target_ids

    for step in range(1, MAX_STEPS + 1):
        step_start = time.time()
        input_ids, target_ids = get_fact_batch()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=True):
            logits, loss = model(input_ids, target_ids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % LOG_EVERY == 0 or step == 1:
            torch.cuda.synchronize()
            step_time = time.time() - step_start
            tokens_per_sec = (BATCH_SIZE * SEQ_LEN) / step_time
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Step {step:5d}/{MAX_STEPS} | Loss {loss.item():.4f} | {tokens_per_sec:,.0f} tok/s | VRAM {peak_vram:.2f} GB")

        if step % CHECKPOINT_EVERY == 0:
            # Save intermediate
            torch.save({
                "step": checkpoint.get("step", 100000) + step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "loss": loss.item(),
            }, CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved")

    # Final save
    torch.save({
        "step": checkpoint.get("step", 100000) + MAX_STEPS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "loss": loss.item(),
    }, CHECKPOINT_PATH)
    print(f"\nFinal checkpoint saved: {CHECKPOINT_PATH}")

    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("FACT INJECTION COMPLETE")
    print("=" * 70)
    print(f"Training time: {total_time/60:.2f} minutes")
    print(f"Final loss: {loss.item():.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()