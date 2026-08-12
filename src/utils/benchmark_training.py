import torch
from model import ModelConfig, SmallEnglishLLM

config = ModelConfig()
device = torch.device("cuda")

print("=" * 60)
print("EXPERIMENT 1 — TRAINING MEMORY BENCHMARK")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2
    ),
    "GB"
)

print()

for batch_size in [1, 2, 4, 8, 16, 32, 64]:

    print(f"Testing batch size: {batch_size}")

    model = SmallEnglishLLM(config).to(device)

    # AdamW is the optimizer we'll initially use
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Random token batch
        x = torch.randint(
            0,
            config.vocab_size,
            (batch_size, config.max_seq_len),
            device=device
        )

        y = torch.randint(
            0,
            config.vocab_size,
            (batch_size, config.max_seq_len),
            device=device
        )

        # Forward
        logits, loss = model(x, y)

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Peak VRAM
        peak_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"  SUCCESS | "
            f"Peak VRAM: {peak_vram:.3f} GB | "
            f"Loss: {loss.item():.4f}"
        )

    except RuntimeError as e:

        if "out of memory" in str(e).lower():

            print("  OUT OF MEMORY")

            torch.cuda.empty_cache()

        else:
            raise

    finally:

        del model
        del optimizer

        if "x" in locals():
            del x

        if "y" in locals():
            del y

        if "logits" in locals():
            del logits

        if "loss" in locals():
            del loss

        torch.cuda.empty_cache()

    print()

print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)