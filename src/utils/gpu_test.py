import torch

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

SEQ_LEN = 512
VOCAB_SIZE = 16_000

for batch_size in [1, 2, 4, 8, 16, 32]:

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        x = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, SEQ_LEN),
            device=device
        )

        # Temporary tensor representing model-like computation
        embedding = torch.nn.Embedding(
            VOCAB_SIZE,
            256
        ).to(device)

        y = embedding(x)

        memory = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"Batch {batch_size:2d} | "
            f"Shape {tuple(x.shape)} | "
            f"Peak VRAM {memory:.3f} GB"
        )

        del x, y, embedding

    except RuntimeError as e:

        if "out of memory" in str(e).lower():
            print(f"Batch {batch_size:2d} | OUT OF MEMORY")
            torch.cuda.empty_cache()
        else:
            raise