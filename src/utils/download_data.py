from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2025-26",
    split="train",
    streaming=True,
)

target_bytes = 10 * 1024**3
total_bytes = 0
documents = 0

with open("data/fineweb/fineweb_2025_26.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"].strip()

        if not text:
            continue

        f.write(text)
        f.write("\n\n")

        total_bytes += len(text.encode("utf-8"))
        documents += 1

        if documents % 1000 == 0:
            print(
                f"Documents: {documents:,} | "
                f"Data: {total_bytes / 1024**3:.2f} GB"
            )

        if total_bytes >= target_bytes:
            break

print("\nDownload complete.")
print(f"Documents: {documents:,}")
print(f"Size: {total_bytes / 1024**3:.2f} GB")