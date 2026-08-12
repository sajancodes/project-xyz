import os
import sys

from datasets import load_dataset

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import FINEWEB_DATA_DIR

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2025-26",
    split="train",
    streaming=True,
)

os.makedirs(FINEWEB_DATA_DIR, exist_ok=True)

target_bytes = 10 * 1024**3
total_bytes = 0
documents = 0

with open(
    os.path.join(FINEWEB_DATA_DIR, "fineweb_2025_26.txt"),
    "w",
    encoding="utf-8",
) as f:
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