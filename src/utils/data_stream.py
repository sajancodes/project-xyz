from datasets import load_dataset
from tqdm import tqdm

TARGET_TOKENS = 100_000_000

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2025-26",
    split="train",
    streaming=True,
)

token_count = 0
document_count = 0

for example in dataset:
    text = example["text"].strip()

    if not text:
        continue

    # Temporary approximation.
    # We will replace this with our actual tokenizer later.
    estimated_tokens = len(text.split())

    token_count += estimated_tokens
    document_count += 1

    if document_count % 1000 == 0:
        print(
            f"Documents: {document_count:,} | "
            f"Estimated tokens: {token_count:,}"
        )

    if token_count >= TARGET_TOKENS:
        break

print("\nStream complete.")
print(f"Documents processed: {document_count:,}")
print(f"Estimated tokens: {token_count:,}")