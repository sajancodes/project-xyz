from datasets import load_dataset
from tokenizers import Tokenizer

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "CC-MAIN-2025-26"

TOKENIZER_PATH = "tokenizer.json"

dataset = load_dataset(
    DATASET,
    name=CONFIG,
    split="train",
    streaming=True,
)

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

total_tokens = 0

for i, example in enumerate(dataset):

    text = example["text"].strip()

    if not text:
        continue

    encoded = tokenizer.encode(text)

    token_ids = encoded.ids

    total_tokens += len(token_ids)

    print(f"\nDocument {i}")
    print(f"Characters: {len(text):,}")
    print(f"Tokens: {len(token_ids):,}")
    print(f"Total tokens: {total_tokens:,}")

    print("First tokens:")
    print(encoded.tokens[:20])

    print("First IDs:")
    print(token_ids[:20])

    if i >= 9:
        break