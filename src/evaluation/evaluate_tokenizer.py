import os
import sys

from datasets import load_dataset
from tokenizers import Tokenizer
import statistics

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import TOKENIZER_PATH

NUM_DOCUMENTS = 10_000

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2025-26",
    split="train",
    streaming=True,
)

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

total_chars = 0
total_tokens = 0
total_words = 0
unk_tokens = 0
documents = 0

tokens_per_document = []

for example in dataset:
    text = example["text"].strip()

    if not text:
        continue

    encoded = tokenizer.encode(text)

    token_count = len(encoded.ids)

    total_chars += len(text)
    total_tokens += token_count
    total_words += len(text.split())

    unk_tokens += encoded.tokens.count("<unk>")

    tokens_per_document.append(token_count)

    documents += 1

    if documents >= NUM_DOCUMENTS:
        break

print("\n===== TOKENIZER EVALUATION =====")

print(f"Documents: {documents:,}")
print(f"Characters: {total_chars:,}")
print(f"Words: {total_words:,}")
print(f"Tokens: {total_tokens:,}")

print(
    f"\nCharacters / token: "
    f"{total_chars / total_tokens:.3f}"
)

print(
    f"Tokens / word: "
    f"{total_tokens / total_words:.3f}"
)

print(
    f"UNK tokens: "
    f"{unk_tokens:,}"
)

print(
    f"UNK rate: "
    f"{unk_tokens / total_tokens * 100:.6f}%"
)

print(
    f"\nAverage tokens/document: "
    f"{statistics.mean(tokens_per_document):.2f}"
)

print(
    f"Median tokens/document: "
    f"{statistics.median(tokens_per_document):.2f}"
)