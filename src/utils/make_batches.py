from datasets import load_dataset
from tokenizers import Tokenizer
import torch

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "CC-MAIN-2025-26"

TOKENIZER_PATH = "tokenizer.json"

SEQ_LEN = 512

dataset = load_dataset(
    DATASET,
    name=CONFIG,
    split="train",
    streaming=True,
)

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

token_buffer = []

for example in dataset:
    text = example["text"].strip()

    if not text:
        continue

    # Tokenize the streamed document
    token_ids = tokenizer.encode(text).ids

    # Add tokens to our continuous buffer
    token_buffer.extend(token_ids)

    # Produce every complete 512-token sequence
    while len(token_buffer) >= SEQ_LEN + 1:

        sequence = token_buffer[:SEQ_LEN + 1]

        # Remove tokens we just consumed
        token_buffer = token_buffer[SEQ_LEN:]

        # Causal language-model inputs and targets
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)

        print("Input shape :", x.shape)
        print("Target shape:", y.shape)

        print("\nFirst 20 input tokens:")
        print(x[:20].tolist())

        print("\nFirst 20 target tokens:")
        print(y[:20].tolist())

        print("\n---")

        # Only demonstrate a few batches
        raise SystemExit