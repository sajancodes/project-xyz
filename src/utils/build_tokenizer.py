import os
import sys

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import TOKENIZER_PATH

DATASET = "HuggingFaceFW/fineweb"
CONFIG = "CC-MAIN-2025-26"

TOKENIZER_VOCAB_SIZE = 16_000
TOKENIZER_SAMPLE_DOCUMENTS = 50_000

dataset = load_dataset(
    DATASET,
    name=CONFIG,
    split="train",
    streaming=True,
)

def text_iterator():
    for i, example in enumerate(dataset):
        text = example["text"].strip()

        if text:
            yield text

        if i + 1 >= TOKENIZER_SAMPLE_DOCUMENTS:
            break


tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=True
)

trainer = trainers.BpeTrainer(
    vocab_size=TOKENIZER_VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)

tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer,
)

tokenizer.decoder = decoders.ByteLevel()

tokenizer.save(TOKENIZER_PATH)

print("Tokenizer created.")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

test_text = "The model learns English from the FineWeb dataset."
encoded = tokenizer.encode(test_text)

print("\nTest:")
print(test_text)
print("\nTokens:")
print(encoded.tokens)
print("\nIDs:")
print(encoded.ids)