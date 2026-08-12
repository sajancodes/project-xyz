from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="CC-MAIN-2025-26",
    split="train",
    streaming=True,
)

for i, example in enumerate(dataset):
    print("=" * 80)
    print(example["text"][:1000])

    if i == 4:
        break