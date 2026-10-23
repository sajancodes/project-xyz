"""Tests for Assistant-Only Loss Masking and Label Collation."""

from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from src.training.qwen4b.collator import DataCollatorForAssistantOnlyLM
from src.training.qwen4b.masking import IGNORE_INDEX, create_assistant_masked_sample


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        # Load small tokenizer or cached tokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    except Exception:
        pytest.skip("Network / Hugging Face access unavailable for tokenizer fixture.")


def test_assistant_only_loss_masking(qwen_tokenizer):
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "What is an intrusion detection system (IDS)?"},
        {"role": "assistant", "content": "An IDS monitors network traffic for malicious signatures."},
    ]

    sample = create_assistant_masked_sample(qwen_tokenizer, messages)
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]

    assert len(input_ids) == len(labels)
    assert len(input_ids) == len(attention_mask)

    # 1. System and User tokens must be masked (-100)
    # The first few tokens must be -100
    assert labels[0].item() == IGNORE_INDEX

    # 2. Assistant answer tokens must be ACTIVE (not -100)
    active_tokens = (labels != IGNORE_INDEX).sum().item()
    masked_tokens = (labels == IGNORE_INDEX).sum().item()

    assert active_tokens > 0, "No assistant tokens were unmasked for loss computation!"
    assert masked_tokens > 0, "Non-assistant tokens were not masked!"

    # 3. Active labels must match input_ids exactly
    active_mask = (labels != IGNORE_INDEX)
    assert torch.equal(labels[active_mask], input_ids[active_mask]), "Unmasked label tokens do not match input IDs!"


def test_collator_padding_with_ignore_index(qwen_tokenizer):
    short_messages = [
        {"role": "user", "content": "Short question?"},
        {"role": "assistant", "content": "Short answer."},
    ]
    long_messages = [
        {"role": "user", "content": "A significantly longer question inquiring about network protocols and firewalls?"},
        {"role": "assistant", "content": "A detailed technical response explaining stateful inspection, TCP tracking, and deep packet analysis in enterprise architectures."},
    ]

    sample1 = create_assistant_masked_sample(qwen_tokenizer, short_messages)
    sample2 = create_assistant_masked_sample(qwen_tokenizer, long_messages)

    collator = DataCollatorForAssistantOnlyLM(tokenizer=qwen_tokenizer, pad_to_multiple_of=8)
    batch = collator([sample1, sample2])

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == labels.shape
    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == 2
    assert input_ids.shape[1] % 8 == 0

    # For the shorter sample, padding positions must have attention_mask = 0 and label = -100
    diff = input_ids.shape[1] - len(sample1["input_ids"])
    if diff > 0:
        # Last positions of first sample are padding
        padded_labels = labels[0, -diff:]
        padded_masks = attention_mask[0, -diff:]
        assert (padded_labels == IGNORE_INDEX).all().item(), "Padding positions in labels are not set to -100!"
        assert (padded_masks == 0).all().item(), "Padding positions in attention_mask are not set to 0!"
