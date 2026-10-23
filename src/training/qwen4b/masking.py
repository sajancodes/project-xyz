"""Assistant-Only Loss Masking for Chat SFT.

Ensures the causal language model is ONLY penalized for predicting assistant tokens:
- System message tokens: masked (-100)
- User message tokens: masked (-100)
- Assistant header (<|im_start|>assistant\\n): masked (-100)
- Assistant response tokens: ACTIVE LOSS (label = input_id)
- Assistant end-of-turn token (<|im_end|>): ACTIVE LOSS (teaches EOS termination)
- Padding tokens: masked (-100)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def create_assistant_masked_sample(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    max_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize conversation and construct labels masking out all non-assistant tokens.
    
    Uses exact template prefix boundaries to guarantee precise token alignments.
    
    Args:
        tokenizer: HF Tokenizer with apply_chat_template support.
        messages: List of message dictionaries with "role" and "content".
        max_length: Optional maximum sequence truncation length.
        
    Returns:
        Dictionary with:
            - input_ids: 1D torch.LongTensor
            - attention_mask: 1D torch.LongTensor
            - labels: 1D torch.LongTensor (with -100 for non-assistant tokens)
    """
    # 1. Full conversation text
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded_full = tokenizer(
        full_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    full_input_ids = encoded_full["input_ids"]

    if max_length is not None and len(full_input_ids) > max_length:
        full_input_ids = full_input_ids[:max_length]

    # Initialize all labels to IGNORE_INDEX (-100)
    labels = [IGNORE_INDEX] * len(full_input_ids)

    # 2. Iterate through messages and unmask assistant segments
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        # Prefix up to the start of this assistant's answer (includes user message + generation prompt)
        prefix_msgs = messages[:idx]
        prefix_text = tokenizer.apply_chat_template(
            prefix_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(prefix_text, return_tensors=None, add_special_tokens=False)["input_ids"]
        start_idx = len(prefix_ids)

        # Sequence including this assistant's message
        with_assistant_msgs = messages[: idx + 1]
        with_assistant_text = tokenizer.apply_chat_template(
            with_assistant_msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        with_assistant_ids = tokenizer(with_assistant_text, return_tensors=None, add_special_tokens=False)["input_ids"]
        end_idx = min(len(with_assistant_ids), len(full_input_ids))

        # Unmask assistant tokens
        if start_idx < len(full_input_ids) and start_idx < end_idx:
            for i in range(start_idx, end_idx):
                labels[i] = full_input_ids[i]

    input_ids_tensor = torch.tensor(full_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.ones_like(input_ids_tensor)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
    }
