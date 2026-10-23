"""Dynamic Batching and Collation with Assistant-Only Masking and Label Padding.

Pads:
- input_ids with pad_token_id
- attention_mask with 0
- labels with -100 (IGNORE_INDEX)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from src.training.qwen4b.masking import IGNORE_INDEX


@dataclass
class DataCollatorForAssistantOnlyLM:
    """Collator that pads dynamic batches with -100 for label padding."""

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_labels = [f["labels"] for f in features]

        # Determine batch max length
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of is not None and (max_len % self.pad_to_multiple_of != 0):
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            cur_len = len(ids)
            diff = max_len - cur_len

            if diff > 0:
                # Right padding
                p_ids = torch.cat([ids, torch.full((diff,), pad_id, dtype=torch.long)])
                p_mask = torch.cat([mask, torch.zeros((diff,), dtype=torch.long)])
                p_labels = torch.cat([labels, torch.full((diff,), IGNORE_INDEX, dtype=torch.long)])
            else:
                p_ids = ids[:max_len]
                p_mask = mask[:max_len]
                p_labels = labels[:max_len]

            padded_input_ids.append(p_ids)
            padded_attention_mask.append(p_mask)
            padded_labels.append(p_labels)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }
