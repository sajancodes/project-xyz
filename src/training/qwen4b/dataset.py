"""Dataset Class for Qwen 4B Cybersecurity SFT.

Loads JSONL records, tokenizes with official Qwen chat template, and masks non-assistant tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from src.training.qwen4b.masking import create_assistant_masked_sample


class CyberChatDataset(Dataset):
    """PyTorch Dataset for Assistant-Masked Cybersecurity SFT."""

    def __init__(
        self,
        data_source: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            self.records = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        self.records.append(json.loads(line_s))
        elif isinstance(data_source, list):
            self.records = list(data_source)
        else:
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        messages = record["messages"]
        return create_assistant_masked_sample(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=self.max_length,
        )
