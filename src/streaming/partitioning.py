"""Deterministic Content-Hash Partitioning for Streaming Datasets.

Assigns streamed records deterministically to train (90%), validation (5%), or test (5%)
based on the cryptographic hash of the normalized prompt and response.
Ensures an example ALWAYS falls into the exact same partition across runs.
"""

from __future__ import annotations

import hashlib
from typing import Literal

PartitionType = Literal["train", "validation", "test"]


def get_partition_for_sample(
    instruction: str,
    response: str,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed_salt: str = "project_xyz_sft_seed_42",
) -> PartitionType:
    """Determine partition deterministically using salted MD5 content hash."""
    clean_inst = " ".join(instruction.strip().lower().split())
    clean_resp = " ".join(response.strip().lower().split())

    key = f"{seed_salt}::{clean_inst}::{clean_resp}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    # Normalize 128-bit hex to [0.0, 1.0)
    fraction = int(digest, 16) / (2**128)

    if fraction < train_ratio:
        return "train"
    elif fraction < (train_ratio + val_ratio):
        return "validation"
    else:
        return "test"
