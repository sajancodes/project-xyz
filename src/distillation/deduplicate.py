"""Deduplication and Split Contamination Detection.

Implements:
1. Exact deduplication via deterministic SHA-256 hash of normalized user prompt.
2. Near deduplication via word-level shingle Jaccard similarity.
3. Cross-split contamination prevention to guarantee independent test/validation benchmarks.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Set, Tuple


def get_user_prompt(record: Dict[str, Any]) -> str:
    """Extract and normalize user prompt text from conversation."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return ""


def compute_exact_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_word_shingles(text: str, k: int = 3) -> Set[str]:
    """Extract k-shingles (word n-grams) from text."""
    # Tokenize words, removing punctuation
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets of shingles."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class Deduplicator:
    """Handles exact and near deduplication of dataset records."""

    def __init__(self, near_dedup_threshold: float = 0.80, shingle_size: int = 3) -> None:
        self.near_dedup_threshold = near_dedup_threshold
        self.shingle_size = shingle_size
        self.seen_ids: Set[str] = set()
        self.exact_hashes: Dict[str, str] = {}  # hash -> record_id
        self.record_shingles: Dict[str, Set[str]] = {}  # record_id -> shingles

    def is_exact_duplicate(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if record ID or exact prompt hash has already been seen."""
        record_id = record.get("id", "")
        if record_id in self.seen_ids:
            return True, f"Duplicate record ID: '{record_id}'"

        prompt = get_user_prompt(record)
        h = compute_exact_hash(prompt)
        if h in self.exact_hashes:
            return True, f"Exact prompt duplicate of record '{self.exact_hashes[h]}'"

        return False, ""

    def is_near_duplicate(self, record: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Check if record prompt is a near-duplicate of an existing record."""
        prompt = get_user_prompt(record)
        shingles = get_word_shingles(prompt, k=self.shingle_size)
        if not shingles:
            return False, "", 0.0

        for existing_id, existing_shingles in self.record_shingles.items():
            sim = jaccard_similarity(shingles, existing_shingles)
            if sim >= self.near_dedup_threshold:
                return True, f"Near duplicate of '{existing_id}' (Jaccard: {sim:.2f})", sim

        return False, "", 0.0

    def add(self, record: Dict[str, Any]) -> None:
        """Register record in deduplication index."""
        record_id = record["id"]
        prompt = get_user_prompt(record)
        h = compute_exact_hash(prompt)
        shingles = get_word_shingles(prompt, k=self.shingle_size)

        self.seen_ids.add(record_id)
        self.exact_hashes[h] = record_id
        self.record_shingles[record_id] = shingles


def check_split_contamination(
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    near_dedup_threshold: float = 0.75,
) -> List[str]:
    """Check whether any eval record is an exact or near duplicate of any train record.
    
    Returns a list of contamination error descriptions.
    """
    train_hashes = {}
    train_shingles = {}

    for r in train_records:
        r_id = r.get("id", "")
        prompt = get_user_prompt(r)
        h = compute_exact_hash(prompt)
        train_hashes[h] = r_id
        train_shingles[r_id] = get_word_shingles(prompt)

    violations = []
    for r in eval_records:
        r_id = r.get("id", "")
        prompt = get_user_prompt(r)
        h = compute_exact_hash(prompt)

        # Check exact match
        if h in train_hashes:
            violations.append(f"Eval record '{r_id}' is exact prompt duplicate of train record '{train_hashes[h]}'")
            continue

        # Check near duplicate
        shingles = get_word_shingles(prompt)
        for t_id, t_shingles in train_shingles.items():
            sim = jaccard_similarity(shingles, t_shingles)
            if sim >= near_dedup_threshold:
                violations.append(
                    f"Eval record '{r_id}' has near-duplicate in train: '{t_id}' (similarity: {sim:.2f})"
                )
                break

    return violations
