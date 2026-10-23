"""Lightweight Streaming Benchmark Decontamination Guard.

Ensures no sample from streaming sources leaks benchmark tasks or solutions into training data.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def normalize_text_for_comparison(text: str) -> str:
    """Normalize text by lowering, removing punctuation, and collapsing whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def get_shingles(text: str, k: int = 3) -> Set[str]:
    """Generate word k-shingles from normalized text."""
    words = normalize_text_for_comparison(text).split()
    if len(words) < k:
        return set(words)
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class BenchmarkDecontaminator:
    """Checks streamed candidate examples against all benchmark prompts."""

    def __init__(self, benchmark_prompts: Optional[List[Dict[str, str]]] = None, jaccard_threshold: float = 0.70) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.exact_prompts: Dict[str, str] = {}  # norm_prompt -> bench_id
        self.bench_shingles: Dict[str, Set[str]] = {}  # bench_id -> shingles
        self.bench_keywords: Dict[str, List[str]] = {}  # bench_id -> important keywords

        if benchmark_prompts:
            for b in benchmark_prompts:
                self.add_benchmark_item(b.get("id", "bench"), b.get("prompt", ""))

    def add_benchmark_item(self, bench_id: str, prompt: str) -> None:
        norm = normalize_text_for_comparison(prompt)
        if norm:
            self.exact_prompts[norm] = bench_id
            self.bench_shingles[bench_id] = get_shingles(prompt)

    def is_contaminated(self, user_prompt: str, assistant_response: str = "") -> Tuple[bool, Optional[str]]:
        """Check if a candidate sample is contaminated against any benchmark task."""
        norm_prompt = normalize_text_for_comparison(user_prompt)
        if not norm_prompt:
            return False, None

        # 1. Exact match
        if norm_prompt in self.exact_prompts:
            return True, f"Exact match with benchmark: {self.exact_prompts[norm_prompt]}"

        # 2. Substring match for substantial prompts
        if len(norm_prompt) > 40:
            for b_norm, b_id in self.exact_prompts.items():
                if len(b_norm) > 40 and (norm_prompt in b_norm or b_norm in norm_prompt):
                    return True, f"Substring match with benchmark: {b_id}"

        # 3. Shingle Jaccard similarity
        sample_shingles = get_shingles(user_prompt)
        if sample_shingles:
            for b_id, b_shingles in self.bench_shingles.items():
                if not b_shingles:
                    continue
                intersection = len(sample_shingles.intersection(b_shingles))
                union = len(sample_shingles.union(b_shingles))
                sim = intersection / union if union > 0 else 0.0
                if sim >= self.jaccard_threshold:
                    return True, f"High Jaccard similarity ({sim:.2f}) with benchmark: {b_id}"

        return False, None
