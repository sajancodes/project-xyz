"""Coding Capability Regression Benchmark Suite."""

from __future__ import annotations

from typing import List
from src.evaluation.qwen4b.cyber import EvalQuestion

CODING_BENCHMARK_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion(
        id="code_algo_01",
        category="coding",
        prompt="Write a Python function `binary_search(arr: list[int], target: int) -> int` that implements binary search on a sorted list, returning the index or -1 if not found. Include time complexity.",
        required_concepts=["def binary_search", "low", "high", "mid", "o(log n)", "while low <= high"],
    ),
    EvalQuestion(
        id="code_algo_02",
        category="coding",
        prompt="Write a Python function `is_valid_parentheses(s: str) -> bool` using a stack to validate balanced brackets '()', '[]', '{}'.",
        required_concepts=["stack", "append", "pop", "def is_valid_parentheses", "return len(stack) == 0"],
    ),
]
