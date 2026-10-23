"""Reasoning Benchmark Suite."""

from __future__ import annotations

from typing import List
from src.evaluation.qwen4b.cyber import EvalQuestion

REASONING_BENCHMARK_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion(
        id="reas_logic_01",
        category="reasoning",
        prompt="Three boxes are labeled 'Apples', 'Oranges', and 'Apples and Oranges'. All three labels are incorrect. You can pick one fruit from one box without looking inside. Which box do you pick from to correctly label all three boxes? Explain step-by-step.",
        required_concepts=["apples and oranges", "incorrect", "deduce", "remaining"],
    ),
    EvalQuestion(
        id="reas_logic_02",
        category="reasoning",
        prompt="If all servers in Cluster A run Linux, and some servers in Cluster A run Kubernetes, does it follow that some servers running Linux run Kubernetes? Provide the formal syllogistic conclusion.",
        required_concepts=["yes", "valid", "some servers", "linux", "kubernetes"],
    ),
]
