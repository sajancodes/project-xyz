"""General Instruction Following and Knowledge Regression Benchmark Suite."""

from __future__ import annotations

from typing import List
from src.evaluation.qwen4b.cyber import EvalQuestion

GENERAL_BENCHMARK_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion(
        id="gen_inst_01",
        category="general_instruction",
        prompt="List exactly 3 benefits of adopting IPv6 over IPv4. Format your answer as a numbered list where each item begins with a bold title.",
        required_concepts=["1.", "2.", "3.", "address space", "**"],
    ),
    EvalQuestion(
        id="gen_inst_02",
        category="general_instruction",
        prompt="Summarize the concept of Occam's Razor in two sentences. Do not use the letter 'z' in your response.",
        required_concepts=["simplest", "explanation", "assumption"],
        forbidden_concepts=["z", "razor"],
    ),
    EvalQuestion(
        id="gen_know_01",
        category="general_knowledge",
        prompt="What is the chemical formula for photosynthesis, and what are the primary reactants and products?",
        required_concepts=["co2", "h2o", "glucose", "oxygen", "c6h12o6"],
    ),
]
