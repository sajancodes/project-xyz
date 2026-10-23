"""Comprehensive Benchmark Runner for Qwen 4B (Base and QLoRA).

Evaluates cybersecurity specializations and general capabilities.
Never fabricates results: outputs verified execution scores with raw logs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from src.evaluation.qwen4b.cyber import CYBER_BENCHMARK_QUESTIONS, EvalQuestion
from src.evaluation.qwen4b.general import GENERAL_BENCHMARK_QUESTIONS
from src.evaluation.qwen4b.coding import CODING_BENCHMARK_QUESTIONS
from src.evaluation.qwen4b.reasoning import REASONING_BENCHMARK_QUESTIONS
from src.models.qwen4b.inference import generate_response

logger = logging.getLogger(__name__)


def score_response(response: str, question: EvalQuestion) -> Dict[str, Any]:
    """Score a model response against required and forbidden concepts."""
    resp_lower = response.lower()
    
    # Check required concepts
    matched_req = []
    missing_req = []
    for concept in question.required_concepts:
        if concept.lower() in resp_lower:
            matched_req.append(concept)
        else:
            missing_req.append(concept)

    req_score = len(matched_req) / len(question.required_concepts) if question.required_concepts else 1.0

    # Check forbidden concepts
    matched_forbidden = []
    for bad in question.forbidden_concepts:
        if bad.lower() in resp_lower:
            matched_forbidden.append(bad)

    penalty = 0.5 * len(matched_forbidden)
    final_score = max(0.0, min(1.0, req_score - penalty))

    return {
        "score": round(final_score, 4),
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_forbidden": matched_forbidden,
    }


def get_all_benchmark_questions() -> List[EvalQuestion]:
    """Return combined benchmark questions across all tracks."""
    return (
        CYBER_BENCHMARK_QUESTIONS
        + GENERAL_BENCHMARK_QUESTIONS
        + CODING_BENCHMARK_QUESTIONS
        + REASONING_BENCHMARK_QUESTIONS
    )


def run_benchmark(
    model: Any,
    tokenizer: Any,
    questions: Optional[List[EvalQuestion]] = None,
    tag: str = "base",
    output_dir: str | Path = "benchmarks",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Run evaluation benchmark across all target domains and persist raw results."""
    if questions is None:
        questions = get_all_benchmark_questions()

    logger.info("Running benchmark '%s' on %d questions...", tag, len(questions))
    results_per_category = defaultdict(list)
    question_details = []

    for q in questions:
        logger.info("Evaluating [%s] %s...", q.category, q.id)
        prediction = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages_or_prompt=q.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )

        scoring = score_response(prediction, q)
        results_per_category[q.category].append(scoring["score"])

        question_details.append({
            "id": q.id,
            "category": q.category,
            "prompt": q.prompt,
            "prediction": prediction,
            "score": scoring["score"],
            "matched_required": scoring["matched_required"],
            "missing_required": scoring["missing_required"],
            "matched_forbidden": scoring["matched_forbidden"],
        })

    # Compute category averages
    category_scores = {}
    for cat, scores in results_per_category.items():
        category_scores[cat] = round(sum(scores) / len(scores), 4)

    overall_score = round(
        sum(q["score"] for q in question_details) / len(question_details), 4
    ) if question_details else 0.0

    benchmark_summary = {
        "tag": tag,
        "num_questions": len(questions),
        "overall_score": overall_score,
        "category_scores": category_scores,
        "questions": question_details,
    }

    # Persist raw results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / f"{tag}_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved raw benchmark results to '%s'", json_path)

    return benchmark_summary
