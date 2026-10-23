"""Tests for Benchmarking, Scoring, and Before/After Comparison Reporting."""

from pathlib import Path
import pytest

from src.evaluation.qwen4b.benchmark import get_all_benchmark_questions, score_response
from src.evaluation.qwen4b.comparison import compare_benchmarks
from src.evaluation.qwen4b.cyber import EvalQuestion


def test_eval_question_scoring_perfect():
    q = EvalQuestion(
        id="test_q1",
        category="networking",
        prompt="Explain SYN flood",
        required_concepts=["syn", "handshake", "cookie"],
        forbidden_concepts=["disable tcp"],
    )
    good_resp = "A SYN flood exploits the TCP handshake mechanism, mitigated by using SYN cookies."
    res = score_response(good_resp, q)

    assert res["score"] == 1.0
    assert len(res["missing_required"]) == 0
    assert len(res["matched_forbidden"]) == 0


def test_eval_question_scoring_partial_and_forbidden():
    q = EvalQuestion(
        id="test_q2",
        category="networking",
        prompt="Explain SYN flood",
        required_concepts=["syn", "handshake", "cookie"],
        forbidden_concepts=["disable tcp"],
    )
    bad_resp = "Just disable tcp when you see a syn attack."
    res = score_response(bad_resp, q)

    # Matched 'syn' (1/3 = 0.33), but penalized for 'disable tcp' (-0.5) -> clamped to 0.0
    assert res["score"] == 0.0
    assert "disable tcp" in res["matched_forbidden"]


def test_compare_benchmarks():
    base_data = {
        "tag": "base",
        "overall_score": 0.5000,
        "category_scores": {
            "networking": 0.4000,
            "secure_coding": 0.6000,
        },
    }
    qlora_data = {
        "tag": "qlora",
        "overall_score": 0.8500,
        "category_scores": {
            "networking": 0.9000,
            "secure_coding": 0.8000,
        },
    }

    report = compare_benchmarks(base_data, qlora_data)

    assert "| Networking | 0.4000 | 0.9000 | +0.5000 |" in report
    assert "| Secure Coding | 0.6000 | 0.8000 | +0.2000 |" in report
    assert "| **Overall Composite Score** | **0.5000** | **0.8500** | **+0.3500** |" in report
