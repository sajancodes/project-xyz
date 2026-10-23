"""Before vs After Benchmark Comparison and Reporting.

Computes exact performance deltas between Base Qwen 4B and Cybersecurity QLoRA adapter.
Outputs Markdown comparison tables. Never fabricates scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def format_score(score: Optional[float]) -> str:
    """Format float score nicely or N/A."""
    return f"{score:.4f}" if score is not None else "N/A"


def format_delta(delta: Optional[float]) -> str:
    """Format signed delta with + or -."""
    if delta is None:
        return "N/A"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def compare_benchmarks(
    base_results: Union[str, Path, Dict[str, Any]],
    qlora_results: Union[str, Path, Dict[str, Any]],
    output_md_path: Optional[Union[str, Path]] = None,
) -> str:
    """Compare baseline benchmark vs fine-tuned QLoRA benchmark."""
    # Load base
    if isinstance(base_results, (str, Path)):
        with open(base_results, "r", encoding="utf-8") as f:
            base_data = json.load(f)
    else:
        base_data = base_results

    # Load qlora
    if isinstance(qlora_results, (str, Path)):
        with open(qlora_results, "r", encoding="utf-8") as f:
            qlora_data = json.load(f)
    else:
        qlora_data = qlora_results

    base_cats = base_data.get("category_scores", {})
    qlora_cats = qlora_data.get("category_scores", {})
    all_cats = sorted(set(list(base_cats.keys()) + list(qlora_cats.keys())))

    lines = [
        "# Qwen 4B Cybersecurity QLoRA Evaluation: Before vs After",
        "",
        "## Summary Category Comparison",
        "",
        "| Benchmark Category | Base | QLoRA | Delta |",
        "|---|---:|---:|---:|",
    ]

    for cat in all_cats:
        b_score = base_cats.get(cat)
        q_score = qlora_cats.get(cat)
        delta = (q_score - b_score) if (q_score is not None and b_score is not None) else None
        lines.append(f"| {cat.replace('_', ' ').title()} | {format_score(b_score)} | {format_score(q_score)} | {format_delta(delta)} |")

    b_overall = base_data.get("overall_score")
    q_overall = qlora_data.get("overall_score")
    overall_delta = (q_overall - b_overall) if (q_overall is not None and b_overall is not None) else None

    lines.extend([
        f"| **Overall Composite Score** | **{format_score(b_overall)}** | **{format_score(q_overall)}** | **{format_delta(overall_delta)}** |",
        "",
        "## Analysis & Regression Verification",
        "",
        "- **Cybersecurity Gains**: Verify positive delta across all targeted security domains.",
        "- **Capability Retention**: Verify that General Instruction, Coding, and Reasoning categories do not suffer severe regression.",
        "",
    ])

    report = "\n".join(lines) + "\n"

    if output_md_path:
        out_p = Path(output_md_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report)

    return report
