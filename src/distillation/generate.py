"""End-to-End Dataset Generation, Validation, Deduplication, and Statistics Pipeline.

Processes raw records into sanitized, split datasets and computes rigorous statistics.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.distillation.deduplicate import Deduplicator, check_split_contamination
from src.distillation.split import deterministic_split, save_jsonl
from src.distillation.validate import validate_record_content

logger = logging.getLogger(__name__)


def compute_statistics(
    raw_records: List[Dict[str, Any]],
    accepted_records: List[Dict[str, Any]],
    rejected_records: List[Tuple[Dict[str, Any], str]],
    duplicate_count: int,
    near_duplicate_count: int,
    train_set: List[Dict[str, Any]],
    val_set: List[Dict[str, Any]],
    test_set: List[Dict[str, Any]],
    benchmark_set: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute exact, non-fabricated metrics and distributions."""
    categories = Counter(r.get("category", "unknown") for r in accepted_records)
    difficulties = Counter(r.get("difficulty", "unknown") for r in accepted_records)

    q_lens = []
    a_lens = []
    approx_tokens = []

    for r in accepted_records:
        q_len = 0
        a_len = 0
        total_len = 0
        for m in r.get("messages", []):
            content = m.get("content", "")
            total_len += len(content)
            if m.get("role") == "user":
                q_len += len(content)
            elif m.get("role") == "assistant":
                a_len += len(content)
        q_lens.append(q_len)
        a_lens.append(a_len)
        approx_tokens.append(int(round(total_len / 4.0)))

    avg_q_len = float(sum(q_lens) / len(q_lens)) if q_lens else 0.0
    avg_a_len = float(sum(a_lens) / len(a_lens)) if a_lens else 0.0
    avg_tokens = float(sum(approx_tokens) / len(approx_tokens)) if approx_tokens else 0.0

    rejection_reasons = Counter(reason for _, reason in rejected_records)

    stats = {
        "raw_records_count": len(raw_records),
        "accepted_records_count": len(accepted_records),
        "rejected_records_count": len(rejected_records),
        "exact_duplicates_filtered": duplicate_count,
        "near_duplicates_filtered": near_duplicate_count,
        "train_count": len(train_set),
        "validation_count": len(val_set),
        "test_count": len(test_set),
        "benchmark_count": len(benchmark_set) if benchmark_set is not None else 0,
        "category_distribution": dict(categories.most_common()),
        "difficulty_distribution": dict(difficulties),
        "average_question_length_chars": round(avg_q_len, 2),
        "average_answer_length_chars": round(avg_a_len, 2),
        "approximate_average_tokens": round(avg_tokens, 2),
        "min_tokens": min(approx_tokens) if approx_tokens else 0,
        "max_tokens": max(approx_tokens) if approx_tokens else 0,
        "rejection_breakdown": dict(rejection_reasons.most_common()),
    }
    return stats


def generate_markdown_stats(stats: Dict[str, Any]) -> str:
    """Format dataset statistics as standard Markdown."""
    lines = [
        "# Cybersecurity Dataset Statistics (Track B — Qwen 4B)",
        "",
        "## Summary Counts",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| Raw Records Ingested | {stats['raw_records_count']} |",
        f"| Accepted High-Quality Records | {stats['accepted_records_count']} |",
        f"| Rejected Records | {stats['rejected_records_count']} |",
        f"| Exact Duplicates Filtered | {stats['exact_duplicates_filtered']} |",
        f"| Near Duplicates Filtered | {stats['near_duplicates_filtered']} |",
        f"| Train Set (90%) | {stats['train_count']} |",
        f"| Validation Set (5%) | {stats['validation_count']} |",
        f"| Test Set (5%) | {stats['test_count']} |",
        f"| Independent Benchmark Set | {stats['benchmark_count']} |",
        "",
        "## Sequence Length Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Average Question Length (chars) | {stats['average_question_length_chars']} |",
        f"| Average Answer Length (chars) | {stats['average_answer_length_chars']} |",
        f"| Approximate Average Tokens | {stats['approximate_average_tokens']} |",
        f"| Min Sequence Tokens | {stats['min_tokens']} |",
        f"| Max Sequence Tokens | {stats['max_tokens']} |",
        "",
        "## Category Distribution",
        "",
        "| Category | Count | Percentage |",
        "|---|---:|---:|",
    ]

    total_acc = stats["accepted_records_count"]
    for cat, count in stats["category_distribution"].items():
        pct = (count / total_acc * 100) if total_acc > 0 else 0
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Difficulty Distribution",
        "",
        "| Difficulty | Count | Percentage |",
        "|---|---:|---:|",
    ])
    for diff, count in stats["difficulty_distribution"].items():
        pct = (count / total_acc * 100) if total_acc > 0 else 0
        lines.append(f"| {diff} | {count} | {pct:.1f}% |")

    if stats.get("rejection_breakdown"):
        lines.extend([
            "",
            "## Rejection Breakdown",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ])
        for rsn, count in stats["rejection_breakdown"].items():
            lines.append(f"| {rsn} | {count} |")

    return "\n".join(lines) + "\n"


def process_dataset(
    raw_records: List[Dict[str, Any]],
    output_dir: str | Path,
    benchmark_records: Optional[List[Dict[str, Any]]] = None,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    near_dedup_threshold: float = 0.80,
) -> Tuple[Dict[str, Any], Path, Path, Path, Optional[Path]]:
    """Complete pipeline: validate, deduplicate, split, persist, and report statistics."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    deduplicator = Deduplicator(near_dedup_threshold=near_dedup_threshold)
    accepted_records: List[Dict[str, Any]] = []
    rejected_records: List[Tuple[Dict[str, Any], str]] = []
    exact_dup_count = 0
    near_dup_count = 0

    for r in raw_records:
        is_valid, reason, cleaned_record = validate_record_content(r)
        if not is_valid:
            rejected_records.append((r, reason or "Validation failure"))
            continue

        # Check exact duplicate
        is_exact, exact_reason = deduplicator.is_exact_duplicate(cleaned_record)
        if is_exact:
            exact_dup_count += 1
            rejected_records.append((r, exact_reason))
            continue

        # Check near duplicate
        is_near, near_reason, _ = deduplicator.is_near_duplicate(cleaned_record)
        if is_near:
            near_dup_count += 1
            rejected_records.append((r, near_reason))
            continue

        # Add to accepted
        deduplicator.add(cleaned_record)
        accepted_records.append(cleaned_record)

    if not accepted_records:
        raise ValueError("All raw records were rejected or duplicated. Cannot create dataset splits.")

    train_set, val_set, test_set = deterministic_split(
        accepted_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_file = save_jsonl(train_set, out_path / "train.jsonl")
    val_file = save_jsonl(val_set, out_path / "val.jsonl")
    test_file = save_jsonl(test_set, out_path / "test.jsonl")

    benchmark_file = None
    if benchmark_records:
        # Verify benchmark records do not contaminate training
        contamination = check_split_contamination(train_set, benchmark_records)
        if contamination:
            raise ValueError(f"Benchmark records contaminate training data! Violations: {contamination}")
        bench_dir = out_path.parent / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        benchmark_file = save_jsonl(benchmark_records, bench_dir / "cyber_benchmark.jsonl")

    stats = compute_statistics(
        raw_records=raw_records,
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        duplicate_count=exact_dup_count,
        near_duplicate_count=near_dup_count,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        benchmark_set=benchmark_records,
    )

    # Save stats JSON and Markdown
    stats_json_path = out_path.parent / "stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    stats_md_path = out_path.parent / "stats.md"
    with open(stats_md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_stats(stats))

    return stats, train_file, val_file, test_file, benchmark_file
