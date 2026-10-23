"""Run Benchmarks and Generate Before/After Comparison for Qwen 4B.

Usage:
    # Run Baseline Benchmark (before QLoRA)
    python scripts/benchmark_qwen4b.py --tag base

    # Run Post-Training Benchmark (after QLoRA) and compare
    python scripts/benchmark_qwen4b.py --tag qlora --adapter_path outputs/qwen3_4b_cyber/final_adapter --compare_with benchmarks/base_benchmark_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.qwen4b.benchmark import run_benchmark
from src.evaluation.qwen4b.comparison import compare_benchmarks
from src.models.qwen4b.inference import load_lora_adapter
from src.models.qwen4b.loader import load_qwen_model, load_qwen_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen 4B")
    parser.add_argument("--config", default="configs/qwen3_4b_t4.yaml", help="Path to config file")
    parser.add_argument("--tag", default="base", help="Evaluation tag (e.g. base, qlora)")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter to attach")
    parser.add_argument("--output_dir", default="benchmarks", help="Output directory for benchmark results")
    parser.add_argument("--compare_with", default=None, help="Path to baseline benchmark JSON for comparison")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model_id"]
    print(f"Loading tokenizer from '{model_id}'...")
    tokenizer = load_qwen_tokenizer(model_id)

    print(f"Loading model '{model_id}'...")
    model = load_qwen_model(
        model_id_or_path=model_id,
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        compute_dtype=cfg["model"].get("bnb_4bit_compute_dtype", "float16"),
        device_map=cfg["model"].get("device_map", "auto"),
    )

    if args.adapter_path:
        print(f"Attaching LoRA adapter from '{args.adapter_path}'...")
        model = load_lora_adapter(model, args.adapter_path)

    results = run_benchmark(
        model=model,
        tokenizer=tokenizer,
        tag=args.tag,
        output_dir=args.output_dir,
    )

    print(f"\nBenchmark '{args.tag}' completed! Overall Score: {results['overall_score']:.4f}")
    print("Category Breakdown:")
    for cat, score in results["category_scores"].items():
        print(f"  - {cat:28s}: {score:.4f}")

    if args.compare_with:
        comp_md = compare_benchmarks(
            base_results=args.compare_with,
            qlora_results=results,
            output_md_path=Path(args.output_dir) / "comparison_report.md",
        )
        print("\n" + comp_md)


if __name__ == "__main__":
    main()
