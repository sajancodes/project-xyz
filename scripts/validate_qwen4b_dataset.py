"""Validate Qwen 4B Cybersecurity Datasets.

Performs strict automated validation on:
- data/qwen4b/processed/train.jsonl
- data/qwen4b/processed/val.jsonl
- data/qwen4b/processed/test.jsonl
- data/qwen4b/benchmarks/cyber_benchmark.jsonl

Checks:
- Schema adherence
- Message roles and non-empty content
- Content sanitization (no secrets, no placeholders, no truncation)
- Cross-split contamination (exact & near-duplicate isolation)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.distillation.deduplicate import check_split_contamination
from src.distillation.split import load_jsonl
from src.distillation.validate import validate_record_content


def validate_file(path: Path) -> bool:
    print(f"\n--- Validating: {path} ---")
    if not path.exists():
        print(f"FAILED: File does not exist: {path}")
        return False

    records = load_jsonl(path)
    print(f"Loaded {len(records)} records.")

    all_valid = True
    for idx, r in enumerate(records, start=1):
        is_valid, reason, _ = validate_record_content(r)
        if not is_valid:
            print(f"  [ERROR] Record #{idx} (id='{r.get('id', 'unknown')}'): {reason}")
            all_valid = False

    if all_valid:
        print(f"  [PASS] All {len(records)} records conform to schema and quality checks.")
    return all_valid


def main() -> bool:
    data_dir = REPO_ROOT / "data" / "qwen4b"
    train_path = data_dir / "processed" / "train.jsonl"
    val_path = data_dir / "processed" / "val.jsonl"
    test_path = data_dir / "processed" / "test.jsonl"
    bench_path = data_dir / "benchmarks" / "cyber_benchmark.jsonl"

    files_to_check = [train_path, val_path, test_path, bench_path]
    success = True

    for p in files_to_check:
        if not validate_file(p):
            success = False

    # Check contamination
    print("\n--- Checking Cross-Split Contamination ---")
    if train_path.exists():
        train_records = load_jsonl(train_path)

        for name, p in [("Validation", val_path), ("Test", test_path), ("Benchmark", bench_path)]:
            if p.exists():
                eval_records = load_jsonl(p)
                issues = check_split_contamination(train_records, eval_records)
                if issues:
                    print(f"  [CONTAMINATION ERROR] Train vs {name}: {issues}")
                    success = False
                else:
                    print(f"  [PASS] No contamination between Train and {name}.")

    if success:
        print("\nALL DATASET VALIDATIONS PASSED SUCCESSFULLY!\n")
    else:
        print("\nSOME DATASET VALIDATIONS FAILED!\n")

    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
