"""Sampler to Materialize Finite Validation and Test Datasets from Streams.

Extracts a fixed number of validation (e.g. 600) and test (e.g. 400) examples
from the streaming mixture to ensure Trainer evaluation terminates and computes exact losses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import datasets

from src.distillation.split import save_jsonl
from src.streaming.decontamination import BenchmarkDecontaminator
from src.streaming.filters import filter_sample
from src.streaming.normalizers import (
    normalize_bigcode,
    normalize_cybersecurity,
    normalize_opencoder,
    normalize_smoltalk,
)
from src.streaming.partitioning import get_partition_for_sample
from src.streaming.streamer import initialize_decontaminator

logger = logging.getLogger(__name__)


def build_finite_eval_datasets(
    output_dir: str | Path = "data/qwen4b/processed",
    val_target: int = 600,
    test_target: int = 400,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Stream and materialize finite validation and test datasets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    val_file = out_path / "stream_val.jsonl"
    test_file = out_path / "stream_test.jsonl"

    if val_file.exists() and test_file.exists():
        logger.info("Finite evaluation datasets already exist: %s, %s", val_file, test_file)
        return val_file, test_file

    decontaminator = initialize_decontaminator()
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # Quotas for eval mixture: 45% OpenCoder, 25% BigCode, 20% SmolTalk, 10% Cyber
    eval_sources = [
        ("opencoder", "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", normalize_opencoder, 0.45),
        ("bigcode", "bigcode/self-oss-instruct-sc2-exec-filter-50k", None, normalize_bigcode, 0.25),
        ("smoltalk", "HuggingFaceTB/smoltalk", "smol-magpie-ultra", normalize_smoltalk, 0.20),
        ("cybersecurity", "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", None, normalize_cybersecurity, 0.10),
    ]

    for name, repo, subset, norm_fn, ratio in eval_sources:
        needed_val = max(5, int(val_target * ratio))
        needed_test = max(5, int(test_target * ratio))
        src_val = 0
        src_test = 0

        logger.info("Extracting eval samples from '%s' (val target: %d, test target: %d)...", name, needed_val, needed_test)
        try:
            kwargs: Dict[str, Any] = {"streaming": True, "split": "train"}
            if subset:
                ds = datasets.load_dataset(repo, subset, **kwargs)
            else:
                ds = datasets.load_dataset(repo, **kwargs)
            ds = ds.shuffle(seed=seed, buffer_size=5000)

            for raw in ds:
                if src_val >= needed_val and src_test >= needed_test:
                    break

                rec = norm_fn(raw)
                if rec is None:
                    continue

                u = rec["messages"][1]["content"]
                a = rec["messages"][2]["content"]

                if not filter_sample(u, a)[0]:
                    continue
                if decontaminator.is_contaminated(u, a)[0]:
                    continue

                part = get_partition_for_sample(u, a)
                if part == "validation" and src_val < needed_val:
                    val_records.append(rec)
                    src_val += 1
                elif part == "test" and src_test < needed_test:
                    test_records.append(rec)
                    src_test += 1

        except Exception as e:
            logger.warning("Error fetching eval samples from %s: %s", name, e)

    save_jsonl(val_records, val_file)
    save_jsonl(test_records, test_file)
    logger.info("Saved %d validation records to %s", len(val_records), val_file)
    logger.info("Saved %d test records to %s", len(test_records), test_file)

    return val_file, test_file
