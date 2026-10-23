"""Finite Deterministic Streaming SFT Dataset Multiplexer with Quotas and Decontamination.

Consumes 4 streaming sources:
- 45% OpenCoder (educational_instruct)
- 25% BigCode (self-oss-instruct-sc2-exec-filter-50k)
- 20% SmolTalk (smol-magpie-ultra)
- 10% Cybersecurity (Trendyol + curated seeds)
Total: ~16,000 unique examples.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import torch
from torch.utils.data import IterableDataset

import datasets
from src.evaluation.coding.tasks import CODING_BENCHMARK_TASKS
from src.evaluation.qwen4b.benchmark import get_all_benchmark_questions
from src.streaming.decontamination import BenchmarkDecontaminator
from src.streaming.filters import filter_sample
from src.streaming.normalizers import (
    normalize_bigcode,
    normalize_cybersecurity,
    normalize_opencoder,
    normalize_smoltalk,
)
from src.streaming.partitioning import PartitionType, get_partition_for_sample
from src.training.qwen4b.masking import create_assistant_masked_sample

logger = logging.getLogger(__name__)


@dataclass
class SourceQuota:
    name: str
    target_count: int
    accepted: int = 0
    rejected_quality: int = 0
    rejected_duplicate: int = 0
    rejected_too_long: int = 0
    rejected_contamination: int = 0
    exhausted: bool = False


class StreamSourceTracker:
    """Tracks and reports real streaming counts across all sources."""

    def __init__(self, quotas: Dict[str, int]) -> None:
        self.quotas = {name: SourceQuota(name=name, target_count=target) for name, target in quotas.items()}
        self.seen_hashes: Set[str] = set()
        self.seq_lengths: List[int] = []

    def is_quota_reached(self, source_name: str) -> bool:
        return self.quotas[source_name].accepted >= self.quotas[source_name].target_count

    def all_quotas_reached(self) -> bool:
        return all(q.accepted >= q.target_count or q.exhausted for q in self.quotas.values())

    def get_summary_table(self) -> str:
        lines = [
            "Source Statistics & Mixture Summary",
            "=" * 92,
            f"{'Source':<20s} | {'Target':<7s} | {'Accepted':<8s} | {'Rejected':<8s} | {'Duplicate':<9s} | {'Too Long':<8s} | {'Contam':<7s} | {'Actual %':<8s}",
            "-" * 92,
        ]
        total_accepted = sum(q.accepted for q in self.quotas.values())
        for q in self.quotas.values():
            pct = (q.accepted / total_accepted * 100.0) if total_accepted > 0 else 0.0
            lines.append(
                f"{q.name:<20s} | {q.target_count:<7d} | {q.accepted:<8d} | {q.rejected_quality:<8d} | "
                f"{q.rejected_duplicate:<9d} | {q.rejected_too_long:<8d} | {q.rejected_contamination:<7d} | {pct:<7.2f}%"
            )
        lines.append("=" * 92)
        lines.append(f"Total Unique Accepted Examples: {total_accepted:,}")
        return "\n".join(lines)

    def get_length_stats(self) -> Dict[str, float]:
        if not self.seq_lengths:
            return {"median": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
        import numpy as np
        arr = np.array(self.seq_lengths)
        return {
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }


def initialize_decontaminator() -> BenchmarkDecontaminator:
    """Load benchmark questions and tasks into the decontamination guard."""
    prompts = []
    # 1. Multi-domain benchmark questions
    for q in get_all_benchmark_questions():
        prompts.append({"id": q.id, "prompt": q.prompt})
    # 2. Coding tasks
    for c in CODING_BENCHMARK_TASKS:
        prompts.append({"id": c.task_id, "prompt": c.prompt})

    decontam = BenchmarkDecontaminator(prompts)
    logger.info("Initialized BenchmarkDecontaminator with %d benchmark prompts.", len(prompts))
    return decontam


class StreamSFTDataset(IterableDataset):
    """PyTorch IterableDataset that streams, filters, decontaminates, masks, and tokenizes samples on the fly."""

    def __init__(
        self,
        tokenizer: Any,
        partition: PartitionType = "train",
        max_seq_length: int = 2048,
        total_target: int = 16000,
        quotas: Optional[Dict[str, int]] = None,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.partition = partition
        self.max_seq_length = max_seq_length
        self.total_target = total_target
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        # Default 45/25/20/10 quotas
        if quotas is None:
            self.quotas = {
                "opencoder": int(round(total_target * 0.45)),    # 7,200
                "bigcode": int(round(total_target * 0.25)),      # 4,000
                "smoltalk": int(round(total_target * 0.20)),     # 3,200
                "cybersecurity": int(round(total_target * 0.10)),# 1,600
            }
        else:
            self.quotas = dict(quotas)

        self.tracker = StreamSourceTracker(self.quotas)
        self.decontaminator = initialize_decontaminator()

    def _open_streams(self) -> Dict[str, Iterator[Dict[str, Any]]]:
        """Open streaming iterators from Hugging Face with buffer shuffling."""
        streams = {}

        # 1. OpenCoder
        try:
            ds_opencoder = datasets.load_dataset(
                "OpenCoder-LLM/opc-sft-stage2",
                "educational_instruct",
                streaming=True,
                split="train",
            ).shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            streams["opencoder"] = iter(ds_opencoder)
        except Exception as e:
            logger.warning("Failed to open OpenCoder stream: %s", e)
            self.tracker.quotas["opencoder"].exhausted = True

        # 2. BigCode
        try:
            ds_bigcode = datasets.load_dataset(
                "bigcode/self-oss-instruct-sc2-exec-filter-50k",
                streaming=True,
                split="train",
            ).shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            streams["bigcode"] = iter(ds_bigcode)
        except Exception as e:
            logger.warning("Failed to open BigCode stream: %s", e)
            self.tracker.quotas["bigcode"].exhausted = True

        # 3. SmolTalk
        try:
            ds_smoltalk = datasets.load_dataset(
                "HuggingFaceTB/smoltalk",
                "smol-magpie-ultra",
                streaming=True,
                split="train",
            ).shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            streams["smoltalk"] = iter(ds_smoltalk)
        except Exception as e:
            logger.warning("Failed to open SmolTalk stream: %s", e)
            self.tracker.quotas["smoltalk"].exhausted = True

        # 4. Cybersecurity
        try:
            ds_cyber = datasets.load_dataset(
                "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
                streaming=True,
                split="train",
            ).shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            streams["cybersecurity"] = iter(ds_cyber)
        except Exception as e:
            logger.warning("Failed to open Trendyol cybersecurity stream: %s", e)
            self.tracker.quotas["cybersecurity"].exhausted = True

        return streams

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        streams = self._open_streams()
        normalizers = {
            "opencoder": normalize_opencoder,
            "bigcode": normalize_bigcode,
            "smoltalk": normalize_smoltalk,
            "cybersecurity": normalize_cybersecurity,
        }

        # Weighted round-robin interleaved stream
        source_order = ["opencoder", "bigcode", "smoltalk", "cybersecurity"]

        while not self.tracker.all_quotas_reached() and any(not q.exhausted for q in self.tracker.quotas.values()):
            yielded_any = False

            for s_name in source_order:
                if s_name not in streams or self.tracker.is_quota_reached(s_name) or self.tracker.quotas[s_name].exhausted:
                    continue

                stream_it = streams[s_name]
                norm_fn = normalizers[s_name]
                found_valid = False

                while not found_valid and not self.tracker.is_quota_reached(s_name):
                    try:
                        raw_sample = next(stream_it)
                    except StopIteration:
                        logger.info("Source '%s' exhausted stream.", s_name)
                        self.tracker.quotas[s_name].exhausted = True
                        break
                    except Exception as e:
                        logger.warning("Error reading from stream '%s': %s", s_name, e)
                        self.tracker.quotas[s_name].exhausted = True
                        break

                    norm_record = norm_fn(raw_sample)
                    if norm_record is None:
                        self.tracker.quotas[s_name].rejected_quality += 1
                        continue

                    user_content = norm_record["messages"][1]["content"]
                    assistant_content = norm_record["messages"][2]["content"]

                    # 1. Quality filter
                    passes_quality, reason = filter_sample(user_content, assistant_content)
                    if not passes_quality:
                        self.tracker.quotas[s_name].rejected_quality += 1
                        continue

                    # 2. Decontamination
                    is_contam, contam_msg = self.decontaminator.is_contaminated(user_content, assistant_content)
                    if is_contam:
                        self.tracker.quotas[s_name].rejected_contamination += 1
                        continue

                    # 3. Exact deduplication
                    content_hash = hashlib.sha256((user_content.strip() + "::" + assistant_content.strip()).encode("utf-8")).hexdigest()
                    if content_hash in self.tracker.seen_hashes:
                        self.tracker.quotas[s_name].rejected_duplicate += 1
                        continue
                    self.tracker.seen_hashes.add(content_hash)

                    # 4. Partition assignment
                    target_partition = get_partition_for_sample(user_content, assistant_content)
                    if target_partition != self.partition:
                        # Belongs to another partition (val or test)
                        continue

                    # 5. Tokenization & Assistant-only masking
                    sample = create_assistant_masked_sample(
                        tokenizer=self.tokenizer,
                        messages=norm_record["messages"],
                        max_length=self.max_seq_length,
                    )
                    seq_len = len(sample["input_ids"])
                    self.tracker.seq_lengths.append(seq_len)

                    self.tracker.quotas[s_name].accepted += 1
                    found_valid = True
                    yielded_any = True
                    yield sample

            if not yielded_any:
                break
