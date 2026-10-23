"""Unit Tests for Streaming SFT Pipeline, Normalizers, Filters, and Quotas."""

from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from src.streaming.decontamination import BenchmarkDecontaminator
from src.streaming.filters import filter_sample
from src.streaming.normalizers import (
    normalize_bigcode,
    normalize_cybersecurity,
    normalize_opencoder,
    normalize_smoltalk,
)
from src.streaming.partitioning import get_partition_for_sample
from src.streaming.streamer import SourceQuota, StreamSourceTracker


def test_opencoder_normalizer():
    sample = {
        "instruction": "Write a binary search in Python.",
        "output": "def binary_search(arr, target): return 0",
        "code": "def binary_search(arr, target): return 0",
        "entry_point": "binary_search",
        "seq_id": 101,
    }
    res = normalize_opencoder(sample)
    assert res is not None
    assert res["id"] == "opencoder_101"
    assert len(res["messages"]) == 3
    assert res["messages"][1]["content"] == "Write a binary search in Python."
    assert res["messages"][2]["content"] == "def binary_search(arr, target): return 0"
    assert res["category"] == "coding_educational"


def test_bigcode_normalizer():
    sample = {
        "instruction": "Write an LRU cache implementation.",
        "response": "class LRUCache: pass",
        "id": "bigcode_task_55",
        "concepts": ["data-structures"],
    }
    res = normalize_bigcode(sample)
    assert res is not None
    assert res["id"] == "bigcode_bigcode_task_55"
    assert res["category"] == "coding_exec_validated"
    assert res["messages"][1]["content"] == "Write an LRU cache implementation."


def test_smoltalk_normalizer():
    sample = {
        "messages": [
            {"role": "user", "content": "Explain photosynthesis."},
            {"role": "assistant", "content": "Plants convert sunlight, CO2, and water into glucose."},
        ],
        "category": "science",
        "difficulty": "beginner",
    }
    res = normalize_smoltalk(sample)
    assert res is not None
    assert len(res["messages"]) == 3
    assert "photosynthesis" in res["messages"][1]["content"]


def test_cybersecurity_normalizer():
    sample = {
        "user": "What is an intrusion detection system?",
        "assistant": "An IDS inspects network traffic for suspicious signatures.",
        "category": "networking",
    }
    res = normalize_cybersecurity(sample)
    assert res is not None
    assert res["messages"][1]["content"] == "What is an intrusion detection system?"
    assert res["messages"][2]["content"] == "An IDS inspects network traffic for suspicious signatures."


def test_deterministic_partitioning_stability():
    inst = "Implement quicksort algorithm in Python."
    resp = "def quicksort(arr): return arr"

    # Same input must always produce the exact same partition
    part1 = get_partition_for_sample(inst, resp)
    part2 = get_partition_for_sample(inst, resp)
    assert part1 == part2
    assert part1 in ["train", "validation", "test"]


def test_benchmark_decontamination():
    decontam = BenchmarkDecontaminator([
        {"id": "bench_01", "prompt": "Explain the difference between TCP SYN flood attacks and UDP amplification attacks."},
    ])

    # Contaminated prompt
    contam_user = "Explain the difference between TCP SYN flood attacks and UDP amplification attacks."
    is_contam, reason = decontam.is_contaminated(contam_user)
    assert is_contam is True
    assert "bench_01" in reason

    # Clean prompt
    clean_user = "How does Python handle garbage collection with reference counts?"
    is_contam2, _ = decontam.is_contaminated(clean_user)
    assert is_contam2 is False


def test_quality_filtering():
    # Pass
    ok, _ = filter_sample("Write a quicksort function in Python.", "def quicksort(arr): return sorted(arr)")
    assert ok is True

    # Empty
    bad1, reason1 = filter_sample("", "def foo(): pass")
    assert bad1 is False
    assert "empty" in reason1.lower()

    # Placeholder
    bad2, reason2 = filter_sample("How do I hack a server?", "As an AI language model, I cannot help you.")
    assert bad2 is False
    assert "placeholder" in reason2.lower()

    # Broken code fence
    bad3, reason3 = filter_sample("Write code.", "```python\ndef broken():\n    return 1")
    assert bad3 is False
    assert "unbalanced" in reason3.lower()


def test_quota_tracker():
    quotas = {"opencoder": 2, "bigcode": 1}
    tracker = StreamSourceTracker(quotas)

    assert not tracker.is_quota_reached("opencoder")
    tracker.quotas["opencoder"].accepted += 1
    assert not tracker.is_quota_reached("opencoder")
    tracker.quotas["opencoder"].accepted += 1
    assert tracker.is_quota_reached("opencoder")

    assert not tracker.all_quotas_reached()
    tracker.quotas["bigcode"].accepted += 1
    assert tracker.all_quotas_reached()

    summary = tracker.get_summary_table()
    assert "opencoder" in summary
    assert "bigcode" in summary
