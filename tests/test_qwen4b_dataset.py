"""Tests for Dataset Schema, Deduplication, Splitting, and Contamination Prevention."""

import json
from pathlib import Path
import pytest

from src.distillation.deduplicate import (
    Deduplicator,
    check_split_contamination,
    compute_exact_hash,
    jaccard_similarity,
)
from src.distillation.split import deterministic_split, load_jsonl, save_jsonl
from src.distillation.validate import validate_record_content, validate_schema


def test_schema_valid_record():
    valid_record = {
        "id": "test_001",
        "category": "networking",
        "difficulty": "intermediate",
        "source": "unit-test",
        "tags": ["test"],
        "messages": [
            {"role": "user", "content": "Explain how the TCP 3-way handshake operates in detail."},
            {"role": "assistant", "content": "The TCP handshake uses SYN, SYN-ACK, and ACK to synchronize sequence numbers safely."},
        ],
    }
    is_valid, err = validate_schema(valid_record)
    assert is_valid is True
    assert err is None

    ok, reason, cleaned = validate_record_content(valid_record)
    assert ok is True
    assert reason is None
    assert cleaned["id"] == "test_001"


def test_schema_invalid_record_missing_field():
    invalid_record = {
        "id": "test_bad",
        "category": "networking",
        # missing messages
    }
    is_valid, err = validate_schema(invalid_record)
    assert is_valid is False
    assert "messages" in err


def test_schema_invalid_roles():
    bad_role = {
        "id": "test_bad_role",
        "category": "networking",
        "difficulty": "intermediate",
        "source": "unit-test",
        "tags": [],
        "messages": [
            {"role": "human", "content": "Invalid role keyword"},
            {"role": "assistant", "content": "Response content here for unit testing."},
        ],
    }
    is_valid, err = validate_schema(bad_role)
    assert is_valid is False
    assert "invalid role" in err.lower()


def test_schema_rejects_empty_content():
    empty_content = {
        "id": "test_empty",
        "category": "networking",
        "difficulty": "intermediate",
        "source": "unit-test",
        "tags": [],
        "messages": [
            {"role": "user", "content": "Valid question here"},
            {"role": "assistant", "content": "   "},
        ],
    }
    is_valid, err = validate_schema(empty_content)
    assert is_valid is False
    assert "empty" in err.lower()


def test_exact_deduplication():
    dedup = Deduplicator()
    rec1 = {
        "id": "rec_1",
        "messages": [{"role": "user", "content": "What is the difference between TCP and UDP?"}],
    }
    rec2 = {
        "id": "rec_2",
        "messages": [{"role": "user", "content": "What is the difference between TCP and UDP?   "}],
    }

    assert dedup.is_exact_duplicate(rec1)[0] is False
    dedup.add(rec1)

    # rec2 has same normalized prompt
    is_dup, reason = dedup.is_exact_duplicate(rec2)
    assert is_dup is True
    assert "rec_1" in reason


def test_near_deduplication():
    dedup = Deduplicator(near_dedup_threshold=0.80)
    rec1 = {
        "id": "rec_alpha",
        "messages": [{"role": "user", "content": "Can you explain how a SYN flood denial of service attack targets the TCP handshake?"}],
    }
    rec2 = {
        "id": "rec_beta",
        "messages": [{"role": "user", "content": "Could you explain how a SYN flood denial of service attack targets the TCP handshake mechanism?"}],
    }

    dedup.add(rec1)
    is_near, reason, sim = dedup.is_near_duplicate(rec2)
    assert is_near is True
    assert sim >= 0.80


def test_deterministic_split(tmp_path):
    records = [
        {
            "id": f"rec_{i}",
            "messages": [
                {"role": "user", "content": f"Unique question prompt index {i} for testing splitting."},
                {"role": "assistant", "content": f"Detailed answer explaining the technical concept {i}."},
            ],
        }
        for i in range(20)
    ]

    train1, val1, test1 = deterministic_split(records, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)
    train2, val2, test2 = deterministic_split(records, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42)

    # Determinism
    assert [r["id"] for r in train1] == [r["id"] for r in train2]
    assert [r["id"] for r in val1] == [r["id"] for r in val2]
    assert [r["id"] for r in test1] == [r["id"] for r in test2]

    # No overlap
    train_ids = {r["id"] for r in train1}
    val_ids = {r["id"] for r in val1}
    test_ids = {r["id"] for r in test1}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_split_contamination_detection():
    train_set = [
        {
            "id": "tr_1",
            "messages": [{"role": "user", "content": "How does cross-site scripting XSS occur in web applications?"}],
        }
    ]
    eval_set_contaminated = [
        {
            "id": "ev_1",
            "messages": [{"role": "user", "content": "How does cross-site scripting XSS occur in web applications?"}],
        }
    ]
    violations = check_split_contamination(train_set, eval_set_contaminated)
    assert len(violations) > 0
    assert "exact prompt duplicate" in violations[0].lower()


def test_jsonl_save_and_load(tmp_path):
    test_file = tmp_path / "test.jsonl"
    data = [{"id": "1", "val": "a"}, {"id": "2", "val": "b"}]
    save_jsonl(data, test_file)

    loaded = load_jsonl(test_file)
    assert loaded == data
