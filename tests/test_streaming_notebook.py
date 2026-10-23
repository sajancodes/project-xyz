"""Unit Tests for Experiment V2 Streaming Colab Notebook."""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_streaming_colab_notebook_structure():
    nb_path = REPO_ROOT / "notebooks" / "qwen3_4b_coding_stream_qlora_colab.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) >= 20, f"Expected at least 20 cells, got {len(cells)}"

    # Check for critical training markers
    all_source = "\n".join("".join(c.get("source", [])) for c in cells)

    assert "FINAL STREAMING TRAINING CELL" in all_source
    assert "qwen3-4b-coding-stream-v2" in all_source
    assert "MAX_SEQ_LENGTH = 2048" in all_source
    assert "MAX_STEPS = 1000" in all_source
    assert "evaluate_coding_benchmark" in all_source
    assert "StreamSFTDataset" in all_source
