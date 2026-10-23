"""Tests for Colab Notebook Structure, Cell Sequence, and Training Trigger."""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_notebook_structure_and_final_training_cell():
    nb_path = REPO_ROOT / "notebooks" / "qwen3_4b_cyber_qlora_colab.ipynb"
    assert nb_path.exists(), f"Colab notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    cells = nb["cells"]
    assert len(cells) == 24, f"Expected exactly 24 cells in notebook, found {len(cells)}"

    # Check required text patterns in specific cells
    # Cell 1: Overview
    assert "Track B" in "".join(cells[0]["source"])

    # Cell 2: GPU Check
    assert "nvidia-smi" in "".join(cells[1]["source"])

    # Cell 5: Google Drive
    assert "drive.mount" in "".join(cells[4]["source"])

    # Cell 6: Config variables
    c6 = "".join(cells[5]["source"])
    assert "BASE_MODEL" in c6
    assert "MAX_SEQ_LENGTH" in c6
    assert "LORA_R" in c6
    assert "LORA_ALPHA" in c6

    # Cell 16: FINAL TRAINING CELL
    c16 = "".join(cells[15]["source"])
    assert "FINAL TRAINING CELL" in c16
    assert "trainer.train" in c16

    # Cell 17: Save Adapter
    assert "save_pretrained" in "".join(cells[16]["source"])

    # Cell 19: Reload Adapter
    assert "load_lora_adapter" in "".join(cells[18]["source"])

    # Cell 21: Post-training benchmark
    assert "run_benchmark" in "".join(cells[20]["source"])

    # Cell 22: Comparison report
    assert "compare_benchmarks" in "".join(cells[21]["source"])
