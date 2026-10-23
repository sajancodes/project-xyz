"""Tests for Google Drive Persistence and Resume Logic."""

from pathlib import Path
import pytest

from src.training.qwen4b.checkpointing import (
    GoogleDriveCheckpointManager,
    find_latest_checkpoint,
    get_checkpoint_step,
)


def test_get_checkpoint_step():
    assert get_checkpoint_step("checkpoint-25") == 25
    assert get_checkpoint_step(Path("/tmp/output/checkpoint-150")) == 150
    assert get_checkpoint_step("invalid_folder") == -1


def test_find_latest_checkpoint(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    (output_dir / "checkpoint-10").mkdir()
    (output_dir / "checkpoint-100").mkdir()
    (output_dir / "checkpoint-50").mkdir()
    (output_dir / "other_folder").mkdir()

    latest = find_latest_checkpoint(output_dir)
    assert latest is not None
    assert latest.name == "checkpoint-100"


def test_resume_priority(tmp_path):
    drive_root = tmp_path / "drive" / "MyDrive" / "ProjectXYZ" / "qwen3-4b-cyber"
    local_dir = tmp_path / "local_outputs"

    drive_ckpt_dir = drive_root / "checkpoints"
    drive_ckpt_dir.mkdir(parents=True)
    local_dir.mkdir(parents=True)

    # Local has checkpoint-50
    (local_dir / "checkpoint-50").mkdir()
    # Drive has checkpoint-100
    (drive_ckpt_dir / "checkpoint-100").mkdir()

    mgr = GoogleDriveCheckpointManager(drive_root)
    resume_target = mgr.get_resume_checkpoint(local_dir)

    # Drive checkpoint must take priority
    assert resume_target is not None
    assert "checkpoint-100" in resume_target
