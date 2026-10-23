"""Google Drive Persistence, Backup Callback, and Resume Logic.

Persists checkpoints and adapters to:
    MyDrive/ProjectXYZ/qwen3-4b-cyber/
        ├── checkpoints/
        ├── adapters/
        ├── benchmarks/
        ├── logs/
        └── reports/
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


def get_checkpoint_step(path_or_name: str | Path) -> int:
    """Extract step number from checkpoint directory name (e.g. checkpoint-150 -> 150)."""
    name = Path(path_or_name).name
    match = re.search(r"checkpoint-(\d+)", name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(directory: str | Path) -> Optional[Path]:
    """Find the checkpoint directory with the highest step number."""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        return None

    checkpoints = [
        p for p in dir_path.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None

    return max(checkpoints, key=get_checkpoint_step)


def copy_tree_safe(src: Path, dst: Path) -> None:
    """Safely copy directory tree, creating destination if needed."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            copy_tree_safe(item, dest_item)
        else:
            shutil.copy2(item, dest_item)


class GoogleDriveCheckpointManager:
    """Manages syncing between local training directory and persistent Google Drive."""

    def __init__(self, drive_root: str | Path) -> None:
        self.drive_root = Path(drive_root)
        self.checkpoints_dir = self.drive_root / "checkpoints"
        self.adapters_dir = self.drive_root / "adapters"
        self.benchmarks_dir = self.drive_root / "benchmarks"
        self.logs_dir = self.drive_root / "logs"
        self.reports_dir = self.drive_root / "reports"

    def is_drive_mounted(self) -> bool:
        """Check if Google Drive mount root exists and is accessible."""
        return self.drive_root.parent.exists()

    def ensure_directories(self) -> None:
        """Create standard folder structure in Google Drive if mounted."""
        if not self.is_drive_mounted():
            logger.warning("Google Drive root '%s' is not mounted or accessible.", self.drive_root)
            return

        for d in [
            self.checkpoints_dir,
            self.adapters_dir,
            self.benchmarks_dir,
            self.logs_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Google Drive directories verified at '%s'", self.drive_root)

    def sync_checkpoint_to_drive(self, local_checkpoint_path: Path) -> Optional[Path]:
        """Copy a local checkpoint directory to Google Drive."""
        if not self.is_drive_mounted():
            return None
        self.ensure_directories()
        dest = self.checkpoints_dir / local_checkpoint_path.name
        logger.info("Syncing checkpoint '%s' -> '%s'...", local_checkpoint_path.name, dest)
        copy_tree_safe(local_checkpoint_path, dest)
        return dest

    def sync_adapter_to_drive(self, local_adapter_path: Path, adapter_name: str = "final_cyber_adapter") -> Optional[Path]:
        """Copy trained adapter to Google Drive."""
        if not self.is_drive_mounted():
            return None
        self.ensure_directories()
        dest = self.adapters_dir / adapter_name
        logger.info("Syncing adapter to Drive: '%s'", dest)
        copy_tree_safe(local_adapter_path, dest)
        return dest

    def get_resume_checkpoint(self, local_dir: Path) -> Optional[str]:
        """Check Google Drive then local directory for the latest checkpoint to resume."""
        # 1. Check Google Drive first
        if self.is_drive_mounted() and self.checkpoints_dir.exists():
            drive_latest = find_latest_checkpoint(self.checkpoints_dir)
            if drive_latest is not None:
                logger.info("Found resume checkpoint on Google Drive: '%s'", drive_latest)
                return str(drive_latest)

        # 2. Check local output directory
        if local_dir.exists():
            local_latest = find_latest_checkpoint(local_dir)
            if local_latest is not None:
                logger.info("Found resume checkpoint in local directory: '%s'", local_latest)
                return str(local_latest)

        return None


class DriveBackupCallback(TrainerCallback):
    """HuggingFace TrainerCallback that automatically backs up saves to Google Drive."""

    def __init__(self, drive_manager: GoogleDriveCheckpointManager) -> None:
        self.drive_manager = drive_manager

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Triggered whenever Trainer saves a checkpoint."""
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            try:
                self.drive_manager.sync_checkpoint_to_drive(checkpoint_dir)
            except Exception as e:
                logger.warning("Failed to sync checkpoint to Google Drive: %s", e)
        return control
