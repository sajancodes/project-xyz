#!/usr/bin/env python3
"""
Centralized project paths for Project XYZ.

All scripts under src/ should import this module so that file paths
remain correct regardless of the current working directory:

    from paths import PROJECT_ROOT, TOKENIZER_PATH, CHECKPOINT_106K, ...

Importing this module also adds the src/ subfolders to sys.path so that
the existing flat imports continue to work:

    from model import SmallEnglishLLM
    from model_config import ModelConfig
"""

import os
import sys

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
#
# This file lives at <root>/src/paths.py, so the project root is
# the parent of this directory.

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


# ------------------------------------------------------------
# sys.path: allow flat imports (model, model_config, ...)
# ------------------------------------------------------------

for _sub in (
    "models",
    "config",
    "training",
    "evaluation",
    "utils",
):
    _path = os.path.join(SRC_DIR, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ------------------------------------------------------------
# Standard paths
# ------------------------------------------------------------

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer.json")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FINEWEB_DATA_DIR = os.path.join(DATA_DIR, "fineweb")

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
PRETRAIN_DIR = os.path.join(CHECKPOINTS_DIR, "pretrain")
SEMANTIC_DIR = os.path.join(CHECKPOINTS_DIR, "semantic")
MIXED_DIR = os.path.join(CHECKPOINTS_DIR, "mixed")


# ------------------------------------------------------------
# Well-known checkpoints
# ------------------------------------------------------------

CHECKPOINT_106K = os.path.join(PRETRAIN_DIR, "checkpoint-106k.pt")
CHECKPOINT_FINEWEB_LATEST = os.path.join(PRETRAIN_DIR, "checkpoint_fineweb.pt")
CHECKPOINT_SEMANTIC_5K = os.path.join(SEMANTIC_DIR, "checkpoint-semantic-5000.pt")
CHECKPOINT_MIXED_2K = os.path.join(MIXED_DIR, "checkpoint-mixed-2k.pt")
CHECKPOINT_MIXED_LATEST = os.path.join(MIXED_DIR, "checkpoint_mixed_10k.pt")


def pretrain_checkpoint(step):
    """Absolute path for a pretraining milestone checkpoint."""
    return os.path.join(PRETRAIN_DIR, f"checkpoint-{step}.pt")


def semantic_checkpoint(step):
    """Absolute path for a semantic fine-tuning checkpoint."""
    return os.path.join(SEMANTIC_DIR, f"checkpoint-semantic-{step}.pt")


def mixed_checkpoint(step):
    """Absolute path for a mixed training milestone checkpoint."""
    return os.path.join(MIXED_DIR, f"checkpoint-mixed-{step}.pt")
