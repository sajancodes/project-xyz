"""Tests for Qwen 4B QLoRA Configuration Parsing."""

from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_config_structure_and_types():
    config_path = REPO_ROOT / "configs" / "qwen3_4b_t4.yaml"
    assert config_path.exists(), "Configuration file missing"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Required top-level sections
    for sec in ["model", "lora", "training", "data", "benchmark"]:
        assert sec in cfg, f"Missing section: {sec}"

    # Model settings
    model_cfg = cfg["model"]
    assert model_cfg["base_model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert model_cfg["load_in_4bit"] is True
    assert model_cfg["bnb_4bit_quant_type"] == "nf4"
    assert model_cfg["bnb_4bit_use_double_quant"] is True
    assert model_cfg["bnb_4bit_compute_dtype"] == "float16"

    # LoRA settings
    lora_cfg = cfg["lora"]
    assert isinstance(lora_cfg["r"], int) and lora_cfg["r"] > 0
    assert isinstance(lora_cfg["lora_alpha"], int) and lora_cfg["lora_alpha"] > 0
    assert 0.0 <= lora_cfg["lora_dropout"] <= 0.2
    assert "target_modules" in lora_cfg
    assert len(lora_cfg["target_modules"]) >= 4

    # Training settings for T4
    train_cfg = cfg["training"]
    assert train_cfg["micro_batch_size"] == 1
    assert train_cfg["gradient_accumulation_steps"] >= 8
    assert train_cfg["fp16"] is True
    assert train_cfg["bf16"] is False
    assert train_cfg["gradient_checkpointing"] is True
    assert train_cfg["optim"] == "paged_adamw_8bit"
