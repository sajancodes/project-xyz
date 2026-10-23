"""QLoRA Training Script for Qwen 4B Instruct on NVIDIA T4.

Features:
- NF4 4-bit loading via bitsandbytes
- Assistant-only loss masking
- Google Drive checkpoint persistence & resume
- Memory-efficient optimizer (paged_adamw_8bit)
- T4 OOM protection & smoke forward/backward step
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import torch
from transformers import Trainer, TrainingArguments

from src.models.qwen4b.loader import get_model_parameters_info, load_qwen_model, load_qwen_tokenizer
from src.models.qwen4b.lora import apply_qlora, get_qlora_config
from src.training.qwen4b.checkpointing import DriveBackupCallback, GoogleDriveCheckpointManager
from src.training.qwen4b.collator import DataCollatorForAssistantOnlyLM
from src.training.qwen4b.dataset import CyberChatDataset
from src.training.qwen4b.utils import clear_cuda_memory, print_gpu_environment, run_smoke_forward_backward_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML training configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(
    config_path: str = "configs/qwen3_4b_t4.yaml",
    resume_from_checkpoint: Optional[str] = None,
    drive_root: Optional[str] = None,
) -> Any:
    """Execute complete QLoRA training pipeline."""
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # 1. Print environment
    print_gpu_environment()

    # 2. Checkpoint manager
    drive_path = drive_root or train_cfg.get("drive_backup_dir", "/content/drive/MyDrive/ProjectXYZ/qwen3-4b-cyber")
    drive_manager = GoogleDriveCheckpointManager(drive_path)
    drive_manager.ensure_directories()

    # Determine resume checkpoint
    local_output_dir = Path(train_cfg.get("output_dir", "outputs/qwen3_4b_cyber"))
    resume_target = resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")
    if not resume_target:
        resume_target = drive_manager.get_resume_checkpoint(local_output_dir)

    # 3. Load Tokenizer
    tokenizer = load_qwen_tokenizer(
        model_id_or_path=model_cfg["base_model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # 4. Load Base Model in NF4 4-bit
    model = load_qwen_model(
        model_id_or_path=model_cfg["base_model_id"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        compute_dtype=model_cfg.get("bnb_4bit_compute_dtype", "float16"),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_cache=False,
        enable_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )

    # 5. Attach LoRA Adapter
    lora_config = get_qlora_config(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
    )
    model = apply_qlora(model, lora_config=lora_config, prepare_kbit=True)

    params_info = get_model_parameters_info(model)
    logger.info("Model parameters info: %s", params_info)

    # 6. Smoke forward/backward pass
    smoke_ok, smoke_msg = run_smoke_forward_backward_step(model, tokenizer, max_seq_len=128)
    if not smoke_ok:
        raise RuntimeError(f"Smoke test failed! {smoke_msg}")

    # 7. Load Datasets
    max_seq_len = train_cfg.get("max_seq_length", 1024)
    train_dataset = CyberChatDataset(
        data_source=data_cfg["train_path"],
        tokenizer=tokenizer,
        max_length=max_seq_len,
    )
    val_dataset = CyberChatDataset(
        data_source=data_cfg["val_path"],
        tokenizer=tokenizer,
        max_length=max_seq_len,
    )
    collator = DataCollatorForAssistantOnlyLM(tokenizer=tokenizer)

    # 8. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(local_output_dir),
        per_device_train_batch_size=train_cfg.get("micro_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        num_train_epochs=int(train_cfg.get("num_train_epochs", 3)),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=int(train_cfg.get("eval_steps", 25)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        seed=int(train_cfg.get("seed", 42)),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    # 9. Trainer with Drive Callback
    callbacks = [DriveBackupCallback(drive_manager)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # 10. Train
    logger.info("Starting QLoRA training... (Resume target: %s)", resume_target)
    train_result = trainer.train(resume_from_checkpoint=resume_target)

    # 11. Save final adapter and tokenizer locally and to Drive
    final_adapter_dir = local_output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    logger.info("Saved final adapter locally to '%s'", final_adapter_dir)

    drive_manager.sync_adapter_to_drive(final_adapter_dir, adapter_name="qwen3_4b_cyber_qlora_final")

    return train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen 4B Cybersecurity QLoRA")
    parser.add_argument("--config", default="configs/qwen3_4b_t4.yaml", help="Path to config file")
    parser.add_argument("--resume", default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--drive_root", default=None, help="Google Drive backup directory")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        drive_root=args.drive_root,
    )
