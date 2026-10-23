"""Streaming SFT Training Script for Qwen 4B Coding & Capability Specialization.

Features:
- Progressive streaming consumption without full disk materialization
- 16,000 unique target mixture (45% OpenCoder, 25% BigCode, 20% SmolTalk, 10% Cybersecurity)
- Sequence length: 2048 with assistant-only loss masking
- Early stopping with patience on finite validation set
- Live training visibility & Google Drive auto-resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from src.models.qwen4b.loader import get_model_parameters_info, load_qwen_model, load_qwen_tokenizer
from src.models.qwen4b.lora import apply_qlora, get_qlora_config
from src.streaming.sampler import build_finite_eval_datasets
from src.streaming.streamer import StreamSFTDataset
from src.training.qwen4b.callbacks import LiveTrainingProgressCallback
from src.training.qwen4b.checkpointing import DriveBackupCallback, GoogleDriveCheckpointManager
from src.training.qwen4b.collator import DataCollatorForAssistantOnlyLM
from src.training.qwen4b.dataset import CyberChatDataset
from src.training.qwen4b.utils import print_gpu_environment, run_smoke_forward_backward_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_streaming_training(
    config_path: str = "configs/qwen3_4b_stream_coding_t4.yaml",
    resume_from_checkpoint: Optional[str] = None,
    drive_root: Optional[str] = None,
) -> Any:
    """Execute complete 16K streaming SFT training pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    stream_cfg = cfg["streaming"]

    # 1. Inspect environment
    print_gpu_environment()

    # 2. Configure Google Drive persistence
    drive_path = drive_root or train_cfg.get("drive_backup_dir", "/content/drive/MyDrive/ProjectXYZ/qwen3-4b-coding-stream-v2")
    drive_manager = GoogleDriveCheckpointManager(drive_path)
    drive_manager.ensure_directories()

    local_output_dir = Path(train_cfg.get("output_dir", "outputs/qwen3_4b_coding_stream_v2"))
    resume_target = resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")
    if not resume_target:
        resume_target = drive_manager.get_resume_checkpoint(local_output_dir)

    # 3. Load Tokenizer
    tokenizer = load_qwen_tokenizer(
        model_id_or_path=model_cfg["base_model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # 4. Load Base Model in NF4
    model = load_qwen_model(
        model_id_or_path=model_cfg["base_model_id"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        compute_dtype=model_cfg.get("bnb_4bit_compute_dtype", "float16"),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        use_cache=False,
        enable_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )

    # 5. Attach LoRA Adapter (r=16, alpha=32)
    lora_config = get_qlora_config(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules"),
    )
    model = apply_qlora(model, lora_config=lora_config, prepare_kbit=True)

    params_info = get_model_parameters_info(model)
    logger.info("Trainable LoRA parameters info: %s", params_info)

    # 6. Verify gradient flow with smoke step
    smoke_ok, smoke_msg = run_smoke_forward_backward_step(model, tokenizer, max_seq_len=256)
    if not smoke_ok:
        raise RuntimeError(f"Smoke test failed: {smoke_msg}")

    # 7. Materialize finite validation dataset if not already present
    val_file, _ = build_finite_eval_datasets(
        output_dir="data/qwen4b/processed",
        val_target=stream_cfg.get("validation_target", 600),
        test_target=stream_cfg.get("test_target", 400),
    )
    val_dataset = CyberChatDataset(val_file, tokenizer=tokenizer, max_length=train_cfg.get("max_seq_length", 2048))

    # 8. Instantiate Streaming Train Dataset
    train_stream_dataset = StreamSFTDataset(
        tokenizer=tokenizer,
        partition="train",
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        total_target=stream_cfg.get("total_target", 16000),
        quotas=stream_cfg.get("quotas"),
        shuffle_buffer_size=stream_cfg.get("shuffle_buffer_size", 10000),
        seed=train_cfg.get("seed", 42),
    )

    collator = DataCollatorForAssistantOnlyLM(tokenizer=tokenizer)

    # 9. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(local_output_dir),
        per_device_train_batch_size=train_cfg.get("micro_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_steps=int(train_cfg.get("max_steps", 1000)),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        seed=int(train_cfg.get("seed", 42)),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_cfg.get("greater_is_better", False),
        report_to="none",
        remove_unused_columns=False,
    )

    # 10. Callbacks
    patience = train_cfg.get("early_stopping_patience", 3)
    callbacks = [
        DriveBackupCallback(drive_manager),
        LiveTrainingProgressCallback(
            max_steps=training_args.max_steps,
            log_every_n_steps=training_args.logging_steps,
            tracker=train_stream_dataset.tracker,
        ),
        EarlyStoppingCallback(early_stopping_patience=patience),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_stream_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    logger.info("Beginning streaming training (Resume target: %s)...", resume_target)
    train_result = trainer.train(resume_from_checkpoint=resume_target)

    # 11. Print Final Source Summary
    print("\n" + train_stream_dataset.tracker.get_summary_table() + "\n")
    print(f"Sequence Length Statistics: {train_stream_dataset.tracker.get_length_stats()}\n")

    # 12. Save final adapter locally and to Google Drive
    final_adapter_dir = local_output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    logger.info("Saved final adapter locally to '%s'", final_adapter_dir)

    drive_manager.sync_adapter_to_drive(final_adapter_dir, adapter_name="final_stream_coding_adapter")

    return train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming SFT for Qwen 4B Coding")
    parser.add_argument("--config", default="configs/qwen3_4b_stream_coding_t4.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--drive_root", default=None, help="Google Drive backup root")
    args = parser.parse_args()

    run_streaming_training(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        drive_root=args.drive_root,
    )
