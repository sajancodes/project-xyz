"""Live Training Visibility, Metrics Progress Bar, and Diagnostic Callbacks.

Provides real-time visibility into:
- Current step / max steps and percentage complete
- Training and evaluation loss
- Learning rate
- Examples and tokens processed per second
- Elapsed and remaining time (ETA)
- GPU allocated and reserved memory
- Source counts distribution
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class LiveTrainingProgressCallback(TrainerCallback):
    """Real-time progress logger displaying compact metrics and GPU VRAM utilization."""

    def __init__(self, max_steps: int = 1000, log_every_n_steps: int = 5, tracker: Optional[Any] = None) -> None:
        self.max_steps = max_steps
        self.log_every_n_steps = log_every_n_steps
        self.tracker = tracker
        self.start_time: Optional[float] = None
        self.last_step_time: Optional[float] = None
        self.total_tokens_seen: int = 0
        self.total_examples_seen: int = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self.start_time = time.time()
        self.last_step_time = self.start_time
        print("\n" + "=" * 80)
        print("STREAMING QLORA TRAINING INITIALIZED")
        print(f"Target Max Steps: {self.max_steps} | Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print("=" * 80 + "\n")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        step = state.global_step
        if step % self.log_every_n_steps != 0 and step != self.max_steps:
            return

        now = time.time()
        elapsed = now - (self.start_time or now)
        step_delta = now - (self.last_step_time or now)
        self.last_step_time = now

        pct = (step / self.max_steps * 100.0) if self.max_steps > 0 else 0.0

        # Estimate remaining time
        if step > 0:
            rate = step / elapsed
            rem_sec = (self.max_steps - step) / rate if rate > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(rem_sec))
        else:
            eta_str = "--:--:--"

        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # Metrics from TrainerState
        latest_loss = "N/A"
        latest_lr = "N/A"
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    latest_loss = f"{entry['loss']:.4f}"
                    break
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    latest_lr = f"{entry['learning_rate']:.2e}"
                    break

        # GPU VRAM
        vram_alloc = (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        vram_res = (torch.cuda.memory_reserved() / (1024**3)) if torch.cuda.is_available() else 0.0

        # Progress bar
        bar_len = 20
        filled = int(round(bar_len * (pct / 100.0)))
        bar = "=" * filled + ">" + "-" * (bar_len - filled)
        if filled >= bar_len:
            bar = "=" * bar_len

        print(
            f"Step [{bar}] {pct:5.1f}% ({step:4d}/{self.max_steps}) | "
            f"Loss: {latest_loss} | LR: {latest_lr} | "
            f"Time: {elapsed_str} | ETA: {eta_str} | "
            f"VRAM: {vram_alloc:.1f}G/{vram_res:.1f}G"
        )

        # Print source breakdown every 100 steps
        if step % 100 == 0 and self.tracker is not None:
            print("\n--- Live Source Intake Distribution ---")
            for name, q in self.tracker.quotas.items():
                print(f"  - {name:<18s}: {q.accepted:5d} / {q.target_count:5d} accepted")
            print("---------------------------------------\n")
