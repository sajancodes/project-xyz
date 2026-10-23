"""GPU Environment Diagnostics, T4 OOM Protection, and Smoke Step Verification.
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any, Dict, Optional, Tuple
import torch

from src.models.qwen4b.quantization import is_bitsandbytes_available

logger = logging.getLogger(__name__)


def print_gpu_environment() -> Dict[str, Any]:
    """Inspect and print complete GPU and framework environment details."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "No CUDA Device (CPU)"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0

    import transformers
    import peft

    bnb_ver = "unavailable"
    if is_bitsandbytes_available():
        import bitsandbytes
        bnb_ver = getattr(bitsandbytes, "__version__", "installed")

    info = {
        "cuda_available": cuda_avail,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "cuda_version": torch.version.cuda if cuda_avail else "N/A",
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "bitsandbytes_version": bnb_ver,
        "python_version": sys.version.split()[0],
    }

    print("\n" + "=" * 60)
    print("HARDWARE & DEPENDENCY ENVIRONMENT")
    print("=" * 60)
    print(f"GPU:                  {info['gpu_name']}")
    print(f"VRAM:                 {info['vram_gb']} GB")
    print(f"CUDA Available:       {info['cuda_available']} (Version: {info['cuda_version']})")
    print(f"PyTorch Version:      {info['pytorch_version']}")
    print(f"Transformers Version: {info['transformers_version']}")
    print(f"PEFT Version:         {info['peft_version']}")
    print(f"BitsAndBytes Version: {info['bitsandbytes_version']}")
    print(f"Python Version:       {info['python_version']}")
    print("=" * 60 + "\n")

    return info


def clear_cuda_memory() -> None:
    """Safely clear Python garbage collection and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_smoke_forward_backward_step(
    model: Any,
    tokenizer: Any,
    max_seq_len: int = 128,
) -> Tuple[bool, Optional[str]]:
    """Run a tiny 1-step forward + backward pass to verify gradients and OOM safety before full training.
    
    Returns (success, error_or_warning_message).
    """
    logger.info("Executing tiny QLoRA forward/backward smoke pass (seq_len=%d)...", max_seq_len)
    clear_cuda_memory()

    dummy_messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "What is the primary function of an intrusion detection system (IDS)?"},
        {"role": "assistant", "content": "An IDS monitors network traffic and host activity for malicious behavior."},
    ]

    from src.training.qwen4b.masking import create_assistant_masked_sample

    try:
        sample = create_assistant_masked_sample(tokenizer, dummy_messages, max_length=max_seq_len)
        device = next(model.parameters()).device

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)

        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        if loss is None or torch.isnan(loss):
            return False, f"Smoke loss computation returned invalid loss: {loss}"

        loss.backward()

        # Zero gradients after test
        model.zero_grad(set_to_none=True)
        clear_cuda_memory()

        logger.info("Smoke step succeeded! Computed initial loss: %.4f", loss.item())
        return True, f"Smoke step succeeded with initial loss: {loss.item():.4f}"

    except torch.cuda.OutOfMemoryError as oom:
        clear_cuda_memory()
        msg = (
            f"CUDA OutOfMemoryError during smoke step! T4 Fallback recommendation: "
            f"1. Reduce max_seq_length to 768 or 512. "
            f"2. Ensure micro_batch_size=1. "
            f"3. Lower LoRA rank r to 16. Error: {oom}"
        )
        logger.error(msg)
        return False, msg
    except Exception as e:
        clear_cuda_memory()
        msg = f"Unexpected error during smoke step: {e}"
        logger.error(msg)
        return False, msg
