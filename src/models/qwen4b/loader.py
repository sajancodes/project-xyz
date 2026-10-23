"""Model and Tokenizer Loader for Qwen 4B Instruct.

Handles runtime NF4 4-bit quantization, tokenizer verification, and T4 GPU optimization.
DO NOT use prequantized GGUF, AWQ, or GPTQ files as base; loads Hugging Face checkpoint directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import torch

from src.models.qwen4b.quantization import get_bnb_4bit_config, is_bitsandbytes_available

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_qwen_tokenizer(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    trust_remote_code: bool = True,
) -> Any:
    """Load the official Qwen tokenizer.
    
    Preserves original Qwen chat template and special tokens.
    DO NOT use Project XYZ's scratch tokenizer.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from '%s'...", model_id_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Standard causal LM SFT padding setup
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token ('%s')", tokenizer.eos_token)
        else:
            tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
            logger.info("Added special pad_token '<|extra_0|>'")

    tokenizer.padding_side = "right"
    return tokenizer


def load_qwen_model(
    model_id_or_path: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = True,
    compute_dtype: str = "float16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_cache: bool = False,
    enable_gradient_checkpointing: bool = True,
) -> Any:
    """Load Qwen 4B Instruct model with optional NF4 4-bit runtime quantization.
    
    Args:
        model_id_or_path: HuggingFace model repo ID or local checkpoint path.
        load_in_4bit: If True, quantize at runtime using bitsandbytes NF4.
        compute_dtype: "float16" for NVIDIA T4 GPU.
        device_map: "auto" or device string.
        trust_remote_code: Required for custom model architectures.
        use_cache: Must be False during training with gradient checkpointing.
        enable_gradient_checkpointing: Saves VRAM for T4 training.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = torch.float16 if compute_dtype == "float16" else torch.bfloat16
    quantization_config = None

    if load_in_4bit:
        if not is_bitsandbytes_available() and not torch.cuda.is_available():
            logger.warning(
                "Bitsandbytes/CUDA is unavailable in this environment. Falling back to non-quantized loading."
            )
        else:
            quantization_config = get_bnb_4bit_config(
                compute_dtype=compute_dtype,
                use_double_quant=True,
                quant_type="nf4",
            )

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = device_map
    elif torch.cuda.is_available():
        kwargs["device_map"] = device_map

    logger.info("Loading model '%s' (load_in_4bit=%s)...", model_id_or_path, load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)

    # Disable KV cache during training to avoid gradient checkpointing conflict
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    return model


def get_model_parameters_info(model: Any) -> Dict[str, Any]:
    """Inspect and return total, trainable, and frozen parameter counts."""
    total_params = 0
    trainable_params = 0

    for _, param in model.named_parameters():
        num_p = param.numel()
        total_params += num_p
        if param.requires_grad:
            trainable_params += num_p

    pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": round(pct, 4),
    }
