"""LoRA and QLoRA Adaptation Module for Qwen 4B.

Targeting linear projection layers:
- Attention: q_proj, k_proj, v_proj, o_proj
- MLP: gate_proj, up_proj, down_proj
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_qlora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Create PEFT LoraConfig for causal language modeling."""
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    logger.info(
        "Created LoraConfig(r=%d, lora_alpha=%d, dropout=%.2f, targets=%s)",
        r,
        lora_alpha,
        lora_dropout,
        target_modules,
    )
    return config


def verify_target_modules(model: Any, target_modules: List[str]) -> Tuple[List[str], List[str]]:
    """Verify which target module names exist within the model."""
    all_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    found = [t for t in target_modules if t in all_module_names]
    missing = [t for t in target_modules if t not in all_module_names]
    return found, missing


def apply_qlora(
    model: Any,
    lora_config: Optional[LoraConfig] = None,
    prepare_kbit: bool = True,
    r: int = 32,
    lora_alpha: int = 64,
) -> Any:
    """Prepare a quantized model and attach LoRA adapter.
    
    Args:
        model: Base model (e.g. 4-bit NF4 quantized).
        lora_config: Optional pre-constructed LoraConfig.
        prepare_kbit: Whether to call prepare_model_for_kbit_training.
        r: Rank to use if lora_config is None.
        lora_alpha: Alpha to use if lora_config is None.
        
    Returns:
        PeftModel wrapping the base model with trainable LoRA adapters.
    """
    if prepare_kbit:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    if lora_config is None:
        lora_config = get_qlora_config(r=r, lora_alpha=lora_alpha)

    # Verify targets
    found, missing = verify_target_modules(model, lora_config.target_modules)
    if missing:
        logger.warning(
            "Some specified LoRA target modules were not found in model: %s. Found: %s",
            missing,
            found,
        )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
