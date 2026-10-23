"""Inference and Adapter Loading Module for Qwen 4B.

Provides inference utilities, adapter hot-reloading from Google Drive or local storage,
and chat template formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def format_messages_with_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages using the official Qwen chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback formatting if chat template missing
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"
    return formatted


def generate_response(
    model: Any,
    tokenizer: Any,
    messages_or_prompt: Union[List[Dict[str, str]], str],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: Optional[str] = None,
) -> str:
    """Generate assistant completion for a given prompt or message history."""
    if isinstance(messages_or_prompt, str):
        messages = [{"role": "user", "content": messages_or_prompt}]
    else:
        messages = messages_or_prompt

    prompt_text = format_messages_with_template(tokenizer, messages, add_generation_prompt=True)
    
    # Resolve device
    if device is None:
        if hasattr(model, "device"):
            device = str(model.device)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample and temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    # Slice off prompt tokens to decode only assistant output
    generated_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response


def load_lora_adapter(base_model: Any, adapter_path: Union[str, Path]) -> PeftModel:
    """Load and attach a trained LoRA adapter from disk / Google Drive."""
    adapter_path_str = str(adapter_path)
    logger.info("Loading LoRA adapter from '%s'...", adapter_path_str)
    model = PeftModel.from_pretrained(base_model, adapter_path_str)
    model.eval()
    return model


def merge_and_export_adapter(peft_model: PeftModel, export_path: Union[str, Path]) -> Any:
    """Merge LoRA weights back into the base model and export."""
    export_path_p = Path(export_path)
    export_path_p.mkdir(parents=True, exist_ok=True)
    logger.info("Merging LoRA adapter into base model and saving to '%s'...", export_path_p)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(str(export_path_p))
    return merged_model
