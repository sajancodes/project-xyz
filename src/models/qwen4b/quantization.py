"""BitsAndBytes NF4 4-bit Quantization Configuration for Qwen 4B.

Configured specifically for NVIDIA T4 (16GB VRAM) constraints:
- 4-bit NormalFloat (NF4) quantization
- Nested/double quantization enabled (bnb_4bit_use_double_quant=True)
- Float16 compute dtype (bnb_4bit_compute_dtype=torch.float16)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore
    BNB_AVAILABLE = False


def is_bitsandbytes_available() -> bool:
    """Check if bitsandbytes and BitsAndBytesConfig are installed and importable."""
    if not BNB_AVAILABLE:
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def get_bnb_4bit_config(
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
    quant_type: str = "nf4",
) -> Any:
    """Create a BitsAndBytesConfig for 4-bit NF4 QLoRA loading.
    
    Args:
        compute_dtype: "float16" or "bfloat16". For NVIDIA T4, use "float16".
        use_double_quant: Nested quantization saves ~0.4 bits/param.
        quant_type: Must be "nf4" for QLoRA.
        
    Returns:
        BitsAndBytesConfig instance.
    """
    if not BNB_AVAILABLE:
        raise ImportError(
            "BitsAndBytesConfig is not available. Please install `transformers` and `bitsandbytes`."
        )

    # Resolve torch dtype
    if compute_dtype == "float16":
        torch_compute_dtype = torch.float16
    elif compute_dtype == "bfloat16":
        torch_compute_dtype = torch.bfloat16
    elif compute_dtype == "float32":
        torch_compute_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported compute_dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch_compute_dtype,
    )
    logger.info(
        "Configured BitsAndBytesConfig(load_in_4bit=True, quant_type='%s', "
        "use_double_quant=%s, compute_dtype=%s)",
        quant_type,
        use_double_quant,
        compute_dtype,
    )
    return bnb_config
