"""Quality Filtering and Sanitization for Streaming Code and Instruction Samples.

Rejects empty, repetitive, truncated, placeholder, or pathological examples.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

PLACEHOLDER_REGEX = re.compile(
    r"(^as an ai (?:language )?model)|"
    r"(^i cannot (?:fulfill|assist|help|comply))|"
    r"(\[insert\s+[a-z_\s]+here\])|"
    r"(\[todo:?\])|"
    r"(lorem ipsum dolor sit amet)|"
    r"(<\|im_start\|>)|"
    r"(<\|im_end\|>)",
    re.IGNORECASE,
)


def validate_code_blocks(text: str) -> bool:
    """Check that markdown code blocks are properly balanced."""
    fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
    # Unbalanced fences indicate broken or abruptly truncated code
    return fence_count % 2 == 0


def check_ngram_repetition(text: str, n: int = 4, threshold: float = 0.45) -> bool:
    """Check for pathological looping generation."""
    words = text.split()
    if len(words) < 30:
        return False
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return (1.0 - unique_ratio) > threshold


def filter_sample(
    instruction: str,
    response: str,
    min_inst_len: int = 10,
    min_resp_len: int = 20,
    max_char_len: int = 24000,
) -> Tuple[bool, Optional[str]]:
    """Validate a candidate instruction/response pair for quality.
    
    Returns (passes, rejection_reason).
    """
    inst = instruction.strip()
    resp = response.strip()

    if not inst:
        return False, "Empty instruction"
    if not resp:
        return False, "Empty response"

    if len(inst) < min_inst_len:
        return False, f"Instruction too short ({len(inst)} < {min_inst_len})"
    if len(resp) < min_resp_len:
        return False, f"Response too short ({len(resp)} < {min_resp_len})"

    total_len = len(inst) + len(resp)
    if total_len > max_char_len:
        return False, f"Total characters ({total_len}) exceeds pathological limit ({max_char_len})"

    # Check placeholder / refusal / leaked prompt delimiters
    if PLACEHOLDER_REGEX.search(inst):
        return False, "Instruction matched placeholder/refusal pattern"
    if PLACEHOLDER_REGEX.search(resp):
        return False, "Response matched placeholder/refusal pattern"

    # Check code block balance if code fences are present
    if "```" in resp and not validate_code_blocks(resp):
        return False, "Unbalanced code fence (truncation indicator)"

    # Check severe repetition
    if check_ngram_repetition(resp):
        return False, "Excessive n-gram repetition detected in response"

    return True, None
