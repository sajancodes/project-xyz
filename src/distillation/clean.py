"""Data Cleaning and Sanitization for Cybersecurity Datasets.

Enforces strict content hygiene:
- Normalizes whitespace
- Detects leaked prompt markers and templates
- Detects and flags secrets, API keys, credentials
- Filters placeholder responses and refusal artifacts
- Detects excessive repetition and severe truncation
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Patterns for secrets, tokens, credentials
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)[\s:=]+['\"][a-zA-Z0-9_\-]{20,}['\"]", re.IGNORECASE),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
    re.compile(r"sk-[a-zA-Z0-9]{20,48}"),  # OpenAI style key
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub personal token
    re.compile(r"eyJ[a-zA-Z0-9_\-]{20,}\.eyJ[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]+"),  # JWT token
]

# Patterns for refusal or boilerplate placeholder generation
PLACEHOLDER_PATTERNS = [
    re.compile(r"^as an ai (?:language )?model", re.IGNORECASE),
    re.compile(r"^i cannot (?:fulfill|assist|help|comply)", re.IGNORECASE),
    re.compile(r"\[insert\s+[a-z_\s]+here\]", re.IGNORECASE),
    re.compile(r"\[your\s+[a-z_\s]+here\]", re.IGNORECASE),
    re.compile(r"\[todo:?\]", re.IGNORECASE),
    re.compile(r"lorem ipsum dolor sit amet", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and strip useless artifacts."""
    if not text:
        return ""
    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # Remove null bytes
    text = text.replace("\x00", "")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def find_secrets(text: str) -> List[str]:
    """Scan text for real-looking credentials, private keys, or API tokens."""
    matched = []
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            matched.append(match.group(0)[:30] + "...")
    return matched


def is_placeholder_or_refusal(text: str) -> Tuple[bool, Optional[str]]:
    """Check if the text is a placeholder, refusal artifact, or boilerplate refusal."""
    text_stripped = text.strip()
    for pattern in PLACEHOLDER_PATTERNS:
        match = pattern.search(text_stripped)
        if match:
            return True, f"Matched placeholder pattern: {pattern.pattern}"
    return False, None


def check_excessive_repetition(text: str, max_rep_ratio: float = 0.4, n: int = 4) -> bool:
    """Check whether text repeats the same n-grams excessively (looping generation)."""
    words = text.split()
    if len(words) < 20:
        return False
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return False
    unique_ngrams = set(ngrams)
    rep_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
    return rep_ratio > max_rep_ratio


def check_severe_truncation(text: str) -> bool:
    """Check if the response appears severely truncated (e.g. unclosed code fences, cut off mid-sentence)."""
    text_stripped = text.strip()
    if not text_stripped:
        return True
    
    # Check unbalanced code fences (odd count of ```)
    fences = len(re.findall(r"^```", text_stripped, flags=re.MULTILINE))
    if fences % 2 != 0:
        return True
        
    # Check if text ends abruptly mid-word or with orphan conjunction
    trailing_bad = [" and", " or", " but", " because", " the", " a", " an", " with", " in", " for"]
    last_20 = text_stripped[-20:].lower()
    for bad in trailing_bad:
        if last_20.endswith(bad):
            return True

    return False


def clean_message_content(content: str) -> Tuple[str, List[str]]:
    """Clean and sanitize a message string. Returns (cleaned_text, issues_found)."""
    issues = []
    cleaned = normalize_text(content)
    
    # Check secrets
    secrets = find_secrets(cleaned)
    if secrets:
        issues.append(f"Contains credential/secret pattern: {', '.join(secrets)}")
        
    # Check placeholder
    is_ph, ph_reason = is_placeholder_or_refusal(cleaned)
    if is_ph:
        issues.append(ph_reason or "Contains placeholder/refusal")
        
    # Check excessive repetition
    if check_excessive_repetition(cleaned):
        issues.append("Excessive n-gram repetition detected")
        
    # Check severe truncation
    if check_severe_truncation(cleaned):
        issues.append("Severe truncation or unclosed code fence detected")
        
    return cleaned, issues
