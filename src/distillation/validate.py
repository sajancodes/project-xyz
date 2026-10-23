"""Dataset Schema Validation and Quality Filtering.

Ensures every example strictly adheres to the required JSONL schema:
- Proper roles (system, user, assistant)
- Non-empty content with sufficient depth
- Category, difficulty, source, tags
- Defensive/lab/CTF grounding for security concepts
- Complete rejection of malformed or unsafe items
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.distillation.clean import clean_message_content

logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_DIFFICULTIES = {"beginner", "intermediate", "advanced"}

# Keywords indicating offensive techniques that require ethical/lab grounding
OFFENSIVE_KEYWORDS = [
    "exploit", "payload", "attack", "sqli", "xss", "rce", "buffer overflow",
    "privilege escalation", "bypass", "shellcode", "reverse shell", "cve-"
]

# Keywords that confirm proper educational/lab/defensive context
GROUNDING_KEYWORDS = [
    "lab", "ctf", "defensive", "remediation", "patch", "mitigat", "authorized",
    "toy", "sandbox", "detection", "siem", "rule", "snort", "suricata", "audit",
    "secure coding", "analysis", "wireshark", "hardening", "forensic", "concept"
]


def validate_schema(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate top-level keys and structure of a dataset record."""
    if not isinstance(record, dict):
        return False, "Record is not a dictionary"

    for field_name in ["id", "messages", "category", "difficulty", "source", "tags"]:
        if field_name not in record:
            return False, f"Missing required field: '{field_name}'"

    if not isinstance(record["id"], str) or not record["id"].strip():
        return False, "Field 'id' must be a non-empty string"

    if not isinstance(record["category"], str) or not record["category"].strip():
        return False, "Field 'category' must be a non-empty string"

    if record["difficulty"] not in ALLOWED_DIFFICULTIES:
        return False, f"Field 'difficulty' must be one of {ALLOWED_DIFFICULTIES}, got '{record['difficulty']}'"

    if not isinstance(record["source"], str) or not record["source"].strip():
        return False, "Field 'source' must be a non-empty string"

    if not isinstance(record["tags"], list):
        return False, "Field 'tags' must be a list"

    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "Field 'messages' must be a list with at least 2 messages (user and assistant)"

    has_user = False
    has_assistant = False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message at index {idx} is not a dictionary"
        role = msg.get("role")
        content = msg.get("content")

        if role not in ALLOWED_ROLES:
            return False, f"Message at index {idx} has invalid role '{role}'. Allowed: {ALLOWED_ROLES}"

        if not isinstance(content, str) or not content.strip():
            return False, f"Message at index {idx} has empty or non-string content"

        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

    if not has_user:
        return False, "Conversation must contain at least one message with role 'user'"
    if not has_assistant:
        return False, "Conversation must contain at least one message with role 'assistant'"

    return True, None


def validate_record_content(
    record: Dict[str, Any],
    min_user_len: int = 15,
    min_assistant_len: int = 40,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Validate content quality, length, safety grounding, and clean artifacts.
    
    Returns (is_valid, rejection_reason, cleaned_record).
    """
    valid_schema, schema_err = validate_schema(record)
    if not valid_schema:
        return False, schema_err, record

    cleaned_messages = []
    total_user_len = 0
    total_assistant_len = 0
    full_text = ""

    for msg in record["messages"]:
        cleaned_content, issues = clean_message_content(msg["content"])
        if issues:
            return False, f"Sanitization failed for role '{msg['role']}': {'; '.join(issues)}", record

        role = msg["role"]
        if role == "user":
            total_user_len += len(cleaned_content)
        elif role == "assistant":
            total_assistant_len += len(cleaned_content)

        full_text += " " + cleaned_content
        cleaned_messages.append({"role": role, "content": cleaned_content})

    if total_user_len < min_user_len:
        return False, f"User message too short ({total_user_len} chars, min {min_user_len})", record

    if total_assistant_len < min_assistant_len:
        return False, f"Assistant message too short ({total_assistant_len} chars, min {min_assistant_len})", record

    # Check offensive grounding
    full_text_lower = full_text.lower()
    has_offensive = any(k in full_text_lower for k in OFFENSIVE_KEYWORDS)
    if has_offensive:
        has_grounding = any(g in full_text_lower for g in GROUNDING_KEYWORDS)
        if not has_grounding:
            return False, "Offensive techniques mentioned without defensive/lab/CTF/educational grounding", record

    cleaned_record = dict(record)
    cleaned_record["messages"] = cleaned_messages
    return True, None, cleaned_record
