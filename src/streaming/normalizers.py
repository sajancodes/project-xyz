"""Normalizers for Upstream Streaming Datasets into Common Conversational Schema.

Converts diverse dataset structures (OpenCoder, BigCode, SmolTalk, Cybersecurity)
into a unified chat schema compatible with Qwen's chat template and assistant masking.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional

DEFAULT_SYSTEM_PROMPT = "You are a precise, capable coding and cybersecurity assistant."


def generate_deterministic_id(source: str, prompt: str, response: str) -> str:
    """Generate a stable unique ID based on source and content hash."""
    h = hashlib.sha256((source + "::" + prompt.strip() + "::" + response.strip()).encode("utf-8")).hexdigest()[:16]
    return f"{source}_{h}"


def normalize_opencoder(sample: Dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
    """Normalize OpenCoder educational_instruct sample."""
    instruction = sample.get("instruction", "").strip()
    output = sample.get("output", "").strip()

    if not instruction or not output:
        return None

    sample_id = sample.get("seq_id")
    if sample_id is None:
        sample_id = generate_deterministic_id("opencoder", instruction, output)
    else:
        sample_id = f"opencoder_{sample_id}"

    metadata = {
        "code": sample.get("code"),
        "entry_point": sample.get("entry_point"),
        "testcase": sample.get("testcase"),
        "seq_id": sample.get("seq_id"),
    }

    return {
        "id": str(sample_id),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "source": "opencoder_educational_instruct",
        "category": "coding_educational",
        "difficulty": "intermediate",
        "metadata": metadata,
    }


def normalize_bigcode(sample: Dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
    """Normalize BigCode self-oss-instruct-sc2-exec-filter-50k sample."""
    instruction = sample.get("instruction", "").strip()
    response = sample.get("response", "").strip()

    if not instruction or not response:
        return None

    sample_id = sample.get("id") or sample.get("sha1")
    if not sample_id:
        sample_id = generate_deterministic_id("bigcode", instruction, response)
    else:
        sample_id = f"bigcode_{sample_id}"

    metadata = {
        "id": sample.get("id"),
        "concepts": sample.get("concepts"),
        "seed": sample.get("seed"),
        "sha1": sample.get("sha1"),
    }

    return {
        "id": str(sample_id),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "source": "bigcode_self_oss_instruct_sc2",
        "category": "coding_exec_validated",
        "difficulty": "advanced",
        "metadata": metadata,
    }


def normalize_smoltalk(sample: Dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
    """Normalize SmolTalk smol-magpie-ultra sample."""
    raw_messages = sample.get("messages", [])
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        return None

    user_content = ""
    assistant_content = ""

    for m in raw_messages:
        role = m.get("role")
        content = m.get("content", "").strip()
        if role == "user" and not user_content:
            user_content = content
        elif role == "assistant" and not assistant_content:
            assistant_content = content

    if not user_content or not assistant_content:
        return None

    sample_id = generate_deterministic_id("smoltalk", user_content, assistant_content)

    category = sample.get("category", "general_instruction")
    difficulty = sample.get("difficulty", "intermediate")

    metadata = {
        "category": category,
        "difficulty": difficulty,
        "quality": sample.get("quality"),
        "reward_model_score": sample.get("reward_model_score"),
        "conversation_tokens": sample.get("conversation_tokens"),
    }

    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "smoltalk_magpie_ultra",
        "category": category,
        "difficulty": difficulty,
        "metadata": metadata,
    }


def normalize_cybersecurity(sample: Dict[str, Any], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Optional[Dict[str, Any]]:
    """Normalize cybersecurity sample (Trendyol, Fenrir, or local curated seeds)."""
    user_content = ""
    assistant_content = ""
    sys_content = system_prompt

    if "messages" in sample and isinstance(sample["messages"], list):
        for m in sample["messages"]:
            role = m.get("role")
            content = m.get("content", "").strip()
            if role == "system" and content:
                sys_content = content
            elif role == "user" and not user_content:
                user_content = content
            elif role == "assistant" and not assistant_content:
                assistant_content = content
    else:
        user_content = sample.get("user", "").strip() or sample.get("instruction", "").strip()
        assistant_content = sample.get("assistant", "").strip() or sample.get("response", "").strip()
        if sample.get("system"):
            sys_content = sample.get("system").strip()

    if not user_content or not assistant_content:
        return None

    sample_id = sample.get("id") or generate_deterministic_id("cyber", user_content, assistant_content)
    category = sample.get("category", "cybersecurity")
    difficulty = sample.get("difficulty", "intermediate")

    return {
        "id": str(sample_id),
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": sample.get("source", "cybersecurity_stream"),
        "category": category,
        "difficulty": difficulty,
        "metadata": {"tags": sample.get("tags", [])},
    }
