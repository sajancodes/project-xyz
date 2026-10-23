"""Provider-neutral Teacher Abstraction for Response Distillation.

This module provides abstractions for generating synthetic cybersecurity examples
using external teacher models or local sources.

NOTE ON DISTILLATION METHODOLOGY:
This project performs response distillation / teacher-generated Supervised Fine-Tuning (SFT).
It trains on teacher-generated completions rather than classical logits-based knowledge
distillation (which would require access to teacher token probability distributions / logits).
"""

from __future__ import annotations

import abc
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class TeacherResponse:
    """Standardized response from any teacher model."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TeacherProvider(abc.ABC):
    """Abstract base class for teacher providers."""

    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 1500) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> TeacherResponse:
        """Generate a response given a list of chat messages."""
        pass

    def generate_batch(self, message_list: List[List[Dict[str, str]]]) -> List[TeacherResponse]:
        """Generate responses for a batch of conversations."""
        return [self.generate(msgs) for msgs in message_list]


class OpenAITeacherProvider(TeacherProvider):
    """OpenAI-compatible HTTP API provider (works with OpenAI, vLLM, Ollama, DeepSeek, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("TEACHER_API_BASE", "https://api.openai.com/v1")).rstrip("/")

    def generate(self, messages: List[Dict[str, str]]) -> TeacherResponse:
        if not self.api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY or provide api_key in constructor. "
                "No secrets are hardcoded in the codebase."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]
                content = choice["message"]["content"]
                usage = data.get("usage", {})
                return TeacherResponse(
                    content=content,
                    model=data.get("model", self.model_name),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    metadata={"finish_reason": choice.get("finish_reason")},
                )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"Teacher HTTP request failed ({e.code}): {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Teacher request encountered unexpected error: {e}") from e


class LocalTeacherProvider(TeacherProvider):
    """Local model provider (e.g. running via transformers, llama.cpp, or mock)."""

    def __init__(
        self,
        model_name: str = "local-cyber-teacher",
        generator_fn: Optional[Any] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.generator_fn = generator_fn

    def generate(self, messages: List[Dict[str, str]]) -> TeacherResponse:
        if self.generator_fn is not None:
            content = self.generator_fn(messages)
            return TeacherResponse(content=content, model=self.model_name)
        # Fallback echo mock for testing / validation
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return TeacherResponse(
            content=f"[Local Teacher Answer for: {user_msg[:60]}...]",
            model=self.model_name,
        )


class ManualDatasetTeacherProvider(TeacherProvider):
    """Loads pre-generated teacher examples from local files or seeds."""

    def __init__(self, records: List[Dict[str, Any]], model_name: str = "curated-expert-seed") -> None:
        super().__init__(model_name=model_name)
        self.records = records
        self._index = 0

    def generate(self, messages: List[Dict[str, str]]) -> TeacherResponse:
        if not self.records:
            raise ValueError("No records available in ManualDatasetTeacherProvider.")
        record = self.records[self._index % len(self.records)]
        self._index += 1
        # Extract assistant content from messages in the record
        assistant_content = ""
        for m in record.get("messages", []):
            if m.get("role") == "assistant":
                assistant_content = m.get("content", "")
                break
        return TeacherResponse(content=assistant_content, model=self.model_name, metadata={"id": record.get("id")})


def get_teacher_provider(
    provider_type: str,
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    records: Optional[List[Dict[str, Any]]] = None,
) -> TeacherProvider:
    """Factory to instantiate teacher providers."""
    provider_type = provider_type.lower()
    if provider_type in ("openai", "vllm", "ollama", "compatible"):
        return OpenAITeacherProvider(model_name=model_name, api_key=api_key, base_url=base_url)
    elif provider_type == "local":
        return LocalTeacherProvider(model_name=model_name)
    elif provider_type == "manual":
        return ManualDatasetTeacherProvider(records=records or [], model_name=model_name)
    else:
        raise ValueError(f"Unknown teacher provider type: {provider_type}")
