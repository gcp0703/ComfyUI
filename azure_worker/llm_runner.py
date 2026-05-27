"""Ollama native-API client for the worker.

Targets Ollama's ``POST /api/chat`` endpoint rather than the OpenAI-compatible
``/v1/chat/completions``. Reason: only the native endpoint exposes the
``think`` toggle that controls reasoning on Qwen3-family GGUFs — the
OpenAI-compat layer silently ignores ``chat_template_kwargs`` and ``think``
and the model keeps reasoning regardless.

The worker does NOT manage the Ollama process — start ``ollama serve``
separately (or rely on the Windows tray icon) and point ``OLLAMA_URL`` at it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .llm_messages import LlmRequest

log = logging.getLogger(__name__)


class LlmJobError(RuntimeError):
    pass


@dataclass
class LlmCompletion:
    """Plain-Python view of one Ollama /api/chat response."""
    completion: str
    reasoning: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    finish_reason: Optional[str]


class LlmRunner:
    def __init__(self, base_url: str, request_timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout_seconds
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def run(self, req: LlmRequest) -> LlmCompletion:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.user_prompt})

        payload = {
            "model": req.model,
            "messages": messages,
            "stream": False,
            # `think` is the native toggle Ollama honors for Qwen3-family
            # reasoning models. When false, the model skips the reasoning
            # phase entirely and goes straight to the answer. Non-thinking
            # models ignore the flag.
            "think": req.thinking,
            "options": {
                "num_predict": req.max_tokens,
                "temperature": req.temperature,
            },
        }

        url = f"{self.base_url}/api/chat"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise LlmJobError(f"ollama request failed: {e}") from e

        if resp.status_code >= 400:
            body = resp.text[:2000]
            raise LlmJobError(f"ollama HTTP {resp.status_code}: {body}")

        try:
            data = resp.json()
        except ValueError as e:
            raise LlmJobError(f"ollama returned non-JSON: {resp.text[:500]}") from e

        message = data.get("message") or {}
        completion = message.get("content") or ""
        # Ollama puts reasoning under `thinking` on /api/chat (and `reasoning`
        # on the OpenAI-compat path — accept either for safety).
        reasoning = message.get("thinking") or message.get("reasoning")

        return LlmCompletion(
            completion=completion,
            reasoning=reasoning if reasoning else None,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )
