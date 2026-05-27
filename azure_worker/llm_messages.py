"""Message contracts for the LLM Storage Queues (parallel to messages.py).

The image pipeline ships images via Blob Storage and posts a SAS URL; the LLM
pipeline returns the completion text inline when it fits in a Storage Queue
message (<48 KiB JSON) and spills to Blob Storage when it doesn't. The
``blob_url``/``blob_name`` fields play the same role as in ``ImageResult``.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Optional


# Prompts can be much larger than image prompts (LLM context windows are >>4K).
# 64 KiB Storage Queue cap minus base64 overhead gives ~48 KiB JSON; reserve
# the rest for the rest of the envelope.
MAX_LLM_PROMPT_CHARS = 32_000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 32_768


class LlmMessageValidationError(ValueError):
    pass


def _coerce_bool(value, field_name: str, default: bool) -> bool:
    """Accept Python bool, or case-insensitive yes/no/true/false strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "y", "true", "1", "on"):
            return True
        if v in ("no", "n", "false", "0", "off", ""):
            return False
    raise LlmMessageValidationError(
        f"{field_name!r} must be a boolean or yes/no/true/false string, got {value!r}"
    )


def _coerce_float(value, field_name: str, default: float, lo: float, hi: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise LlmMessageValidationError(f"{field_name!r} must be a number, got bool")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError as e:
            raise LlmMessageValidationError(f"{field_name!r}={value!r} is not numeric") from e
    else:
        raise LlmMessageValidationError(f"{field_name!r} must be a number or numeric string")
    if not (lo <= f <= hi):
        raise LlmMessageValidationError(f"{field_name!r}={f} must be in [{lo}, {hi}]")
    return f


def _coerce_int(value, field_name: str, default: int, lo: int, hi: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise LlmMessageValidationError(f"{field_name!r} must be an integer, got bool")
    if isinstance(value, int):
        i = value
    elif isinstance(value, str):
        try:
            i = int(value.strip())
        except ValueError as e:
            raise LlmMessageValidationError(f"{field_name!r}={value!r} is not an integer") from e
    else:
        raise LlmMessageValidationError(f"{field_name!r} must be an integer")
    if not (lo <= i <= hi):
        raise LlmMessageValidationError(f"{field_name!r}={i} must be in [{lo}, {hi}]")
    return i


@dataclass
class LlmRequest:
    job_id: str
    user_prompt: str
    model: str
    name: str = ""
    system_prompt: str = ""
    thinking: bool = False
    temperature: float = 1.0
    max_tokens: int = 2048

    @classmethod
    def from_json(cls, raw: str) -> "LlmRequest":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LlmMessageValidationError(f"message is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise LlmMessageValidationError("message must be a JSON object")

        # job_id: accept int or string (producer-language-neutral); store as str.
        raw_job_id = data.get("job_id")
        if raw_job_id is None:
            job_id = str(uuid.uuid4())
        elif isinstance(raw_job_id, (int, str)):
            job_id = str(raw_job_id).strip() or str(uuid.uuid4())
        else:
            raise LlmMessageValidationError("'job_id' must be an integer or string if provided")

        name = data.get("name", "") or ""
        if not isinstance(name, str):
            raise LlmMessageValidationError("'name' must be a string if provided")

        user_prompt = data.get("user_prompt")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise LlmMessageValidationError(
                "'user_prompt' is required and must be a non-empty string"
            )
        if len(user_prompt) > MAX_LLM_PROMPT_CHARS:
            raise LlmMessageValidationError(
                f"'user_prompt' exceeds {MAX_LLM_PROMPT_CHARS} characters"
            )

        system_prompt = data.get("system_prompt", "") or ""
        if not isinstance(system_prompt, str):
            raise LlmMessageValidationError("'system_prompt' must be a string if provided")
        if len(system_prompt) > MAX_LLM_PROMPT_CHARS:
            raise LlmMessageValidationError(
                f"'system_prompt' exceeds {MAX_LLM_PROMPT_CHARS} characters"
            )

        model = data.get("model")
        if not isinstance(model, str) or not model.strip():
            raise LlmMessageValidationError(
                "'model' is required and must be a non-empty string"
            )

        thinking = _coerce_bool(data.get("thinking"), "thinking", default=False)
        # Accept both "temperature" and the producer's "temp" alias.
        temperature = _coerce_float(
            data.get("temperature", data.get("temp")),
            "temperature",
            default=1.0,
            lo=MIN_TEMPERATURE,
            hi=MAX_TEMPERATURE,
        )
        max_tokens = _coerce_int(
            data.get("max_tokens"),
            "max_tokens",
            default=2048,
            lo=MIN_MAX_TOKENS,
            hi=MAX_MAX_TOKENS,
        )

        return cls(
            job_id=job_id,
            name=name.strip(),
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model.strip(),
            thinking=thinking,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass
class LlmResult:
    job_id: str
    name: str
    status: str  # "success" | "error"
    model: str
    completion: Optional[str] = None
    reasoning: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def success(
        cls,
        req: LlmRequest,
        completion: str,
        reasoning: Optional[str],
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        finish_reason: Optional[str],
    ) -> "LlmResult":
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="success",
            model=req.model,
            completion=completion,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def with_blob_spill(self, blob_name: str, blob_url: str) -> "LlmResult":
        """Return a copy of this result with completion/reasoning replaced by a blob pointer."""
        return LlmResult(
            job_id=self.job_id,
            name=self.name,
            status=self.status,
            model=self.model,
            completion=None,
            reasoning=None,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            finish_reason=self.finish_reason,
            blob_url=blob_url,
            blob_name=blob_name,
            error=self.error,
        )

    @classmethod
    def error_for(
        cls,
        req: Optional[LlmRequest],
        message: str,
        raw_job_id: str = "",
    ) -> "LlmResult":
        if req is None:
            return cls(
                job_id=raw_job_id or "unknown",
                name="unknown",
                status="error",
                model="unknown",
                error=message,
            )
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="error",
            model=req.model,
            error=message,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))
