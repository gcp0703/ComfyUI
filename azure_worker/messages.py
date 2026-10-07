"""Message contracts for the inbound and outbound Azure Storage Queues."""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional


# prompt + negative_prompt may total this many characters. Not a model limit —
# no active profile's text encoder truncates — but a transport one: the Storage
# Queue message is 64 KiB after base64 (~48 KiB raw), shared by every field.
MAX_PROMPT_CHARS = 32_000
# Byte backstop on the raw request. The queue would already have rejected
# anything over 48 KiB; this lower bar leaves ~2 KiB for the result message's
# added fields (render block, SAS URL) so the *reply* fits too.
MAX_REQUEST_BYTES = 46 * 1024
# Contract v1 request fields the worker now owns. Accepted, ignored, reported.
LEGACY_FIELDS = ("width", "height", "steps", "cfg")


class MessageValidationError(ValueError):
    pass


@dataclass
class ImageRequest:
    job_id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    # Which of LEGACY_FIELDS the client sent, in LEGACY_FIELDS order. Surfaced
    # on the result as a warning so stale clients can be found in the field.
    ignored_fields: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: str) -> "ImageRequest":
        size = len(raw.encode("utf-8"))
        if size > MAX_REQUEST_BYTES:
            raise MessageValidationError(
                f"message is {size} bytes; the limit is {MAX_REQUEST_BYTES} bytes"
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MessageValidationError(f"message is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise MessageValidationError("message must be a JSON object")

        job_id = str(data.get("job_id") or uuid.uuid4())
        name = data.get("name")
        prompt = data.get("prompt")

        if not isinstance(name, str) or not name.strip():
            raise MessageValidationError("'name' is required and must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MessageValidationError("'prompt' is required and must be a non-empty string")

        negative = data.get("negative_prompt", "") or ""
        if not isinstance(negative, str):
            raise MessageValidationError("'negative_prompt' must be a string")
        total = len(prompt) + len(negative)
        if total > MAX_PROMPT_CHARS:
            raise MessageValidationError(
                f"'prompt' + 'negative_prompt' total {total} characters; the limit is {MAX_PROMPT_CHARS}"
            )

        seed_raw = data.get("seed")
        if seed_raw is None:
            seed = random.randint(0, 2**63 - 1)
        elif isinstance(seed_raw, int):
            seed = seed_raw
        else:
            raise MessageValidationError("'seed' must be an integer if provided")

        ignored = tuple(f for f in LEGACY_FIELDS if f in data)

        return cls(
            job_id=job_id,
            name=name.strip(),
            prompt=prompt,
            negative_prompt=negative,
            seed=seed,
            ignored_fields=ignored,
        )


@dataclass
class ImageResult:
    job_id: str
    name: str
    status: str  # "success" | "error"
    prompt: str
    width: int
    height: int
    seed: int
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def success(
        cls,
        req: ImageRequest,
        blob_name: str,
        blob_url: str,
    ) -> "ImageResult":
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="success",
            prompt=req.prompt,
            width=0,
            height=0,
            seed=req.seed,
            blob_name=blob_name,
            blob_url=blob_url,
        )

    @classmethod
    def error_for(cls, req: Optional[ImageRequest], message: str, raw_job_id: str = "") -> "ImageResult":
        if req is None:
            return cls(
                job_id=raw_job_id or "unknown",
                name="unknown",
                status="error",
                prompt="",
                width=0,
                height=0,
                seed=0,
                error=message,
            )
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="error",
            prompt=req.prompt,
            width=0,
            height=0,
            seed=req.seed,
            error=message,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str, fallback: str = "image") -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or fallback
