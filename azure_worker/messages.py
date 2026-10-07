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


def warnings_for(req: "ImageRequest") -> list[str]:
    """Non-fatal notes for the client. Today: which contract-v1 fields were ignored."""
    if not req.ignored_fields:
        return []
    return [f"ignored client-supplied fields: {', '.join(req.ignored_fields)}"]


@dataclass
class ImageResult:
    """One outbound message. Every key is always present (contract v2 §3)."""

    job_id: str
    name: str
    status: str  # "success" | "error"
    prompt: str
    negative_prompt: str
    # The PNG's real dimensions on success; the graph's intended size on a
    # runtime error; 0 when validation failed before a graph existed.
    width: int
    height: int
    seed: int
    # workflow.render_report() output, or None when no graph was built.
    render: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def success(
        cls,
        req: ImageRequest,
        blob_name: str,
        blob_url: str,
        *,
        width: int,
        height: int,
        render: dict,
    ) -> "ImageResult":
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="success",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            seed=req.seed,
            render=render,
            warnings=warnings_for(req),
            blob_name=blob_name,
            blob_url=blob_url,
        )

    @classmethod
    def error_for(
        cls,
        req: Optional[ImageRequest],
        message: str,
        *,
        raw_job_id: str = "",
        width: int = 0,
        height: int = 0,
        render: Optional[dict] = None,
    ) -> "ImageResult":
        if req is None:
            return cls(
                job_id=raw_job_id or "unknown",
                name="unknown",
                status="error",
                prompt="",
                negative_prompt="",
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
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            seed=req.seed,
            render=render,
            warnings=warnings_for(req),
            error=message,
        )

    def to_json(self) -> str:
        # ensure_ascii=False: \u-escaping a non-ASCII prompt roughly doubles
        # its byte count (6 bytes/char vs 2-4 for raw UTF-8), which can push a
        # message that fit the request-side byte check over the queue's cap
        # on the way out. azure_io.send_result's guard assumes raw UTF-8.
        return json.dumps(asdict(self), ensure_ascii=False)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str, fallback: str = "image") -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or fallback
