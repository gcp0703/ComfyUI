"""Message contracts for the inbound and outbound Azure Storage Queues."""
from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional


MAX_PROMPT_CHARS = 4096
MIN_DIM = 64
MAX_DIM = 4096
# Flux 2's EmptyFlux2LatentImage requires width/height in 16-pixel increments
# (see nodes_flux.py: step=16 on the int inputs).
DIM_MULTIPLE = 16


class MessageValidationError(ValueError):
    pass


@dataclass
class ImageRequest:
    job_id: str
    name: str
    prompt: str
    width: int
    height: int
    negative_prompt: str = ""
    seed: int = 0
    steps: int = 20
    cfg: float = 7.0

    @classmethod
    def from_json(cls, raw: str) -> "ImageRequest":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MessageValidationError(f"message is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise MessageValidationError("message must be a JSON object")

        job_id = str(data.get("job_id") or uuid.uuid4())
        name = data.get("name")
        prompt = data.get("prompt")
        width = data.get("width")
        height = data.get("height")

        if not isinstance(name, str) or not name.strip():
            raise MessageValidationError("'name' is required and must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MessageValidationError("'prompt' is required and must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise MessageValidationError(f"'prompt' exceeds {MAX_PROMPT_CHARS} characters")
        if not isinstance(width, int) or not isinstance(height, int):
            raise MessageValidationError("'width' and 'height' must be integers")
        _validate_dim("width", width)
        _validate_dim("height", height)

        negative = data.get("negative_prompt", "") or ""
        if not isinstance(negative, str):
            raise MessageValidationError("'negative_prompt' must be a string")
        if len(negative) > MAX_PROMPT_CHARS:
            raise MessageValidationError(f"'negative_prompt' exceeds {MAX_PROMPT_CHARS} characters")

        seed_raw = data.get("seed")
        if seed_raw is None:
            seed = random.randint(0, 2**63 - 1)
        elif isinstance(seed_raw, int):
            seed = seed_raw
        else:
            raise MessageValidationError("'seed' must be an integer if provided")

        steps = data.get("steps", 20)
        if not isinstance(steps, int) or not (1 <= steps <= 200):
            raise MessageValidationError("'steps' must be an integer in [1, 200]")

        cfg_raw = data.get("cfg", 7.0)
        if isinstance(cfg_raw, int):
            cfg_raw = float(cfg_raw)
        if not isinstance(cfg_raw, float) or not (0.0 <= cfg_raw <= 30.0):
            raise MessageValidationError("'cfg' must be a number in [0, 30]")

        return cls(
            job_id=job_id,
            name=name.strip(),
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg_raw,
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
            width=req.width,
            height=req.height,
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
            width=req.width,
            height=req.height,
            seed=req.seed,
            error=message,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str, fallback: str = "image") -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or fallback


def _validate_dim(field_name: str, value: int) -> None:
    if not (MIN_DIM <= value <= MAX_DIM):
        raise MessageValidationError(
            f"'{field_name}'={value} must be in [{MIN_DIM}, {MAX_DIM}]"
        )
    if value % DIM_MULTIPLE != 0:
        raise MessageValidationError(
            f"'{field_name}'={value} must be a multiple of {DIM_MULTIPLE}"
        )
