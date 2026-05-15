"""Environment-variable configuration for the Azure → ComfyUI worker."""
from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    storage_connection_string: str
    inbound_queue: str
    outbound_queue: str
    blob_container: str
    flux_unet: str
    flux_clip: str
    flux_vae: str
    sas_expiry_hours: int = 24
    poll_interval_seconds: float = 2.0
    visibility_timeout_seconds: int = 300
    max_dequeue_count: int = 3


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"environment variable {name} is required")
    return value


def _optional_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"environment variable {name}={raw!r} is not an integer") from e


def _optional_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"environment variable {name}={raw!r} is not a number") from e


def load_config() -> Config:
    return Config(
        storage_connection_string=_require("AZURE_STORAGE_CONNECTION_STRING"),
        inbound_queue=_require("AZURE_INBOUND_QUEUE"),
        outbound_queue=_require("AZURE_OUTBOUND_QUEUE"),
        blob_container=_require("AZURE_BLOB_CONTAINER"),
        flux_unet=_require("COMFY_FLUX_UNET"),
        flux_clip=_require("COMFY_FLUX_CLIP"),
        flux_vae=_require("COMFY_FLUX_VAE"),
        sas_expiry_hours=_optional_int("SAS_EXPIRY_HOURS", 24),
        poll_interval_seconds=_optional_float("POLL_INTERVAL_SECONDS", 2.0),
        visibility_timeout_seconds=_optional_int("VISIBILITY_TIMEOUT_SECONDS", 300),
        max_dequeue_count=_optional_int("MAX_DEQUEUE_COUNT", 3),
    )
