"""Environment-variable configuration for the Azure → ComfyUI worker."""
from __future__ import annotations

import os
from dataclasses import dataclass


PROFILE_FLUX1_DEV = "flux1-dev"
PROFILE_FLUX2_KLEIN = "flux2-klein"
PROFILE_CHROMA1 = "chroma1"
PROFILE_FLUXED_UP = "fluxed-up"
PROFILE_QWEN_IMAGE_2512 = "qwen-image-2512"
PROFILE_OPENFLUX1 = "openflux1"
PROFILE_QWEN_RAPID_AIO = "qwen-rapid-aio"
KNOWN_PROFILES = (
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_CHROMA1,
    PROFILE_FLUXED_UP,
    PROFILE_QWEN_IMAGE_2512,
    PROFILE_OPENFLUX1,
    PROFILE_QWEN_RAPID_AIO,
)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    storage_connection_string: str
    inbound_queue: str
    outbound_queue: str
    blob_container: str
    profile: str
    # Flux 1 dev profile
    flux1_unet: str
    flux1_clip_l: str
    flux1_t5: str
    flux1_vae: str
    # Flux 2 Klein profile
    flux2_unet: str
    flux2_clip: str
    flux2_vae: str
    # Chroma1 profile (de-distilled Flux derivative — T5 only, real CFG)
    chroma_unet: str
    chroma_clip: str
    chroma_vae: str
    # Fluxed Up profile (NSFW Flux 1 dev finetune — same arch, reuses flux1 CLIP-L/T5/VAE)
    fluxedup_unet: str
    # Qwen-Image 2512 profile (Alibaba — Qwen 2.5 VL text encoder, own VAE, real CFG)
    qwen_unet: str
    qwen_clip: str
    qwen_vae: str
    # OpenFLUX.1 profile (de-distilled Flux 1 schnell — same arch, real CFG; reuses flux1 CLIP-L/T5/VAE)
    openflux_unet: str
    # Qwen-Image-Edit Rapid AIO profile (Phr00t — all-in-one checkpoint: UNet+CLIP+VAE
    # merged into one file, loaded with CheckpointLoaderSimple. 4-step distilled, cfg=1.)
    qwen_rapid_checkpoint: str
    # LLM-side queues + Ollama HTTP endpoint (separate workload, polled with
    # priority over the image queue in the main loop). Uses Ollama's native
    # /api/chat (not OpenAI-compat) so the `think` toggle works on Qwen3.
    llm_inbound_queue: str
    llm_outbound_queue: str
    ollama_url: str
    llm_request_timeout_seconds: int = 300
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
    profile = _require("COMFY_PROFILE")
    if profile not in KNOWN_PROFILES:
        raise ConfigError(
            f"COMFY_PROFILE={profile!r} not recognized; expected one of {KNOWN_PROFILES}"
        )

    return Config(
        storage_connection_string=_require("AZURE_STORAGE_CONNECTION_STRING"),
        inbound_queue=_require("AZURE_INBOUND_QUEUE"),
        outbound_queue=_require("AZURE_OUTBOUND_QUEUE"),
        blob_container=_require("AZURE_BLOB_CONTAINER"),
        profile=profile,
        flux1_unet=_require("COMFY_FLUX1_UNET"),
        flux1_clip_l=_require("COMFY_FLUX1_CLIP_L"),
        flux1_t5=_require("COMFY_FLUX1_T5"),
        flux1_vae=_require("COMFY_FLUX1_VAE"),
        flux2_unet=_require("COMFY_FLUX2_UNET"),
        flux2_clip=_require("COMFY_FLUX2_CLIP"),
        flux2_vae=_require("COMFY_FLUX2_VAE"),
        chroma_unet=_require("COMFY_CHROMA_UNET"),
        chroma_clip=_require("COMFY_CHROMA_CLIP"),
        chroma_vae=_require("COMFY_CHROMA_VAE"),
        fluxedup_unet=_require("COMFY_FLUXEDUP_UNET"),
        qwen_unet=_require("COMFY_QWEN_UNET"),
        qwen_clip=_require("COMFY_QWEN_CLIP"),
        qwen_vae=_require("COMFY_QWEN_VAE"),
        openflux_unet=_require("COMFY_OPENFLUX_UNET"),
        qwen_rapid_checkpoint=_require("COMFY_QWEN_RAPID_CHECKPOINT"),
        llm_inbound_queue=_require("LLM_INBOUND_QUEUE"),
        llm_outbound_queue=_require("LLM_OUTBOUND_QUEUE"),
        ollama_url=_require("OLLAMA_URL"),
        llm_request_timeout_seconds=_optional_int("LLM_REQUEST_TIMEOUT_SECONDS", 300),
        sas_expiry_hours=_optional_int("SAS_EXPIRY_HOURS", 24),
        poll_interval_seconds=_optional_float("POLL_INTERVAL_SECONDS", 2.0),
        visibility_timeout_seconds=_optional_int("VISIBILITY_TIMEOUT_SECONDS", 300),
        max_dequeue_count=_optional_int("MAX_DEQUEUE_COUNT", 3),
    )
