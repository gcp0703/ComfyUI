"""Environment-variable configuration for the Azure → ComfyUI worker."""
from __future__ import annotations

import os
from dataclasses import dataclass


PROFILE_FLUX1_DEV = "flux1-dev"
PROFILE_FLUX2_KLEIN = "flux2-klein"
PROFILE_CHROMA1 = "chroma1"
PROFILE_FLUXED_UP = "fluxed-up"
PROFILE_QWEN_IMAGE_2512 = "qwen-image-2512"
PROFILE_QWEN_IMAGE_2_1 = "qwen-image-2.1"
PROFILE_OPENFLUX1 = "openflux1"
PROFILE_QWEN_RAPID_AIO = "qwen-rapid-aio"
PROFILE_SDXL_DREAMSHAPER = "sdxl-dreamshaper"
KNOWN_PROFILES = (
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_CHROMA1,
    PROFILE_FLUXED_UP,
    PROFILE_QWEN_IMAGE_2512,
    PROFILE_QWEN_IMAGE_2_1,
    PROFILE_OPENFLUX1,
    PROFILE_QWEN_RAPID_AIO,
    PROFILE_SDXL_DREAMSHAPER,
)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoraSpec:
    """One entry in a profile's LoRA stack.

    ``name`` is a path relative to ``models/loras`` using the *native* separator
    — the same string ComfyUI's ``LoraLoader`` validates against, since
    ``folder_paths.recursive_search`` builds its filename list with
    ``os.path.relpath`` (backslashes on Windows). Configuration is written with
    forward slashes and normalized by :func:`_optional_loras`.
    """

    name: str
    model_strength: float = 1.0
    clip_strength: float = 1.0


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
    # Qwen-Image 2.1 profile (Alibaba — Qwen3-VL 8B encoder, own 64-channel VAE,
    # the TextEncodeQwenImage21 node that emits both conditioning branches)
    qwen21_unet: str
    qwen21_clip: str
    qwen21_vae: str
    # OpenFLUX.1 profile (de-distilled Flux 1 schnell — same arch, real CFG; reuses flux1 CLIP-L/T5/VAE)
    openflux_unet: str
    # Qwen-Image-Edit Rapid AIO profile (Phr00t — all-in-one checkpoint: UNet+CLIP+VAE
    # merged into one file, loaded with CheckpointLoaderSimple. 4-step distilled, cfg=1.)
    qwen_rapid_checkpoint: str
    # SDXL DreamShaper XL profile (Lykon DreamShaper XL Turbo v2.1 — plain SDXL
    # 1.0 architecture in a single checkpoint with baked CLIP + VAE, plus an
    # ordered stack of fantasy-race LoRAs chained onto it.)
    sdxl_checkpoint: str
    sdxl_loras: tuple[LoraSpec, ...]
    # When set, a race named in the prompt appends that race's own LoRA to the
    # stack above (see lora_router). The multi-race LoRA alone does not reshape
    # body morphology strongly enough -- dwarves render as humans without it.
    sdxl_lora_autoroute: bool
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


def _optional_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"environment variable {name}={raw!r} is not a boolean")


def _optional_loras(name: str) -> tuple[LoraSpec, ...]:
    """Parse a comma-separated LoRA stack from one environment variable.

    Each entry is ``file[:model_strength[:clip_strength]]``; omitted strengths
    default to 1.0, and a lone model strength is reused for the CLIP strength
    (the usual case). An unset or empty variable means "no LoRAs".

        COMFY_SDXL_LORAS=dnd/RPGElfXL.safetensors:0.8,dnd/Elf_Ears_XL.safetensors:0.6:0.4

    Subfolder separators are written as ``/`` and rewritten to ``os.sep`` so the
    name matches ComfyUI's filename list on Windows as well as POSIX.
    """
    raw = os.environ.get(name) or ""
    specs: list[LoraSpec] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) > 3:
            raise ConfigError(
                f"environment variable {name} entry {entry!r} has too many ':' fields; "
                "expected file[:model_strength[:clip_strength]]"
            )
        lora_name = parts[0].replace("/", os.sep)
        if not lora_name:
            raise ConfigError(f"environment variable {name} entry {entry!r} has an empty filename")
        strengths: list[float] = []
        for field, part in zip(("model_strength", "clip_strength"), parts[1:]):
            try:
                strengths.append(float(part))
            except ValueError as e:
                raise ConfigError(
                    f"environment variable {name} entry {entry!r} has a non-numeric {field}: {part!r}"
                ) from e
        model_strength = strengths[0] if strengths else 1.0
        clip_strength = strengths[1] if len(strengths) > 1 else model_strength
        specs.append(LoraSpec(lora_name, model_strength, clip_strength))
    return tuple(specs)


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
        qwen21_unet=_require("COMFY_QWEN21_UNET"),
        qwen21_clip=_require("COMFY_QWEN21_CLIP"),
        qwen21_vae=_require("COMFY_QWEN21_VAE"),
        openflux_unet=_require("COMFY_OPENFLUX_UNET"),
        qwen_rapid_checkpoint=_require("COMFY_QWEN_RAPID_CHECKPOINT"),
        sdxl_checkpoint=_require("COMFY_SDXL_CHECKPOINT"),
        sdxl_loras=_optional_loras("COMFY_SDXL_LORAS"),
        sdxl_lora_autoroute=_optional_bool("COMFY_SDXL_LORA_AUTOROUTE", True),
        llm_inbound_queue=_require("LLM_INBOUND_QUEUE"),
        llm_outbound_queue=_require("LLM_OUTBOUND_QUEUE"),
        ollama_url=_require("OLLAMA_URL"),
        llm_request_timeout_seconds=_optional_int("LLM_REQUEST_TIMEOUT_SECONDS", 300),
        sas_expiry_hours=_optional_int("SAS_EXPIRY_HOURS", 24),
        poll_interval_seconds=_optional_float("POLL_INTERVAL_SECONDS", 2.0),
        visibility_timeout_seconds=_optional_int("VISIBILITY_TIMEOUT_SECONDS", 300),
        max_dequeue_count=_optional_int("MAX_DEQUEUE_COUNT", 3),
    )
