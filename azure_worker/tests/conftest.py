"""Shared test helpers for the azure_worker suite."""
from __future__ import annotations

from typing import Optional

from azure_worker.config import Config
from azure_worker.profiles import RenderSettings, load_profiles

_PROFILES = load_profiles()


def make_config(
    profile: str,
    sdxl_loras: tuple = (),
    sdxl_lora_autoroute: bool = False,
    render: Optional[RenderSettings] = None,
) -> Config:
    """A fully-populated Config for ``profile`` with the shipped render settings.

    Pass ``render`` to override the profile's settings for one test (e.g. a
    different resolution to exercise the Qwen 2.1 shift).
    """
    return Config(
        storage_connection_string="x",
        inbound_queue="i",
        outbound_queue="o",
        blob_container="c",
        profile=profile,
        flux1_unet="flux1-dev.safetensors",
        flux1_clip_l="clip_l.safetensors",
        flux1_t5="t5xxl_fp16.safetensors",
        flux1_vae="ae.safetensors",
        flux2_unet="flux-2-klein-9b-fp8.safetensors",
        flux2_clip="qwen_3_8b_fp8mixed.safetensors",
        flux2_vae="full_encoder_small_decoder.safetensors",
        chroma_unet="Chroma1-HD-fp8mixed.safetensors",
        chroma_clip="t5xxl_fp16.safetensors",
        chroma_vae="ae.safetensors",
        fluxedup_unet="fluxedUpFluxNSFW_40DevFp8.safetensors",
        qwen_unet="qwen_image_2512_fp8_e4m3fn.safetensors",
        qwen_clip="qwen_2.5_vl_7b_fp8_scaled.safetensors",
        qwen_vae="qwen_image_vae.safetensors",
        qwen21_unet="qwen_image_2.1_int8_convrot.safetensors",
        qwen21_clip="qwen3vl_8b_int8_convrot.safetensors",
        qwen21_vae="qwen_image_2.1_vae_bf16.safetensors",
        openflux_unet="openflux1-v0.1.0-fp8.safetensors",
        qwen_rapid_checkpoint="Qwen-Rapid-AIO-NSFW-v23.safetensors",
        sdxl_checkpoint="DreamShaperXL_Turbo_v2_1.safetensors",
        sdxl_loras=sdxl_loras,
        sdxl_lora_autoroute=sdxl_lora_autoroute,
        render=render or _PROFILES[profile],
        llm_inbound_queue="llm-requests",
        llm_outbound_queue="llm-results",
        ollama_url="http://localhost:11434",
    )
