"""ComfyUI workflow builders.

Four profiles are supported, selected at startup via `COMFY_PROFILE`:

- ``flux1-dev`` — the classic Flux 1 dev pipeline (UNETLoader + DualCLIPLoader
  with clip_l + T5-XXL + the Flux 1 VAE), shaped to match the official
  ``flux_dev_full_text_to_image`` workflow template that ships with ComfyUI.
- ``flux2-klein`` — the Flux 2 Klein pipeline (UNETLoader fp8 + CLIPLoader
  type=flux2 with Qwen3 + a 128-channel VAE), driven by ``Flux2Scheduler``
  feeding ``SamplerCustomAdvanced``.
- ``chroma1`` — Chroma1 (a de-distilled Flux derivative by lodestone-rock):
  UNETLoader + CLIPLoader(type=chroma) with T5 only + Flux 1 VAE, driven by
  ``ModelSamplingAuraFlow`` (sigma shift 1.0) + ``CFGGuider`` + ``BasicScheduler``
  (beta) + ``SamplerCustomAdvanced``. Unlike the Flux profiles, Chroma supports
  proper CFG and a real negative prompt.
- ``fluxed-up`` — NSFW Flux 1 dev finetune. Same workflow shape as flux1-dev
  (DualCLIPLoader/KSampler/ConditioningZeroOut), but loads the Fluxed Up UNet
  in fp8 mode. Reuses the flux1 CLIP-L/T5/VAE since the architecture is
  identical.

A single ``build_workflow(req, cfg)`` dispatcher picks the right builder.
"""
from __future__ import annotations

from .config import (
    Config,
    PROFILE_CHROMA1,
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_FLUXED_UP,
)
from .messages import ImageRequest, sanitize_name


# Save node IDs differ between profiles; main.py doesn't actually need them
# (PromptExecutor auto-discovers output nodes), but tests reference them.
FLUX1_SAVE_NODE_ID = "9"
FLUX2_SAVE_NODE_ID = "12"
CHROMA1_SAVE_NODE_ID = "14"
FLUXED_UP_SAVE_NODE_ID = "9"


# Chroma sampling defaults baked into the workflow — these are not user-tunable
# per request because they're tied to the model's training and the official
# lodestone-rock recipe (Euler + Beta + shift=1.0).
CHROMA_SAMPLER = "euler"
CHROMA_SCHEDULER = "beta"
CHROMA_SHIFT = 1.0


def build_workflow(req: ImageRequest, cfg: Config) -> dict:
    if cfg.profile == PROFILE_FLUX1_DEV:
        return build_flux1_dev_workflow(req, cfg)
    if cfg.profile == PROFILE_FLUX2_KLEIN:
        return build_flux2_klein_workflow(req, cfg)
    if cfg.profile == PROFILE_CHROMA1:
        return build_chroma1_workflow(req, cfg)
    if cfg.profile == PROFILE_FLUXED_UP:
        return build_fluxed_up_workflow(req, cfg)
    raise ValueError(f"unknown profile {cfg.profile!r}")  # pragma: no cover


def build_flux1_dev_workflow(req: ImageRequest, cfg: Config) -> dict:
    """Mirror of ComfyUI's ``flux_dev_full_text_to_image`` template.

    Uses KSampler with cfg=1 and ConditioningZeroOut for the negative — this is
    the recommended driver for guidance-distilled Flux 1 dev (cheaper than
    FluxGuidance + BasicGuider for plain text-to-image).
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.flux1_unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": cfg.flux1_clip_l,
                "clip_name2": cfg.flux1_t5,
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg.flux1_vae},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.prompt},
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": req.seed,
                "steps": req.steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        FLUX1_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def build_flux2_klein_workflow(req: ImageRequest, cfg: Config) -> dict:
    """Flux 2 Klein t2i — single CLIP (Qwen3) + 128-channel VAE + custom scheduler.

    Flux 2 has no single-file checkpoint and uses a different latent format
    from Flux 1 (128 channels, 1/16 spatial). Sampling is driven by
    ``Flux2Scheduler`` feeding ``RandomNoise`` → ``BasicGuider`` →
    ``SamplerCustomAdvanced`` rather than the stock ``KSampler``.
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.flux2_unet, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": cfg.flux2_clip, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg.flux2_vae},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.prompt},
        },
        "5": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": req.steps,
                "width": req.width,
                "height": req.height,
            },
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": req.seed},
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["1", 0], "conditioning": ["4", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["6", 0],
                "latent_image": ["5", 0],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        FLUX2_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["11", 0]},
        },
    }


def build_chroma1_workflow(req: ImageRequest, cfg: Config) -> dict:
    """Mirror of ComfyUI's ``image_chroma_text_to_image`` template.

    Chroma is a de-distilled Flux derivative — it uses the Flux 1 VAE and a
    16-channel latent (``EmptySD3LatentImage``), a T5-only CLIP path
    (``CLIPLoader`` ``type=chroma``), and crucially the Beta noise schedule
    that the model was trained against. Sigma shift is patched to 1.0 via
    ``ModelSamplingAuraFlow`` (the "Flow Shift" node).

    Unlike the Flux profiles, Chroma supports real CFG and a real negative
    prompt — ``req.cfg`` flows into ``CFGGuider`` and ``req.negative_prompt``
    is encoded by a second ``CLIPTextEncode``.
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.chroma_unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": cfg.chroma_clip, "type": "chroma"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg.chroma_vae},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": CHROMA_SHIFT},
        },
        "5": {
            "class_type": "T5TokenizerOptions",
            "inputs": {"clip": ["2", 0], "min_padding": 1, "min_length": 0},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": req.prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": req.negative_prompt or ""},
        },
        "8": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "cfg": req.cfg,
            },
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": CHROMA_SAMPLER},
        },
        "10": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["4", 0],
                "scheduler": CHROMA_SCHEDULER,
                "steps": req.steps,
                "denoise": 1.0,
            },
        },
        "11": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": req.seed},
        },
        "12": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["8", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["12", 0],
            },
        },
        CHROMA1_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["15", 0],
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["3", 0]},
        },
    }


def build_fluxed_up_workflow(req: ImageRequest, cfg: Config) -> dict:
    """NSFW Flux 1 dev finetune (Scorpion06/FluxedUp).

    Identical pipeline shape to :func:`build_flux1_dev_workflow` — same
    DualCLIPLoader (clip_l + T5-XXL, type=flux), same Flux 1 VAE
    (``ae.safetensors``), same KSampler with ``cfg=1`` + ``ConditioningZeroOut``
    for the negative (Flux 1 dev is guidance-distilled). The only differences
    are the UNet file (the finetune) and ``weight_dtype="fp8_e4m3fn"`` because
    the Fluxed Up release is already fp8-mixed quantized.

    Reuses ``cfg.flux1_clip_l`` / ``cfg.flux1_t5`` / ``cfg.flux1_vae`` since
    those files are byte-identical for any Flux 1 dev derivative.
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.fluxedup_unet, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": cfg.flux1_clip_l,
                "clip_name2": cfg.flux1_t5,
                "type": "flux",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg.flux1_vae},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.prompt},
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": req.seed,
                "steps": req.steps,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        FLUXED_UP_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }
