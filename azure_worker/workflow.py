"""ComfyUI workflow builders.

Two profiles are supported, selected at startup via `COMFY_PROFILE`:

- ``flux1-dev`` — the classic Flux 1 dev pipeline (UNETLoader + DualCLIPLoader
  with clip_l + T5-XXL + the Flux 1 VAE), shaped to match the official
  ``flux_dev_full_text_to_image`` workflow template that ships with ComfyUI.
- ``flux2-klein`` — the Flux 2 Klein pipeline (UNETLoader fp8 + CLIPLoader
  type=flux2 with Qwen3 + a 128-channel VAE), driven by ``Flux2Scheduler``
  feeding ``SamplerCustomAdvanced``.

A single ``build_workflow(req, cfg)`` dispatcher picks the right builder.
"""
from __future__ import annotations

from .config import Config, PROFILE_FLUX1_DEV, PROFILE_FLUX2_KLEIN
from .messages import ImageRequest, sanitize_name


# Save node IDs differ between profiles; main.py doesn't actually need them
# (PromptExecutor auto-discovers output nodes), but tests reference them.
FLUX1_SAVE_NODE_ID = "9"
FLUX2_SAVE_NODE_ID = "12"


def build_workflow(req: ImageRequest, cfg: Config) -> dict:
    if cfg.profile == PROFILE_FLUX1_DEV:
        return build_flux1_dev_workflow(req, cfg)
    if cfg.profile == PROFILE_FLUX2_KLEIN:
        return build_flux2_klein_workflow(req, cfg)
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
