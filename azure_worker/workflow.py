"""ComfyUI workflow builders.

Seven profiles are supported, selected at startup via `COMFY_PROFILE`:

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
- ``qwen-image-2512`` — Alibaba Qwen-Image (December 2025 release). UNETLoader
  fp8 + CLIPLoader(type=qwen_image) with Qwen 2.5 VL 7B + its own VAE,
  ``ModelSamplingAuraFlow`` (sigma shift 3.1), stock KSampler (euler/simple).
  Like chroma1, honors ``req.cfg`` and ``req.negative_prompt``.
- ``openflux1`` — ostris/OpenFLUX.1, a de-distilled Flux 1 schnell. Same Flux 1
  architecture as flux1-dev (reuses CLIP-L + T5-XXL + ae.safetensors VAE) but
  loads the OpenFLUX UNet in fp8 mode and drives KSampler with real ``req.cfg``
  + a real second ``CLIPTextEncode`` for ``req.negative_prompt`` instead of
  ``ConditioningZeroOut``.
- ``qwen-rapid-aio`` — Phr00t/Qwen-Image-Edit-Rapid-AIO, an all-in-one merged
  checkpoint (UNet + CLIP + VAE in one file) loaded with
  ``CheckpointLoaderSimple`` + ``TextEncodeQwenImageEditPlus`` (no images = pure
  text-to-image). A 4-step distilled accelerator merge: cfg=1 +
  ``ConditioningZeroOut`` with ``euler_ancestral``/``beta``. Like flux1-dev,
  ``req.cfg`` and ``req.negative_prompt`` are no-ops.

A single ``build_workflow(req, cfg)`` dispatcher picks the right builder.
"""
from __future__ import annotations

from .config import (
    Config,
    PROFILE_CHROMA1,
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_FLUXED_UP,
    PROFILE_OPENFLUX1,
    PROFILE_QWEN_IMAGE_2512,
    PROFILE_QWEN_RAPID_AIO,
)
from .messages import ImageRequest, sanitize_name


# Save node IDs differ between profiles; main.py doesn't actually need them
# (PromptExecutor auto-discovers output nodes), but tests reference them.
FLUX1_SAVE_NODE_ID = "9"
FLUX2_SAVE_NODE_ID = "12"
CHROMA1_SAVE_NODE_ID = "14"
FLUXED_UP_SAVE_NODE_ID = "9"
QWEN_IMAGE_SAVE_NODE_ID = "10"
OPENFLUX1_SAVE_NODE_ID = "9"
QWEN_RAPID_SAVE_NODE_ID = "7"


# Chroma sampling defaults baked into the workflow — these are not user-tunable
# per request because they're tied to the model's training and the official
# lodestone-rock recipe (Euler + Beta + shift=1.0).
CHROMA_SAMPLER = "euler"
CHROMA_SCHEDULER = "beta"
CHROMA_SHIFT = 1.0


# Qwen-Image 2512 sampling defaults (from the official ComfyUI template).
QWEN_IMAGE_SAMPLER = "euler"
QWEN_IMAGE_SCHEDULER = "simple"
QWEN_IMAGE_SHIFT = 3.1


# Qwen-Image-Edit Rapid AIO sampling defaults (Phr00t model card, v23):
# 4-step distilled accelerator merge — run at cfg=1 with euler_ancestral/beta.
# These are baked in, not user-tunable, because they're tied to the merge.
QWEN_RAPID_SAMPLER = "euler_ancestral"
QWEN_RAPID_SCHEDULER = "beta"


def build_workflow(req: ImageRequest, cfg: Config) -> dict:
    if cfg.profile == PROFILE_FLUX1_DEV:
        return build_flux1_dev_workflow(req, cfg)
    if cfg.profile == PROFILE_FLUX2_KLEIN:
        return build_flux2_klein_workflow(req, cfg)
    if cfg.profile == PROFILE_CHROMA1:
        return build_chroma1_workflow(req, cfg)
    if cfg.profile == PROFILE_FLUXED_UP:
        return build_fluxed_up_workflow(req, cfg)
    if cfg.profile == PROFILE_QWEN_IMAGE_2512:
        return build_qwen_image_2512_workflow(req, cfg)
    if cfg.profile == PROFILE_OPENFLUX1:
        return build_openflux1_workflow(req, cfg)
    if cfg.profile == PROFILE_QWEN_RAPID_AIO:
        return build_qwen_rapid_aio_workflow(req, cfg)
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


def build_qwen_image_2512_workflow(req: ImageRequest, cfg: Config) -> dict:
    """Mirror of ComfyUI's ``image_qwen_image`` template (December 2025 release).

    Qwen-Image is a separate model family from Flux — it uses Qwen 2.5 VL 7B as
    the text encoder (``CLIPLoader`` ``type=qwen_image``, NOT the Qwen3 encoders
    used by Flux 2 Klein), its own VAE, and ``ModelSamplingAuraFlow`` with
    sigma shift 3.1 (different from Chroma's 1.0). Sampling is plain
    ``KSampler`` with ``euler`` + ``simple`` — no custom sampler chain needed.

    Like Chroma1, Qwen-Image supports real CFG and a real negative prompt:
    ``req.cfg`` flows into KSampler and ``req.negative_prompt`` is encoded by a
    second ``CLIPTextEncode``. The official recipe recommends ~20-50 steps at
    cfg=4.0 for the base 2512 model (without the 2-step Turbo LoRA).
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.qwen_unet, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": cfg.qwen_clip, "type": "qwen_image"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg.qwen_vae},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": QWEN_IMAGE_SHIFT},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.negative_prompt or ""},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": req.seed,
                "steps": req.steps,
                "cfg": req.cfg,
                "sampler_name": QWEN_IMAGE_SAMPLER,
                "scheduler": QWEN_IMAGE_SCHEDULER,
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        QWEN_IMAGE_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]},
        },
    }


def build_openflux1_workflow(req: ImageRequest, cfg: Config) -> dict:
    """OpenFLUX.1 (ostris) — de-distilled Flux 1 schnell.

    Architecturally identical to Flux 1 dev (same DualCLIPLoader with
    clip_l + T5-XXL, same Flux 1 VAE ``ae.safetensors``) — only the UNet differs
    and is loaded in fp8 mode. Unlike ``flux1-dev`` / ``fluxed-up``, the
    distillation has been trained out, so the workflow uses **real CFG and a
    real negative prompt**: ``req.cfg`` flows into KSampler and
    ``req.negative_prompt`` is encoded by a second ``CLIPTextEncode`` (no
    ``ConditioningZeroOut``).

    Reuses ``cfg.flux1_clip_l`` / ``cfg.flux1_t5`` / ``cfg.flux1_vae`` since
    those files are byte-identical for any Flux 1 derivative. Recommended
    request params per the ostris model card: ``cfg≈3.5``, ``steps=20`` and up.
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg.openflux_unet, "weight_dtype": "fp8_e4m3fn"},
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
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": req.negative_prompt or ""},
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
                "cfg": req.cfg,
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
        OPENFLUX1_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def build_qwen_rapid_aio_workflow(req: ImageRequest, cfg: Config) -> dict:
    """Phr00t/Qwen-Image-Edit-Rapid-AIO — all-in-one checkpoint, pure text-to-image.

    Unlike every other profile, this is a single merged checkpoint (UNet + CLIP +
    VAE in one file), so it loads with ``CheckpointLoaderSimple`` rather than the
    separate UNETLoader/CLIPLoader/VAELoader chain. The prompt is encoded by
    ``TextEncodeQwenImageEditPlus`` with no input images — per the model card,
    "provide no images to just do pure text to image."

    It is a 4-step distilled accelerator merge driven at ``cfg=1`` with
    ``euler_ancestral``/``beta`` (the v23 recommendation). Like ``flux1-dev`` and
    ``fluxed-up`` it is guidance-distilled, so ``req.cfg`` and ``req.negative_prompt``
    are no-ops — the negative branch is a ``ConditioningZeroOut`` placeholder.
    Recommended ``req.steps`` is 4 (4-8 works).

    The ``NSFW-v23`` build merges the NSFW LoRAs directly into the weights, so no
    trigger keyword is required; the SFW build is the same graph with a different
    checkpoint file.
    """
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": cfg.qwen_rapid_checkpoint},
        },
        "2": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["1", 1], "prompt": req.prompt},
        },
        "3": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["2", 0]},
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": req.width,
                "height": req.height,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": req.seed,
                "steps": req.steps,
                "cfg": 1,
                "sampler_name": QWEN_RAPID_SAMPLER,
                "scheduler": QWEN_RAPID_SCHEDULER,
                "denoise": 1,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        QWEN_RAPID_SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["6", 0]},
        },
    }
