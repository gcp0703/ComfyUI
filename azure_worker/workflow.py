"""Builds the ComfyUI workflow dict for a single Flux 2 Klein text-to-image request.

Flux 2 has no single-file checkpoint — the UNet, the text encoder, and the VAE
load through three separate nodes, the latent uses a 128-channel `EmptyFlux2LatentImage`
shape, and sampling is driven by `Flux2Scheduler` feeding a custom
`RandomNoise` → `BasicGuider` → `SamplerCustomAdvanced` chain rather than `KSampler`.

The shape mirrors ComfyUI's stock Flux 2 graph; see `comfy_extras/nodes_flux.py`
and `comfy_extras/nodes_custom_sampler.py` for the underlying node definitions.
"""
from __future__ import annotations

from .messages import ImageRequest, sanitize_name


SAVE_NODE_ID = "12"


def build_workflow(req: ImageRequest, unet: str, clip: str, vae: str) -> dict:
    filename_prefix = sanitize_name(req.name)
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "fp8_e4m3fn"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
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
        SAVE_NODE_ID: {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["11", 0]},
        },
    }
