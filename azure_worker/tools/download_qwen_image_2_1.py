"""One-shot downloader for the Comfy-Org Qwen-Image 2.1 repack (int8_convrot).

Pulls the three files ComfyUI's Qwen-Image 2.1 template loads — the int8 UNet,
the int8 Qwen3-VL-8B text encoder and the bf16 VAE — straight into the matching
``models/`` subdirectories (no symlinks, no HF cache indirection; these are the
exact filenames ComfyUI's loaders read).

int8_convrot is the template's primary weight set: ~17 GiB total, which leaves
headroom on a 32 GB card. The bf16 pair is ~32 GiB for the weights alone.

Requires a ComfyUI new enough to ship ``comfy/text_encoders/qwen_image21.py``
and the ``int8_tensorwise``/convrot quant layout (v0.37.0+).
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Comfy-Org/Qwen-Image-2.1"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Repo-relative paths double as ComfyUI model-dir-relative paths.
FILES = (
    "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)

MIN_PLAUSIBLE_BYTES = 100_000_000


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(FILES, start=1):
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > MIN_PLAUSIBLE_BYTES:
            size_gib = dest.stat().st_size / 1024**3
            print(f"[{index}/{len(FILES)}] already present: {dest} ({size_gib:.2f} GiB)")
            continue
        print(f"[{index}/{len(FILES)}] downloading {REPO_ID}/{filename}")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(MODELS_DIR))
        size_gib = dest.stat().st_size / 1024**3
        print(f"[{index}/{len(FILES)}] OK: {dest} ({size_gib:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
