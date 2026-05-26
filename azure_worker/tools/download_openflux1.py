"""One-shot downloader for ostris/OpenFLUX.1 fp8 UNet.

Pulls only the single ~17.2 GB safetensors file the worker needs, straight into
ComfyUI's ``models/diffusion_models/`` directory (no symlinks, no HF cache
indirection — this is the file ComfyUI's UNETLoader will read by name).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ostris/OpenFLUX.1"
FILENAME = "openflux1-v0.1.0-fp8.safetensors"
TARGET_DIR = Path(__file__).resolve().parents[2] / "models" / "diffusion_models"


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target = TARGET_DIR / FILENAME
    if target.exists() and target.stat().st_size > 1_000_000_000:
        print(f"already present: {target} ({target.stat().st_size / 1024**3:.2f} GiB)")
        return 0

    print(f"downloading {REPO_ID}/{FILENAME} -> {TARGET_DIR}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(TARGET_DIR),
    )
    p = Path(path)
    size_gib = p.stat().st_size / 1024**3
    print(f"OK: {p} ({size_gib:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
