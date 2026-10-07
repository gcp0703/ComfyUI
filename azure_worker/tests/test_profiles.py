"""profiles.toml loading and validation (no GPU, no network)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from azure_worker.config import KNOWN_PROFILES
from azure_worker.profiles import (
    DEFAULT_PROFILES_PATH,
    ProfileError,
    RenderSettings,
    load_profiles,
)


def test_shipped_file_covers_every_known_profile():
    profiles = load_profiles(required=KNOWN_PROFILES)
    assert set(profiles) == set(KNOWN_PROFILES)


def test_shipped_values_match_the_contract_table():
    p = load_profiles()
    assert p["qwen-image-2.1"] == RenderSettings(
        width=2048, height=2048, steps=45, cfg=1.0, sampler="euler", scheduler="simple", shift=None
    )
    assert p["qwen-image-2512"] == RenderSettings(
        width=1328, height=1328, steps=20, cfg=4.0, sampler="euler", scheduler="simple", shift=3.1
    )
    assert p["sdxl-dreamshaper"] == RenderSettings(
        width=1024, height=1024, steps=6, cfg=2.0, sampler="dpmpp_sde", scheduler="karras", shift=None
    )
    assert p["qwen-rapid-aio"].steps == 4
    assert p["chroma1"].shift == 1.0


def test_flux2_klein_has_no_cfg_and_no_scheduler():
    p = load_profiles()["flux2-klein"]
    assert p.cfg is None
    assert p.scheduler is None
    assert p.sampler == "euler"


def _write(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "profiles.toml"
    f.write_text(body, encoding="utf-8")
    return f


def test_missing_required_profile_is_an_error(tmp_path):
    f = _write(
        tmp_path,
        '[flux1-dev]\nwidth = 1024\nheight = 1024\nsteps = 20\nsampler = "euler"\ncfg = 1.0\nscheduler = "simple"\n',
    )
    with pytest.raises(ProfileError, match="chroma1"):
        load_profiles(f, required=("flux1-dev", "chroma1"))


@pytest.mark.parametrize(
    "bad_line,needle",
    [
        ("width = 1032", "multiple of 16"),
        ("width = 32", "[64, 4096]"),
        ("steps = 0", "steps"),
        ("steps = 201", "steps"),
        ("cfg = 31.0", "cfg"),
        ("cfg = -1", "cfg"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, bad_line, needle):
    base = {"width": "width = 1024", "height": "height = 1024", "steps": "steps = 20", "cfg": "cfg = 1.0"}
    key = bad_line.split(" ")[0]
    base[key] = bad_line
    body = "[p]\n" + "\n".join(base.values()) + '\nsampler = "euler"\n'
    with pytest.raises(ProfileError, match=re.escape(needle)):
        load_profiles(_write(tmp_path, body))


def test_missing_required_key_is_an_error(tmp_path):
    f = _write(tmp_path, '[p]\nwidth = 1024\nheight = 1024\nsteps = 20\n')  # no sampler
    with pytest.raises(ProfileError, match="sampler"):
        load_profiles(f)


def test_unknown_key_is_an_error(tmp_path):
    f = _write(tmp_path, '[p]\nwidth = 1024\nheight = 1024\nsteps = 20\nsampler = "euler"\nstep = 4\n')
    with pytest.raises(ProfileError, match="step"):
        load_profiles(f)


def test_chroma1_without_shift_is_an_error(tmp_path):
    f = _write(
        tmp_path,
        '[chroma1]\nwidth = 1024\nheight = 1024\nsteps = 26\ncfg = 3.5\nsampler = "euler"\nscheduler = "beta"\n',
    )
    with pytest.raises(ProfileError, match="shift"):
        load_profiles(f)


def test_qwen_image_2_1_with_shift_is_an_error(tmp_path):
    f = _write(
        tmp_path,
        '["qwen-image-2.1"]\nwidth = 2048\nheight = 2048\nsteps = 45\ncfg = 1.0\n'
        'sampler = "euler"\nscheduler = "simple"\nshift = 1.0\n',
    )
    with pytest.raises(ProfileError, match="unexpected"):
        load_profiles(f)


def test_flux2_klein_with_cfg_is_an_error(tmp_path):
    f = _write(
        tmp_path,
        '[flux2-klein]\nwidth = 1024\nheight = 1024\nsteps = 20\nsampler = "euler"\ncfg = 1.0\n',
    )
    with pytest.raises(ProfileError):
        load_profiles(f)


def test_shipped_file_still_loads_under_the_per_profile_schema():
    # test_shipped_file_covers_every_known_profile already loads the shipped
    # file; this just names the guarantee this finding is about explicitly.
    profiles = load_profiles()
    assert profiles["chroma1"].shift == 1.0
    assert profiles["qwen-image-2.1"].shift is None
    assert profiles["flux2-klein"].cfg is None and profiles["flux2-klein"].scheduler is None


def test_default_path_points_inside_the_package():
    assert DEFAULT_PROFILES_PATH.name == "profiles.toml"
    assert DEFAULT_PROFILES_PATH.parent.name == "azure_worker"


def test_load_config_attaches_active_profile_render_settings(monkeypatch):
    from azure_worker.config import load_config

    env = {
        "COMFY_PROFILE": "qwen-rapid-aio",
        "AZURE_STORAGE_CONNECTION_STRING": "x", "AZURE_INBOUND_QUEUE": "i",
        "AZURE_OUTBOUND_QUEUE": "o", "AZURE_BLOB_CONTAINER": "c",
        "COMFY_FLUX1_UNET": "a", "COMFY_FLUX1_CLIP_L": "a", "COMFY_FLUX1_T5": "a", "COMFY_FLUX1_VAE": "a",
        "COMFY_FLUX2_UNET": "a", "COMFY_FLUX2_CLIP": "a", "COMFY_FLUX2_VAE": "a",
        "COMFY_CHROMA_UNET": "a", "COMFY_CHROMA_CLIP": "a", "COMFY_CHROMA_VAE": "a",
        "COMFY_FLUXEDUP_UNET": "a",
        "COMFY_QWEN_UNET": "a", "COMFY_QWEN_CLIP": "a", "COMFY_QWEN_VAE": "a",
        "COMFY_QWEN21_UNET": "a", "COMFY_QWEN21_CLIP": "a", "COMFY_QWEN21_VAE": "a",
        "COMFY_OPENFLUX_UNET": "a", "COMFY_QWEN_RAPID_CHECKPOINT": "a", "COMFY_SDXL_CHECKPOINT": "a",
        "LLM_INBOUND_QUEUE": "l", "LLM_OUTBOUND_QUEUE": "m", "OLLAMA_URL": "http://x",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cfg = load_config()
    assert cfg.render.steps == 4
    assert cfg.render.sampler == "euler_ancestral"
    assert (cfg.render.width, cfg.render.height) == (1024, 1024)
