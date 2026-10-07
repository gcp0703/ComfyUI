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
    f = _write(tmp_path, '[flux1-dev]\nwidth = 1024\nheight = 1024\nsteps = 20\nsampler = "euler"\n')
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


def test_default_path_points_inside_the_package():
    assert DEFAULT_PROFILES_PATH.name == "profiles.toml"
    assert DEFAULT_PROFILES_PATH.parent.name == "azure_worker"
