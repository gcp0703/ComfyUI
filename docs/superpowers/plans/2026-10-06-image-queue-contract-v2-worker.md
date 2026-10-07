# Image Queue Contract v2 — Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The worker owns per-profile size/steps/cfg/sampler in a tracked `profiles.toml`, the job message drops those fields, and every result carries a `render` block read back from the executed graph.

**Architecture:** A new `profiles.py` loads `profiles.toml` into a frozen `RenderSettings` per profile and `Config` gains a `render` field for the active one; the nine workflow builders read size/steps/cfg/sampler from `cfg.render` instead of the request. `summarize_workflow()` is replaced by `render_report()`, which returns the spec's `render` dict (including the display `summary`) and feeds both the log line and the result message. `ImageRequest` tolerates the four legacy fields and records them; `ImageResult` gains `negative_prompt`, `render`, `warnings`, and reports graph-derived `width`/`height`.

**Tech Stack:** Python 3.13 (`tomllib` stdlib), dataclasses, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md`

## Global Constraints

- Run tests with: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q` (from `D:\Projects\ComfyUI`). Baseline before this plan: **129 passed**. Every task ends green.
- `prompt` + `negative_prompt` combined ≤ **32,000 characters** (spec §2).
- Raw request JSON ≤ **46 KiB** (`46 * 1024` bytes) — worker-side byte backstop under the queue's 48 KiB raw ceiling, leaving room for the result's added fields (spec §2, §9).
- Legacy fields `width`, `height`, `steps`, `cfg` are **accepted, ignored, and listed in `warnings`** as `"ignored client-supplied fields: width, height, steps, cfg"` (spec §2, §6).
- `render` is `null` on validation errors and **populated** on runtime errors; `width`/`height` on runtime errors are the graph's intended size, `0` on validation errors (spec §3, §4).
- `render` field set, exactly these ten keys: `profile, model, loras, steps, cfg, sampler, scheduler, shift, negative_honored, summary` (spec §4).
- `cfg` and `scheduler` are `null` on `flux2-klein`; `shift` is `null` on profiles without a `ModelSamplingAuraFlow` node (spec §4).
- Profile values are exactly the spec §5 table. Reproduced here because the executor needs them in every task:

| Profile | width | height | steps | cfg | sampler | scheduler | shift |
|---|---|---|---|---|---|---|---|
| `flux1-dev` | 1024 | 1024 | 20 | 1.0 | euler | simple | — |
| `flux2-klein` | 1024 | 1024 | 20 | — | euler | — | — |
| `chroma1` | 1024 | 1024 | 26 | 3.5 | euler | beta | 1.0 |
| `fluxed-up` | 1024 | 1024 | 20 | 1.0 | euler | simple | — |
| `qwen-image-2512` | 1328 | 1328 | 20 | 4.0 | euler | simple | 3.1 |
| `qwen-image-2.1` | 2048 | 2048 | 45 | 1.0 | euler | simple | derived |
| `openflux1` | 1024 | 1024 | 20 | 3.5 | euler | simple | — |
| `qwen-rapid-aio` | 1024 | 1024 | 4 | 1.0 | euler_ancestral | beta | — |
| `sdxl-dreamshaper` | 1024 | 1024 | 6 | 2.0 | dpmpp_sde | karras | — |

- Commit after every task. Commit messages end with: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
- Branch: `qwen-image-2.1-upgrade` (pushes to `origin/azure-worker`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `azure_worker/profiles.toml` | **create** | The admin-visible render settings table. One `[profile]` block each. |
| `azure_worker/profiles.py` | **create** | `RenderSettings` dataclass; `load_profiles()` parses + validates the TOML. No imports from `config.py` (avoids a cycle). |
| `azure_worker/config.py` | modify | `Config.render: RenderSettings`; `load_config()` loads the active profile's settings. |
| `azure_worker/workflow.py` | modify | Builders read `cfg.render.*`; per-profile constants deleted; `render_report()` + `workflow_dimensions()` replace `summarize_workflow()`. |
| `azure_worker/messages.py` | modify | `ImageRequest` v2 (legacy fields tolerated, 32k/46KiB limits); `ImageResult` v2 (`negative_prompt`, `render`, `warnings`). |
| `azure_worker/main.py` | modify | Wire `render_report` into success and runtime-error results; startup log of active settings. |
| `azure_worker/tests/conftest.py` | **create** | `make_config(profile, ...)` shared by workflow and main-loop tests. |
| `azure_worker/tests/test_profiles.py` | **create** | TOML loading and validation. |
| `azure_worker/tests/test_workflow.py` | modify | Builders assert against `cfg.render`; `render_report` tests replace `summarize_workflow` tests. |
| `azure_worker/tests/test_main_loop.py` | modify | Add `_process_one` tests: success, runtime error, legacy-field warning. |
| `azure_worker/README.md`, `azure_worker/SPEC.md`, `azure_worker/sample_request.json` | modify | Point at v2 contract and `profiles.toml`. |
| `docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md` | modify | Fix the `shift` example (3.7169 at 2048², not 1.9937). |

**Task order matters.** Builders must switch to `cfg.render` (Task 3) *before* the request drops its fields (Task 4), or the suite breaks mid-plan.

---

### Task 1: `profiles.toml` and the loader

**Files:**
- Create: `azure_worker/profiles.toml`
- Create: `azure_worker/profiles.py`
- Test: `azure_worker/tests/test_profiles.py`

**Interfaces:**
- Produces: `profiles.RenderSettings` (frozen dataclass: `width: int, height: int, steps: int, sampler: str, cfg: float | None = None, scheduler: str | None = None, shift: float | None = None`)
- Produces: `profiles.load_profiles(path: Path = DEFAULT_PROFILES_PATH, required: Iterable[str] = ()) -> dict[str, RenderSettings]`
- Produces: `profiles.ProfileError(RuntimeError)`, `profiles.DEFAULT_PROFILES_PATH`
- Produces: `profiles.MIN_DIM = 64`, `MAX_DIM = 4096`, `DIM_MULTIPLE = 16` (moved here from `messages.py` in Task 4)

- [ ] **Step 1: Write the failing tests**

Create `azure_worker/tests/test_profiles.py`:

```python
"""profiles.toml loading and validation (no GPU, no network)."""
from __future__ import annotations

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
    with pytest.raises(ProfileError, match=needle):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_profiles.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'azure_worker.profiles'`

- [ ] **Step 3: Create `profiles.toml`**

Create `azure_worker/profiles.toml`. TOML needs one key per line (no `;` separators). Table names containing a dot must be quoted.

```toml
# Per-profile render settings. The worker owns these; job messages no longer
# carry steps/cfg/width/height. Every value here is reported back on the
# result message's `render` block, read out of the executed graph.
#
# Model *filenames* stay in .env — those are "what's on this machine".
# This file is "how this model wants to be run", and ships with the repo.
#
# Keys: width, height (pixels, multiples of 16, 64-4096), steps (1-200),
# cfg (0-30, omit when the profile has no guidance node), sampler,
# scheduler (omit when the sampler node supplies its own curve),
# shift (sigma shift for ModelSamplingAuraFlow; omit when derived or absent).

[flux1-dev]
width = 1024
height = 1024
steps = 20
cfg = 1.0                      # guidance-distilled, baked
sampler = "euler"
scheduler = "simple"

[flux2-klein]
width = 1024
height = 1024
steps = 20
# no cfg — BasicGuider has no guidance node at all
sampler = "euler"              # Flux2Scheduler supplies the sigma curve

[chroma1]
width = 1024
height = 1024
steps = 26
cfg = 3.5                      # de-distilled: real CFG, real negative
sampler = "euler"
scheduler = "beta"
shift = 1.0

[fluxed-up]
width = 1024
height = 1024
steps = 20
cfg = 1.0                      # same distilled driving as flux1-dev
sampler = "euler"
scheduler = "simple"

[qwen-image-2512]
width = 1328
height = 1328
steps = 20
cfg = 4.0
sampler = "euler"
scheduler = "simple"
shift = 3.1

["qwen-image-2.1"]
width = 2048
height = 2048
steps = 45
cfg = 1.0
sampler = "euler"
scheduler = "simple"
# shift intentionally absent — derived per-size by workflow.qwen21_shift()

[openflux1]
width = 1024
height = 1024
steps = 20
cfg = 3.5                      # de-distilled schnell
sampler = "euler"
scheduler = "simple"

[qwen-rapid-aio]
width = 1024
height = 1024
steps = 4
cfg = 1.0
sampler = "euler_ancestral"
scheduler = "beta"

[sdxl-dreamshaper]
width = 1024
height = 1024
steps = 6
cfg = 2.0
sampler = "dpmpp_sde"
scheduler = "karras"
```

- [ ] **Step 4: Create `profiles.py`**

Create `azure_worker/profiles.py`:

```python
"""Per-profile render settings, loaded from the tracked ``profiles.toml``.

The worker owns output size, step count, guidance scale and sampler for
whatever model it has loaded; the job message no longer carries them. They
live in a TOML file rather than in code so an administrator can read and
retune them without a deploy, and rather than in ``.env`` so they ship with
the repo instead of being retyped per machine.

This module deliberately does not import :mod:`azure_worker.config` —
``config`` imports this module to populate ``Config.render``.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "profiles.toml"

MIN_DIM = 64
MAX_DIM = 4096
# Flux 2's EmptyFlux2LatentImage requires width/height in 16-pixel increments
# (see nodes_flux.py: step=16 on the int inputs); every other latent node
# accepts /8 or /16, so 16 is the common denominator.
DIM_MULTIPLE = 16
MIN_STEPS, MAX_STEPS = 1, 200
MIN_CFG, MAX_CFG = 0.0, 30.0

_REQUIRED_KEYS = ("width", "height", "steps", "sampler")
_OPTIONAL_KEYS = ("cfg", "scheduler", "shift")


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderSettings:
    width: int
    height: int
    steps: int
    sampler: str
    cfg: Optional[float] = None
    scheduler: Optional[str] = None
    shift: Optional[float] = None


def load_profiles(
    path: Path = DEFAULT_PROFILES_PATH,
    required: Iterable[str] = (),
) -> dict[str, RenderSettings]:
    """Parse ``path`` into one :class:`RenderSettings` per TOML table.

    ``required`` names profiles that must be present (the caller passes
    ``KNOWN_PROFILES``); a missing one is a :class:`ProfileError` at startup
    rather than a ``KeyError`` on the first job.
    """
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError as e:
        raise ProfileError(f"profiles file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"{path} is not valid TOML: {e}") from e

    profiles: dict[str, RenderSettings] = {}
    for name, table in data.items():
        if not isinstance(table, dict):
            raise ProfileError(f"{path}: [{name}] must be a table")
        profiles[name] = _parse_table(name, table, path)

    missing = [p for p in required if p not in profiles]
    if missing:
        raise ProfileError(f"{path} is missing profiles: {', '.join(missing)}")
    return profiles


def _parse_table(name: str, table: dict, path: Path) -> RenderSettings:
    unknown = set(table) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)
    if unknown:
        raise ProfileError(f"{path}: [{name}] has unknown keys: {', '.join(sorted(unknown))}")
    for key in _REQUIRED_KEYS:
        if key not in table:
            raise ProfileError(f"{path}: [{name}] is missing required key '{key}'")

    width = _int(name, "width", table["width"], path)
    height = _int(name, "height", table["height"], path)
    for field_name, value in (("width", width), ("height", height)):
        if not (MIN_DIM <= value <= MAX_DIM):
            raise ProfileError(f"{path}: [{name}] {field_name}={value} must be in [{MIN_DIM}, {MAX_DIM}]")
        if value % DIM_MULTIPLE != 0:
            raise ProfileError(f"{path}: [{name}] {field_name}={value} must be a multiple of {DIM_MULTIPLE}")

    steps = _int(name, "steps", table["steps"], path)
    if not (MIN_STEPS <= steps <= MAX_STEPS):
        raise ProfileError(f"{path}: [{name}] steps={steps} must be in [{MIN_STEPS}, {MAX_STEPS}]")

    sampler = table["sampler"]
    if not isinstance(sampler, str) or not sampler:
        raise ProfileError(f"{path}: [{name}] sampler must be a non-empty string")

    cfg = None
    if "cfg" in table:
        cfg = _float(name, "cfg", table["cfg"], path)
        if not (MIN_CFG <= cfg <= MAX_CFG):
            raise ProfileError(f"{path}: [{name}] cfg={cfg} must be in [{MIN_CFG}, {MAX_CFG}]")

    scheduler = table.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, str) or not scheduler):
        raise ProfileError(f"{path}: [{name}] scheduler must be a non-empty string")

    shift = _float(name, "shift", table["shift"], path) if "shift" in table else None

    return RenderSettings(
        width=width, height=height, steps=steps, sampler=sampler,
        cfg=cfg, scheduler=scheduler, shift=shift,
    )


def _int(name: str, key: str, value: object, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{path}: [{name}] {key} must be an integer, got {value!r}")
    return value


def _float(name: str, key: str, value: object, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{path}: [{name}] {key} must be a number, got {value!r}")
    return float(value)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_profiles.py -q`
Expected: all PASS (10 tests). Then the full suite — expected **139 passed** (129 + 10).

- [ ] **Step 6: Commit**

```bash
git add azure_worker/profiles.toml azure_worker/profiles.py azure_worker/tests/test_profiles.py
git commit -m "azure_worker: add profiles.toml and loader for worker-owned render settings

Per-profile width/height/steps/cfg/sampler/scheduler/shift in a tracked TOML
file so an admin can see and retune them. Nothing consumes it yet.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `Config.render` and the shared test config helper

**Files:**
- Modify: `azure_worker/config.py` (dataclass fields ~line 100, `load_config()` ~line 196)
- Create: `azure_worker/tests/conftest.py`
- Modify: `azure_worker/tests/test_workflow.py:60-93` (replace `_cfg`)

**Interfaces:**
- Consumes: `profiles.load_profiles`, `profiles.RenderSettings`, `profiles.ProfileError` (Task 1)
- Produces: `Config.render: RenderSettings` — the active profile's settings
- Produces: `tests.conftest.make_config(profile: str, sdxl_loras: tuple = (), sdxl_lora_autoroute: bool = False, render: RenderSettings | None = None) -> Config`

- [ ] **Step 1: Write the failing test**

Append to `azure_worker/tests/test_profiles.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_profiles.py::test_load_config_attaches_active_profile_render_settings -q`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'render'`

- [ ] **Step 3: Add `render` to `Config` and load it**

In `azure_worker/config.py`, add the import after `from dataclasses import dataclass`:

```python
from .profiles import ProfileError, RenderSettings, load_profiles
```

In the `Config` dataclass, directly after the `sdxl_lora_autoroute: bool` line (it must come before the first defaulted field, `llm_request_timeout_seconds`), add:

```python
    # The active profile's render settings (size, steps, cfg, sampler) from
    # profiles.toml. The job message no longer carries these; the worker owns
    # them and reports what it used on every result.
    render: RenderSettings
```

In `load_config()`, after the `KNOWN_PROFILES` check and before `return Config(`:

```python
    try:
        profiles = load_profiles(required=KNOWN_PROFILES)
    except ProfileError as e:
        raise ConfigError(str(e)) from e
```

And inside the `Config(...)` call, after `sdxl_lora_autoroute=...,`:

```python
        render=profiles[profile],
```

- [ ] **Step 4: Create the shared `make_config` helper**

Create `azure_worker/tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Replace `_cfg` in `test_workflow.py` with the shared helper**

Delete the whole `def _cfg(...)` function (`azure_worker/tests/test_workflow.py:60-93`) and add this import below the existing `from azure_worker.workflow import (...)` block:

```python
from azure_worker.tests.conftest import make_config as _cfg
```

Every existing `_cfg(PROFILE_X, ...)` call keeps working unchanged.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: **140 passed**.

- [ ] **Step 7: Commit**

```bash
git add azure_worker/config.py azure_worker/tests/conftest.py azure_worker/tests/test_workflow.py azure_worker/tests/test_profiles.py
git commit -m "azure_worker: attach the active profile's render settings to Config

Config.render is loaded from profiles.toml at startup; a missing or malformed
file is a ConfigError. Tests share one make_config() helper.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Builders read `cfg.render`; per-profile constants deleted

**Files:**
- Modify: `azure_worker/workflow.py` — constants at lines 86-166, all nine builders (282-994)
- Modify: `azure_worker/tests/test_workflow.py` — builder shape tests

**Interfaces:**
- Consumes: `Config.render` (Task 2)
- Produces: every builder now ignores `req.width/height/steps/cfg` entirely. `req` still has those attributes until Task 4 removes them.
- Deletes: `CHROMA_SAMPLER`, `CHROMA_SCHEDULER`, `CHROMA_SHIFT`, `QWEN_IMAGE_SAMPLER`, `QWEN_IMAGE_SCHEDULER`, `QWEN_IMAGE_SHIFT`, `QWEN21_SAMPLER`, `QWEN21_SCHEDULER`, `QWEN21_CFG`, `QWEN_RAPID_SAMPLER`, `QWEN_RAPID_SCHEDULER`, `SDXL_SAMPLER`, `SDXL_SCHEDULER`, `SDXL_CFG`. **Keeps** `QWEN21_RESOLUTION`, `qwen21_shift()`, `QWEN21_SHIFT_*`, `SDXL_LORA_NODE_BASE`, all `*_SAVE_NODE_ID`.

- [ ] **Step 1: Write the failing test**

Add to `azure_worker/tests/test_workflow.py` (anywhere after the imports):

```python
# -- Builders take size/steps/cfg/sampler from Config.render, not the request --

@pytest.mark.parametrize(
    "profile",
    [
        PROFILE_FLUX1_DEV, PROFILE_FLUX2_KLEIN, PROFILE_CHROMA1, PROFILE_FLUXED_UP,
        PROFILE_QWEN_IMAGE_2512, PROFILE_QWEN_IMAGE_2_1, PROFILE_OPENFLUX1,
        PROFILE_QWEN_RAPID_AIO, PROFILE_SDXL_DREAMSHAPER,
    ],
)
def test_builders_ignore_request_size_and_steps(profile):
    """Whatever a legacy client sends, the graph is sized and stepped from profiles.toml."""
    cfg = _cfg(profile)
    req = ImageRequest.from_json(_sample_payload(width=512, height=512, steps=99, cfg=9.0))
    wf = build_workflow(req, cfg)

    latents = [n for n in wf.values() if n["class_type"].startswith("Empty") and n["class_type"].endswith("LatentImage")]
    assert len(latents) == 1, f"{profile}: expected one latent node"
    assert latents[0]["inputs"]["width"] == cfg.render.width
    assert latents[0]["inputs"]["height"] == cfg.render.height

    steps = [
        n["inputs"]["steps"] for n in wf.values()
        if n["class_type"] in ("KSampler", "BasicScheduler", "Flux2Scheduler")
    ]
    assert steps == [cfg.render.steps], f"{profile}: steps {steps!r}"
    assert effective_cfg(wf) == cfg.render.cfg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_workflow.py::test_builders_ignore_request_size_and_steps -q`
Expected: 9 FAIL — `assert 512 == 1024` (latent width taken from the request).

- [ ] **Step 3: Delete the constants**

In `azure_worker/workflow.py` delete these blocks entirely (comments included):
- `CHROMA_SAMPLER` / `CHROMA_SCHEDULER` / `CHROMA_SHIFT` and their comment (lines 86-92)
- `QWEN_IMAGE_SAMPLER` / `QWEN_IMAGE_SCHEDULER` / `QWEN_IMAGE_SHIFT` and comment (95-98)
- `QWEN_RAPID_SAMPLER` / `QWEN_RAPID_SCHEDULER` and comment (101-105)
- `QWEN21_SAMPLER` / `QWEN21_SCHEDULER` / `QWEN21_CFG` and their comments (108-116) — **keep** the `QWEN21_RESOLUTION` block (117-121) and everything from `# The 2.1 scheduler config ships` through `def qwen21_shift` (122-151)
- `SDXL_SAMPLER` / `SDXL_SCHEDULER` / `SDXL_CFG` and comment (153-161) — **keep** `SDXL_LORA_NODE_BASE`

Replace the deleted Qwen-2.1 comment with one line above `QWEN21_RESOLUTION`:

```python
# Qwen-Image 2.1 sampling defaults (steps/cfg/sampler) live in profiles.toml;
# the sigma shift is the one value derived here, per output size.
```

- [ ] **Step 4: Rewrite each builder's consumption sites**

Apply these exact substitutions. Line numbers are pre-edit; search by content.

**`build_flux1_dev_workflow`** (node "6" latent, node "7" KSampler):
```python
            "inputs": {
                "width": cfg.render.width,
                "height": cfg.render.height,
                "batch_size": 1,
            },
```
```python
                "seed": req.seed,
                "steps": cfg.render.steps,
                "cfg": cfg.render.cfg,
                "sampler_name": cfg.render.sampler,
                "scheduler": cfg.render.scheduler,
```

**`build_flux2_klein_workflow`**: the `EmptyFlux2LatentImage` node and the `Flux2Scheduler` node both carry width/height; all three become `cfg.render.*`. `"steps": req.steps` → `"steps": cfg.render.steps`. `KSamplerSelect` `"sampler_name": "euler"` → `"sampler_name": cfg.render.sampler`.

**`build_chroma1_workflow`**:
- `ModelSamplingAuraFlow` `"shift": CHROMA_SHIFT` → `"shift": cfg.render.shift`
- `CFGGuider` `"cfg": req.cfg` → `"cfg": cfg.render.cfg`
- `KSamplerSelect` `"sampler_name": CHROMA_SAMPLER` → `cfg.render.sampler`
- `BasicScheduler` `"scheduler": CHROMA_SCHEDULER` → `cfg.render.scheduler`, `"steps": req.steps` → `cfg.render.steps`
- `EmptySD3LatentImage` width/height → `cfg.render.width/height`

**`build_fluxed_up_workflow`**: identical pattern to flux1-dev (latent + KSampler: width, height, steps, cfg, sampler_name, scheduler all from `cfg.render`).

**`build_qwen_image_2512_workflow`**:
- `ModelSamplingAuraFlow` `"shift": QWEN_IMAGE_SHIFT` → `cfg.render.shift`
- latent width/height → `cfg.render.*`
- KSampler: `steps`, `cfg`, `sampler_name`, `scheduler` → `cfg.render.steps/cfg/sampler/scheduler`

**`build_qwen_image_2_1_workflow`**:
- `"shift": qwen21_shift(req.width, req.height)` → `"shift": qwen21_shift(cfg.render.width, cfg.render.height)`
- latent width/height → `cfg.render.*`
- KSampler: `"steps": req.steps` → `cfg.render.steps`; `"cfg": QWEN21_CFG` → `cfg.render.cfg`; `"sampler_name": QWEN21_SAMPLER` → `cfg.render.sampler`; `"scheduler": QWEN21_SCHEDULER` → `cfg.render.scheduler`
- Update the docstring sentence `the shift is computed per request from ``req.width``/``req.height``` → `the shift is computed from the profile's ``cfg.render.width``/``height``` and delete the final paragraph starting `The official recipe drives ``cfg=1``` (it describes request fields that no longer exist).

**`build_openflux1_workflow`**: latent + KSampler, same as flux1-dev but `"cfg": req.cfg` → `"cfg": cfg.render.cfg`.

**`build_qwen_rapid_aio_workflow`**: latent width/height → `cfg.render.*`; KSampler `steps` → `cfg.render.steps`, `"cfg": 1` → `cfg.render.cfg`, `QWEN_RAPID_SAMPLER` → `cfg.render.sampler`, `QWEN_RAPID_SCHEDULER` → `cfg.render.scheduler`.

**`build_sdxl_dreamshaper_workflow`**: `EmptyLatentImage` width/height → `cfg.render.*`; KSampler `"steps": req.steps` → `cfg.render.steps`, `"cfg": SDXL_CFG` → `cfg.render.cfg`, `SDXL_SAMPLER` → `cfg.render.sampler`, `SDXL_SCHEDULER` → `cfg.render.scheduler`. In its docstring delete `see ``SDXL_CFG`` for why ``req.cfg`` is not used.`

**Module docstring** (lines 1-52): replace every mention of `req.cfg`, `req.steps` with a single sentence added at the end of the profile list:

```
Output size, step count, guidance scale and sampler for every profile come
from ``profiles.toml`` (``Config.render``); the request supplies only the
prompts and seed.
```
and delete the per-profile sentences that start with "Like chroma1, honors", "``req.cfg`` and ``req.negative_prompt`` are no-ops", "so ``req.cfg`` is a no-op; ``req.steps`` is honored", and "drives KSampler with real ``req.cfg``". Keep the negative-prompt facts (which profiles use `ConditioningZeroOut` vs a real encode) — those still describe the graph.

- [ ] **Step 5: Update the existing builder tests**

In `azure_worker/tests/test_workflow.py`:

1. Remove `SDXL_CFG`, `SDXL_SAMPLER`, `SDXL_SCHEDULER` from the `from azure_worker.workflow import (...)` block.
2. In `test_chroma1_workflow_shape` (line ~186): delete the `steps=26,` and `cfg=3.5,` lines from the `_sample_payload(...)` call. The asserts `cfg == 3.5` and `steps == 26` stay (same values from the TOML).
3. In `test_qwen_image_2512_workflow_shape` (~294): delete `steps=30,` and `cfg=4.0,` from the payload; change `assert ks["inputs"]["steps"] == 30` → `== 20`.
4. In `test_qwen_image_2_1_workflow_shape` (~349): delete `steps=40,` and `cfg=7.0,`; change `assert ks["inputs"]["steps"] == 40` → `== 45`; the `cfg == 1` assert stays. Change the comment `# Official path runs at cfg=1, so a request cfg of 7.0 is ignored` → `# cfg comes from profiles.toml (1.0 on the official path)`. The latent-size assert in this test, if it compares to 1024, becomes `== 2048`.
5. Replace `test_qwen_image_2_1_shift_grows_with_resolution` (~409) with:

```python
def test_qwen_image_2_1_shift_grows_with_resolution():
    from dataclasses import replace
    from azure_worker.workflow import qwen21_shift

    base = _cfg(PROFILE_QWEN_IMAGE_2_1)
    small = replace(base, render=replace(base.render, width=1024, height=1024))
    large = replace(base, render=replace(base.render, width=2048, height=2048))
    req = ImageRequest.from_json(_sample_payload())

    shift_small = build_workflow(req, small)["4"]["inputs"]["shift"]
    shift_large = build_workflow(req, large)["4"]["inputs"]["shift"]
    assert shift_small == pytest.approx(qwen21_shift(1024, 1024))
    assert shift_large == pytest.approx(qwen21_shift(2048, 2048))
    assert shift_large > shift_small
    assert shift_large == pytest.approx(3.7169, abs=1e-3)
```

6. `test_qwen_image_2_1_cfg_is_reported_as_baked` (~425) and `test_qwen_image_2_1_shift_is_reported_per_resolution` (~433) reference `summarize_workflow`; leave them for Task 5, but delete `cfg=7.0` from their payloads now if present. If `test_qwen_image_2_1_shift_is_reported_per_resolution` builds requests at two sizes via the payload, rewrite it with the `replace(...)` pattern above.
7. `test_openflux1_workflow_shape` (~449): delete `steps=20,`/`cfg=3.5,` from the payload; asserts unchanged.
8. `test_qwen_rapid_aio_workflow_shape` (~503): delete `steps=4,` and `cfg=8.0,  # ignored`; asserts `steps == 4`, `cfg == 1` unchanged.
9. `test_sdxl_dreamshaper_workflow_shape_without_loras` (~586): payload `_sample_payload(steps=6, negative_prompt="blurry")` → `_sample_payload(negative_prompt="blurry")`; `assert sampler["cfg"] == SDXL_CFG == 2.0` → `assert sampler["cfg"] == 2.0`; `steps == 6` stays. Any `sampler["sampler_name"] == SDXL_SAMPLER` → `== "dpmpp_sde"`, `SDXL_SCHEDULER` → `"karras"`.
10. `test_effective_cfg_reports_the_baked_value` (~737): the request `cfg=7.0` is now simply ignored; keep the test, update its comment to `# Request cfg is ignored entirely; the value comes from profiles.toml.` and remove `assert req.cfg == 7.0`.
11. `test_effective_cfg_reports_honored_value_for_real_cfg_profiles` (~752): change `_sample_payload(cfg=4.5)` → `_sample_payload()` and `== 4.5` → `== 3.5`.
12. `test_summarize_workflow_echoes_models_loras_and_sampler` and `test_summarize_workflow_covers_every_profile`: they pass `steps=6` and assert `"steps=6"`; `sdxl-dreamshaper` is 6 so the first still passes. The parametrized one will fail for other profiles — change its assert to `assert f"steps={_cfg(profile).render.steps}" in line`. (Both are rewritten in Task 5 anyway.)

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: **149 passed** (140 + 9 parametrized).

- [ ] **Step 7: Commit**

```bash
git add azure_worker/workflow.py azure_worker/tests/test_workflow.py
git commit -m "azure_worker: builders take size/steps/cfg/sampler from profiles.toml

Every workflow builder now reads Config.render instead of the request. The
per-profile sampler constants in workflow.py are gone; profiles.toml is the
single visible source of truth.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `ImageRequest` v2 — legacy fields tolerated, new limits

**Files:**
- Modify: `azure_worker/messages.py:1-105`
- Modify: `azure_worker/tests/test_workflow.py` — `_sample_payload`, request tests

**Interfaces:**
- Produces: `ImageRequest(job_id, name, prompt, negative_prompt="", seed=0, ignored_fields: tuple[str, ...] = ())`
- Produces: `messages.MAX_PROMPT_CHARS = 32_000` (combined), `messages.MAX_REQUEST_BYTES = 46 * 1024`, `messages.LEGACY_FIELDS = ("width", "height", "steps", "cfg")`
- Removes: `ImageRequest.width/height/steps/cfg`, `messages.MIN_DIM/MAX_DIM/DIM_MULTIPLE`, `messages._validate_dim`

- [ ] **Step 1: Write the failing tests**

In `azure_worker/tests/test_workflow.py`, replace the three existing request tests (`test_request_round_trip_defaults`, `test_request_rejects_non_multiple_of_16`, `test_request_rejects_missing_prompt`) with:

```python
# -- Message validation (contract v2) --

def test_request_round_trip_defaults():
    req = ImageRequest.from_json(json.dumps({"job_id": "abc", "name": "test-image", "prompt": "a cat", "seed": 7}))
    assert req.job_id == "abc"
    assert req.name == "test-image"
    assert req.negative_prompt == ""
    assert req.seed == 7
    assert req.ignored_fields == ()
    assert not hasattr(req, "width")
    assert not hasattr(req, "steps")


def test_request_rejects_missing_prompt():
    with pytest.raises(MessageValidationError, match="prompt"):
        ImageRequest.from_json(json.dumps({"name": "x"}))


def test_request_records_legacy_fields_in_order():
    req = ImageRequest.from_json(_sample_payload(width=1024, height=1024, steps=20, cfg=7.0))
    assert req.ignored_fields == ("width", "height", "steps", "cfg")


def test_request_records_only_the_legacy_fields_present():
    req = ImageRequest.from_json(_sample_payload(steps=20))
    assert req.ignored_fields == ("steps",)


def test_request_legacy_fields_are_not_validated():
    # A v1 worker rejected width=1032; v2 doesn't even look at it.
    req = ImageRequest.from_json(_sample_payload(width=1032, steps=0, cfg="nonsense"))
    assert req.ignored_fields == ("width", "steps", "cfg")


def test_request_prompt_budget_is_combined():
    ok = ImageRequest.from_json(_sample_payload(prompt="a" * 20_000, negative_prompt="b" * 12_000))
    assert len(ok.prompt) + len(ok.negative_prompt) == 32_000
    with pytest.raises(MessageValidationError, match="32000"):
        ImageRequest.from_json(_sample_payload(prompt="a" * 20_000, negative_prompt="b" * 12_001))


def test_request_rejects_oversized_message_bytes():
    # 31k em-dashes is under the character budget but 93 KB of UTF-8.
    raw = _sample_payload(prompt="\u2014" * 31_000)
    with pytest.raises(MessageValidationError, match="bytes"):
        ImageRequest.from_json(raw)
```

And change `_sample_payload` so it no longer sends the removed fields by default:

```python
def _sample_payload(**overrides):
    payload = {
        "job_id": "abc",
        "name": "test-image",
        "prompt": "a cat",
        "seed": 7,
    }
    payload.update(overrides)
    return json.dumps(payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_workflow.py -q -k "request"`
Expected: FAIL — `'width' and 'height' must be integers` on the defaults test; `ignored_fields` AttributeError on the others.

- [ ] **Step 3: Rewrite `ImageRequest`**

In `azure_worker/messages.py`, replace everything from `MAX_PROMPT_CHARS = 4096` through the end of `ImageRequest.from_json` with:

```python
# prompt + negative_prompt may total this many characters. Not a model limit —
# no active profile's text encoder truncates — but a transport one: the Storage
# Queue message is 64 KiB after base64 (~48 KiB raw), shared by every field.
MAX_PROMPT_CHARS = 32_000
# Byte backstop on the raw request. The queue would already have rejected
# anything over 48 KiB; this lower bar leaves ~2 KiB for the result message's
# added fields (render block, SAS URL) so the *reply* fits too.
MAX_REQUEST_BYTES = 46 * 1024
# Contract v1 request fields the worker now owns. Accepted, ignored, reported.
LEGACY_FIELDS = ("width", "height", "steps", "cfg")


class MessageValidationError(ValueError):
    pass


@dataclass
class ImageRequest:
    job_id: str
    name: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 0
    # Which of LEGACY_FIELDS the client sent, in LEGACY_FIELDS order. Surfaced
    # on the result as a warning so stale clients can be found in the field.
    ignored_fields: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, raw: str) -> "ImageRequest":
        size = len(raw.encode("utf-8"))
        if size > MAX_REQUEST_BYTES:
            raise MessageValidationError(
                f"message is {size} bytes; the limit is {MAX_REQUEST_BYTES} bytes"
            )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise MessageValidationError(f"message is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise MessageValidationError("message must be a JSON object")

        job_id = str(data.get("job_id") or uuid.uuid4())
        name = data.get("name")
        prompt = data.get("prompt")

        if not isinstance(name, str) or not name.strip():
            raise MessageValidationError("'name' is required and must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise MessageValidationError("'prompt' is required and must be a non-empty string")

        negative = data.get("negative_prompt", "") or ""
        if not isinstance(negative, str):
            raise MessageValidationError("'negative_prompt' must be a string")
        total = len(prompt) + len(negative)
        if total > MAX_PROMPT_CHARS:
            raise MessageValidationError(
                f"'prompt' + 'negative_prompt' total {total} characters; the limit is {MAX_PROMPT_CHARS}"
            )

        seed_raw = data.get("seed")
        if seed_raw is None:
            seed = random.randint(0, 2**63 - 1)
        elif isinstance(seed_raw, int):
            seed = seed_raw
        else:
            raise MessageValidationError("'seed' must be an integer if provided")

        ignored = tuple(f for f in LEGACY_FIELDS if f in data)

        return cls(
            job_id=job_id,
            name=name.strip(),
            prompt=prompt,
            negative_prompt=negative,
            seed=seed,
            ignored_fields=ignored,
        )
```

Delete `MIN_DIM`, `MAX_DIM`, `DIM_MULTIPLE` (and their comment) from the top of the file, and delete the `_validate_dim` function at the bottom. Keep `_SAFE_NAME_RE` and `sanitize_name`.

`ImageResult` still references `req.width`/`req.height` — it is rewritten in Task 6. For now, make it compile by replacing in `ImageResult.success` and `ImageResult.error_for` every `width=req.width,` with `width=0,` and `height=req.height,` with `height=0,`. (Temporary; Task 6 replaces both methods.)

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: **153 passed** (149 − 3 removed + 7 new). If `test_result_success_serializes` fails on `width`, that's the temporary `0` — it only asserts `status`/`blob_name`/`error`, so it should pass.

- [ ] **Step 5: Commit**

```bash
git add azure_worker/messages.py azure_worker/tests/test_workflow.py
git commit -m "azure_worker: image request contract v2

width/height/steps/cfg leave the request; if sent they are recorded in
ignored_fields and otherwise untouched. prompt + negative_prompt share a
32,000-character budget, with a 46 KiB byte backstop under the queue cap.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `render_report()` and `workflow_dimensions()` replace `summarize_workflow()`

**Files:**
- Modify: `azure_worker/workflow.py:166-258` (the `effective_cfg` / `summarize_workflow` region)
- Modify: `azure_worker/tests/test_workflow.py` — summarize tests
- Modify: `azure_worker/main.py:27` (import) and `:107-113` (log line) — minimal, to keep it importing

**Interfaces:**
- Produces: `workflow.render_report(workflow: dict, profile: str) -> dict` — exactly the spec §4 block with keys `profile, model, loras, steps, cfg, sampler, scheduler, shift, negative_honored, summary`
- Produces: `workflow.workflow_dimensions(workflow: dict) -> tuple[int, int]`
- Keeps: `workflow.effective_cfg(workflow) -> float | None`
- Removes: `workflow.summarize_workflow`

- [ ] **Step 1: Write the failing tests**

In `azure_worker/tests/test_workflow.py`, replace the import of `summarize_workflow` with `render_report, workflow_dimensions` in the `from azure_worker.workflow import (...)` block. Replace `test_summarize_workflow_echoes_models_loras_and_sampler`, `test_summarize_workflow_says_none_when_no_loras`, `test_summarize_workflow_covers_every_profile`, `test_qwen_image_2_1_cfg_is_reported_as_baked`, and `test_qwen_image_2_1_shift_is_reported_per_resolution` with:

```python
# -- render_report: the result's `render` block, read back out of the graph --

_RENDER_KEYS = {
    "profile", "model", "loras", "steps", "cfg", "sampler", "scheduler", "shift",
    "negative_honored", "summary",
}

_ALL_PROFILES = [
    PROFILE_FLUX1_DEV, PROFILE_FLUX2_KLEIN, PROFILE_CHROMA1, PROFILE_FLUXED_UP,
    PROFILE_QWEN_IMAGE_2512, PROFILE_QWEN_IMAGE_2_1, PROFILE_OPENFLUX1,
    PROFILE_QWEN_RAPID_AIO, PROFILE_SDXL_DREAMSHAPER,
]


@pytest.mark.parametrize("profile", _ALL_PROFILES)
def test_render_report_has_exactly_the_contract_keys(profile):
    cfg = _cfg(profile)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), profile)
    assert set(report) == _RENDER_KEYS
    assert report["profile"] == profile
    assert report["model"].endswith(".safetensors")
    assert report["steps"] == cfg.render.steps
    assert report["cfg"] == cfg.render.cfg
    assert report["sampler"] == cfg.render.sampler
    assert report["scheduler"] == cfg.render.scheduler
    assert isinstance(report["loras"], list)
    assert isinstance(report["negative_honored"], bool)
    assert isinstance(report["summary"], str) and report["summary"]
    json.dumps(report)  # must be plain JSON types


def test_render_report_model_is_the_loaded_file():
    cfg = _cfg(PROFILE_QWEN_IMAGE_2_1)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["model"] == "qwen_image_2.1_int8_convrot.safetensors"

    cfg = _cfg(PROFILE_SDXL_DREAMSHAPER)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["model"] == "DreamShaperXL_Turbo_v2_1.safetensors"


def test_render_report_lists_loras_in_chain_order():
    cfg = _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=(_ELF, _EARS))
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["loras"] == [
        {"name": _ELF.name, "model_strength": 0.8, "clip_strength": 0.8},
        {"name": _EARS.name, "model_strength": 0.6, "clip_strength": 0.4},
    ]
    assert "+ 2 loras" in report["summary"]


def test_render_report_includes_autorouted_race_lora():
    cfg = _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=(_ELF,), sdxl_lora_autoroute=True)
    req = ImageRequest.from_json(_sample_payload(prompt="a stout dwarf blacksmith"))
    report = render_report(build_workflow(req, cfg), cfg.profile)
    assert len(report["loras"]) == 2
    assert "dwarf" in report["loras"][1]["name"].lower()


def test_render_report_empty_loras_is_an_empty_list():
    cfg = _cfg(PROFILE_FLUX1_DEV)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["loras"] == []
    assert "lora" not in report["summary"]


@pytest.mark.parametrize(
    "profile,expected",
    [
        (PROFILE_FLUX1_DEV, False),        # ConditioningZeroOut, cfg 1
        (PROFILE_FLUX2_KLEIN, False),      # BasicGuider, no cfg at all
        (PROFILE_CHROMA1, True),
        (PROFILE_FLUXED_UP, False),
        (PROFILE_QWEN_IMAGE_2512, True),
        (PROFILE_QWEN_IMAGE_2_1, False),   # real negative node, but cfg 1 makes it inert
        (PROFILE_OPENFLUX1, True),
        (PROFILE_QWEN_RAPID_AIO, False),
        (PROFILE_SDXL_DREAMSHAPER, True),
    ],
)
def test_render_report_negative_honored_matches_contract_table(profile, expected):
    cfg = _cfg(profile)
    req = ImageRequest.from_json(_sample_payload(negative_prompt="blurry"))
    assert render_report(build_workflow(req, cfg), profile)["negative_honored"] is expected


def test_render_report_flux2_klein_has_null_cfg_and_scheduler():
    cfg = _cfg(PROFILE_FLUX2_KLEIN)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["cfg"] is None
    assert report["scheduler"] is None
    assert report["summary"].endswith("· cfg — · euler")


def test_render_report_shift_is_derived_for_qwen21_and_null_elsewhere():
    from azure_worker.workflow import qwen21_shift

    cfg = _cfg(PROFILE_QWEN_IMAGE_2_1)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["shift"] == pytest.approx(qwen21_shift(2048, 2048))

    cfg = _cfg(PROFILE_CHROMA1)
    assert render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)["shift"] == 1.0

    cfg = _cfg(PROFILE_SDXL_DREAMSHAPER)
    assert render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)["shift"] is None


def test_render_report_summary_format():
    cfg = _cfg(PROFILE_QWEN_IMAGE_2_1)
    report = render_report(build_workflow(ImageRequest.from_json(_sample_payload()), cfg), cfg.profile)
    assert report["summary"] == (
        "qwen-image-2.1 · qwen_image_2.1_int8_convrot · 2048×2048 · 45 steps · cfg 1.0 · euler/simple"
    )


@pytest.mark.parametrize("profile", _ALL_PROFILES)
def test_workflow_dimensions_reads_the_latent_node(profile):
    cfg = _cfg(profile)
    wf = build_workflow(ImageRequest.from_json(_sample_payload()), cfg)
    assert workflow_dimensions(wf) == (cfg.render.width, cfg.render.height)
```

`_ELF` and `_EARS` already exist in the file (used by the old summarize test). If they are defined *below* the new tests, move the new tests below them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_workflow.py -q -k "render_report or workflow_dimensions"`
Expected: FAIL — `ImportError: cannot import name 'render_report'`

- [ ] **Step 3: Implement `render_report` and `workflow_dimensions`**

In `azure_worker/workflow.py`, replace the whole `summarize_workflow` function (keep `_KSAMPLER_CLASSES`, `_STEPS_CLASSES`, and `effective_cfg` above it) with:

```python
_LATENT_CLASSES = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage")


def workflow_dimensions(workflow: dict) -> tuple[int, int]:
    """The output size a built graph will render, from its empty-latent node."""
    for node in workflow.values():
        if node.get("class_type") in _LATENT_CLASSES:
            i = node["inputs"]
            return int(i["width"]), int(i["height"])
    raise ValueError("workflow has no empty-latent node")


def render_report(workflow: dict, profile: str) -> dict:
    """The result message's ``render`` block for a built graph.

    Read back out of the graph rather than off ``Config`` so the reported
    values cannot drift from what actually executes, and so every profile
    reports through the same code path regardless of how its sampler is wired.
    The dict is plain JSON types; ``summary`` is the one-line display form.
    """
    model = next(
        (
            name
            for node in workflow.values()
            for key in ("ckpt_name", "unet_name")
            if (name := node.get("inputs", {}).get(key))
        ),
        None,
    )

    lora_ids = sorted(
        (nid for nid, node in workflow.items() if node.get("class_type") == "LoraLoader"),
        key=lambda nid: int(nid) if nid.isdigit() else nid,
    )
    loras = [
        {
            "name": workflow[nid]["inputs"]["lora_name"],
            "model_strength": workflow[nid]["inputs"]["strength_model"],
            "clip_strength": workflow[nid]["inputs"]["strength_clip"],
        }
        for nid in lora_ids
    ]

    shift = next(
        (
            float(node["inputs"]["shift"])
            for node in workflow.values()
            if node.get("class_type") == "ModelSamplingAuraFlow"
        ),
        None,
    )

    steps = sampler = scheduler = None
    for node in workflow.values():
        if node.get("class_type") in _KSAMPLER_CLASSES:
            i = node["inputs"]
            steps, sampler, scheduler = i.get("steps"), i.get("sampler_name"), i.get("scheduler")
            break
    else:
        # SamplerCustomAdvanced profiles: steps, sampler and scheduler live on
        # separate nodes. Flux2Scheduler has no scheduler name at all.
        for node in workflow.values():
            ct, i = node.get("class_type"), node.get("inputs", {})
            if ct in _STEPS_CLASSES:
                steps = i.get("steps") if steps is None else steps
                scheduler = i.get("scheduler") if scheduler is None else scheduler
            if ct == "KSamplerSelect" and sampler is None:
                sampler = i.get("sampler_name")

    cfg = effective_cfg(workflow)
    width, height = workflow_dimensions(workflow)

    # A negative prompt only reaches the model if there is a real guidance
    # scale (cfg != 1 makes uncond matter) and nothing zeroed it out.
    zeroed = any(node.get("class_type") == "ConditioningZeroOut" for node in workflow.values())
    negative_honored = cfg is not None and cfg != 1.0 and not zeroed

    return {
        "profile": profile,
        "model": model,
        "loras": loras,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": scheduler,
        "shift": shift,
        "negative_honored": negative_honored,
        "summary": _summary(profile, model, loras, width, height, steps, cfg, sampler, scheduler),
    }


def _summary(profile, model, loras, width, height, steps, cfg, sampler, scheduler) -> str:
    model_part = model.rsplit(".", 1)[0] if model else "?"
    if loras:
        model_part += f" + {len(loras)} lora" + ("s" if len(loras) != 1 else "")
    cfg_part = "cfg —" if cfg is None else f"cfg {cfg}"
    sampler_part = sampler if scheduler is None else f"{sampler}/{scheduler}"
    return f"{profile} · {model_part} · {width}×{height} · {steps} steps · {cfg_part} · {sampler_part}"
```

- [ ] **Step 4: Keep `main.py` importing**

In `azure_worker/main.py` line 27 change:
```python
from .workflow import build_workflow, effective_cfg, summarize_workflow
```
to:
```python
from .workflow import build_workflow, render_report, workflow_dimensions
```
and replace lines 107-113 (from `settings = summarize_workflow(workflow)` through `log.info("job %s settings: %s", ...)`) with:
```python
        render = render_report(workflow, clients.config.profile)
        log.info("job %s render: %s", req.job_id, render["summary"])
```
(Task 7 finishes the wiring; this just keeps the module importable.)

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: **177 passed** (153 − 5 removed + 29 new: 9 + 1 + 1 + 1 + 1 + 9 + 1 + 1 + 1 + 9... count what pytest reports; the point is all green and no `summarize_workflow` references remain: `grep -rn summarize_workflow azure_worker/` returns nothing).

- [ ] **Step 6: Commit**

```bash
git add azure_worker/workflow.py azure_worker/main.py azure_worker/tests/test_workflow.py
git commit -m "azure_worker: render_report() builds the result's render block from the graph

Replaces summarize_workflow(): same read-back-from-the-graph approach, but
returns the structured dict the v2 contract puts on every result, with the
display summary derived from it. workflow_dimensions() reads the latent size.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `ImageResult` v2

**Files:**
- Modify: `azure_worker/messages.py` — the `ImageResult` class
- Modify: `azure_worker/tests/test_workflow.py` — result tests

**Interfaces:**
- Produces: `ImageResult.success(req, blob_name, blob_url, *, width: int, height: int, render: dict) -> ImageResult`
- Produces: `ImageResult.error_for(req, message, *, raw_job_id: str = "", width: int = 0, height: int = 0, render: dict | None = None) -> ImageResult`
- Produces: `messages.warnings_for(req: ImageRequest) -> list[str]`
- Result JSON keys, in order: `job_id, name, status, prompt, negative_prompt, width, height, seed, render, warnings, blob_url, blob_name, error`

- [ ] **Step 1: Write the failing tests**

Replace `test_result_success_serializes` and `test_result_error_when_request_was_invalid` in `azure_worker/tests/test_workflow.py` with:

```python
# -- Result message (contract v2) --

_RENDER = {
    "profile": "qwen-image-2.1", "model": "m.safetensors", "loras": [], "steps": 45,
    "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "shift": 3.7169,
    "negative_honored": False, "summary": "s",
}

_RESULT_KEYS = [
    "job_id", "name", "status", "prompt", "negative_prompt", "width", "height", "seed",
    "render", "warnings", "blob_url", "blob_name", "error",
]


def test_result_success_serializes_contract_v2():
    req = ImageRequest.from_json(_sample_payload(negative_prompt="blurry"))
    result = ImageResult.success(req, "x/y.png", "https://example/y.png?sas", width=2048, height=2048, render=_RENDER)
    parsed = json.loads(result.to_json())
    assert list(parsed) == _RESULT_KEYS
    assert parsed["status"] == "success"
    assert parsed["negative_prompt"] == "blurry"
    assert (parsed["width"], parsed["height"]) == (2048, 2048)
    assert parsed["render"] == _RENDER
    assert parsed["warnings"] == []
    assert parsed["blob_name"] == "x/y.png"
    assert parsed["error"] is None


def test_result_warns_about_legacy_fields():
    req = ImageRequest.from_json(_sample_payload(width=1024, height=1024, steps=20, cfg=7.0))
    result = ImageResult.success(req, "x/y.png", "u", width=2048, height=2048, render=_RENDER)
    assert json.loads(result.to_json())["warnings"] == [
        "ignored client-supplied fields: width, height, steps, cfg"
    ]


def test_result_validation_error_has_null_render_and_zero_size():
    parsed = json.loads(ImageResult.error_for(None, "bad json").to_json())
    assert list(parsed) == _RESULT_KEYS
    assert parsed["status"] == "error"
    assert parsed["error"] == "bad json"
    assert parsed["job_id"] == "unknown"
    assert parsed["render"] is None
    assert (parsed["width"], parsed["height"]) == (0, 0)
    assert parsed["warnings"] == []
    assert parsed["blob_url"] is None


def test_result_runtime_error_keeps_render_and_intended_size():
    req = ImageRequest.from_json(_sample_payload(steps=20))
    result = ImageResult.error_for(req, "CUDA out of memory", width=2048, height=2048, render=_RENDER)
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "error"
    assert parsed["render"] == _RENDER
    assert (parsed["width"], parsed["height"]) == (2048, 2048)
    assert parsed["seed"] == 7
    assert parsed["warnings"] == ["ignored client-supplied fields: steps"]
    assert parsed["blob_url"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_workflow.py -q -k "result"`
Expected: FAIL — `TypeError: success() got an unexpected keyword argument 'width'`

- [ ] **Step 3: Rewrite `ImageResult`**

In `azure_worker/messages.py`, replace the entire `ImageResult` class with:

```python
def warnings_for(req: "ImageRequest") -> list[str]:
    """Non-fatal notes for the client. Today: which contract-v1 fields were ignored."""
    if not req.ignored_fields:
        return []
    return [f"ignored client-supplied fields: {', '.join(req.ignored_fields)}"]


@dataclass
class ImageResult:
    """One outbound message. Every key is always present (contract v2 §3)."""

    job_id: str
    name: str
    status: str  # "success" | "error"
    prompt: str
    negative_prompt: str
    # The PNG's real dimensions on success; the graph's intended size on a
    # runtime error; 0 when validation failed before a graph existed.
    width: int
    height: int
    seed: int
    # workflow.render_report() output, or None when no graph was built.
    render: Optional[dict] = None
    warnings: list[str] = field(default_factory=list)
    blob_url: Optional[str] = None
    blob_name: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def success(
        cls,
        req: ImageRequest,
        blob_name: str,
        blob_url: str,
        *,
        width: int,
        height: int,
        render: dict,
    ) -> "ImageResult":
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="success",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            seed=req.seed,
            render=render,
            warnings=warnings_for(req),
            blob_name=blob_name,
            blob_url=blob_url,
        )

    @classmethod
    def error_for(
        cls,
        req: Optional[ImageRequest],
        message: str,
        *,
        raw_job_id: str = "",
        width: int = 0,
        height: int = 0,
        render: Optional[dict] = None,
    ) -> "ImageResult":
        if req is None:
            return cls(
                job_id=raw_job_id or "unknown",
                name="unknown",
                status="error",
                prompt="",
                negative_prompt="",
                width=0,
                height=0,
                seed=0,
                error=message,
            )
        return cls(
            job_id=req.job_id,
            name=req.name,
            status="error",
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            seed=req.seed,
            render=render,
            warnings=warnings_for(req),
            error=message,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))
```

`field` is already imported (`from dataclasses import asdict, dataclass, field`). `main.py` still calls `ImageResult.success(req, blob_name, sas_url)` — it will fail at runtime but not at import; Task 7 fixes it.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: all green; count = previous + 2.

- [ ] **Step 5: Commit**

```bash
git add azure_worker/messages.py azure_worker/tests/test_workflow.py
git commit -m "azure_worker: image result contract v2

Results carry negative_prompt, the render block, and a warnings array;
width/height describe the render rather than echoing the request. Runtime
errors keep render and the intended size; validation errors null them.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Wire `main.py` and test `_process_one`

**Files:**
- Modify: `azure_worker/main.py:94-134` (`_process_one`), `:180-190` (`main()` startup log)
- Modify: `azure_worker/tests/test_main_loop.py`

**Interfaces:**
- Consumes: `render_report`, `workflow_dimensions` (Task 5); `ImageResult.success/error_for` keyword forms (Task 6); `ImageRequest.ignored_fields` (Task 4)

- [ ] **Step 1: Write the failing tests**

Append to `azure_worker/tests/test_main_loop.py`:

```python
# -- _process_one: one image job end to end, with the Azure and ComfyUI seams faked --

import json
from pathlib import Path
from types import SimpleNamespace

from azure_worker.comfy_runner import ComfyJobError
from azure_worker.config import PROFILE_QWEN_IMAGE_2_1
from azure_worker.tests.conftest import make_config


class _Msg:
    def __init__(self, payload: dict):
        self.content = json.dumps(payload).encode("utf-8")
        self.id = "m1"
        self.pop_receipt = "r1"
        self.dequeue_count = 1


class _Runner:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def run(self, workflow, prompt_id, timeout_seconds=600.0):
        self.calls.append((workflow, prompt_id))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _wire(monkeypatch, payload: dict, runner: _Runner):
    """Fake every azure_io call _process_one makes; return the captured results list."""
    sent = []
    deleted = []
    monkeypatch.setattr(main.azure_io, "receive_one", lambda clients: _Msg(payload))
    monkeypatch.setattr(main.azure_io, "upload_image", lambda clients, path, name: f"https://blob/{name}?sas")
    monkeypatch.setattr(main.azure_io, "send_result", lambda clients, result: sent.append(json.loads(result.to_json())))
    monkeypatch.setattr(main.azure_io, "delete_message", lambda clients, msg: deleted.append(msg.id))
    clients = SimpleNamespace(config=make_config(PROFILE_QWEN_IMAGE_2_1))
    return clients, sent, deleted


def test_process_one_success_reports_render_and_graph_size(monkeypatch):
    runner = _Runner([Path("out/test-image_00001_.png")])
    clients, sent, deleted = _wire(monkeypatch, {"job_id": "j1", "name": "test-image", "prompt": "a cat", "seed": 3}, runner)

    assert main._process_one(runner, clients) is True

    assert len(sent) == 1
    r = sent[0]
    assert r["status"] == "success"
    assert r["job_id"] == "j1"
    assert (r["width"], r["height"]) == (2048, 2048)
    assert r["render"]["profile"] == "qwen-image-2.1"
    assert r["render"]["steps"] == 45
    assert r["render"]["model"] == "qwen_image_2.1_int8_convrot.safetensors"
    assert r["warnings"] == []
    assert r["blob_name"] == "test-image/test-image_00001_.png"
    assert r["blob_url"].startswith("https://blob/")
    assert deleted == ["m1"]
    # The graph the runner received was sized from profiles.toml, not the request.
    workflow, prompt_id = runner.calls[0]
    assert prompt_id == "j1"
    assert workflow["6"]["inputs"]["width"] == 2048


def test_process_one_legacy_fields_are_ignored_and_warned(monkeypatch):
    runner = _Runner([Path("out/x_00001_.png")])
    clients, sent, _ = _wire(
        monkeypatch,
        {"job_id": "j2", "name": "x", "prompt": "a cat", "width": 1024, "height": 1024, "steps": 20, "cfg": 7.0},
        runner,
    )

    main._process_one(runner, clients)

    r = sent[0]
    assert r["status"] == "success"
    assert (r["width"], r["height"]) == (2048, 2048)   # the render, not the request
    assert r["render"]["steps"] == 45
    assert r["warnings"] == ["ignored client-supplied fields: width, height, steps, cfg"]


def test_process_one_runtime_error_keeps_render(monkeypatch):
    runner = _Runner(ComfyJobError("workflow execution failed: CUDA out of memory"))
    clients, sent, deleted = _wire(monkeypatch, {"job_id": "j3", "name": "x", "prompt": "a cat"}, runner)

    main._process_one(runner, clients)

    r = sent[0]
    assert r["status"] == "error"
    assert "CUDA out of memory" in r["error"]
    assert r["render"]["steps"] == 45
    assert (r["width"], r["height"]) == (2048, 2048)
    assert r["blob_url"] is None
    assert deleted == ["m1"]   # always deleted; the result queue carries the signal


def test_process_one_validation_error_has_null_render(monkeypatch):
    runner = _Runner([])
    clients, sent, deleted = _wire(monkeypatch, {"name": "x"}, runner)   # no prompt

    main._process_one(runner, clients)

    r = sent[0]
    assert r["status"] == "error"
    assert "prompt" in r["error"]
    assert r["render"] is None
    assert (r["width"], r["height"]) == (0, 0)
    assert runner.calls == []
    assert deleted == ["m1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests/test_main_loop.py -q -k process_one`
Expected: FAIL — `TypeError: ImageResult.success() missing 3 required keyword-only arguments`

- [ ] **Step 3: Rewrite `_process_one`**

In `azure_worker/main.py`, replace the whole `_process_one` function with:

```python
def _process_one(runner: ComfyRunner, clients: azure_io.AzureClients) -> bool:
    """Return True if a message was processed (success or failure), False if queue empty."""
    msg = azure_io.receive_one(clients)
    if msg is None:
        return False

    raw_body = azure_io.message_body_text(msg)
    req: Optional[ImageRequest] = None
    # Filled in once a graph exists, so a runtime failure can still report
    # what was being attempted (contract v2 §4: render is populated on
    # runtime errors, null on validation errors).
    render: Optional[dict] = None
    width = height = 0
    try:
        req = ImageRequest.from_json(raw_body)
        log.info("job %s name=%s", req.job_id, req.name)
        log.info("job %s prompt: %s", req.job_id, req.prompt)
        if req.negative_prompt:
            log.info("job %s negative_prompt: %s", req.job_id, req.negative_prompt)
        if req.ignored_fields:
            log.warning(
                "job %s sent contract-v1 fields the worker now owns, ignoring: %s",
                req.job_id, ", ".join(req.ignored_fields),
            )
        workflow = build_workflow(req, clients.config)
        render = render_report(workflow, clients.config.profile)
        width, height = workflow_dimensions(workflow)
        log.info("job %s render: %s", req.job_id, render["summary"])
        outputs = runner.run(workflow, prompt_id=req.job_id)
        if not outputs:
            raise ComfyJobError("workflow produced no output files")
        local_path: Path = outputs[0]
        blob_name = f"{sanitize_name(req.name)}/{local_path.name}"
        sas_url = azure_io.upload_image(clients, local_path, blob_name)
        azure_io.send_result(
            clients,
            ImageResult.success(req, blob_name, sas_url, width=width, height=height, render=render),
        )
        log.info("job %s complete: %s", req.job_id, blob_name)
    except MessageValidationError as e:
        log.warning("invalid message (dequeue_count=%s): %s", msg.dequeue_count, e)
        azure_io.send_result(clients, ImageResult.error_for(None, str(e)))
    except (ComfyJobError, Exception) as e:  # noqa: BLE001 - we want every failure on the result queue
        log.exception("job failed: %s", e)
        azure_io.send_result(
            clients,
            ImageResult.error_for(req, str(e), width=width, height=height, render=render),
        )
    finally:
        # Always delete: result queue carries the success/error signal.
        # Switch to a dequeue_count check + leave-for-retry once we have a retry policy.
        try:
            azure_io.delete_message(clients, msg)
        except Exception:
            log.exception("failed to delete inbound message %s", msg.id)
    return True
```

- [ ] **Step 4: Log the active render settings at startup**

In `main()`, replace `log.info("starting ComfyUI runner (profile=%s)", cfg.profile)` with:

```python
    r = cfg.render
    log.info(
        "starting ComfyUI runner (profile=%s) render: %dx%d steps=%d cfg=%s sampler=%s/%s shift=%s",
        cfg.profile, r.width, r.height, r.steps, r.cfg, r.sampler, r.scheduler, r.shift,
    )
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: all green; count = previous + 4. Then `grep -rn "req\.width\|req\.height\|req\.steps\|req\.cfg\|summarize_workflow\|SDXL_CFG\|QWEN21_CFG\|CHROMA_SHIFT" azure_worker/` returns **nothing**.

- [ ] **Step 6: Commit**

```bash
git add azure_worker/main.py azure_worker/tests/test_main_loop.py
git commit -m "azure_worker: send contract-v2 results with the render block

Success and runtime-error results carry render_report() output and the
graph's size; validation errors carry neither. Legacy request fields are
logged and reported as a warning. Startup logs the active render settings.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Documentation and sample payload

**Files:**
- Modify: `azure_worker/README.md` — profile table (lines 16-27), `## Configuration` (45-80), `## Message contracts` (90-141)
- Modify: `azure_worker/SPEC.md` — §5, §6, §12
- Modify: `azure_worker/sample_request.json`
- Modify: `docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md` — `shift` example values

No code; verification is by reading and by the one JSON sanity check in Step 5.

- [ ] **Step 1: Fix the spec's `shift` example**

In `docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md`, `qwen21_shift(2048, 2048) = exp(0.5 + 0.4 * (16384 - 256) / (8192 - 256)) = exp(1.3129) ≈ 3.7169`. The value `1.9937` shown is the 1024² value. Replace every `"shift": 1.9937` (three occurrences: §4 block, §6 success example, and any other) with `"shift": 3.7169`.

- [ ] **Step 2: Update `sample_request.json`**

Replace the file's contents with:

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "prompt": "a serene sunset over snowy mountains, oil painting, dramatic lighting",
  "negative_prompt": "",
  "seed": 42
}
```

- [ ] **Step 3: Update `SPEC.md`**

Replace the body of **§5** (from the `A single JSON object:` line through the end of the `### Validation failures` subsection) with:

```markdown
A single JSON object:

```json
{
  "job_id": "string, optional",
  "name": "string, required",
  "prompt": "string, required",
  "negative_prompt": "string, optional, default \"\"",
  "seed": "integer, optional"
}
```

### Field rules

| Field | Type | Required | Constraints |
|---|---|---|---|
| `job_id` | string | no | If omitted, the worker generates a UUID. **Provide your own UUID** if you want to correlate requests with results — see §8. |
| `name` | string | yes | Non-empty. Becomes the filename prefix for the generated PNG (sanitized — non-alphanumeric chars are replaced with `_`). Does not need to be unique. |
| `prompt` | string | yes | Non-empty. `prompt` + `negative_prompt` combined ≤ **32,000 characters**. This is a transport limit (64 KiB queue message), not a model one. |
| `negative_prompt` | string | no | Shares the 32,000-character budget. Whether it reached the model is reported on the result as `render.negative_honored`. |
| `seed` | integer | no | 64-bit unsigned. If omitted, the worker picks a random seed and returns it in the result so the run is reproducible. |

**Output size, step count, guidance scale and sampler are owned by the
worker** and configured per profile in `azure_worker/profiles.toml`. The
contract-v1 fields `width`, `height`, `steps` and `cfg` are accepted if sent,
ignored, and listed in the result's `warnings`. The full v2 contract, with the
profile table and migration notes, is
`docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md`.

### Example

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "prompt": "a serene sunset over snowy mountains, oil painting, dramatic lighting",
  "seed": 42
}
```

### Validation failures

If the message is unparseable JSON, missing a required field, over the
prompt budget, or over 46 KiB of raw JSON, the worker:
- Sends a result message with `status="error"` and an `error` string explaining what was wrong.
- Deletes the inbound message (no retries).
- Does not run the model.

If `job_id` couldn't be recovered (e.g. the JSON was completely malformed),
the error result has `job_id="unknown"` and `name="unknown"` — see §6.
```

Replace the body of **§6** (from `A single JSON object. Always contains` through the end of the `### Error result` subsection and its `Common error messages` list) with:

```markdown
A single JSON object. Always contains the same set of keys; `status` tells you
whether to look at `blob_url` or `error`.

```json
{
  "job_id": "string",
  "name": "string",
  "status": "success" | "error",
  "prompt": "string",
  "negative_prompt": "string",
  "width": "integer",
  "height": "integer",
  "seed": "integer",
  "render": "object | null",
  "warnings": ["string"],
  "blob_url": "string | null",
  "blob_name": "string | null",
  "error": "string | null"
}
```

- `width` / `height` are the PNG's real dimensions (the graph's intended size
  on a runtime error; `0` on a validation error). They are **not** an echo of
  the request.
- `render` describes what actually ran — model file, LoRA stack, steps, cfg,
  sampler, scheduler, shift, whether the negative prompt was honored, and a
  one-line `summary` — read back from the executed graph. It is `null` only
  when validation failed before a graph was built. Field-by-field definition:
  `docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md` §4.
- `warnings` is always an array, `[]` when empty. Today it carries
  `"ignored client-supplied fields: ..."` for clients still sending v1 fields.
- `seed` reflects the seed the worker actually used.
- `blob_url` is a pre-signed read-only HTTPS URL, valid 24 hours. `blob_name`
  is the path within the `generated-images` container.

### Success result

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "status": "success",
  "prompt": "a serene sunset over snowy mountains, oil painting, dramatic lighting",
  "negative_prompt": "",
  "width": 2048,
  "height": 2048,
  "seed": 42,
  "render": {
    "profile": "qwen-image-2.1",
    "model": "qwen_image_2.1_int8_convrot.safetensors",
    "loras": [],
    "steps": 45,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "shift": 3.7169,
    "negative_honored": false,
    "summary": "qwen-image-2.1 · qwen_image_2.1_int8_convrot · 2048×2048 · 45 steps · cfg 1.0 · euler/simple"
  },
  "warnings": [],
  "blob_url": "https://nomadimagegen.blob.core.windows.net/generated-images/sunset-mountains/sunset-mountains_00001_.png?se=...&sig=...",
  "blob_name": "sunset-mountains/sunset-mountains_00001_.png",
  "error": null
}
```

### Error result

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "status": "error",
  "prompt": "a serene sunset over snowy mountains",
  "negative_prompt": "",
  "width": 2048,
  "height": 2048,
  "seed": 42,
  "render": { "profile": "qwen-image-2.1", "steps": 45, "...": "..." },
  "warnings": [],
  "blob_url": null,
  "blob_name": null,
  "error": "workflow execution failed: [...]"
}
```

Common `error` messages:
- `"'prompt' + 'negative_prompt' total 32500 characters; the limit is 32000"` — validation failure
- `"message is 48000 bytes; the limit is 47104 bytes"` — validation failure
- `"'prompt' is required and must be a non-empty string"` — schema failure
- `"workflow execution failed: ..."` — runtime failure inside ComfyUI (OOM, model load error, etc.)
- `"workflow failed validation: ..."` — workflow couldn't even start
- `"workflow %s did not complete within %ds"` — generation timed out (default 600s)

If the original message was so malformed that no fields could be recovered,
the error result has `job_id="unknown"`, `name="unknown"`, empty prompts,
`render: null`, and zeros for `width`/`height`/`seed`.
```

In **§12** append one paragraph:

```markdown
Contract v2 (2026-10-06) removed `width`/`height`/`steps`/`cfg` from the
request and added `negative_prompt`, `render` and `warnings` to the result.
The removed request fields are still accepted and ignored, so a v1 producer
keeps working against a v2 worker. A v2 producer against a v1 worker fails
validation (v1 required `width`/`height`): **deploy the worker first.**
```

Also in **§1** change `describing an image (prompt + size)` to `describing an image (prompts + seed)`.

- [ ] **Step 4: Update `README.md`**

1. In the profile table (lines 16-27), delete every parenthetical about `steps`/`cfg` being honored, no-ops, or recommended values (e.g. "Recommended: steps=20-50, cfg=4.0", "`cfg` and `negative_prompt` are no-ops", "`steps` is honored (4-8 recommended)"). Keep model lists and the negative-prompt behaviour. Directly below the table replace the sentence `All profile blocks must be filled in `.env`; only the active profile is actually loaded into VRAM.` with:

```markdown
Model filenames for all profile blocks must be filled in `.env`; only the
active profile is actually loaded into VRAM. Output size, step count, guidance
scale, sampler and scheduler per profile live in **`profiles.toml`** (tracked
in git) — edit that file to retune a profile; no code change needed. Every
result message reports the values that actually ran in its `render` block.
```

2. In `## Configuration (environment variables)` add at the end of the section:

```markdown
### Render settings (`profiles.toml`)

Not environment variables. `azure_worker/profiles.toml` holds one table per
profile with `width`, `height`, `steps`, `cfg`, `sampler`, `scheduler` and
optionally `shift`. The file is validated at startup (dimensions 64-4096 and
multiples of 16, steps 1-200, cfg 0-30); a bad value is a config error before
the first job. The active profile's settings are logged on the `starting
ComfyUI runner` line.
```

3. In `## Message contracts` replace the inbound and outbound JSON examples with the two from SPEC.md §5 and §6 above, and add under the heading: `The authoritative contract is SPEC.md §5–§6 and docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md.`

- [ ] **Step 5: Verify**

Run: `.venv/Scripts/python.exe -c "import json; json.load(open('azure_worker/sample_request.json'))" && .venv/Scripts/python.exe -m pytest azure_worker/tests -q`
Expected: no JSON error; full suite green.
Run: `grep -n "1.9937" docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md azure_worker/SPEC.md`
Expected: no output.
Run: `grep -n "Max 4000\|Max 4096\|steps.*honored\|cfg.*no-op" azure_worker/README.md azure_worker/SPEC.md`
Expected: no output.

- [ ] **Step 6: Commit and push**

```bash
git add azure_worker/README.md azure_worker/SPEC.md azure_worker/sample_request.json docs/superpowers/specs/2026-10-06-image-queue-contract-v2.md
git commit -m "azure_worker: document contract v2 and profiles.toml

SPEC.md §5/§6 carry the v2 request and result schemas; README points at
profiles.toml for render settings. Corrects the spec's Qwen 2.1 shift
example (3.7169 at 2048², not the 1024² value).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push origin HEAD:azure-worker
```

---

## Self-review against the spec

**Spec coverage**

| Spec requirement | Task |
|---|---|
| §2 request down to `job_id/name/prompt/negative_prompt/seed` | 4 |
| §2 combined 32,000-char budget | 4 |
| §2 byte backstop, clean validation error | 4 (`MAX_REQUEST_BYTES`) |
| §2 legacy fields accepted, ignored, warned | 4 (`ignored_fields`), 6 (`warnings_for`), 7 (log) |
| §3 result keys always present, in order | 6 |
| §3 `width`/`height` = real render size; 0 on validation error | 5 (`workflow_dimensions`), 6, 7 |
| §3 feature detection via `render` presence | 6 (`render` key always emitted) |
| §4 ten `render` fields, graph-derived | 5 |
| §4 `cfg`/`scheduler` null on flux2-klein | 5 (`effective_cfg`, scheduler fallthrough) |
| §4 `shift` derived on qwen-2.1, null elsewhere | 3, 5 |
| §4 `negative_honored` rule | 5 |
| §4 `summary` format | 5 (`_summary`) |
| §4 `render` null on validation / populated on runtime error | 7 |
| §5 profile table, admin-editable, tracked | 1 |
| §9 `profiles.toml` + `tomllib` | 1 |
| §9 `summarize_workflow` → render dict | 5 |
| §9 constants deleted, `qwen21_shift` kept | 3 |
| §9 `messages.py` limits | 4 |
| §9 SPEC.md / README updated | 8 |

**Placeholder scan:** none. Every code step shows the code; every test step shows the test.

**Type consistency:** `RenderSettings` field names (`width, height, steps, sampler, cfg, scheduler, shift`) match between Task 1 (definition), Task 2 (`make_config(render=...)`), Task 3 (`cfg.render.*` in builders), and Task 7 (startup log). `render_report(workflow, profile)` signature matches between Task 5 (definition/tests) and Task 7 (call). `ImageResult.success(req, blob_name, blob_url, *, width, height, render)` and `error_for(req, message, *, raw_job_id, width, height, render)` match between Task 6 and Task 7. `make_config` import path `azure_worker.tests.conftest` is the same in Tasks 2 and 7.

**Known gap, intentionally out of scope:** the spec says `width`/`height` on success are "the PNG's actual dimensions". The plan reads them from the latent node, which is what the PNG will be — no profile resizes after decode. Verifying against the file with PIL would add a dependency on the output path in tests for no observable difference; if a future profile adds an upscale node, `workflow_dimensions` is the one place to change.
