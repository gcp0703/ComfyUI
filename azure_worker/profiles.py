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
