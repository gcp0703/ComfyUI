"""Pick a per-race LoRA out of the prompt text.

``COMFY_SDXL_LORAS`` is a single always-on stack, but every job names its own
race in the prompt, so a fixed stack cannot serve all of them. The multi-race
``Fantasy_Races_XL`` was the compromise; measured against no LoRA at all it
does move the image, but not enough to reshape body morphology -- dwarf
prompts still rendered as ordinary humans at strength 0.8 and at 1.0, with
``(dwarf:1.5)`` weighting, and with a short prompt. Swapping in the dedicated
``RPGDwarfXL`` was the only change that fixed them.

So: scan the prompt, and when it names a race, append that race's own LoRA to
the configured stack.

Two details that matter for this to actually work:

- Some LoRAs in the RPG series were trained on an invented token rather than
  the plain English word -- ``rpgelf``, ``rpggnome``, ``rpggoblin``. Prompts
  say "elf", so the trigger has to be injected or the LoRA barely fires.
- Longer race names must win over the shorter ones they contain
  (``hobgoblin`` over ``goblin``, ``dark elf`` over ``elf``), which falls out
  of preferring the earliest match and then the longest one.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from .config import LoraSpec

# Strength for a routed race LoRA. 0.9 is what the dwarf comparison was run at;
# the RPG series is trained light enough that 0.8-1.0 all behave.
DEFAULT_RACE_STRENGTH = 0.9

# Subdirectory under models/loras holding the race LoRAs.
RACE_LORA_DIR = "dnd"


@dataclass(frozen=True)
class RaceLora:
    pattern: str      # alternation matched case-insensitively, on word boundaries
    filename: str     # relative to RACE_LORA_DIR
    trigger: str      # the token the LoRA was actually trained on


RACE_LORAS: tuple[RaceLora, ...] = (
    RaceLora(r"dwarf|dwarves|dwarven", "RPGDwarfXL.safetensors", "dwarf"),
    RaceLora(r"drow|dark elf|dark elves", "RPGDrowXL.safetensors", "drow"),
    RaceLora(r"elf|elves|elven", "RPGElfXL.safetensors", "rpgelf"),
    RaceLora(r"tiefling|tieflings", "RPGTieflingXL.safetensors", "tiefling"),
    RaceLora(r"hobgoblin|hobgoblins", "RPGHobgoblinXL.safetensors", "hobgoblin"),
    RaceLora(r"goblin|goblins", "RPGGoblinXL.safetensors", "rpggoblin"),
    RaceLora(r"bugbear|bugbears", "RPGBugbearXL.safetensors", "bugbear"),
    RaceLora(r"orc|orcs|orcish|half-orc|half-orcs", "RPGOrcXL.safetensors", "orc"),
    RaceLora(r"gnome|gnomes", "RPGGnomeXL.safetensors", "rpggnome"),
    RaceLora(r"tabaxi", "RPGTabaxiXL.safetensors", "tabaxi"),
    RaceLora(r"lizardfolk|lizard folk", "RPGLizardFolkXL.safetensors", "lizard folk"),
    RaceLora(r"ratfolk|rat folk", "RPGRatFolkXL.safetensors", "rat folk"),
    RaceLora(r"warforged", "RPGWarforgedXL.safetensors", "warforged"),
    RaceLora(r"firbolg|firbolgs", "RPGFirbolgXL.safetensors", "firbolg"),
    RaceLora(r"centaur|centaurs", "RPGCentaurXL.safetensors", "centaur"),
    RaceLora(r"minotaur|minotaurs", "RPGMinotaurXL.safetensors", "minotaur"),
    RaceLora(r"aarakocra", "RPGAarakocraXL.safetensors", "aarakocra"),
    RaceLora(r"kenku", "RPGKenkuXL.safetensors", "kenku"),
    RaceLora(r"tortle|tortles", "RPGTortleXL.safetensors", "tortle"),
    RaceLora(r"harpy|harpies", "RPGHarpyXL.safetensors", "harpy"),
    RaceLora(r"aasimar|angelic|angel", "RPGAngelsXL.safetensors", "angel"),
)

_COMPILED = tuple(
    (race, re.compile(rf"\b(?:{race.pattern})\b", re.IGNORECASE)) for race in RACE_LORAS
)


@dataclass(frozen=True)
class RouteResult:
    spec: LoraSpec
    trigger: str
    matched: str   # the literal text in the prompt that selected this LoRA


def route_race_lora(prompt: str, strength: float = DEFAULT_RACE_STRENGTH) -> Optional[RouteResult]:
    """Choose a race LoRA for ``prompt``, or None when it names no known race.

    The earliest mention wins -- character prompts lead with their subject
    ("a regal dwarf woman...") -- and on a tie the longer match wins so
    ``hobgoblin`` is not read as ``goblin``.
    """
    best: Optional[tuple[int, int, RaceLora, str]] = None
    for race, rx in _COMPILED:
        m = rx.search(prompt)
        if not m:
            continue
        # (position, -length) sorts earliest-then-longest to the front.
        key = (m.start(), -len(m.group(0)))
        if best is None or key < (best[0], best[1]):
            best = (m.start(), -len(m.group(0)), race, m.group(0))
    if best is None:
        return None

    _, _, race, matched = best
    # os.sep, not '/': ComfyUI builds its lora filename list with
    # os.path.relpath, so LoraLoader validates against backslashes on Windows.
    name = os.path.join(RACE_LORA_DIR, race.filename)
    return RouteResult(
        spec=LoraSpec(name, strength, strength),
        trigger=race.trigger,
        matched=matched,
    )


def apply_trigger(prompt: str, trigger: str) -> str:
    """Prepend a LoRA's trigger token unless the prompt already carries it.

    Needed for the RPG entries trained on invented tokens (``rpgelf``,
    ``rpggnome``, ``rpggoblin``); for the rest the trigger is the same word
    that matched, so this is a no-op.
    """
    if re.search(rf"\b{re.escape(trigger)}\b", prompt, re.IGNORECASE):
        return prompt
    return f"{trigger}, {prompt}"
