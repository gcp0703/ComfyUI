"""Race routing: which LoRA a prompt selects, and what it does to the graph."""
from __future__ import annotations

import json
import os

import pytest

from azure_worker.config import PROFILE_SDXL_DREAMSHAPER, LoraSpec
from azure_worker.lora_router import (
    RACE_LORAS,
    apply_trigger,
    route_race_lora,
)
from azure_worker.messages import ImageRequest
from azure_worker.workflow import build_workflow, render_report

from .test_workflow import _cfg, _sample_payload


def _routed_name(prompt: str):
    r = route_race_lora(prompt)
    return None if r is None else r.spec.name


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a regal dwarf woman on a throne", "RPGDwarfXL.safetensors"),
        ("portrait of a tiefling rogue", "RPGTieflingXL.safetensors"),
        ("a stout dwarven smith", "RPGDwarfXL.safetensors"),
        ("an elf ranger in a forest", "RPGElfXL.safetensors"),
        ("a warforged sentinel", "RPGWarforgedXL.safetensors"),
        ("a tabaxi monk", "RPGTabaxiXL.safetensors"),
        ("a lizard folk shaman", "RPGLizardFolkXL.safetensors"),
        ("a lizardfolk shaman", "RPGLizardFolkXL.safetensors"),
    ],
)
def test_routes_common_races(prompt, expected):
    assert _routed_name(prompt) == os.path.join("dnd", expected)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        # Longer names must beat the shorter ones they contain.
        ("a hobgoblin warlord", "RPGHobgoblinXL.safetensors"),
        ("a goblin scout", "RPGGoblinXL.safetensors"),
        ("a drow assassin", "RPGDrowXL.safetensors"),
        ("a dark elf assassin", "RPGDrowXL.safetensors"),
    ],
)
def test_longer_race_names_win_over_substrings(prompt, expected):
    assert _routed_name(prompt) == os.path.join("dnd", expected)


def test_earliest_mention_wins_when_two_races_appear():
    # Character prompts lead with their subject.
    assert _routed_name("a dwarf merchant haggling with an elf") == os.path.join(
        "dnd", "RPGDwarfXL.safetensors"
    )
    assert _routed_name("an elf merchant haggling with a dwarf") == os.path.join(
        "dnd", "RPGElfXL.safetensors"
    )


@pytest.mark.parametrize(
    "prompt", ["a human knight in plate armor", "a sunlit desert caravan camp", ""]
)
def test_returns_none_when_no_race_named(prompt):
    assert route_race_lora(prompt) is None


def test_word_boundaries_prevent_false_matches():
    # 'elf' inside 'shelf', 'orc' inside 'orchard'.
    assert route_race_lora("a book on a shelf") is None
    assert route_race_lora("walking through an orchard") is None


def test_apply_trigger_injects_invented_tokens_only_when_absent():
    # The RPG elf/gnome/goblin LoRAs were trained on invented tokens.
    assert apply_trigger("an elf ranger", "rpgelf") == "rpgelf, an elf ranger"
    # Already present -> untouched.
    assert apply_trigger("a dwarf smith", "dwarf") == "a dwarf smith"
    assert apply_trigger("A DWARF SMITH", "dwarf") == "A DWARF SMITH"


def test_every_race_entry_points_at_a_downloaded_lora():
    """Guard against a typo silently routing to a file that isn't there."""
    lora_dir = os.path.join("models", "loras", "dnd")
    for race in RACE_LORAS:
        assert os.path.exists(os.path.join(lora_dir, race.filename)), race.filename


# --- integration with the sdxl builder --------------------------------------


def test_autoroute_appends_race_lora_after_configured_stack():
    base = LoraSpec(os.path.join("dnd", "Fantasy_Races_XL.safetensors"), 0.8, 0.8)
    req = ImageRequest.from_json(
        _sample_payload(prompt="a regal dwarf woman on a throne", steps=7)
    )
    wf = build_workflow(
        req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=(base,), sdxl_lora_autoroute=True)
    )
    loras = render_report(wf, PROFILE_SDXL_DREAMSHAPER)["loras"]

    # Configured stack first, routed race LoRA last.
    assert "Fantasy_Races_XL.safetensors" in loras[0]["name"]
    assert "RPGDwarfXL.safetensors" in loras[1]["name"]
    assert loras[0]["model_strength"] == 0.8
    assert loras[0]["clip_strength"] == 0.8
    assert loras[1]["model_strength"] == 0.9
    assert loras[1]["clip_strength"] == 0.9


def test_autoroute_off_leaves_the_stack_alone():
    req = ImageRequest.from_json(_sample_payload(prompt="a regal dwarf woman"))
    wf = build_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_lora_autoroute=False))
    assert render_report(wf, PROFILE_SDXL_DREAMSHAPER)["loras"] == []


def test_autoroute_injects_trigger_into_the_positive_prompt_only():
    req = ImageRequest.from_json(
        _sample_payload(prompt="an elf ranger", negative_prompt="blurry")
    )
    wf = build_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_lora_autoroute=True))

    assert wf["2"]["inputs"]["text"] == "rpgelf, an elf ranger"
    assert wf["3"]["inputs"]["text"] == "blurry"


def test_autoroute_leaves_prompt_untouched_when_trigger_already_present():
    req = ImageRequest.from_json(_sample_payload(prompt="a regal dwarf woman"))
    wf = build_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_lora_autoroute=True))
    assert wf["2"]["inputs"]["text"] == "a regal dwarf woman"


def test_autoroute_is_a_noop_for_raceless_prompts():
    req = ImageRequest.from_json(_sample_payload(prompt="a human knight"))
    wf = build_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_lora_autoroute=True))
    assert render_report(wf, PROFILE_SDXL_DREAMSHAPER)["loras"] == []
    assert wf["2"]["inputs"]["text"] == "a human knight"
