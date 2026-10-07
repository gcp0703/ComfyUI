"""GPU-free unit tests for the message + workflow layer."""
from __future__ import annotations

import json
import math
import os

import pytest

from azure_worker.config import (
    PROFILE_CHROMA1,
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_FLUXED_UP,
    PROFILE_OPENFLUX1,
    PROFILE_QWEN_IMAGE_2512,
    PROFILE_QWEN_IMAGE_2_1,
    PROFILE_QWEN_RAPID_AIO,
    PROFILE_SDXL_DREAMSHAPER,
    Config,
    ConfigError,
    LoraSpec,
    _optional_loras,
)
from azure_worker.messages import (
    ImageRequest,
    ImageResult,
    MessageValidationError,
    sanitize_name,
)
from azure_worker.workflow import (
    CHROMA1_SAVE_NODE_ID,
    FLUX1_SAVE_NODE_ID,
    FLUX2_SAVE_NODE_ID,
    FLUXED_UP_SAVE_NODE_ID,
    OPENFLUX1_SAVE_NODE_ID,
    QWEN_IMAGE_SAVE_NODE_ID,
    QWEN21_SAVE_NODE_ID,
    QWEN_RAPID_SAVE_NODE_ID,
    SDXL_LORA_NODE_BASE,
    SDXL_SAVE_NODE_ID,
    build_chroma1_workflow,
    build_flux1_dev_workflow,
    build_flux2_klein_workflow,
    build_fluxed_up_workflow,
    build_openflux1_workflow,
    build_qwen_image_2512_workflow,
    build_qwen_image_2_1_workflow,
    build_qwen_rapid_aio_workflow,
    build_sdxl_dreamshaper_workflow,
    build_workflow,
    effective_cfg,
    render_report,
    workflow_dimensions,
)
from azure_worker.tests.conftest import make_config as _cfg


def _sample_payload(**overrides):
    payload = {
        "job_id": "abc",
        "name": "test-image",
        "prompt": "a cat",
        "seed": 7,
    }
    payload.update(overrides)
    return json.dumps(payload)


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
    raw = _sample_payload(prompt="—" * 31_000)
    with pytest.raises(MessageValidationError, match="bytes"):
        ImageRequest.from_json(raw)


# -- Flux 1 dev workflow --

def test_flux1_workflow_shape():
    # width/height are deliberately mismatched from the profile's render
    # settings here, to prove the builder ignores the request and uses
    # cfg.render instead (see test_builders_ignore_request_size_and_steps).
    req = ImageRequest.from_json(_sample_payload(prompt="dragon", seed=99, width=1024, height=768))
    wf = build_flux1_dev_workflow(req, _cfg(PROFILE_FLUX1_DEV))

    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "flux1-dev.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "default"

    assert wf["2"]["class_type"] == "DualCLIPLoader"
    assert wf["2"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert wf["2"]["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"
    assert wf["2"]["inputs"]["type"] == "flux"

    assert wf["3"]["inputs"]["vae_name"] == "ae.safetensors"
    assert wf["4"]["inputs"]["text"] == "dragon"
    assert wf["5"]["class_type"] == "ConditioningZeroOut"
    assert wf["6"]["class_type"] == "EmptySD3LatentImage"
    assert wf["6"]["inputs"]["width"] == 1024 and wf["6"]["inputs"]["height"] == 1024

    ks = wf["7"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["seed"] == 99
    assert ks["inputs"]["cfg"] == 1
    assert ks["inputs"]["sampler_name"] == "euler"
    assert ks["inputs"]["scheduler"] == "simple"
    assert ks["inputs"]["positive"] == ["4", 0]
    assert ks["inputs"]["negative"] == ["5", 0]

    assert wf[FLUX1_SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[FLUX1_SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


# -- Flux 2 Klein workflow --

def test_flux2_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(prompt="dragon", seed=99, width=1024, height=768))
    wf = build_flux2_klein_workflow(req, _cfg(PROFILE_FLUX2_KLEIN))

    assert wf["1"]["inputs"]["unet_name"] == "flux-2-klein-9b-fp8.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"
    assert wf["2"]["inputs"]["clip_name"] == "qwen_3_8b_fp8mixed.safetensors"
    assert wf["2"]["inputs"]["type"] == "flux2"
    assert wf["3"]["inputs"]["vae_name"] == "full_encoder_small_decoder.safetensors"
    assert wf["5"]["class_type"] == "EmptyFlux2LatentImage"
    assert wf["6"]["class_type"] == "Flux2Scheduler"
    assert wf["8"]["inputs"]["noise_seed"] == 99

    sampler = wf["10"]
    assert sampler["class_type"] == "SamplerCustomAdvanced"
    assert sampler["inputs"]["sigmas"] == ["6", 0]

    assert wf[FLUX2_SAVE_NODE_ID]["class_type"] == "SaveImage"


# -- Chroma1 workflow --

def test_chroma1_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",
        seed=99,
        width=1024,
        height=1024,
    ))
    wf = build_chroma1_workflow(req, _cfg(PROFILE_CHROMA1))

    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "Chroma1-HD-fp8mixed.safetensors"

    assert wf["2"]["class_type"] == "CLIPLoader"
    assert wf["2"]["inputs"]["clip_name"] == "t5xxl_fp16.safetensors"
    assert wf["2"]["inputs"]["type"] == "chroma"

    assert wf["3"]["inputs"]["vae_name"] == "ae.safetensors"
    assert wf["4"]["class_type"] == "ModelSamplingAuraFlow"
    assert wf["4"]["inputs"]["shift"] == 1.0
    assert wf["5"]["class_type"] == "T5TokenizerOptions"

    # Real negative prompt (unlike flux profiles)
    assert wf["6"]["inputs"]["text"] == "dragon"
    assert wf["7"]["inputs"]["text"] == "blurry, low quality"

    # Real CFG flowing through CFGGuider (not BasicGuider)
    assert wf["8"]["class_type"] == "CFGGuider"
    assert wf["8"]["inputs"]["cfg"] == 3.5

    # Beta scheduler is the whole point of using chroma
    assert wf["9"]["inputs"]["sampler_name"] == "euler"
    assert wf["10"]["class_type"] == "BasicScheduler"
    assert wf["10"]["inputs"]["scheduler"] == "beta"
    assert wf["10"]["inputs"]["steps"] == 26

    assert wf["11"]["inputs"]["noise_seed"] == 99
    assert wf["12"]["class_type"] == "EmptySD3LatentImage"

    sampler = wf["13"]
    assert sampler["class_type"] == "SamplerCustomAdvanced"
    assert sampler["inputs"]["guider"] == ["8", 0]
    assert sampler["inputs"]["sigmas"] == ["10", 0]

    assert wf[CHROMA1_SAVE_NODE_ID]["class_type"] == "SaveImage"


# -- Dispatcher --

def test_dispatcher_picks_flux1_for_flux1_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_FLUX1_DEV))
    assert wf["2"]["class_type"] == "DualCLIPLoader"  # only flux1 has this
    assert "14" not in wf  # chroma1's save node id


def test_dispatcher_picks_flux2_for_flux2_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_FLUX2_KLEIN))
    assert wf["6"]["class_type"] == "Flux2Scheduler"  # only flux2 has this
    assert wf[FLUX2_SAVE_NODE_ID]["class_type"] == "SaveImage"


def test_dispatcher_picks_chroma1_for_chroma1_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_CHROMA1))
    assert wf["4"]["class_type"] == "ModelSamplingAuraFlow"  # only chroma1 has this
    assert wf[CHROMA1_SAVE_NODE_ID]["class_type"] == "SaveImage"


# -- Fluxed Up workflow --

def test_fluxed_up_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(prompt="dragon", seed=99, width=1024, height=1024))
    wf = build_fluxed_up_workflow(req, _cfg(PROFILE_FLUXED_UP))

    # Different UNet from vanilla flux1-dev, loaded in fp8
    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "fluxedUpFluxNSFW_40DevFp8.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"

    # Reuses the flux1 CLIP-L + T5 + VAE
    assert wf["2"]["class_type"] == "DualCLIPLoader"
    assert wf["2"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert wf["2"]["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"
    assert wf["2"]["inputs"]["type"] == "flux"
    assert wf["3"]["inputs"]["vae_name"] == "ae.safetensors"

    # Same guidance-distilled driving as flux1-dev (cfg=1 + ConditioningZeroOut)
    assert wf["5"]["class_type"] == "ConditioningZeroOut"
    ks = wf["7"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["cfg"] == 1
    assert ks["inputs"]["seed"] == 99
    assert wf[FLUXED_UP_SAVE_NODE_ID]["class_type"] == "SaveImage"


def test_dispatcher_picks_fluxed_up_for_fluxed_up_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_FLUXED_UP))
    assert wf["1"]["inputs"]["unet_name"] == "fluxedUpFluxNSFW_40DevFp8.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"  # distinguishes from flux1-dev


# -- Qwen-Image 2512 workflow --

def test_qwen_image_2512_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",
        seed=99,
        width=1328,
        height=1328,
    ))
    wf = build_qwen_image_2512_workflow(req, _cfg(PROFILE_QWEN_IMAGE_2512))

    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "qwen_image_2512_fp8_e4m3fn.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"

    assert wf["2"]["class_type"] == "CLIPLoader"
    assert wf["2"]["inputs"]["clip_name"] == "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    assert wf["2"]["inputs"]["type"] == "qwen_image"

    assert wf["3"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"

    # Sigma shift = 3.1 (different from Chroma's 1.0)
    assert wf["4"]["class_type"] == "ModelSamplingAuraFlow"
    assert wf["4"]["inputs"]["shift"] == 3.1

    # Real negative prompt path
    assert wf["5"]["inputs"]["text"] == "dragon"
    assert wf["6"]["inputs"]["text"] == "blurry, low quality"

    assert wf["7"]["class_type"] == "EmptySD3LatentImage"

    # Stock KSampler with real CFG
    ks = wf["8"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["seed"] == 99
    assert ks["inputs"]["steps"] == 20
    assert ks["inputs"]["cfg"] == 4.0
    assert ks["inputs"]["sampler_name"] == "euler"
    assert ks["inputs"]["scheduler"] == "simple"
    assert ks["inputs"]["positive"] == ["5", 0]
    assert ks["inputs"]["negative"] == ["6", 0]

    assert wf[QWEN_IMAGE_SAVE_NODE_ID]["class_type"] == "SaveImage"


def test_dispatcher_picks_qwen_image_for_qwen_image_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_QWEN_IMAGE_2512))
    assert wf["2"]["inputs"]["type"] == "qwen_image"
    assert wf[QWEN_IMAGE_SAVE_NODE_ID]["class_type"] == "SaveImage"


# -- Qwen-Image 2.1 workflow --

def test_qwen_image_2_1_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",
        seed=99,
        width=1024,
        height=1024,
    ))
    wf = build_qwen_image_2_1_workflow(req, _cfg(PROFILE_QWEN_IMAGE_2_1))

    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "qwen_image_2.1_int8_convrot.safetensors"
    # int8_convrot carries its own quant metadata, so no forced weight_dtype
    assert wf["1"]["inputs"]["weight_dtype"] == "default"

    assert wf["2"]["class_type"] == "CLIPLoader"
    assert wf["2"]["inputs"]["clip_name"] == "qwen3vl_8b_int8_convrot.safetensors"
    assert wf["2"]["inputs"]["type"] == "qwen_image"

    assert wf["3"]["inputs"]["vae_name"] == "qwen_image_2.1_vae_bf16.safetensors"

    # Dynamic shift: the model's own settings pin the 1024x1024 mu (0.69), so
    # the builder patches exp(mu) in from the profile's size. This profile's
    # default is 2048x2048, which is e^1.3129.
    shifter = wf["4"]
    assert shifter["class_type"] == "ModelSamplingAuraFlow"
    assert shifter["inputs"]["model"] == ["1", 0]
    assert shifter["inputs"]["shift"] == pytest.approx(math.exp(1.3129032258064517), rel=1e-6)

    # One encoder node emits both conditioning branches
    enc = wf["5"]
    assert enc["class_type"] == "TextEncodeQwenImage21"
    assert enc["inputs"]["prompt"] == "dragon"
    assert enc["inputs"]["negative_prompt"] == "blurry, low quality"
    assert enc["inputs"]["clip"] == ["2", 0]
    # Required by the v3 schema; only sizes reference images, inert for t2i
    assert enc["inputs"]["resolution"] == 1024

    assert wf["6"]["class_type"] == "EmptyLatentImage"
    assert wf["6"]["inputs"]["width"] == 2048 and wf["6"]["inputs"]["height"] == 2048

    ks = wf["7"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["seed"] == 99
    assert ks["inputs"]["steps"] == 45
    assert ks["inputs"]["sampler_name"] == "euler"
    assert ks["inputs"]["scheduler"] == "simple"
    # cfg comes from profiles.toml (1.0 on the official path)
    assert ks["inputs"]["cfg"] == 1
    # Sampling off the shift-patched model, not the raw loader
    assert ks["inputs"]["model"] == ["4", 0]
    assert ks["inputs"]["positive"] == ["5", 0]
    assert ks["inputs"]["negative"] == ["5", 1]

    assert wf["8"]["class_type"] == "VAEDecode"
    assert wf[QWEN21_SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[QWEN21_SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


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


def test_dispatcher_picks_qwen_image_2_1_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_QWEN_IMAGE_2_1))
    assert wf["5"]["class_type"] == "TextEncodeQwenImage21"
    assert wf[QWEN21_SAVE_NODE_ID]["class_type"] == "SaveImage"


# -- OpenFLUX.1 workflow --

def test_openflux1_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",
        seed=99,
        width=1024,
        height=1024,
    ))
    wf = build_openflux1_workflow(req, _cfg(PROFILE_OPENFLUX1))

    # OpenFLUX UNet, loaded in fp8 (the published file is fp8-quantized)
    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "openflux1-v0.1.0-fp8.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"

    # Reuses flux1 CLIP-L + T5 + VAE (same Flux 1 architecture)
    assert wf["2"]["class_type"] == "DualCLIPLoader"
    assert wf["2"]["inputs"]["clip_name1"] == "clip_l.safetensors"
    assert wf["2"]["inputs"]["clip_name2"] == "t5xxl_fp16.safetensors"
    assert wf["2"]["inputs"]["type"] == "flux"
    assert wf["3"]["inputs"]["vae_name"] == "ae.safetensors"

    # Real negative prompt (de-distilled — no ConditioningZeroOut)
    assert wf["4"]["inputs"]["text"] == "dragon"
    assert wf["5"]["class_type"] == "CLIPTextEncode"
    assert wf["5"]["inputs"]["text"] == "blurry, low quality"

    # Real CFG flowing into KSampler
    ks = wf["7"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["seed"] == 99
    assert ks["inputs"]["steps"] == 20
    assert ks["inputs"]["cfg"] == 3.5
    assert ks["inputs"]["sampler_name"] == "euler"
    assert ks["inputs"]["scheduler"] == "simple"
    assert ks["inputs"]["positive"] == ["4", 0]
    assert ks["inputs"]["negative"] == ["5", 0]

    assert wf[OPENFLUX1_SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[OPENFLUX1_SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


def test_dispatcher_picks_openflux1_for_openflux1_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_OPENFLUX1))
    assert wf["1"]["inputs"]["unet_name"] == "openflux1-v0.1.0-fp8.safetensors"
    # Distinguishes from flux1-dev (which uses ConditioningZeroOut at node 5)
    assert wf["5"]["class_type"] == "CLIPTextEncode"


# -- Qwen-Image-Edit Rapid AIO workflow --

def test_qwen_rapid_aio_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",  # ignored (distilled, cfg=1)
        seed=99,
        width=1024,
        height=1024,
    ))
    wf = build_qwen_rapid_aio_workflow(req, _cfg(PROFILE_QWEN_RAPID_AIO))

    # All-in-one checkpoint via CheckpointLoaderSimple (not UNETLoader)
    assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["1"]["inputs"]["ckpt_name"] == "Qwen-Rapid-AIO-NSFW-v23.safetensors"

    # Prompt encoded by the Qwen edit node, fed only clip (pure text-to-image)
    assert wf["2"]["class_type"] == "TextEncodeQwenImageEditPlus"
    assert wf["2"]["inputs"]["prompt"] == "dragon"
    assert wf["2"]["inputs"]["clip"] == ["1", 1]
    assert "image1" not in wf["2"]["inputs"]

    # Guidance-distilled: ConditioningZeroOut negative, cfg pinned to 1
    assert wf["3"]["class_type"] == "ConditioningZeroOut"
    assert wf["4"]["class_type"] == "EmptySD3LatentImage"

    ks = wf["5"]
    assert ks["class_type"] == "KSampler"
    assert ks["inputs"]["seed"] == 99
    assert ks["inputs"]["steps"] == 4
    assert ks["inputs"]["cfg"] == 1
    assert ks["inputs"]["sampler_name"] == "euler_ancestral"
    assert ks["inputs"]["scheduler"] == "beta"
    assert ks["inputs"]["positive"] == ["2", 0]
    assert ks["inputs"]["negative"] == ["3", 0]

    # VAE comes from the checkpoint loader's third output
    assert wf["6"]["class_type"] == "VAEDecode"
    assert wf["6"]["inputs"]["vae"] == ["1", 2]

    assert wf[QWEN_RAPID_SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[QWEN_RAPID_SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


def test_dispatcher_picks_qwen_rapid_aio_for_qwen_rapid_aio_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_QWEN_RAPID_AIO))
    # Only this profile loads an all-in-one checkpoint
    assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
    assert wf[QWEN_RAPID_SAVE_NODE_ID]["class_type"] == "SaveImage"


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


def test_sanitize_name_strips_unsafe_chars():
    assert sanitize_name("../weird name!.png") == "weird_name_.png"
    assert sanitize_name("") == "image"


# --- sdxl-dreamshaper -------------------------------------------------------


_ELF = LoraSpec("dnd/RPGElfXL.safetensors", 0.8, 0.8)
_EARS = LoraSpec("dnd/Elf_Ears_XL.safetensors", 0.6, 0.4)


def test_sdxl_dreamshaper_workflow_shape_without_loras():
    req = ImageRequest.from_json(_sample_payload(negative_prompt="blurry"))
    wf = build_sdxl_dreamshaper_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER))

    assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["1"]["inputs"]["ckpt_name"] == "DreamShaperXL_Turbo_v2_1.safetensors"

    # No LoRAs configured -> no LoraLoader nodes, encoders read the checkpoint.
    assert not [n for n in wf.values() if n["class_type"] == "LoraLoader"]
    assert wf["2"]["inputs"]["clip"] == ["1", 1]
    assert wf["3"]["inputs"]["clip"] == ["1", 1]
    assert wf["5"]["inputs"]["model"] == ["1", 0]

    # Real negative prompt, unlike the guidance-distilled profiles.
    assert wf["2"]["class_type"] == "CLIPTextEncode"
    assert wf["2"]["inputs"]["text"] == "a cat"
    assert wf["3"]["class_type"] == "CLIPTextEncode"
    assert wf["3"]["inputs"]["text"] == "blurry"

    # SDXL uses the 4-channel latent, not EmptySD3LatentImage.
    assert wf["4"]["class_type"] == "EmptyLatentImage"
    assert wf["4"]["inputs"]["width"] == 1024
    assert wf["4"]["inputs"]["height"] == 1024

    sampler = wf["5"]["inputs"]
    assert wf["5"]["class_type"] == "KSampler"
    assert sampler["sampler_name"] == "dpmpp_sde"
    assert sampler["scheduler"] == "karras"
    assert sampler["cfg"] == 2.0
    assert sampler["steps"] == 6
    assert sampler["seed"] == 7

    assert wf["6"]["inputs"]["vae"] == ["1", 2]
    assert wf[SDXL_SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[SDXL_SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


def test_sdxl_dreamshaper_chains_lora_stack_in_order():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_sdxl_dreamshaper_workflow(
        req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=(_ELF, _EARS))
    )

    first, second = str(SDXL_LORA_NODE_BASE), str(SDXL_LORA_NODE_BASE + 1)
    assert wf[first]["class_type"] == "LoraLoader"
    assert wf[first]["inputs"]["lora_name"] == "dnd/RPGElfXL.safetensors"
    assert wf[first]["inputs"]["strength_model"] == 0.8
    assert wf[first]["inputs"]["strength_clip"] == 0.8
    # First LoRA hangs off the checkpoint...
    assert wf[first]["inputs"]["model"] == ["1", 0]
    assert wf[first]["inputs"]["clip"] == ["1", 1]

    # ...and each subsequent one off the previous LoRA.
    assert wf[second]["inputs"]["lora_name"] == "dnd/Elf_Ears_XL.safetensors"
    assert wf[second]["inputs"]["strength_model"] == 0.6
    assert wf[second]["inputs"]["strength_clip"] == 0.4
    assert wf[second]["inputs"]["model"] == [first, 0]
    assert wf[second]["inputs"]["clip"] == [first, 1]

    # Tail of the chain drives both encoders and the sampler; VAE stays on the
    # checkpoint because LoraLoader has no VAE output.
    assert wf["2"]["inputs"]["clip"] == [second, 1]
    assert wf["3"]["inputs"]["clip"] == [second, 1]
    assert wf["5"]["inputs"]["model"] == [second, 0]
    assert wf["6"]["inputs"]["vae"] == ["1", 2]


def test_sdxl_dreamshaper_lora_nodes_do_not_collide_with_core_nodes():
    req = ImageRequest.from_json(_sample_payload())
    stack = tuple(LoraSpec(f"dnd/r{i}.safetensors") for i in range(12))
    wf = build_sdxl_dreamshaper_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=stack))

    assert len([n for n in wf.values() if n["class_type"] == "LoraLoader"]) == 12
    for core in ("1", "2", "3", "4", "5", "6", SDXL_SAVE_NODE_ID):
        assert wf[core]["class_type"] != "LoraLoader"


def test_dispatcher_picks_sdxl_dreamshaper_for_sdxl_dreamshaper_profile():
    req = ImageRequest.from_json(_sample_payload())
    wf = build_workflow(req, _cfg(PROFILE_SDXL_DREAMSHAPER, sdxl_loras=(_ELF,)))

    assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
    assert wf[str(SDXL_LORA_NODE_BASE)]["class_type"] == "LoraLoader"
    assert wf[SDXL_SAVE_NODE_ID]["class_type"] == "SaveImage"


# --- COMFY_SDXL_LORAS parsing ----------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ()),
        ("   ", ()),
        ("a.safetensors", (LoraSpec("a.safetensors", 1.0, 1.0),)),
        # A lone strength applies to both model and CLIP.
        ("a.safetensors:0.7", (LoraSpec("a.safetensors", 0.7, 0.7),)),
        ("a.safetensors:0.7:0.3", (LoraSpec("a.safetensors", 0.7, 0.3),)),
        # Subfolders are normalized to the native separator so the name matches
        # ComfyUI's os.path.relpath-derived filename list (backslashes on Windows).
        (
            " dnd/a.safetensors:0.8 , dnd/b.safetensors ",
            (
                LoraSpec(os.path.join("dnd", "a.safetensors"), 0.8, 0.8),
                LoraSpec(os.path.join("dnd", "b.safetensors"), 1.0, 1.0),
            ),
        ),
        # Trailing/duplicate commas are ignored rather than fatal.
        ("a.safetensors,,", (LoraSpec("a.safetensors", 1.0, 1.0),)),
    ],
)
def test_optional_loras_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("COMFY_SDXL_LORAS", raw)
    assert _optional_loras("COMFY_SDXL_LORAS") == expected


def test_optional_loras_unset_is_empty(monkeypatch):
    monkeypatch.delenv("COMFY_SDXL_LORAS", raising=False)
    assert _optional_loras("COMFY_SDXL_LORAS") == ()


@pytest.mark.parametrize(
    "raw", ["a.safetensors:notanumber", "a.safetensors:1:2:3", ":0.5"]
)
def test_optional_loras_rejects_malformed_entries(monkeypatch, raw):
    monkeypatch.setenv("COMFY_SDXL_LORAS", raw)
    with pytest.raises(ConfigError):
        _optional_loras("COMFY_SDXL_LORAS")


# --- log echo: effective_cfg -----------------------------------------------


@pytest.mark.parametrize(
    "profile,expected_cfg",
    [
        (PROFILE_SDXL_DREAMSHAPER, 2.0),
        (PROFILE_QWEN_RAPID_AIO, 1),
        (PROFILE_FLUX1_DEV, 1),
    ],
)
def test_effective_cfg_reports_the_baked_value(profile, expected_cfg):
    # Request cfg is ignored entirely; the value comes from profiles.toml.
    req = ImageRequest.from_json(_sample_payload(cfg=7.0))
    assert effective_cfg(build_workflow(req, _cfg(profile))) == expected_cfg


def test_effective_cfg_reports_honored_value_for_real_cfg_profiles():
    req = ImageRequest.from_json(_sample_payload())
    assert effective_cfg(build_workflow(req, _cfg(PROFILE_OPENFLUX1))) == 3.5


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
