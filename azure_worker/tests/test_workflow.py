"""GPU-free unit tests for the message + workflow layer."""
from __future__ import annotations

import json

import pytest

from azure_worker.config import (
    PROFILE_CHROMA1,
    PROFILE_FLUX1_DEV,
    PROFILE_FLUX2_KLEIN,
    PROFILE_FLUXED_UP,
    PROFILE_OPENFLUX1,
    PROFILE_QWEN_IMAGE_2512,
    PROFILE_QWEN_RAPID_AIO,
    Config,
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
    QWEN_RAPID_SAVE_NODE_ID,
    build_chroma1_workflow,
    build_flux1_dev_workflow,
    build_flux2_klein_workflow,
    build_fluxed_up_workflow,
    build_openflux1_workflow,
    build_qwen_image_2512_workflow,
    build_qwen_rapid_aio_workflow,
    build_workflow,
)


def _cfg(profile: str) -> Config:
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
        openflux_unet="openflux1-v0.1.0-fp8.safetensors",
        qwen_rapid_checkpoint="Qwen-Rapid-AIO-NSFW-v23.safetensors",
        llm_inbound_queue="llm-requests",
        llm_outbound_queue="llm-results",
        ollama_url="http://localhost:11434",
    )


def _sample_payload(**overrides):
    payload = {
        "job_id": "abc",
        "name": "test-image",
        "prompt": "a cat",
        "width": 1024,
        "height": 1024,
        "seed": 7,
    }
    payload.update(overrides)
    return json.dumps(payload)


# -- Message validation --

def test_request_round_trip_defaults():
    req = ImageRequest.from_json(_sample_payload())
    assert req.job_id == "abc"
    assert req.name == "test-image"
    assert req.steps == 20
    assert req.negative_prompt == ""


def test_request_rejects_non_multiple_of_16():
    with pytest.raises(MessageValidationError):
        ImageRequest.from_json(_sample_payload(width=1032))


def test_request_rejects_missing_prompt():
    with pytest.raises(MessageValidationError):
        ImageRequest.from_json(json.dumps({"name": "x", "width": 1024, "height": 1024}))


# -- Flux 1 dev workflow --

def test_flux1_workflow_shape():
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
    assert wf["6"]["inputs"]["width"] == 1024 and wf["6"]["inputs"]["height"] == 768

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
        steps=26,
        cfg=3.5,
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
        steps=30,
        cfg=4.0,
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
    assert ks["inputs"]["steps"] == 30
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


# -- OpenFLUX.1 workflow --

def test_openflux1_workflow_shape():
    req = ImageRequest.from_json(_sample_payload(
        prompt="dragon",
        negative_prompt="blurry, low quality",
        seed=99,
        width=1024,
        height=1024,
        steps=20,
        cfg=3.5,
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
        steps=4,
        cfg=8.0,  # ignored
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


# -- Result message --

def test_result_success_serializes():
    req = ImageRequest.from_json(_sample_payload())
    result = ImageResult.success(req, blob_name="x/y.png", blob_url="https://example/y.png?sas")
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "success"
    assert parsed["blob_name"] == "x/y.png"
    assert parsed["error"] is None


def test_result_error_when_request_was_invalid():
    result = ImageResult.error_for(None, "bad json")
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "error"
    assert parsed["error"] == "bad json"
    assert parsed["blob_url"] is None


def test_sanitize_name_strips_unsafe_chars():
    assert sanitize_name("../weird name!.png") == "weird_name_.png"
    assert sanitize_name("") == "image"
