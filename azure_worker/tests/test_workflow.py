"""GPU-free unit tests for the message + workflow layer."""
from __future__ import annotations

import json

import pytest

from azure_worker.messages import (
    ImageRequest,
    ImageResult,
    MessageValidationError,
    sanitize_name,
)
from azure_worker.workflow import SAVE_NODE_ID, build_workflow


UNET = "flux-2-klein-9b-fp8.safetensors"
CLIP = "qwen_3_8b_fp8mixed.safetensors"
VAE = "ae.safetensors"


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


def test_request_round_trip_defaults():
    req = ImageRequest.from_json(_sample_payload())
    assert req.job_id == "abc"
    assert req.name == "test-image"
    assert req.steps == 20
    assert req.negative_prompt == ""


def test_request_rejects_non_multiple_of_16():
    # Flux 2 requires 16-pixel alignment, so 1024+8 should be rejected.
    with pytest.raises(MessageValidationError):
        ImageRequest.from_json(_sample_payload(width=1032))


def test_request_rejects_missing_prompt():
    with pytest.raises(MessageValidationError):
        ImageRequest.from_json(json.dumps({"name": "x", "width": 1024, "height": 1024}))


def test_workflow_wires_request_and_models():
    req = ImageRequest.from_json(_sample_payload(prompt="dragon", seed=99, width=1024, height=768))
    wf = build_workflow(req, unet=UNET, clip=CLIP, vae=VAE)

    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == UNET
    assert wf["1"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"

    assert wf["2"]["class_type"] == "CLIPLoader"
    assert wf["2"]["inputs"]["clip_name"] == CLIP
    assert wf["2"]["inputs"]["type"] == "flux2"

    assert wf["3"]["class_type"] == "VAELoader"
    assert wf["3"]["inputs"]["vae_name"] == VAE

    assert wf["4"]["inputs"]["text"] == "dragon"
    assert wf["5"]["class_type"] == "EmptyFlux2LatentImage"
    assert wf["5"]["inputs"]["width"] == 1024
    assert wf["5"]["inputs"]["height"] == 768

    assert wf["6"]["class_type"] == "Flux2Scheduler"
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["noise_seed"] == 99

    sampler = wf["10"]
    assert sampler["class_type"] == "SamplerCustomAdvanced"
    assert sampler["inputs"]["noise"] == ["8", 0]
    assert sampler["inputs"]["guider"] == ["9", 0]
    assert sampler["inputs"]["sampler"] == ["7", 0]
    assert sampler["inputs"]["sigmas"] == ["6", 0]
    assert sampler["inputs"]["latent_image"] == ["5", 0]

    assert wf[SAVE_NODE_ID]["class_type"] == "SaveImage"
    assert wf[SAVE_NODE_ID]["inputs"]["filename_prefix"] == "test-image"


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
