"""Unit tests for azure_io.send_result's size guard (no network, no Azure SDK calls).

ImageRequest.from_json's own byte check (MAX_REQUEST_BYTES = 46 KiB) already
leaves ~2 KiB of headroom for the result's added fields, and measured
overhead (render block, blob_url, etc.) in this repo's fixtures is only a few
hundred bytes — so a prompt that legitimately passes request validation does
not, by itself, push a *correctly UTF-8-serialized* result over the 48 KiB
cap (see test_main_loop.py's end-to-end tests for that boundary). These tests
instead build ImageResult objects directly, bypassing from_json, so the
guard's truncation behavior can be exercised deterministically regardless of
that margin.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from azure_worker import azure_io
from azure_worker.messages import ImageResult

_RESULT_KEYS = {
    "job_id", "name", "status", "prompt", "negative_prompt", "width", "height", "seed",
    "render", "warnings", "blob_url", "blob_name", "error",
}


class _FakeQueue:
    def __init__(self):
        self.sent: list[bytes] = []

    def send_message(self, body: bytes) -> None:
        self.sent.append(body)


def _result(prompt: str = "a cat", negative_prompt: str = "") -> ImageResult:
    return ImageResult(
        job_id="j1",
        name="x",
        status="success",
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=2048,
        height=2048,
        seed=1,
        render={"profile": "qwen-image-2.1", "model": "m.safetensors", "loras": []},
        warnings=[],
        blob_name="x/y.png",
        blob_url="https://blob/x/y.png?sas",
    )


def test_send_result_passes_through_a_fitting_result_unmodified():
    outbound = _FakeQueue()
    clients = SimpleNamespace(outbound=outbound)

    azure_io.send_result(clients, _result("a cat"))

    assert len(outbound.sent) == 1
    body = outbound.sent[0]
    assert len(base64.b64encode(body)) <= 65_536
    parsed = json.loads(body.decode("utf-8"))
    assert set(parsed) == _RESULT_KEYS
    assert parsed["warnings"] == []
    assert parsed["prompt"] == "a cat"


def test_send_result_truncates_an_oversize_prompt():
    outbound = _FakeQueue()
    clients = SimpleNamespace(outbound=outbound)
    huge_prompt = "中" * 40_000  # 120,000 raw UTF-8 bytes on its own

    azure_io.send_result(clients, _result(huge_prompt))

    assert len(outbound.sent) == 1
    body = outbound.sent[0]
    assert len(body) <= azure_io.IMAGE_RESULT_MAX_BYTES
    assert len(base64.b64encode(body)) <= 65_536
    parsed = json.loads(body.decode("utf-8"))
    assert set(parsed) == _RESULT_KEYS
    assert parsed["warnings"] == ["prompt echo truncated to fit the result message"]
    assert len(parsed["prompt"]) < len(huge_prompt)
    assert parsed["status"] == "success"
    assert parsed["job_id"] == "j1"


def test_send_result_truncates_negative_prompt_after_prompt_is_emptied():
    outbound = _FakeQueue()
    clients = SimpleNamespace(outbound=outbound)
    huge = "中" * 40_000

    azure_io.send_result(clients, _result(prompt=huge, negative_prompt=huge))

    assert len(outbound.sent) == 1
    body = outbound.sent[0]
    assert len(body) <= azure_io.IMAGE_RESULT_MAX_BYTES
    parsed = json.loads(body.decode("utf-8"))
    assert set(parsed) == _RESULT_KEYS
    # Exactly one truncation warning no matter how many fields had to shrink.
    assert parsed["warnings"] == ["prompt echo truncated to fit the result message"]
    assert parsed["prompt"] == ""
    assert len(parsed["negative_prompt"]) < len(huge)
