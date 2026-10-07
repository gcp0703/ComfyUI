"""Resilience tests for the worker poll loop (no network, no GPU)."""
from __future__ import annotations

from azure.core.exceptions import (
    ClientAuthenticationError,
    ServiceRequestError,
    ServiceResponseError,
)

from azure_worker import main


def _stop_after(n):
    """Return a should_stop() callable that allows exactly n loop iterations."""
    state = {"i": 0}

    def should_stop():
        state["i"] += 1
        return state["i"] > n

    return should_stop


def test_loop_survives_transient_request_error(monkeypatch):
    """A DNS/connection failure on poll (ServiceRequestError) must not kill the worker."""
    calls = {"n": 0}

    def boom(_llm, _clients):
        calls["n"] += 1
        raise ServiceRequestError("getaddrinfo failed")

    monkeypatch.setattr(main, "_process_one_llm", boom)
    monkeypatch.setattr(main, "_process_one", lambda *_: False)

    sleeps = []
    main._run_loop(
        None, None, None,
        poll_interval=0.0,
        should_stop=_stop_after(3),
        sleep=sleeps.append,
    )

    assert calls["n"] == 3            # retried each iteration instead of crashing
    assert sleeps == [0.0, 0.0, 0.0]  # backed off after every transient failure


def test_loop_survives_transient_response_error(monkeypatch):
    """A read-timeout (ServiceResponseError) is also transient and must be survived."""
    def boom(_llm, _clients):
        raise ServiceResponseError("read timed out")

    monkeypatch.setattr(main, "_process_one_llm", boom)
    monkeypatch.setattr(main, "_process_one", lambda *_: False)

    sleeps = []
    main._run_loop(None, None, None, poll_interval=0.0,
                   should_stop=_stop_after(2), sleep=sleeps.append)

    assert sleeps == [0.0, 0.0]


def test_loop_propagates_auth_error(monkeypatch):
    """A genuine config/auth failure should crash loudly, not loop forever."""
    def boom(_llm, _clients):
        raise ClientAuthenticationError("403 forbidden — bad account key")

    monkeypatch.setattr(main, "_process_one_llm", boom)
    monkeypatch.setattr(main, "_process_one", lambda *_: False)

    raised = False
    try:
        main._run_loop(None, None, None, poll_interval=0.0,
                       should_stop=_stop_after(1), sleep=lambda _x: None)
    except ClientAuthenticationError:
        raised = True
    assert raised


def test_loop_skips_sleep_when_work_was_done(monkeypatch):
    """When a job is processed, the loop polls again immediately (no idle sleep)."""
    monkeypatch.setattr(main, "_process_one_llm", lambda *_: True)
    monkeypatch.setattr(main, "_process_one", lambda *_: False)

    sleeps = []
    main._run_loop(None, None, None, poll_interval=5.0,
                   should_stop=_stop_after(3), sleep=sleeps.append)

    assert sleeps == []  # never idled because there was always work


def test_loop_sleeps_when_both_queues_empty(monkeypatch):
    """When both queues are empty, the loop idles for the poll interval."""
    monkeypatch.setattr(main, "_process_one_llm", lambda *_: False)
    monkeypatch.setattr(main, "_process_one", lambda *_: False)

    sleeps = []
    main._run_loop(None, None, None, poll_interval=5.0,
                   should_stop=_stop_after(2), sleep=sleeps.append)

    assert sleeps == [5.0, 5.0]


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
