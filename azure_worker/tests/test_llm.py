"""Unit tests for the LLM message contract."""
from __future__ import annotations

import json

import pytest

from azure_worker.llm_messages import (
    LlmMessageValidationError,
    LlmRequest,
    LlmResult,
)


def _payload(**overrides):
    payload = {
        "job_id": 12345,
        "user_prompt": "explain photosynthesis",
        "model": "Qwen/Qwen3-8B-Instruct",
    }
    payload.update(overrides)
    return json.dumps(payload)


# -- Request defaults --

def test_request_round_trip_defaults():
    req = LlmRequest.from_json(_payload())
    assert req.job_id == "12345"  # int coerced to string
    assert req.model == "Qwen/Qwen3-8B-Instruct"
    assert req.system_prompt == ""
    assert req.thinking is False
    assert req.temperature == 1.0
    assert req.max_tokens == 2048
    assert req.name == ""


def test_request_generates_job_id_when_missing():
    payload = json.dumps({"user_prompt": "x", "model": "m"})
    req = LlmRequest.from_json(payload)
    # uuid4 string form
    assert len(req.job_id) == 36
    assert req.job_id.count("-") == 4


# -- job_id polymorphism --

def test_job_id_accepts_int():
    req = LlmRequest.from_json(_payload(job_id=42))
    assert req.job_id == "42"


def test_job_id_accepts_string():
    req = LlmRequest.from_json(_payload(job_id="abc-123"))
    assert req.job_id == "abc-123"


def test_job_id_rejects_other_types():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(job_id=[1, 2, 3]))


# -- thinking coercion --

@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False),
    ("yes", True), ("YES", True), ("y", True), ("true", True), ("1", True), ("on", True),
    ("no", False), ("NO", False), ("n", False), ("false", False), ("0", False), ("off", False),
    ("", False), (None, False),
])
def test_thinking_coercion(raw, expected):
    req = LlmRequest.from_json(_payload(thinking=raw))
    assert req.thinking is expected


def test_thinking_rejects_garbage():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(thinking="maybe"))


# -- temperature coercion --

@pytest.mark.parametrize("raw,expected", [
    (0.3, 0.3), (0, 0.0), (2, 2.0),
    ("0.7", 0.7), ("1.5", 1.5),
])
def test_temperature_coercion(raw, expected):
    req = LlmRequest.from_json(_payload(temperature=raw))
    assert req.temperature == pytest.approx(expected)


def test_temp_alias_accepted():
    """Producer's 'temp' field is accepted as an alias for 'temperature'."""
    payload = json.dumps({
        "user_prompt": "x", "model": "m", "temp": "0.42",
    })
    req = LlmRequest.from_json(payload)
    assert req.temperature == pytest.approx(0.42)


def test_temperature_out_of_range():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(temperature=3.0))
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(temperature=-0.1))


def test_temperature_non_numeric_string():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(temperature="hot"))


# -- max_tokens --

def test_max_tokens_accepts_int_and_string():
    req = LlmRequest.from_json(_payload(max_tokens=512))
    assert req.max_tokens == 512
    req2 = LlmRequest.from_json(_payload(max_tokens="1024"))
    assert req2.max_tokens == 1024


def test_max_tokens_out_of_range():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(max_tokens=0))
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(max_tokens=100_000))


# -- required fields --

def test_user_prompt_required():
    payload = json.dumps({"model": "m"})
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(payload)


def test_user_prompt_must_be_nonempty():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(user_prompt="   "))


def test_model_required():
    payload = json.dumps({"user_prompt": "x"})
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(payload)


def test_rejects_non_object():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json("[1, 2, 3]")


def test_rejects_bad_json():
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json("{not-json")


# -- prompt length cap --

def test_user_prompt_length_cap():
    huge = "x" * 32_001
    with pytest.raises(LlmMessageValidationError):
        LlmRequest.from_json(_payload(user_prompt=huge))


# -- Result serialization --

def test_result_success_serializes():
    req = LlmRequest.from_json(_payload())
    result = LlmResult.success(
        req,
        completion="hello",
        reasoning=None,
        prompt_tokens=10,
        completion_tokens=5,
        finish_reason="stop",
    )
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "success"
    assert parsed["completion"] == "hello"
    assert parsed["reasoning"] is None
    assert parsed["prompt_tokens"] == 10
    assert parsed["completion_tokens"] == 5
    assert parsed["finish_reason"] == "stop"
    assert parsed["blob_url"] is None
    assert parsed["error"] is None
    assert parsed["model"] == "Qwen/Qwen3-8B-Instruct"


def test_result_error_for_no_request():
    result = LlmResult.error_for(None, "totally malformed json")
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "error"
    assert parsed["job_id"] == "unknown"
    assert parsed["model"] == "unknown"
    assert parsed["error"] == "totally malformed json"
    assert parsed["completion"] is None


def test_result_error_for_request():
    req = LlmRequest.from_json(_payload())
    result = LlmResult.error_for(req, "vLLM HTTP 500")
    parsed = json.loads(result.to_json())
    assert parsed["status"] == "error"
    assert parsed["job_id"] == "12345"
    assert parsed["model"] == "Qwen/Qwen3-8B-Instruct"
    assert parsed["error"] == "vLLM HTTP 500"


def test_result_with_blob_spill():
    req = LlmRequest.from_json(_payload())
    result = LlmResult.success(
        req, completion="a" * 100_000, reasoning=None,
        prompt_tokens=10, completion_tokens=99, finish_reason="length",
    )
    spilled = result.with_blob_spill("llm/12345.json", "https://example/llm/12345.json?sas")
    parsed = json.loads(spilled.to_json())
    assert parsed["completion"] is None
    assert parsed["reasoning"] is None
    assert parsed["blob_name"] == "llm/12345.json"
    assert parsed["blob_url"] == "https://example/llm/12345.json?sas"
    # Metadata preserved
    assert parsed["finish_reason"] == "length"
    assert parsed["completion_tokens"] == 99


# -- LLM runner payload shape (no network) --

def test_llm_runner_builds_correct_payload():
    """Verify the runner posts Ollama's native /api/chat body shape."""
    from azure_worker import llm_runner

    captured = {}

    class _FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "model": "Qwen/Qwen3-8B-Instruct",
                "message": {"content": "answer", "thinking": "thinking..."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
            }

    class _FakeSession:
        def post(self, url, json, timeout):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse()

        def close(self):
            pass

    runner = llm_runner.LlmRunner("http://localhost:11434", 300)
    runner._session = _FakeSession()

    req = LlmRequest.from_json(_payload(
        system_prompt="be brief",
        user_prompt="hi",
        thinking=True,
        temperature=0.5,
        max_tokens=100,
    ))
    completion = runner.run(req)

    assert captured["url"] == "http://localhost:11434/api/chat"
    body = captured["json"]
    assert body["model"] == "Qwen/Qwen3-8B-Instruct"
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert body["stream"] is False
    assert body["think"] is True
    assert body["options"] == {"num_predict": 100, "temperature": 0.5}

    assert completion.completion == "answer"
    assert completion.reasoning == "thinking..."
    assert completion.prompt_tokens == 12
    assert completion.completion_tokens == 3
    assert completion.finish_reason == "stop"


def test_llm_runner_drops_empty_system_prompt():
    from azure_worker import llm_runner

    captured = {}

    class _FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": "ok"}, "done_reason": "stop"}

    class _FakeSession:
        def post(self, url, json, timeout):  # noqa: A002
            captured["json"] = json
            return _FakeResponse()

        def close(self):
            pass

    runner = llm_runner.LlmRunner("http://localhost:11434", 300)
    runner._session = _FakeSession()

    req = LlmRequest.from_json(_payload())  # no system_prompt; thinking defaults False
    runner.run(req)

    assert captured["json"]["messages"] == [{"role": "user", "content": "explain photosynthesis"}]
    assert captured["json"]["think"] is False


def test_llm_runner_propagates_http_error():
    from azure_worker import llm_runner

    class _FakeResponse:
        status_code = 404
        text = '{"error": "model \'foo\' not found, try pulling it first"}'

    class _FakeSession:
        def post(self, *a, **kw):
            return _FakeResponse()

        def close(self):
            pass

    runner = llm_runner.LlmRunner("http://localhost:11434", 300)
    runner._session = _FakeSession()
    req = LlmRequest.from_json(_payload())

    with pytest.raises(llm_runner.LlmJobError) as ei:
        runner.run(req)
    assert "404" in str(ei.value)
    assert "not found" in str(ei.value)
