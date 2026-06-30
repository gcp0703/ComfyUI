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
