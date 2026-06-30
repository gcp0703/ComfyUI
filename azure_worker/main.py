"""Worker entry point: poll Azure → run ComfyUI → upload + reply.

Run with:
    python -m azure_worker.main

All configuration is environment-variable driven; see config.py / README.md.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from azure.core.exceptions import ServiceRequestError, ServiceResponseError
from dotenv import load_dotenv

from . import azure_io
from .comfy_runner import ComfyJobError, ComfyRunner
from .config import ConfigError, load_config
from .llm_messages import LlmMessageValidationError, LlmRequest, LlmResult
from .llm_runner import LlmJobError, LlmRunner
from .messages import ImageRequest, ImageResult, MessageValidationError, sanitize_name
from .workflow import build_workflow

log = logging.getLogger("azure_worker")


_shutdown = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _shutdown
        log.info("received signal %s, shutting down after current job", signum)
        _shutdown = True

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


def _process_one_llm(llm: LlmRunner, clients: azure_io.AzureClients) -> bool:
    """Return True if an LLM message was processed (success or failure), False if queue empty."""
    msg = azure_io.receive_one_llm(clients)
    if msg is None:
        return False

    raw_body = azure_io.message_body_text(msg)
    req: Optional[LlmRequest] = None
    try:
        req = LlmRequest.from_json(raw_body)
        log.info(
            "llm job %s model=%s thinking=%s temp=%.2f max_tokens=%d",
            req.job_id, req.model, req.thinking, req.temperature, req.max_tokens,
        )
        if req.system_prompt:
            log.info("llm job %s system_prompt: %s", req.job_id, req.system_prompt)
        log.info("llm job %s user_prompt: %s", req.job_id, req.user_prompt)
        completion = llm.run(req)
        result = LlmResult.success(
            req,
            completion=completion.completion,
            reasoning=completion.reasoning,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            finish_reason=completion.finish_reason,
        )
        azure_io.send_llm_result(clients, result)
        log.info(
            "llm job %s complete (finish=%s, tokens=%s/%s)",
            req.job_id, completion.finish_reason,
            completion.prompt_tokens, completion.completion_tokens,
        )
    except LlmMessageValidationError as e:
        log.warning("invalid llm message (dequeue_count=%s): %s", msg.dequeue_count, e)
        azure_io.send_llm_result(clients, LlmResult.error_for(None, str(e)))
    except (LlmJobError, Exception) as e:  # noqa: BLE001 - we want every failure on the result queue
        log.exception("llm job failed: %s", e)
        azure_io.send_llm_result(clients, LlmResult.error_for(req, str(e)))
    finally:
        try:
            azure_io.delete_llm_message(clients, msg)
        except Exception:
            log.exception("failed to delete inbound llm message %s", msg.id)
    return True


def _process_one(runner: ComfyRunner, clients: azure_io.AzureClients) -> bool:
    """Return True if a message was processed (success or failure), False if queue empty."""
    msg = azure_io.receive_one(clients)
    if msg is None:
        return False

    raw_body = azure_io.message_body_text(msg)
    req: Optional[ImageRequest] = None
    try:
        req = ImageRequest.from_json(raw_body)
        log.info("job %s name=%s %dx%d", req.job_id, req.name, req.width, req.height)
        log.info("job %s prompt: %s", req.job_id, req.prompt)
        if req.negative_prompt:
            log.info("job %s negative_prompt: %s", req.job_id, req.negative_prompt)
        workflow = build_workflow(req, clients.config)
        outputs = runner.run(workflow, prompt_id=req.job_id)
        if not outputs:
            raise ComfyJobError("workflow produced no output files")
        local_path: Path = outputs[0]
        blob_name = f"{sanitize_name(req.name)}/{local_path.name}"
        sas_url = azure_io.upload_image(clients, local_path, blob_name)
        azure_io.send_result(clients, ImageResult.success(req, blob_name, sas_url))
        log.info("job %s complete: %s", req.job_id, blob_name)
    except MessageValidationError as e:
        log.warning("invalid message (dequeue_count=%s): %s", msg.dequeue_count, e)
        azure_io.send_result(clients, ImageResult.error_for(None, str(e)))
    except (ComfyJobError, Exception) as e:  # noqa: BLE001 - we want every failure on the result queue
        log.exception("job failed: %s", e)
        azure_io.send_result(clients, ImageResult.error_for(req, str(e)))
    finally:
        # Always delete: result queue carries the success/error signal.
        # Switch to a dequeue_count check + leave-for-retry once we have a retry policy.
        try:
            azure_io.delete_message(clients, msg)
        except Exception:
            log.exception("failed to delete inbound message %s", msg.id)
    return True


# Transport-level errors raised before any HTTP response is parsed: DNS failures
# (getaddrinfo failed), connection resets, read timeouts. These are transient — a
# laptop sleep, VPN reconnect, or brief network blip — and must NOT take down a
# long-running poll worker. Genuine HttpResponseError subclasses (auth/config
# failures like a bad account key) are deliberately *not* caught: those should
# crash loudly rather than spin in a silent retry loop.
_TRANSIENT_POLL_ERRORS = (ServiceRequestError, ServiceResponseError)


def _run_loop(
    llm_runner: LlmRunner,
    runner: ComfyRunner,
    clients: azure_io.AzureClients,
    poll_interval: float,
    should_stop: Callable[[], bool] = lambda: _shutdown,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll both queues until `should_stop()`, surviving transient network errors.

    LLM-first priority: drain the LLM queue before touching the image queue.
    A flood of LLM requests will starve image jobs, which is the intended
    ordering — image jobs are minutes long, LLM jobs are seconds.
    """
    while not should_stop():
        try:
            if _process_one_llm(llm_runner, clients):
                continue
            if _process_one(runner, clients):
                continue
        except _TRANSIENT_POLL_ERRORS as e:
            # Log and fall through to the backoff sleep; the queue endpoint will
            # resolve again once connectivity returns.
            log.warning("transient Azure error during poll, backing off %.1fs: %s", poll_interval, e)
        sleep(poll_interval)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # azure-* SDKs log every HTTP request/response at INFO by default — drowns the worker's own logs.
    logging.getLogger("azure").setLevel(logging.WARNING)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    log.info("starting ComfyUI runner (profile=%s)", cfg.profile)
    runner = ComfyRunner()
    llm_runner = LlmRunner(cfg.ollama_url, cfg.llm_request_timeout_seconds)
    clients = azure_io.build_clients(cfg)
    _install_signal_handlers()

    log.info(
        "polling llm-queue %r (priority) then image-queue %r every %.1fs",
        cfg.llm_inbound_queue, cfg.inbound_queue, cfg.poll_interval_seconds,
    )
    _run_loop(llm_runner, runner, clients, cfg.poll_interval_seconds)

    log.info("shutdown complete")
    runner.close()
    llm_runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
