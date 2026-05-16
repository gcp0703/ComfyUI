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
from typing import Optional

from dotenv import load_dotenv

from . import azure_io
from .comfy_runner import ComfyJobError, ComfyRunner
from .config import ConfigError, load_config
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
    clients = azure_io.build_clients(cfg)
    _install_signal_handlers()

    log.info("polling queue %r every %.1fs", cfg.inbound_queue, cfg.poll_interval_seconds)
    while not _shutdown:
        processed = _process_one(runner, clients)
        if not processed:
            time.sleep(cfg.poll_interval_seconds)

    log.info("shutdown complete")
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
