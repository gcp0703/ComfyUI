"""End-to-end smoke test for the LLM pipeline.

Enqueues a real message onto llm-requests, drives one iteration of the
worker's _process_one_llm, then reads + verifies the result from llm-results.
Does NOT boot ComfyUI — the LLM path is fully independent.

Run: python -m azure_worker.tools.smoke_llm
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Make `azure_worker.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from azure_worker import azure_io, main as worker_main  # noqa: E402
from azure_worker.config import load_config  # noqa: E402
from azure_worker.llm_runner import LlmRunner  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    cfg = load_config()
    clients = azure_io.build_clients(cfg)
    llm = LlmRunner(cfg.ollama_url, cfg.llm_request_timeout_seconds)

    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "name": "smoke-arithmetic",
        "user_prompt": "What is 17 * 23? Reply with just the number.",
        "model": "qwen3.6:35b-a3b-q4_K_M",
        "thinking": False,
        "temperature": 0.1,
        "max_tokens": 50,
    }
    body = json.dumps(payload).encode("utf-8")

    print(f"\n--- enqueueing job {job_id} ---")
    clients.llm_inbound.send_message(body)

    print("--- processing via _process_one_llm ---")
    t0 = time.time()
    processed = worker_main._process_one_llm(llm, clients)
    elapsed = time.time() - t0
    if not processed:
        print("FAIL: queue was empty (the message we just enqueued didn't appear)")
        return 1
    print(f"processed in {elapsed:.2f}s")

    print("--- draining llm-results to find our job ---")
    deadline = time.time() + 30
    found = None
    while time.time() < deadline and found is None:
        msgs = clients.llm_outbound.receive_messages(max_messages=10, visibility_timeout=30)
        for m in msgs:
            raw = m.content.decode("utf-8") if isinstance(m.content, bytes) else m.content
            data = json.loads(raw)
            if data.get("job_id") == job_id:
                found = data
                clients.llm_outbound.delete_message(m.id, m.pop_receipt)
                break
            # Not ours — leave for someone else (visibility will expire and it returns).
        if found is None:
            time.sleep(1)

    if found is None:
        print("FAIL: did not see a result with our job_id within 30s")
        return 1

    print("--- result ---")
    print(json.dumps(found, indent=2))

    if found["status"] != "success":
        print(f"FAIL: status={found['status']} error={found['error']}")
        return 1
    if not found["completion"]:
        print("FAIL: completion is empty")
        return 1
    # 17 * 23 = 391
    if "391" not in found["completion"]:
        print(f"WARN: expected '391' in completion, got: {found['completion']!r}")

    print(f"\nOK: smoke passed. completion={found['completion']!r}")
    print(f"     tokens: prompt={found['prompt_tokens']} completion={found['completion_tokens']}")
    print(f"     finish_reason={found['finish_reason']}")
    llm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
