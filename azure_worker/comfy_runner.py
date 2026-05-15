"""In-process ComfyUI invocation.

Bootstraps ComfyUI exactly the same way `python main.py` does, including the
background `prompt_worker` thread, but never starts the aiohttp web server.
Jobs are submitted directly to `prompt_server.prompt_queue` (same path the
POST /prompt HTTP handler uses) and we poll the queue's history for results.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


class ComfyJobError(RuntimeError):
    pass


class ComfyRunner:
    """Wraps a ComfyUI PromptServer + prompt_worker for synchronous one-at-a-time use."""

    def __init__(self) -> None:
        self._bootstrap()
        # Imports must happen after _bootstrap so the ComfyUI repo root is on sys.path
        # and all module-level side effects in main.py have run.
        import main as comfy_main  # noqa: WPS433 - intentional late import
        import execution  # noqa: WPS433

        self._comfy_main = comfy_main
        self._execution = execution

        _loop, prompt_server, _start_all = comfy_main.start_comfyui()
        self._server = prompt_server
        self._queue = prompt_server.prompt_queue
        # Dedicated loop for awaiting validate_prompt from this synchronous worker.
        # We never run it on the loop that start_comfyui() created, since that loop
        # is reserved for the aiohttp server we are deliberately NOT starting.
        self._loop = asyncio.new_event_loop()

    @staticmethod
    def _bootstrap() -> None:
        """Add the ComfyUI repo root to sys.path and neutralize CLI args.

        ComfyUI's `comfy.cli_args` calls `parser.parse_args()` at import time when
        `comfy.options.args_parsing` is True. main.py enables that flag, so any
        unrecognised argv would crash us. We trim argv to the program name only;
        the worker takes all configuration through environment variables.
        """
        repo_root = Path(__file__).resolve().parent.parent
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        # Working directory matters: ComfyUI resolves models/, output/, etc.
        # relative to the repo root via folder_paths.
        os.chdir(repo_root_str)
        sys.argv = [sys.argv[0]]

    def run(self, workflow: dict, prompt_id: str, timeout_seconds: float = 600.0) -> List[Path]:
        """Submit `workflow` to ComfyUI and block until the SaveImage node has written its files.

        Returns absolute paths to the generated image(s) on disk.
        """
        valid = self._loop.run_until_complete(
            self._execution.validate_prompt(prompt_id, workflow, None)
        )
        if not valid[0]:
            raise ComfyJobError(f"workflow failed validation: {valid[1]}")
        outputs_to_execute = valid[2]

        # Queue tuple shape mirrors server.py POST /prompt (see server.py:956):
        # (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
        extra_data = {"create_time": int(time.time() * 1000)}
        number = self._server.number
        self._server.number += 1
        self._queue.put((number, prompt_id, workflow, extra_data, outputs_to_execute, {}))

        deadline = time.monotonic() + timeout_seconds
        while True:
            history = self._queue.get_history(prompt_id=prompt_id)
            entry = history.get(prompt_id)
            if entry is not None and entry.get("status") is not None:
                status = entry["status"]
                if status.get("status_str") != "success" or not status.get("completed"):
                    messages = status.get("messages") or []
                    raise ComfyJobError(f"workflow execution failed: {messages}")
                paths = self._comfy_main._collect_output_absolute_paths(entry)
                return [Path(p) for p in paths]
            if time.monotonic() > deadline:
                raise ComfyJobError(f"workflow {prompt_id} did not complete within {timeout_seconds}s")
            time.sleep(0.25)

    def close(self) -> None:
        try:
            self._loop.close()
        except Exception:
            log.exception("error closing comfy_runner asyncio loop")
