"""Create the llm-requests + llm-results queues if they don't already exist."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.queue import QueueServiceClient
from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        print("AZURE_STORAGE_CONNECTION_STRING missing from azure_worker/.env", file=sys.stderr)
        return 2
    svc = QueueServiceClient.from_connection_string(conn)

    queues = [
        os.environ.get("LLM_INBOUND_QUEUE", "llm-requests"),
        os.environ.get("LLM_OUTBOUND_QUEUE", "llm-results"),
    ]
    for name in queues:
        try:
            svc.create_queue(name)
            print(f"created queue: {name}")
        except ResourceExistsError:
            print(f"already exists: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
