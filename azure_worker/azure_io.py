"""Azure Storage Queue + Blob helpers for the worker."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)
from azure.storage.queue import (
    BinaryBase64DecodePolicy,
    BinaryBase64EncodePolicy,
    QueueClient,
)

from .config import Config
from .llm_messages import LlmResult
from .messages import ImageResult

log = logging.getLogger(__name__)


# Storage Queue hard cap is 64 KiB after base64; reserve some headroom for
# JSON quoting overhead and the message envelope. Below this, the completion
# is inlined; above it, we spill to Blob Storage and replace the inline text
# with a SAS URL.
LLM_RESULT_INLINE_CAP_BYTES = 50_000


@dataclass
class AzureClients:
    inbound: QueueClient
    outbound: QueueClient
    llm_inbound: QueueClient
    llm_outbound: QueueClient
    blobs: BlobServiceClient
    config: Config


def build_clients(cfg: Config) -> AzureClients:
    # Base64 encoding/decoding makes the queue payload safe regardless of the
    # producer's choice of binary or text-mode bodies.
    def _queue(name: str) -> QueueClient:
        return QueueClient.from_connection_string(
            cfg.storage_connection_string,
            name,
            message_decode_policy=BinaryBase64DecodePolicy(),
            message_encode_policy=BinaryBase64EncodePolicy(),
        )

    blobs = BlobServiceClient.from_connection_string(cfg.storage_connection_string)
    return AzureClients(
        inbound=_queue(cfg.inbound_queue),
        outbound=_queue(cfg.outbound_queue),
        llm_inbound=_queue(cfg.llm_inbound_queue),
        llm_outbound=_queue(cfg.llm_outbound_queue),
        blobs=blobs,
        config=cfg,
    )


def receive_one(clients: AzureClients):
    """Return the next message from the image inbound queue, or None if empty."""
    return _receive_one(clients.inbound, clients.config.visibility_timeout_seconds)


def receive_one_llm(clients: AzureClients):
    """Return the next message from the LLM inbound queue, or None if empty."""
    return _receive_one(clients.llm_inbound, clients.config.visibility_timeout_seconds)


def _receive_one(queue: QueueClient, visibility_timeout: int):
    msgs = queue.receive_messages(
        max_messages=1,
        visibility_timeout=visibility_timeout,
    )
    for msg in msgs:
        return msg
    return None


def message_body_text(msg) -> str:
    """Decode the queue message body to a UTF-8 string regardless of byte/str storage."""
    body = msg.content
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body)


def delete_message(clients: AzureClients, msg) -> None:
    clients.inbound.delete_message(msg.id, msg.pop_receipt)


def delete_llm_message(clients: AzureClients, msg) -> None:
    clients.llm_inbound.delete_message(msg.id, msg.pop_receipt)


def upload_image(clients: AzureClients, local_path: Path, blob_name: str) -> str:
    """Upload `local_path` to the configured container, return a read-only SAS URL."""
    container = clients.blobs.get_container_client(clients.config.blob_container)
    blob = container.get_blob_client(blob_name)
    with local_path.open("rb") as fh:
        blob.upload_blob(
            fh,
            overwrite=True,
            content_settings=ContentSettings(content_type="image/png"),
        )

    expiry = datetime.now(timezone.utc) + timedelta(hours=clients.config.sas_expiry_hours)
    sas_token = generate_blob_sas(
        account_name=clients.blobs.account_name,
        container_name=clients.config.blob_container,
        blob_name=blob_name,
        account_key=_account_key(clients.blobs),
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"{blob.url}?{sas_token}"


def send_result(clients: AzureClients, result: ImageResult) -> None:
    # Outbound queue is configured with BinaryBase64EncodePolicy, which expects bytes.
    clients.outbound.send_message(result.to_json().encode("utf-8"))


def send_llm_result(clients: AzureClients, result: LlmResult) -> None:
    """Send an LLM result, spilling the completion to Blob Storage if it won't fit."""
    body = result.to_json().encode("utf-8")
    if len(body) <= LLM_RESULT_INLINE_CAP_BYTES:
        clients.llm_outbound.send_message(body)
        return

    # Too big: spill the completion (and reasoning, if any) to a blob and
    # re-send with the inline text fields cleared and a SAS URL attached.
    spill_payload = {"completion": result.completion or "", "reasoning": result.reasoning or ""}
    import json as _json

    blob_name = f"llm/{result.job_id}.json"
    blob_url = _upload_json_blob(clients, _json.dumps(spill_payload).encode("utf-8"), blob_name)
    spilled = result.with_blob_spill(blob_name=blob_name, blob_url=blob_url)
    clients.llm_outbound.send_message(spilled.to_json().encode("utf-8"))


def _upload_json_blob(clients: AzureClients, body: bytes, blob_name: str) -> str:
    container = clients.blobs.get_container_client(clients.config.blob_container)
    blob = container.get_blob_client(blob_name)
    blob.upload_blob(
        body,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
    )
    expiry = datetime.now(timezone.utc) + timedelta(hours=clients.config.sas_expiry_hours)
    sas_token = generate_blob_sas(
        account_name=clients.blobs.account_name,
        container_name=clients.config.blob_container,
        blob_name=blob_name,
        account_key=_account_key(clients.blobs),
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"{blob.url}?{sas_token}"


def _account_key(service: BlobServiceClient) -> Optional[str]:
    # `credential` is set when the client was built from a connection string with a key.
    # For Managed Identity / SAS-bound clients this would need a different SAS generator;
    # we intentionally fail loud rather than silently produce broken URLs.
    credential = service.credential
    account_key = getattr(credential, "account_key", None)
    if not account_key:
        raise RuntimeError(
            "Blob client has no account key; SAS generation requires a connection string "
            "with AccountKey, or switch to user-delegation SAS."
        )
    return account_key
