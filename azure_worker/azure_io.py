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
from .messages import ImageResult

log = logging.getLogger(__name__)


@dataclass
class AzureClients:
    inbound: QueueClient
    outbound: QueueClient
    blobs: BlobServiceClient
    config: Config


def build_clients(cfg: Config) -> AzureClients:
    # Base64 encoding/decoding makes the queue payload safe regardless of the
    # producer's choice of binary or text-mode bodies.
    inbound = QueueClient.from_connection_string(
        cfg.storage_connection_string,
        cfg.inbound_queue,
        message_decode_policy=BinaryBase64DecodePolicy(),
        message_encode_policy=BinaryBase64EncodePolicy(),
    )
    outbound = QueueClient.from_connection_string(
        cfg.storage_connection_string,
        cfg.outbound_queue,
        message_decode_policy=BinaryBase64DecodePolicy(),
        message_encode_policy=BinaryBase64EncodePolicy(),
    )
    blobs = BlobServiceClient.from_connection_string(cfg.storage_connection_string)
    return AzureClients(inbound=inbound, outbound=outbound, blobs=blobs, config=cfg)


def receive_one(clients: AzureClients):
    """Return the next message from the inbound queue, or None if empty."""
    msgs = clients.inbound.receive_messages(
        max_messages=1,
        visibility_timeout=clients.config.visibility_timeout_seconds,
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
