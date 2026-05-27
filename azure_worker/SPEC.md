# Worker Queue — Client Specification

Contract for any application that wants to **request an image generation** or
**run an LLM chat completion**, by talking to the four Azure Storage Queues
backing the worker. This document is self-contained: a developer writing a
client should not need to read anything else.

The worker exposes two independent pipelines:
- **Image pipeline** — `image-requests` → ComfyUI → Blob → `image-results` (§5–§6).
- **LLM pipeline** — `llm-requests` → Ollama (native `/api/chat` HTTP) → `llm-results` (§13–§14).
  The LLM queue is **polled first every iteration**; image jobs only run when
  the LLM queue is empty.

---

## 1. Architecture in one paragraph

A producer enqueues a JSON request describing an image (prompt + size) onto
the **inbound** Storage Queue. A pool of one or more ComfyUI workers polls
that queue, runs the generation, uploads the resulting PNG to a Blob Storage
container, and enqueues a JSON result message — containing a short-lived
download URL — onto the **outbound** Storage Queue. The consumer reads the
result message, downloads the PNG via the embedded URL, and deletes the
result message.

The worker deletes every inbound message it handles (success or failure).
Every inbound message produces exactly one outbound message.

---

## 2. Resources

These are the deployed resource names today. Treat the account/queue/container
names as configuration; don't hard-code them past a config file.

| Resource | Name | Endpoint |
|---|---|---|
| Storage account | `nomadimagegen` | `*.core.windows.net` |
| Image inbound queue | `image-requests` | `https://nomadimagegen.queue.core.windows.net/image-requests` |
| Image outbound queue | `image-results` | `https://nomadimagegen.queue.core.windows.net/image-results` |
| LLM inbound queue | `llm-requests` | `https://nomadimagegen.queue.core.windows.net/llm-requests` |
| LLM outbound queue | `llm-results` | `https://nomadimagegen.queue.core.windows.net/llm-results` |
| Blob container (PNGs + LLM overflow) | `generated-images` | `https://nomadimagegen.blob.core.windows.net/generated-images/` |

Region: `centralus`. Redundancy: `Standard_LRS`. TLS 1.2 minimum, HTTPS only,
public blob access disabled.

---

## 3. Authentication

The producer and consumer need different permissions and can authenticate
independently. Pick **one** option from each row.

### 3.1 Producer (sends to `image-requests`)

| Mechanism | Permission required | Notes |
|---|---|---|
| Connection string with `AccountKey` | full account access | Simple. Treat the key like a password — never embed in a client app. |
| SAS token | `Add` on `image-requests` | Recommended for client apps; mint server-side, hand out short-lived tokens. |
| Managed identity (Entra) | `Storage Queue Data Message Sender` on the queue | Best for first-party services running on Azure. |

### 3.2 Consumer (reads from `image-results`)

| Mechanism | Permission required | Notes |
|---|---|---|
| Connection string with `AccountKey` | full account access | Simple. |
| SAS token | `Read` + `Process` on `image-results` | Use this for headless integrations. |
| Managed identity (Entra) | `Storage Queue Data Message Processor` on the queue | Recommended. |

### 3.3 Downloading the generated PNG

**No authentication required on the consumer side.** Every successful result
message embeds a pre-signed SAS URL with `read` permission and a fixed expiry
(24 hours by default). Issue a plain HTTPS `GET` against the URL.

---

## 4. Message encoding rules

Both queues are configured with **base64 message encoding**. This is the
behavior of `BinaryBase64EncodePolicy` / `BinaryBase64DecodePolicy` in the
official Azure SDK.

A message body on the wire is:

```
base64( utf8_bytes( json_string ) )
```

If you use the official SDK with the policies above, encoding/decoding is
automatic — pass `bytes` to `send_message`, and `receive_messages` returns
already-decoded `bytes`. If you call the REST API directly, you must
base64-encode the JSON yourself before `PUT` and base64-decode after `GET`.

Storage Queue cap is **64 KiB per message** (after base64). With ~33% base64
overhead, the practical JSON payload limit is **~48 KiB**. Normal requests
are well under 2 KiB.

---

## 5. Request schema (producer → `image-requests`)

A single JSON object:

```json
{
  "job_id": "string, optional",
  "name": "string, required",
  "prompt": "string, required",
  "negative_prompt": "string, optional, default \"\"",
  "width": "integer, required",
  "height": "integer, required",
  "seed": "integer, optional",
  "steps": "integer, optional, default 20",
  "cfg": "number, optional, default 7.0"
}
```

### Field rules

| Field | Type | Required | Constraints |
|---|---|---|---|
| `job_id` | string | no | If omitted, the worker generates a UUID. **Provide your own UUID** if you want to correlate requests with results — see §8. |
| `name` | string | yes | Non-empty. Becomes the filename prefix for the generated PNG (sanitized — non-alphanumeric chars are replaced with `_`). Does not need to be unique. |
| `prompt` | string | yes | Non-empty. Max 4000 chars. |
| `negative_prompt` | string | no | Max 4000 chars. **Honored on `chroma1`, `qwen-image-2512`, and `openflux1`** (all three support real CFG with a real negative branch). **Ignored** on `flux1-dev` and `fluxed-up` (both use `ConditioningZeroOut`) and `flux2-klein` (uses `BasicGuider` — no negative path). |
| `width` | integer | yes | 64 ≤ w ≤ 4096, **multiple of 16**. |
| `height` | integer | yes | 64 ≤ h ≤ 4096, **multiple of 16**. |
| `seed` | integer | no | 64-bit unsigned. If omitted, the worker picks a random seed and returns it in the result so the run is reproducible. |
| `steps` | integer | no | 1 ≤ steps ≤ 200. Default 20. Both Flux profiles work well at 20; `chroma1`'s recommended baseline is **26** (workable from 20 for iteration up to 35–50 for finer detail). |
| `cfg` | number | no | 0.0 ≤ cfg ≤ 30.0. Default 7.0. **Honored on `chroma1`** (recommend 3.5; workable 3.5–7), **`qwen-image-2512`** (recommend 4.0), and **`openflux1`** (recommend ≈3.5). **Ignored** on `flux1-dev`, `flux2-klein`, and `fluxed-up` (all guidance-distilled). |

### Example

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "prompt": "a serene sunset over snowy mountains, oil painting, dramatic lighting",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

### Validation failures

If the message is unparseable JSON, missing a required field, has an
out-of-range value, or has dimensions not aligned to 16, the worker:
- Sends a result message with `status="error"` and a `error` string explaining what was wrong.
- Deletes the inbound message (no retries).
- Does not run the model.

If `job_id` couldn't be recovered (e.g. the JSON was completely malformed),
the error result has `job_id="unknown"` and `name="unknown"` — see §6.

---

## 6. Result schema (consumer ← `image-results`)

A single JSON object. Always contains the same set of keys; `status` tells you
whether to look at `blob_url` or `error`.

```json
{
  "job_id": "string",
  "name": "string",
  "status": "success" | "error",
  "prompt": "string",
  "width": "integer",
  "height": "integer",
  "seed": "integer",
  "blob_url": "string | null",
  "blob_name": "string | null",
  "error": "string | null"
}
```

### Success result

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "status": "success",
  "prompt": "a serene sunset over snowy mountains, oil painting, dramatic lighting",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "blob_url": "https://nomadimagegen.blob.core.windows.net/generated-images/sunset-mountains/sunset-mountains_00001_.png?se=...&sig=...",
  "blob_name": "sunset-mountains/sunset-mountains_00001_.png",
  "error": null
}
```

- `blob_url` is a pre-signed read-only HTTPS URL. Valid for 24 hours from
  generation. Download with a plain `GET`; no headers required.
- `blob_name` is the path within the `generated-images` container, useful if
  you authenticate to the blob service yourself and don't want the SAS.
- `seed` reflects the seed the worker actually used — important if the request
  omitted it.
- `width` / `height` echo the request; the PNG is exactly those dimensions.

### Error result

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "sunset-mountains",
  "status": "error",
  "prompt": "a serene sunset over snowy mountains",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "blob_url": null,
  "blob_name": null,
  "error": "workflow execution failed: [...]"
}
```

Common `error` messages:
- `"'width'=99 must be a multiple of 16"` — validation failure
- `"'prompt' is required and must be a non-empty string"` — schema failure
- `"workflow execution failed: ..."` — runtime failure inside ComfyUI (OOM, model load error, etc.)
- `"workflow failed validation: ..."` — workflow couldn't even start
- `"workflow %s did not complete within %ds"` — generation timed out (default 600s)

If the original message was so malformed that no fields could be recovered,
the error result has `job_id="unknown"`, `name="unknown"`, empty `prompt`,
zeros for `width`/`height`/`seed`. Use `job_id="unknown"` as your signal that
correlation is impossible for that one.

---

## 7. Operational guarantees

| Property | Value |
|---|---|
| **Result-per-request** | Exactly one outbound message per inbound message handled. |
| **Ordering** | Not guaranteed. Don't rely on results arriving in submission order. |
| **At-most-once delivery to the worker** | The worker deletes the inbound message after processing — no retries. If your producer dropped the request, it's lost. (We can add a retry/DLQ policy later if needed.) |
| **Duplicate suppression** | None. If you enqueue the same `job_id` twice, you get two result messages. |
| **Concurrency** | Single-threaded per worker process. Throughput scales by running more worker processes. |
| **Cold start** | First request after a worker boot takes 30-90 s extra to load models. Subsequent requests on the same worker are warm. |
| **Generation latency (warm)** | ~13-30 s for 1024×1024 at 20 steps on an RTX 5090. Scales linearly with pixel count and steps. |
| **Result retention** | Storage Queue retains messages 7 days by default. Consumer should poll regularly. |
| **Blob retention** | Indefinite. SAS URL expires in 24 h. To extend, either the consumer needs container-level credentials, or `SAS_EXPIRY_HOURS` must be bumped server-side. |

---

## 8. Correlating requests and results

The worker preserves `job_id` from the request to the result message. **Always
generate a UUID client-side and set `job_id`** — that's your only durable
correlation key. Don't rely on submission order, message timestamps, or
filename prefixes.

If you need to wait for a specific result, the recommended pattern is:

1. Generate `job_id = uuid.uuid4()` and enqueue the request.
2. Receive messages from the outbound queue in a loop.
3. For each message, parse `job_id`. If it's yours, process it; otherwise put
   it back (don't delete) so another consumer / your other in-flight requests
   can pick it up.

For a heavy workload with many in-flight requests, run a single dispatcher
that drains the outbound queue and fans results out by `job_id` to waiters —
don't have N consumers all peeking and putting back, that's pathological under
contention.

---

## 9. Reference: Python producer (≈30 lines)

Requires `azure-storage-queue >= 12.8`.

```python
import json
import uuid
from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy

CONNECTION_STRING = "..."  # or use SAS / managed identity

queue = QueueClient.from_connection_string(
    CONNECTION_STRING,
    "image-requests",
    message_encode_policy=BinaryBase64EncodePolicy(),
)

job_id = str(uuid.uuid4())
payload = json.dumps({
    "job_id": job_id,
    "name": "sunset-mountains",
    "prompt": "a serene sunset over snowy mountains, oil painting",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "steps": 20,
}).encode("utf-8")

queue.send_message(payload)
print("submitted", job_id)
```

---

## 10. Reference: Python consumer (≈40 lines)

```python
import json
import time
import urllib.request
from azure.storage.queue import QueueClient, BinaryBase64DecodePolicy

CONNECTION_STRING = "..."
WANTED_JOB_ID = "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01"

queue = QueueClient.from_connection_string(
    CONNECTION_STRING,
    "image-results",
    message_decode_policy=BinaryBase64DecodePolicy(),
)

while True:
    received = 0
    for msg in queue.receive_messages(max_messages=8, visibility_timeout=30):
        received += 1
        body = msg.content
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        result = json.loads(body)

        if result["job_id"] != WANTED_JOB_ID:
            # Not ours — leave it for another consumer by NOT deleting.
            continue

        if result["status"] == "success":
            with urllib.request.urlopen(result["blob_url"]) as resp:
                with open("output.png", "wb") as f:
                    f.write(resp.read())
            print("got", result["blob_name"])
        else:
            print("failed:", result["error"])

        queue.delete_message(msg.id, msg.pop_receipt)
        raise SystemExit(0)

    if received == 0:
        time.sleep(2)
```

The `visibility_timeout` is the window in which you must call
`delete_message` (or the message reappears for someone else to retry). 30 s
is plenty for "parse JSON + download PNG"; bump it if you do heavy
post-processing.

---

## 11. REST (no SDK) example for the producer

For environments without the Azure SDK, you can hit the Storage Queue REST
API directly. Authentication via a queue-scoped SAS is simplest. The body
must be wrapped in `<QueueMessage><MessageText>BASE64</MessageText></QueueMessage>`.

```bash
SAS="?sv=...&sig=..."
BODY=$(printf '%s' '{"name":"x","prompt":"a cat","width":512,"height":512}' \
  | base64 -w0)

curl -X POST \
  "https://nomadimagegen.queue.core.windows.net/image-requests/messages${SAS}" \
  -H "Content-Type: application/xml" \
  --data "<QueueMessage><MessageText>${BODY}</MessageText></QueueMessage>"
```

Receiving via REST is similar but returns XML with `<MessageText>` containing
the base64 body; decode and parse as JSON.

---

## 12. Compatibility note

The schema is conservative-additive: adding new optional fields to the
request, or new keys to the result message, is **not** considered a breaking
change. Producers MUST tolerate unknown keys in results. Consumers MUST treat
any absent-or-unknown-status as a failure.

Removing or renaming a field, or changing the type/encoding of an existing
field, requires a coordinated migration and a new spec version.

---

## 13. LLM request schema (producer → `llm-requests`)

The LLM queue runs **chat completions** through an Ollama daemon using its
native `POST /api/chat` endpoint. Same base64 envelope and 64 KiB cap as
the image queue (§4).

```json
{
  "job_id": "string or integer, optional",
  "name": "string, optional",
  "system_prompt": "string, optional, default \"\"",
  "user_prompt": "string, required",
  "model": "string, required",
  "thinking": "boolean or yes/no string, optional, default false",
  "temperature": "number 0.0-2.0, optional, default 1.0",
  "max_tokens": "integer 1-32768, optional, default 2048"
}
```

### Field rules

| Field | Type | Required | Constraints |
|---|---|---|---|
| `job_id` | int or string | no | Accepted in either form; coerced to string internally. Worker generates a UUID if omitted. **Provide your own** to correlate requests with results. |
| `name` | string | no | Optional human-readable label; surfaces in worker logs and in the result message. |
| `system_prompt` | string | no | Max 32000 chars. Empty/omitted → no system message is sent to vLLM. |
| `user_prompt` | string | **yes** | Non-empty. Max 32000 chars. |
| `model` | string | **yes** | Must match a model pulled into the Ollama daemon (verify with `ollama list` or `GET {OLLAMA_URL}/api/tags`). Example: `qwen3.6:35b-a3b-q4_K_M`. Mismatch → error result with Ollama's error body. |
| `thinking` | bool or string | no | `true`/`false`, or case-insensitive `yes`/`no`/`true`/`false`/`1`/`0`/`on`/`off`. Sets Ollama's `think` flag — honored by Qwen3-family reasoning GGUFs, ignored by others. **Default `false`**: Qwen3.6 *defaults to thinking ON*, so leaving it unspecified will burn tokens on chain-of-thought; pass `false` explicitly for short answers. |
| `temperature` | number | no | 0.0 ≤ t ≤ 2.0. Default 1.0. String like `"0.7"` is also accepted. The field name `temp` is accepted as a back-compat alias. |
| `max_tokens` | int | no | 1 ≤ n ≤ 32768. Default 2048. **Set this thoughtfully** — large completions force the result to spill to Blob Storage (§14). |

### Example

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "summarize-meeting",
  "system_prompt": "You are a concise meeting-notes assistant.",
  "user_prompt": "Summarize the following transcript in three bullet points: ...",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "thinking": false,
  "temperature": 0.3,
  "max_tokens": 512
}
```

### Validation failures

Same policy as the image queue: malformed/missing/out-of-range fields produce
an `error` result and the inbound message is deleted (no retries).

---

## 14. LLM result schema (consumer ← `llm-results`)

```json
{
  "job_id": "string",
  "name": "string",
  "status": "success" | "error",
  "model": "string",
  "completion": "string | null",
  "reasoning": "string | null",
  "prompt_tokens": "integer | null",
  "completion_tokens": "integer | null",
  "finish_reason": "stop | length | tool_calls | content_filter | null",
  "blob_url": "string | null",
  "blob_name": "string | null",
  "error": "string | null"
}
```

### Success result — inline

For typical short completions, `completion` carries the text directly and
`blob_url`/`blob_name` are `null`:

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "summarize-meeting",
  "status": "success",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "completion": "- Agreed timeline...\n- Budget approved...\n- Next meeting Friday.",
  "reasoning": null,
  "prompt_tokens": 412,
  "completion_tokens": 88,
  "finish_reason": "stop",
  "blob_url": null,
  "blob_name": null,
  "error": null
}
```

### Success result — spilled to blob

If the total JSON would exceed ~50 KiB (Storage Queue's 64 KiB cap minus
envelope overhead), the worker uploads `{"completion": ..., "reasoning": ...}`
as a JSON blob and replaces the inline fields with a SAS URL:

```json
{
  "job_id": "...",
  "name": "long-essay",
  "status": "success",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "completion": null,
  "reasoning": null,
  "prompt_tokens": 1024,
  "completion_tokens": 7400,
  "finish_reason": "stop",
  "blob_url": "https://nomadimagegen.blob.core.windows.net/generated-images/llm/<job_id>.json?<SAS>",
  "blob_name": "llm/<job_id>.json",
  "error": null
}
```

The blob is a single JSON object: `{"completion": "...", "reasoning": "..."}`.
Same 24 h SAS lifetime as image blobs.

### Reasoning content

For reasoning-enabled models (Qwen3 family with `thinking: true`), `reasoning`
carries the chain-of-thought separately from `completion`. Ollama returns it
under `message.thinking` on `/api/chat`; the worker exposes it as `reasoning`
in the result message. Most workloads should display `completion` to the end
user and treat `reasoning` as debug-only.

### Error result

```json
{
  "job_id": "...",
  "name": "...",
  "status": "error",
  "model": "...",
  "completion": null,
  "reasoning": null,
  "prompt_tokens": null,
  "completion_tokens": null,
  "finish_reason": null,
  "blob_url": null,
  "blob_name": null,
  "error": "ollama HTTP 404: {\"error\": \"model 'foo' not found\"}"
}
```

Common errors:
- `"'user_prompt' is required and must be a non-empty string"` — schema failure
- `"'temperature'=3.0 must be in [0.0, 2.0]"` — validation failure
- `"ollama HTTP 404: ..."` — model not pulled into the daemon
- `"ollama request failed: ConnectionError"` — Ollama not running / wrong URL
- `"ollama HTTP 400: {...}"` — daemon rejected the request body

### Operational notes

| Property | Value |
|---|---|
| Polling priority | LLM queue is polled **first every iteration**; image jobs only run when LLM queue is empty. |
| Concurrency | Same worker process serializes both queues. One LLM call at a time, one image generation at a time. |
| Ollama lifecycle | The worker does NOT start Ollama. Start it separately (`ollama serve`, or the Windows tray icon). Models must be pre-pulled with `ollama pull <name>`. |
| Result correlation | Same pattern as images — generate `job_id` client-side. |
