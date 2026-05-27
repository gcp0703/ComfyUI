# Building an LLM Client for the Azure-Queue Worker

Instructions for an AI assistant (or human) writing code that talks to the
LLM pipeline exposed by `azure_worker`. Read this end-to-end before writing
any code — the gotchas section is load-bearing.

You are building **two things**:
1. A **producer** that places JSON LLM-query messages onto `llm-requests`.
2. A **consumer** that reads JSON result messages off `llm-results`, correlates
   them back to the original request, and downloads any blob-spilled content.

Both queues live in the Azure Storage account `nomadimagegen`
(region `centralus`). They are independent FIFO-ish queues; the worker drains
inbound and posts to outbound on its own schedule.

---

## 0. Mental model in one paragraph

You enqueue a JSON request (base64-wrapped) onto `llm-requests`. A worker
pulls it, calls Ollama, and posts a JSON result onto `llm-results` carrying
the **same `job_id`** you sent. You then poll `llm-results`, look for *your*
`job_id`, and act on the result. **Exactly one result message is produced per
inbound message** (success or error). The worker deletes the inbound message
after processing — there are no retries on the inbound side.

---

## 1. Wire encoding (read this once, then forget it)

Both queues are configured for **base64 message encoding**:

```
on_the_wire = base64( utf8_bytes( json_string ) )
```

If you use the official Azure SDK with `BinaryBase64EncodePolicy` /
`BinaryBase64DecodePolicy` (shown in the examples below), encoding/decoding
is automatic — pass and receive `bytes`. **Do not double-encode.**

Per-message cap: **64 KiB after base64** (~48 KiB JSON). Your prompts won't
hit this; the worker handles oversized *responses* by spilling to Blob
Storage (see §5).

---

## 2. Request schema — `llm-requests`

A single JSON object. Required fields are bold.

```json
{
  "job_id": 12345,
  "name": "summarize-meeting",
  "system_prompt": "You are a concise notes assistant.",
  "user_prompt": "Summarize: ...",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "thinking": false,
  "temperature": 0.3,
  "max_tokens": 512
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `job_id` | int or string | no, but always set it | Producer-language-neutral: int OK, string OK. Worker coerces to string. **Set this yourself** — it's your only durable correlation key. Use `uuid.uuid4()` if you don't already have an ID. |
| `name` | string | no | Optional human-readable label. Echoed in result + logs. |
| `system_prompt` | string | no | Max 32000 chars. Omit or pass `""` for no system message. |
| **`user_prompt`** | **string** | **yes** | Non-empty. Max 32000 chars. |
| **`model`** | **string** | **yes** | Must match a model already pulled into Ollama. Verify with `ollama list` or `GET http://<host>:11434/api/tags`. Today's loaded models include `qwen3.6:35b-a3b-q4_K_M`. |
| `thinking` | bool or string | no, **default `false`** | Accepts `true`/`false`, or case-insensitive `yes`/`no`/`true`/`false`/`1`/`0`/`on`/`off`. **See §6 — this matters more than you think.** |
| `temperature` | number or numeric string | no, default `1.0` | Range `[0.0, 2.0]`. `temp` is accepted as an alias. |
| `max_tokens` | int | no, default `2048` | Range `[1, 32768]`. Cap on completion length. Don't make it huge by default; large completions force result-spill to blob. |

**Anything else you put in the JSON is ignored, not rejected.** That keeps
the protocol forward-compatible.

---

## 3. Result schema — `llm-results`

Always contains the same set of keys. `status` tells you which to read.

```json
{
  "job_id": "12345",
  "name": "summarize-meeting",
  "status": "success",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "completion": "- Agreed timeline...\n- Budget approved...",
  "reasoning": null,
  "prompt_tokens": 412,
  "completion_tokens": 88,
  "finish_reason": "stop",
  "blob_url": null,
  "blob_name": null,
  "error": null
}
```

| Field | Notes |
|---|---|
| `job_id` | Always a string. If the original was an int, it was coerced. Compare as strings. |
| `name` | Echo of the request, or `"unknown"` if the request couldn't be parsed. |
| `status` | `"success"` or `"error"`. **Check this first.** |
| `model` | Echo of the request, or `"unknown"` on a totally malformed request. |
| `completion` | Final answer text. **`null` if blob-spilled** (see §5). |
| `reasoning` | Chain-of-thought, only populated when `thinking: true` and the model supports it. `null` otherwise. Display to end users sparingly — it's verbose and model-internal. |
| `prompt_tokens` / `completion_tokens` | Usage stats. `null` only on hard errors. |
| `finish_reason` | `"stop"` = natural end; `"length"` = hit `max_tokens` (truncated — consider retrying with higher cap). |
| `blob_url` | `null` for inline results. For spilled results: a SAS URL valid 24 h pointing to a JSON object `{"completion": "...", "reasoning": "..."}`. |
| `blob_name` | `null` for inline results, else path inside the `generated-images` container (e.g. `llm/<job_id>.json`). |
| `error` | `null` on success; error message string on failure. |

---

## 4. Auth

Pick one mechanism per side. The producer and consumer can use different
ones independently.

**Producer (sends to `llm-requests`):**
- Connection string with `AccountKey` — simple, full account access.
- SAS scoped to `Add` on `llm-requests` — recommended for client apps.
- Managed Identity (Entra) with `Storage Queue Data Message Sender`.

**Consumer (reads from `llm-results`):**
- Connection string with `AccountKey`.
- SAS scoped to `Read` + `Process` on `llm-results`.
- Managed Identity with `Storage Queue Data Message Processor`.

**Blob download (if `blob_url` is set in a result):** no auth — the embedded
SAS handles it.

---

## 5. The blob-spill case

If the JSON result would exceed ~50 KiB, the worker:
1. Uploads `{"completion": ..., "reasoning": ...}` as a JSON blob at
   `generated-images/llm/<job_id>.json`.
2. Returns a result with `completion: null`, `reasoning: null`,
   `blob_name: "llm/<job_id>.json"`, and `blob_url: "<24h SAS URL>"`.

Your consumer **must** handle both shapes:

```python
def extract_completion(result: dict) -> tuple[str, str | None]:
    """Return (completion, reasoning) regardless of inline-vs-spilled."""
    if result["blob_url"]:
        import urllib.request, json
        with urllib.request.urlopen(result["blob_url"]) as resp:
            spilled = json.loads(resp.read())
        return spilled["completion"], spilled.get("reasoning") or None
    return result["completion"] or "", result["reasoning"]
```

24-hour SAS expiry is short for long-lived jobs. If you need durable
storage, copy the blob into your own container immediately after download.

---

## 6. Critical gotchas (these will bite you)

### 6a. Qwen3.6 defaults to **thinking ON**

If you omit `thinking` for `qwen3.6:35b-a3b-q4_K_M` (or any Qwen3 reasoning
model), it defaults to **on**. Your `max_tokens` budget will be consumed
generating chain-of-thought before the model emits a single character of
the actual answer. The most common symptom is `completion: ""` and a giant
`reasoning` string — the answer was about to start when `max_tokens` ran out.

**Always pass `"thinking": false` for short tasks** (classification,
extraction, naming, quick questions). Only enable it when you actually want
reasoning and have set `max_tokens` to 2000+.

### 6b. `job_id` is your problem

The worker preserves `job_id` from request to result. It never matches
results to consumers — *you* do that. Run multiple jobs concurrently? Run a
single consumer that fans out by `job_id`, **not** N consumers each peeking
and putting back. The peek-and-putback pattern is pathological under
contention.

### 6c. Always delete the result message

`receive_messages` makes a message invisible for `visibility_timeout` seconds.
If you don't call `delete_message(msg.id, msg.pop_receipt)`, it reappears for
someone else (or you) to re-process. **Delete after you've persisted the
result** — not before.

### 6d. `status: "error"` is normal

Validation failures (bad JSON, missing fields, Ollama down, model not
pulled, `max_tokens` out of range) all produce a result with
`status: "error"` and a populated `error` string. **Always check `status`
before reading `completion`.** A failed request still consumes the inbound
message — it does not retry.

### 6e. Don't put the same job_id into both queues

The image pipeline (`image-requests` / `image-results`) and the LLM
pipeline (`llm-requests` / `llm-results`) are completely separate. A
message in the wrong queue will fail validation (different required
fields) and you'll get an error result back on whichever output queue is
paired with the input queue you used.

### 6f. Ordering is not guaranteed

Don't rely on FIFO. Two requests enqueued in order A then B can return as
B then A. Use `job_id` for correlation, full stop.

---

## 7. Reference: Python producer (≈25 lines)

```python
import json
import uuid
from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy

CONNECTION_STRING = "..."  # or SAS / managed identity

queue = QueueClient.from_connection_string(
    CONNECTION_STRING,
    "llm-requests",
    message_encode_policy=BinaryBase64EncodePolicy(),
)

job_id = str(uuid.uuid4())
payload = json.dumps({
    "job_id": job_id,
    "name": "summarize",
    "system_prompt": "You are a concise summarizer.",
    "user_prompt": "Summarize this in three bullets:\n\n" + long_text,
    "model": "qwen3.6:35b-a3b-q4_K_M",
    "thinking": False,    # critical — see §6a
    "temperature": 0.3,
    "max_tokens": 512,
}).encode("utf-8")

queue.send_message(payload)
print("submitted", job_id)
```

---

## 8. Reference: Python consumer (≈45 lines)

```python
import json
import time
import urllib.request
from azure.storage.queue import QueueClient, BinaryBase64DecodePolicy

CONNECTION_STRING = "..."
WANTED_JOB_ID = "..."     # the str(uuid.uuid4()) from the producer
TIMEOUT_SECONDS = 300

queue = QueueClient.from_connection_string(
    CONNECTION_STRING,
    "llm-results",
    message_decode_policy=BinaryBase64DecodePolicy(),
)

deadline = time.time() + TIMEOUT_SECONDS
while time.time() < deadline:
    saw_any = False
    for msg in queue.receive_messages(max_messages=8, visibility_timeout=30):
        saw_any = True
        body = msg.content
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        result = json.loads(body)

        if result["job_id"] != WANTED_JOB_ID:
            # Not ours — leave it for another consumer by NOT deleting.
            # It becomes visible again after visibility_timeout.
            continue

        # It's ours: delete before processing to claim it.
        queue.delete_message(msg.id, msg.pop_receipt)

        if result["status"] != "success":
            raise RuntimeError(f"LLM job failed: {result['error']}")

        # Handle inline vs blob-spilled
        if result["blob_url"]:
            with urllib.request.urlopen(result["blob_url"]) as resp:
                spilled = json.loads(resp.read())
            completion = spilled["completion"]
        else:
            completion = result["completion"] or ""

        print(f"got {result['completion_tokens']} tokens in "
              f"{result['finish_reason']} state:\n{completion}")
        raise SystemExit(0)

    if not saw_any:
        time.sleep(2)

raise TimeoutError(f"no result for {WANTED_JOB_ID} within {TIMEOUT_SECONDS}s")
```

---

## 9. Multi-request dispatcher pattern (recommended for >1 in-flight)

For workloads with many concurrent requests, **do not** spin up N consumers
each peeking-and-putting-back. Instead run one dispatcher that drains the
queue and routes by `job_id` to in-process waiters:

```python
import asyncio, json

pending: dict[str, asyncio.Future] = {}

async def submit(payload: dict) -> dict:
    """Enqueue and await the matching result."""
    fut = asyncio.get_event_loop().create_future()
    pending[payload["job_id"]] = fut
    await asyncio.to_thread(inbound.send_message,
                            json.dumps(payload).encode("utf-8"))
    return await fut

async def dispatcher():
    while True:
        msgs = await asyncio.to_thread(outbound.receive_messages,
                                       max_messages=8, visibility_timeout=30)
        any_message = False
        for msg in msgs:
            any_message = True
            body = msg.content.decode("utf-8") if isinstance(msg.content, bytes) else msg.content
            result = json.loads(body)
            fut = pending.pop(result["job_id"], None)
            if fut and not fut.done():
                fut.set_result(result)
                await asyncio.to_thread(outbound.delete_message, msg.id, msg.pop_receipt)
            # If not ours, leave it (visibility expires naturally).
        if not any_message:
            await asyncio.sleep(1)
```

Caveats:
- A crash drops in-flight `pending` futures; rely on a result-side timeout
  per request (the dispatcher doesn't enforce one).
- If multiple dispatchers run in the same account, they'll race for the
  same messages. One dispatcher per process; horizontal scale by sharding
  on the producer side (different queue per shard).

---

## 10. Local testing without spinning up everything

You can hit the worker's pipeline locally even without Ollama running on
your machine — the worker is what calls Ollama, and the request just sits
on the queue if no worker is consuming. To test schema validation:

```bash
# In one terminal, start a local worker pointed at Azure + your Ollama:
python -m azure_worker.main

# In another, run the end-to-end smoke (does not load ComfyUI):
python -m azure_worker.tools.smoke_llm
```

For pure schema-only validation without any Azure or Ollama access:

```python
from azure_worker.llm_messages import LlmRequest, LlmMessageValidationError
import json

try:
    LlmRequest.from_json(json.dumps(your_payload))
    print("payload OK")
except LlmMessageValidationError as e:
    print(f"would be rejected: {e}")
```

---

## 11. Currently available models

Pre-pulled into Ollama on the worker host:

| `model` field value | Notes |
|---|---|
| `qwen3.6:35b-a3b-q4_K_M` | 35B MoE (3B active), Q4_K_M, ~23 GB. Reasoning-capable. **Pass `thinking: false` for short answers.** |
| `qwen3.6:27b` | Dense, 17 GB. |
| `qwen3.5:27b` | Older Qwen, 17 GB. |

To request a new model: have someone run `ollama pull <name>` on the worker
host. The worker doesn't auto-pull on demand.

---

## 12. What this protocol does NOT do (yet)

- **Streaming responses** — every result lands as a single message after the
  full completion is generated. There is no token-streaming surface.
- **Tool calling / function calling** — request schema has no `tools` field.
- **Multi-turn conversations** — each request is a fresh chat. Send the
  entire history in `user_prompt` (or split into a series of role-tagged
  messages — but the request schema currently only supports one
  `system_prompt` + one `user_prompt`, so multi-turn requires concatenation).
- **Retries** — failed jobs return one error result and stop. The producer
  decides whether to re-enqueue.
- **Backpressure / queue depth** — Storage Queue accepts as much as you
  enqueue. There's no flow-control signal to the producer.

If you need any of the above, surface it as a request before writing
workarounds — the protocol is conservative-additive, so adding optional
fields is non-breaking.
