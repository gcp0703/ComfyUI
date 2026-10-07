# Image Generation Queue — Client Contract v2

**Status:** proposed, not yet implemented
**Date:** 2026-10-06
**Supersedes:** §5 and §6 of `azure_worker/SPEC.md`
**Audience:** whoever owns the NomadImage client

Everything else in `azure_worker/SPEC.md` — resources (§2), authentication
(§3), the base64 envelope (§4), correlation (§8), and the entire LLM pipeline
(§13–§14) — is unchanged. This document replaces only the image request and
result schemas.

---

## 1. What changes, in one paragraph

The client stops telling the worker **how** to render and only says **what** to
render. `steps`, `cfg`, `width` and `height` leave the request: the worker owns
them, because they are properties of the model that happens to be loaded, not
of the picture being asked for. In exchange, every result now carries a
`render` block describing what actually ran — the real model file, the real
step count, the real guidance scale, the real sampler, read back out of the
executed graph rather than copied from a config label. The saved prompt and the
Prompt button can show the truth instead of a guess.

### Why the old shape was wrong

A client sending `steps=20, cfg=7.0` had no way to know those numbers were
frequently discarded. Six of the nine profiles bake their own guidance scale
and ignore `cfg` outright; `qwen-rapid-aio` is a 4-step distill where
`steps=20` wastes five times the work for no gain; `qwen-image-2.1` wants 40–50.
The request fields looked authoritative and were not. The result echoed them
back, so a saved prompt recorded the request rather than the render, and the
Prompt button displayed numbers that never reached the sampler.

---

## 2. Request schema (producer → `image-requests`)

```json
{
  "job_id": "string, optional",
  "name": "string, required",
  "prompt": "string, required",
  "negative_prompt": "string, optional, default \"\"",
  "seed": "integer, optional"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `job_id` | string | no | Worker generates a UUID if omitted. **Always send your own** — it is the only durable correlation key (SPEC §8). |
| `name` | string | **yes** | Non-empty. Becomes the PNG filename prefix, sanitized (non-alphanumerics → `_`). Need not be unique. |
| `prompt` | string | **yes** | Non-empty. `prompt` + `negative_prompt` combined ≤ **32,000 characters** — see *Prompt length* below. |
| `negative_prompt` | string | no | Counts toward the same 32,000-character budget as `prompt`. Honored only on some profiles — the result tells you which, see `render.negative_honored`. |
| `seed` | integer | no | 64-bit unsigned. Omitted → the worker picks one at random and reports it back, so any render stays reproducible. |

### Prompt length

**`prompt` and `negative_prompt` together may total at most 32,000
characters.** This replaces the per-field 4096 limit in v1.

The limit is profile-independent, so the client can enforce it before sending
without knowing which model the worker has loaded.

Where the number comes from, so nobody mistakes it for a model property:

- The v1 cap of 4096 (`messages.py: MAX_PROMPT_CHARS`) was arbitrary. No
  text encoder in the active profiles truncates input — the Qwen-VL encoders
  behind `qwen-image-2512`, `qwen-image-2.1` and `qwen-rapid-aio` pass every
  token through.
- The only hard ceiling is transport: an Azure Storage Queue message is
  **64 KiB after base64 encoding**, about 48 KB of raw JSON, shared by every
  field in the message.
- The queue counts **bytes, not characters**. Non-ASCII text (em-dashes are 3
  bytes in UTF-8) and JSON escaping both inflate the byte count. 32,000
  characters leaves enough margin that the message fits even if the prompt is
  heavy with such characters.

The worker enforces the 32,000-character rule and, as a backstop, the true
byte limit of the encoded message. Exceeding either produces an `error` result
with a message naming the limit; it never surfaces as an SDK exception.

The result message is serialized as UTF-8, not `\u`-escaped, and is itself
guarded against the same transport ceiling: if a result would not fit the
queue, the worker truncates the `prompt`/`negative_prompt` echo and adds
`"prompt echo truncated to fit the result message"` to `warnings` rather than
letting the send fail.

**This is a transport limit, not a quality promise.** The Qwen-Image models
were trained on long structured captions and tolerate length far better than
CLIP-based models, but nobody has measured output quality at 30,000
characters. Length past a few thousand characters is permitted, not
recommended.

### Removed fields

`width`, `height`, `steps` and `cfg` are **no longer part of the request.**

The worker accepts them if sent, ignores them, and lists them in the result's
`warnings` array. Nothing breaks during a staged rollout — but a client still
sending them is not steering anything, and the warning is there so you can find
such a client in the field.

### Example

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "guildmaster-assassins-1",
  "prompt": "Perspective: front-facing view. The scene is set in a shadowy chamber behind a false wall...",
  "negative_prompt": "",
  "seed": 42
}
```

---

## 3. Result schema (consumer ← `image-results`)

Every key is always present. `status` tells you whether to read `blob_url` or
`error`.

```json
{
  "job_id": "string",
  "name": "string",
  "status": "success" | "error",
  "prompt": "string",
  "negative_prompt": "string",
  "width": "integer",
  "height": "integer",
  "seed": "integer",
  "render": "object | null",
  "warnings": ["string"],
  "blob_url": "string | null",
  "blob_name": "string | null",
  "error": "string | null"
}
```

### Changes from v1

| Key | v1 | v2 |
|---|---|---|
| `width` / `height` | echoed the request | **the PNG's actual dimensions** |
| `negative_prompt` | absent | present, echoes the request |
| `render` | absent | new — see §4 |
| `warnings` | absent | new, always an array, `[]` when empty |

`width`/`height` keep their names and types but strengthen their meaning from
"what you asked for" to "what you got". A client that was already treating them
as the image's real size needs no change. On a runtime error they report the
size that was being rendered (read from the built graph); on a validation error
no size exists and both are `0`.

### Feature detection

`render` is absent on a v1 worker and a non-`null` object on a v2 worker for
any job that reached the sampler. Branch on its presence; no version field is
sent, and none is needed.

---

## 4. The `render` block

```json
"render": {
  "profile": "qwen-image-2.1",
  "model": "qwen_image_2.1_int8_convrot.safetensors",
  "loras": [],
  "steps": 45,
  "cfg": 1.0,
  "sampler": "euler",
  "scheduler": "simple",
  "shift": 3.7169,
  "negative_honored": false,
  "summary": "qwen-image-2.1 · qwen_image_2.1_int8_convrot · 2048×2048 · 45 steps · cfg 1.0 · euler/simple"
}
```

| Field | Type | Meaning |
|---|---|---|
| `profile` | string | The worker's configured profile name. A label — use `model` for identity. |
| `model` | string | The checkpoint or UNet **filename actually loaded**. This is the ground truth of "which model made this". |
| `loras` | array | Ordered LoRA stack, `[]` when none. Each entry `{"name", "model_strength", "clip_strength"}`. |
| `steps` | integer | Sampler steps actually executed. |
| `cfg` | number \| null | Guidance scale actually sampled at. **`null` on `flux2-klein`**, which uses `BasicGuider` and has no guidance node at all. |
| `sampler` | string | e.g. `euler`, `dpmpp_sde`, `euler_ancestral`. |
| `scheduler` | string \| null | e.g. `simple`, `beta`, `karras`. `null` on `flux2-klein` (`Flux2Scheduler` supplies the curve itself). |
| `shift` | number \| null | Sigma shift, when the profile states one. On `qwen-image-2.1` it is derived from the output size per render, so it genuinely varies. `null` when the profile has no shift node. |
| `negative_honored` | boolean | Whether `negative_prompt` reached the model, or was discarded by a `ConditioningZeroOut` / `BasicGuider` path. |
| `summary` | string | One-line human-readable digest, ready to display verbatim. |

### On `summary`

Formatted worker-side so every consumer shows the same string and the format
lives in one place. If the Prompt button needs different wording, build it from
the structured fields instead — but then only that client benefits.

### Where these values come from

Not from configuration. The worker builds the ComfyUI graph, then reads the
sampler settings **back out of the built graph** before executing it
(`workflow.py: summarize_workflow`). A value in `render` is a value that was
wired into the node that ran. It cannot drift from a config label, because no
config label is consulted on the way out.

Two consequences worth relying on:

- If a profile bakes its own cfg, `render.cfg` is the baked number, not a default.
- If the LoRA autorouter appended a race-specific LoRA because the prompt said
  "dwarf", that LoRA appears in `render.loras`. The saved prompt records it.

### When `render` is `null`

| Situation | `render` | Why |
|---|---|---|
| Success | object | Graph built and executed. |
| Validation error | **`null`** | Rejected before a graph existed. |
| Runtime error (OOM, timeout, model load failure) | **object** | The graph existed; this tells you what was being attempted when it died. |

A runtime error with a populated `render` is the useful case: it is how you
learn that the OOM happened at 2048×2048 and 45 steps.

---

## 5. Profile reference

What the worker will produce, per configured profile. The client cannot select
these — the worker is started with one profile and keeps it until restarted —
but they tell you what sizes and shapes to expect.

| Profile | Size | Steps | cfg | Sampler / scheduler | `negative_honored` |
|---|---|---|---|---|---|
| `flux1-dev` | 1024×1024 | 20 | 1.0 | euler / simple | false |
| `flux2-klein` | 1024×1024 | 20 | `null` | euler / `null` | false |
| `chroma1` | 1024×1024 | 26 | 3.5 | euler / beta | **true** |
| `fluxed-up` | 1024×1024 | 20 | 1.0 | euler / simple | false |
| `qwen-image-2512` | 1328×1328 | 20 | 4.0 | euler / simple | **true** |
| `qwen-image-2.1` | 2048×2048 | 45 | 1.0 | euler / simple | false |
| `openflux1` | 1024×1024 | 20 | 3.5 | euler / simple | **true** |
| `qwen-rapid-aio` | 1024×1024 | 4 | 1.0 | euler_ancestral / beta | false |
| `sdxl-dreamshaper` | 1024×1024 | 6 | 2.0 | dpmpp_sde / karras | **true** |

These live in `azure_worker/profiles.toml`, tracked in git and editable by an
administrator without a code change. **Treat this table as informational, not
as a contract** — an admin may retune any row. The `render` block on each
result is the authoritative record; never infer settings from the profile name.

---

## 6. Worked examples

### Success

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "guildmaster-assassins-1",
  "status": "success",
  "prompt": "Perspective: front-facing view...",
  "negative_prompt": "",
  "width": 2048,
  "height": 2048,
  "seed": 8419203371,
  "render": {
    "profile": "qwen-image-2.1",
    "model": "qwen_image_2.1_int8_convrot.safetensors",
    "loras": [],
    "steps": 45,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "shift": 3.7169,
    "negative_honored": false,
    "summary": "qwen-image-2.1 · qwen_image_2.1_int8_convrot · 2048×2048 · 45 steps · cfg 1.0 · euler/simple"
  },
  "warnings": [],
  "blob_url": "https://nomadimagegen.blob.core.windows.net/generated-images/guildmaster-assassins-1/guildmaster-assassins-1_00001_.png?se=...&sig=...",
  "blob_name": "guildmaster-assassins-1/guildmaster-assassins-1_00001_.png",
  "error": null
}
```

### Success with a LoRA stack

`sdxl-dreamshaper` with the fantasy-race autorouter firing on a prompt that
says "dwarf":

```json
{
  "render": {
    "profile": "sdxl-dreamshaper",
    "model": "DreamShaperXL_Turbo_v2_1.safetensors",
    "loras": [
      {"name": "dnd/Fantasy_Races_XL.safetensors", "model_strength": 0.8, "clip_strength": 0.8},
      {"name": "dnd/Dwarf_XL.safetensors",         "model_strength": 0.7, "clip_strength": 0.7}
    ],
    "steps": 6,
    "cfg": 2.0,
    "sampler": "dpmpp_sde",
    "scheduler": "karras",
    "shift": null,
    "negative_honored": true,
    "summary": "sdxl-dreamshaper · DreamShaperXL_Turbo_v2_1 + 2 loras · 1024×1024 · 6 steps · cfg 2.0 · dpmpp_sde/karras"
  }
}
```

### Legacy client still sending removed fields

Request: `{"name": "x", "prompt": "...", "width": 1024, "height": 1024, "steps": 20, "cfg": 7.0}`

```json
{
  "status": "success",
  "width": 2048,
  "height": 2048,
  "render": { "steps": 45, "cfg": 1.0, "...": "..." },
  "warnings": ["ignored client-supplied fields: width, height, steps, cfg"]
}
```

The job succeeds. Note `width` reports 2048, **not** the 1024 that was asked
for — the result describes the PNG, not the request.

### Validation error

```json
{
  "job_id": "9c4f7e90-4f4a-4d6e-9b04-39d1b62b3a01",
  "name": "unknown",
  "status": "error",
  "prompt": "",
  "negative_prompt": "",
  "width": 0,
  "height": 0,
  "seed": 0,
  "render": null,
  "warnings": [],
  "blob_url": null,
  "blob_name": null,
  "error": "'prompt' is required and must be a non-empty string"
}
```

### Runtime error

```json
{
  "status": "error",
  "width": 2048,
  "height": 2048,
  "seed": 8419203371,
  "render": {
    "profile": "qwen-image-2.1",
    "model": "qwen_image_2.1_int8_convrot.safetensors",
    "steps": 45,
    "cfg": 1.0,
    "...": "..."
  },
  "warnings": [],
  "blob_url": null,
  "blob_name": null,
  "error": "workflow execution failed: CUDA out of memory"
}
```

No PNG exists, but the graph did, so `width`/`height` and `render` together say
exactly what was being attempted: 2048×2048 at 45 steps. That is the whole
point of populating `render` on runtime failures.

---

## 7. Client migration checklist

1. **Stop sending** `width`, `height`, `steps`, `cfg`. Remove the UI controls
   that set them, or repurpose them as read-only display of the last result's
   `render`.
2. **Stop assuming a known output size.** Read `width`/`height` off the result.
   A client that hard-codes 1024×1024 layout will break on `qwen-image-2.1`
   (2048²) and `qwen-image-2512` (1328²).
3. **Persist the whole `render` object** alongside the saved prompt. Store it
   verbatim; new keys may be added (§8).
4. **Point the Prompt button at `render.summary`**, falling back to the
   structured fields if you want custom wording.
5. **Handle `render: null`** — a validation error produced no render.
6. **Handle `cfg: null` and `scheduler: null`** — real on `flux2-klein`.
7. **Surface `negative_honored: false`** if your UI offers a negative prompt
   field. On five of nine profiles that text is silently discarded, and the user
   should be told rather than left to wonder why it did nothing.
8. **Log `warnings`** during rollout to catch stale clients.

### Rollout order

The two sides are not symmetric:

- **Worker first:** safe. Old clients keep working; their steps/cfg/size are
  ignored and reported in `warnings`.
- **Client first:** breaks. `width`/`height` are *required* on a v1 worker, so
  every job from a v2 client **fails validation** until the worker is upgraded.

**Deploy the worker first.** A v2 client requires a v2 worker.

---

## 8. Compatibility rules

Unchanged from `azure_worker/SPEC.md` §12, restated for the new keys:

- Adding keys to `render`, or new strings to `warnings`, is **not** a breaking
  change. Store `render` verbatim and tolerate unknown keys.
- Treat any unrecognized `status` as a failure.
- Removing or renaming a key, or changing its type, requires a coordinated
  migration and a new contract version.

`render` fields may be added as profiles gain features. The ten fields in §4
are the guaranteed set.

---

## 9. Open items for the worker implementation

Not client concerns, listed so the two sides stay in step:

- `azure_worker/profiles.toml` is new; `tomllib` is stdlib on Python 3.13.
- `summarize_workflow()` changes from returning a string to returning the
  `render` dict; the existing log line formats that dict.
- The per-profile sampler constants in `workflow.py` (`CHROMA_*`,
  `QWEN_IMAGE_*`, `QWEN21_CFG/SAMPLER/SCHEDULER`, `QWEN_RAPID_*`, `SDXL_*`)
  move into `profiles.toml` and are deleted from code.
- `qwen21_shift()` stays in code — it is a function of output size, not a
  constant.
- `messages.py`: `MAX_PROMPT_CHARS` becomes a combined 32,000-character check
  across `prompt` + `negative_prompt`, plus a byte-length check on the
  serialized message against the queue's 64 KiB post-base64 ceiling (raw JSON
  ≤ 48 KiB). Both reject with a validation error naming the limit.
- `azure_worker/SPEC.md` §5/§6 and the README profile table get updated to
  match, or replaced by a pointer to this document and to `profiles.toml`.
- `azure_io.send_result` guards the outbound message against the same 64 KiB
  (post-base64) ceiling the request is checked against: `ImageResult.to_json`
  serializes as UTF-8 (`ensure_ascii=False`), and if the result would still
  exceed `IMAGE_RESULT_MAX_BYTES`, the worker truncates the `prompt` (then
  `negative_prompt`) echo and sends rather than raising.
