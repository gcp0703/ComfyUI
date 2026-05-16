# azure_worker — Azure Storage Queue ⇄ ComfyUI bridge

A small worker that consumes text-to-image requests from an Azure Storage
Queue, runs them through ComfyUI in-process, uploads the resulting PNG to
Azure Blob Storage, and posts a result message (with a SAS URL) to a second
Storage Queue.

Two model profiles are supported and selected at startup via `COMFY_PROFILE`:

| Profile | Models |
|---|---|
| `flux1-dev` (default) | `flux1-dev.safetensors` + DualCLIP (`clip_l` + `t5xxl_fp16`) + Flux 1 VAE (`ae.safetensors`) |
| `flux2-klein` | `flux-2-klein-9b-fp8.safetensors` + CLIPLoader(type=flux2) with Qwen3 + 128-ch Flux 2 VAE |

Both profile blocks must be filled in `.env`; only the active profile is
actually loaded into VRAM. Switching profiles requires restarting the worker.

The worker uses ComfyUI's production execution path — it boots a
`PromptServer` and submits to `prompt_queue` exactly like the HTTP `/prompt`
endpoint does, but never starts the aiohttp listener. Models stay resident in
GPU memory between jobs.

## Requirements

```
pip install -r azure_worker/requirements.txt
```

(ComfyUI's own dependencies must already be installed in the same environment.)

## Configuration (environment variables)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | yes | — | Must include `AccountKey` so the worker can mint SAS tokens. |
| `AZURE_INBOUND_QUEUE` | yes | — | Storage Queue name for incoming requests. |
| `AZURE_OUTBOUND_QUEUE` | yes | — | Storage Queue name for results. |
| `AZURE_BLOB_CONTAINER` | yes | — | Blob container for generated PNGs. |
| `COMFY_PROFILE` | yes | — | `flux1-dev` or `flux2-klein`. |
| `COMFY_FLUX1_UNET` | yes | — | Flux 1 UNet under `models/diffusion_models/` (e.g. `flux1-dev.safetensors`). |
| `COMFY_FLUX1_CLIP_L` | yes | — | CLIP-L text encoder under `models/text_encoders/`. |
| `COMFY_FLUX1_T5` | yes | — | T5-XXL text encoder under `models/text_encoders/`. |
| `COMFY_FLUX1_VAE` | yes | — | Flux 1 VAE under `models/vae/` (typically `ae.safetensors`). |
| `COMFY_FLUX2_UNET` | yes | — | Flux 2 UNet under `models/diffusion_models/`. |
| `COMFY_FLUX2_CLIP` | yes | — | Qwen3 text encoder under `models/text_encoders/`. |
| `COMFY_FLUX2_VAE` | yes | — | Flux 2 128-channel VAE under `models/vae/`. |
| `SAS_EXPIRY_HOURS` | no | `24` | Lifetime of generated SAS URLs. |
| `POLL_INTERVAL_SECONDS` | no | `2.0` | How often to poll an empty inbound queue. |
| `VISIBILITY_TIMEOUT_SECONDS` | no | `300` | Inbound message visibility timeout while a job runs. |
| `MAX_DEQUEUE_COUNT` | no | `3` | Reserved for future retry policy; currently unused. |

## Running

```
python -m azure_worker.main
```

The worker takes no CLI args — configuration is env-only so the embedded
ComfyUI argparser never sees foreign options.

## Message contracts

### Inbound (`AZURE_INBOUND_QUEUE`), base64-encoded JSON

```json
{
  "job_id": "uuid",
  "name": "sunset-over-mountains",
  "prompt": "a serene sunset over snowy mountains, oil painting",
  "negative_prompt": "blurry, low quality",
  "width": 768,
  "height": 512,
  "seed": 12345,
  "steps": 20,
  "cfg": 7.0
}
```

- `job_id`, `negative_prompt`, `seed`, `steps`, `cfg` are optional.
- `width` and `height` must be in `[64, 4096]` and multiples of 16
  (Flux 2's `EmptyFlux2LatentImage` requires 16-pixel alignment).
- `negative_prompt` and `cfg` are accepted but currently ignored in both
  profiles: Flux 2 Klein uses `BasicGuider` with no negative path; Flux 1 dev
  uses `KSampler(cfg=1)` with `ConditioningZeroOut` per ComfyUI's official
  template.
- A missing/invalid message produces an error result and is deleted from the
  inbound queue (no retries yet).

### Outbound (`AZURE_OUTBOUND_QUEUE`), base64-encoded JSON

```json
{
  "job_id": "uuid",
  "name": "sunset-over-mountains",
  "status": "success",
  "prompt": "...",
  "width": 768, "height": 512, "seed": 12345,
  "blob_url": "https://<acct>.blob.core.windows.net/<container>/<blob>?<SAS>",
  "blob_name": "sunset-over-mountains/sunset-over-mountains_00001.png",
  "error": null
}
```

`status` is `"error"` on any failure; `error` carries the message, and
`blob_url`/`blob_name` are `null`.

## End-to-end smoke test (Azurite)

```
docker run -p 10000-10002:10000-10002 mcr.microsoft.com/azure-storage/azurite

# Against Azurite's well-known connection string:
$env:AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
$env:AZURE_INBOUND_QUEUE = "image-requests"
$env:AZURE_OUTBOUND_QUEUE = "image-results"
$env:AZURE_BLOB_CONTAINER = "generated-images"
$env:COMFY_PROFILE = "flux1-dev"
$env:COMFY_FLUX1_UNET = "flux1-dev.safetensors"
$env:COMFY_FLUX1_CLIP_L = "clip_l.safetensors"
$env:COMFY_FLUX1_T5 = "t5xxl_fp16.safetensors"
$env:COMFY_FLUX1_VAE = "ae.safetensors"
$env:COMFY_FLUX2_UNET = "flux-2-klein-9b-fp8.safetensors"
$env:COMFY_FLUX2_CLIP = "qwen_3_8b_fp8mixed.safetensors"
$env:COMFY_FLUX2_VAE = "full_encoder_small_decoder.safetensors"

# Create the queues + container with `az storage` against Azurite, then:
python -m azure_worker.main
```

Enqueue `sample_request.json` (base64-encoded) onto the inbound queue and
watch the outbound queue for the result.

## Out of scope

Multiple workflow templates, ControlNet, img2img, Managed Identity,
dead-letter handling, health endpoint, container manifests — future work.
