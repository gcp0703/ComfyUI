# azure_worker — Azure Storage Queue ⇄ ComfyUI bridge

A small worker that consumes text-to-image requests from an Azure Storage
Queue, runs them through ComfyUI in-process (Flux 2 Klein pipeline), uploads
the resulting PNG to Azure Blob Storage, and posts a result message (with a
SAS URL) to a second Storage Queue.

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
| `COMFY_FLUX_UNET` | yes | — | Flux 2 UNet filename under `models/diffusion_models/` (e.g. `flux-2-klein-9b-fp8.safetensors`). |
| `COMFY_FLUX_CLIP` | yes | — | Text encoder filename under `models/text_encoders/` (e.g. `qwen_3_8b_fp8mixed.safetensors`). |
| `COMFY_FLUX_VAE` | yes | — | Flux VAE filename under `models/vae/` (typically `ae.safetensors`). |
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
- `negative_prompt` and `cfg` are accepted but currently ignored: Flux 2 Klein
  is a guidance-distilled model driven by `BasicGuider` with no negative path.
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
$env:COMFY_FLUX_UNET = "flux-2-klein-9b-fp8.safetensors"
$env:COMFY_FLUX_CLIP = "qwen_3_8b_fp8mixed.safetensors"
$env:COMFY_FLUX_VAE = "ae.safetensors"

# Create the queues + container with `az storage` against Azurite, then:
python -m azure_worker.main
```

Enqueue `sample_request.json` (base64-encoded) onto the inbound queue and
watch the outbound queue for the result.

## Out of scope

Multiple workflow templates, ControlNet, img2img, Managed Identity,
dead-letter handling, health endpoint, container manifests — future work.
