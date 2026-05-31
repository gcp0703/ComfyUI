# azure_worker — Azure Storage Queue ⇄ ComfyUI + Ollama bridge

A small worker that consumes two kinds of requests off Azure Storage Queues:

1. **Image-generation requests** — runs them through ComfyUI in-process,
   uploads the resulting PNG to Azure Blob Storage, posts a result message
   (with a SAS URL) to a second Storage Queue.
2. **LLM chat-completion requests** — forwards them to a separately-running
   Ollama daemon (native `/api/chat` API), and posts the completion back to
   a second Storage Queue (spilling oversize completions to Blob Storage).

The LLM queue is polled **first every iteration**; image jobs only run when
the LLM queue is empty.

Seven model profiles are supported and selected at startup via `COMFY_PROFILE`:

| Profile | Models | Notes |
|---|---|---|
| `flux1-dev` | `flux1-dev.safetensors` + DualCLIP (`clip_l` + `t5xxl_fp16`) + Flux 1 VAE (`ae.safetensors`) | Guidance-distilled — `cfg` and `negative_prompt` are no-ops. |
| `flux2-klein` | `flux-2-klein-9b-fp8.safetensors` + CLIPLoader(type=flux2) with Qwen3 + 128-ch Flux 2 VAE | Guidance-distilled — `cfg` and `negative_prompt` are no-ops. |
| `chroma1` | `Chroma1-HD-fp8mixed.safetensors` + CLIPLoader(type=chroma) with T5-XXL + Flux 1 VAE | De-distilled — **`cfg` and `negative_prompt` are honored.** Beta scheduler, Euler sampler, sigma shift 1.0 are baked in. |
| `fluxed-up` | `fluxedUpFluxNSFW_40DevFp8.safetensors` (fp8) + reuses flux1 DualCLIP + Flux 1 VAE | NSFW Flux 1 dev finetune. Same guidance-distilled driving as `flux1-dev` — `cfg` and `negative_prompt` are no-ops. |
| `qwen-image-2512` | `qwen_image_2512_fp8_e4m3fn.safetensors` + CLIPLoader(type=qwen_image) with Qwen 2.5 VL 7B + `qwen_image_vae.safetensors` | Alibaba Qwen-Image (Dec 2025). **`cfg` and `negative_prompt` are honored.** Euler + simple, sigma shift 3.1. Recommended: steps=20-50, cfg=4.0. |
| `openflux1` | `openflux1-v0.1.0-fp8.safetensors` (fp8) + reuses flux1 DualCLIP + Flux 1 VAE | ostris/OpenFLUX.1 — de-distilled Flux 1 schnell. Same Flux 1 architecture. **`cfg` and `negative_prompt` are honored.** Recommended: cfg≈3.5, steps≥20. |
| `qwen-rapid-aio` | `Qwen-Rapid-AIO-NSFW-v23.safetensors` — **all-in-one** checkpoint (UNet+CLIP+VAE merged), loaded with `CheckpointLoaderSimple` + `TextEncodeQwenImageEditPlus` | Phr00t/Qwen-Image-Edit-Rapid-AIO. 4-step distilled accelerator merge — `cfg=1` + `euler_ancestral`/`beta` baked in, so **`cfg` and `negative_prompt` are no-ops**. NSFW LoRAs merged in (no trigger word). Recommended: steps=4 (4-8). |

All seven profile blocks must be filled in `.env`; only the active profile is
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
| `AZURE_INBOUND_QUEUE` | yes | — | Storage Queue name for incoming image requests. |
| `AZURE_OUTBOUND_QUEUE` | yes | — | Storage Queue name for image results. |
| `AZURE_BLOB_CONTAINER` | yes | — | Blob container for generated PNGs (and overflow LLM completions). |
| `LLM_INBOUND_QUEUE` | yes | — | Storage Queue name for incoming LLM requests (polled with priority over images). |
| `LLM_OUTBOUND_QUEUE` | yes | — | Storage Queue name for LLM results. |
| `OLLAMA_URL` | yes | — | Base URL of a running Ollama daemon (e.g. `http://localhost:11434`). Worker POSTs to `{OLLAMA_URL}/api/chat`. |
| `LLM_REQUEST_TIMEOUT_SECONDS` | no | `300` | Per-request HTTP timeout for the Ollama call. |
| `COMFY_PROFILE` | yes | — | `flux1-dev`, `flux2-klein`, `chroma1`, `fluxed-up`, `qwen-image-2512`, or `openflux1`. |
| `COMFY_FLUX1_UNET` | yes | — | Flux 1 UNet under `models/diffusion_models/` (e.g. `flux1-dev.safetensors`). |
| `COMFY_FLUX1_CLIP_L` | yes | — | CLIP-L text encoder under `models/text_encoders/`. |
| `COMFY_FLUX1_T5` | yes | — | T5-XXL text encoder under `models/text_encoders/`. |
| `COMFY_FLUX1_VAE` | yes | — | Flux 1 VAE under `models/vae/` (typically `ae.safetensors`). |
| `COMFY_FLUX2_UNET` | yes | — | Flux 2 UNet under `models/diffusion_models/`. |
| `COMFY_FLUX2_CLIP` | yes | — | Qwen3 text encoder under `models/text_encoders/`. |
| `COMFY_FLUX2_VAE` | yes | — | Flux 2 128-channel VAE under `models/vae/`. |
| `COMFY_CHROMA_UNET` | yes | — | Chroma UNet under `models/diffusion_models/` (e.g. `Chroma1-HD-fp8mixed.safetensors`). |
| `COMFY_CHROMA_CLIP` | yes | — | T5-XXL text encoder under `models/text_encoders/` (used with type=chroma). |
| `COMFY_CHROMA_VAE` | yes | — | VAE under `models/vae/` (Chroma uses the Flux 1 VAE `ae.safetensors`). |
| `COMFY_FLUXEDUP_UNET` | yes | — | Fluxed Up UNet under `models/diffusion_models/` (e.g. `fluxedUpFluxNSFW_40DevFp8.safetensors`). Reuses the flux1 CLIP-L / T5 / VAE — no separate text-encoder/VAE env vars. |
| `COMFY_QWEN_UNET` | yes | — | Qwen-Image UNet under `models/diffusion_models/` (e.g. `qwen_image_2512_fp8_e4m3fn.safetensors`). |
| `COMFY_QWEN_CLIP` | yes | — | Qwen 2.5 VL 7B text encoder under `models/text_encoders/` (e.g. `qwen_2.5_vl_7b_fp8_scaled.safetensors`). |
| `COMFY_QWEN_VAE` | yes | — | Qwen-Image VAE under `models/vae/` (e.g. `qwen_image_vae.safetensors`). |
| `COMFY_OPENFLUX_UNET` | yes | — | OpenFLUX.1 UNet under `models/diffusion_models/` (e.g. `openflux1-v0.1.0-fp8.safetensors`). Reuses the flux1 CLIP-L / T5 / VAE — no separate text-encoder/VAE env vars. |
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
- `width` and `height` must be in `[64, 4096]` and multiples of 16.
- `negative_prompt` and `cfg` are **honored on `chroma1`, `qwen-image-2512`,
  and `openflux1`**. Flux 1 dev and Fluxed Up both use `KSampler(cfg=1)` with
  `ConditioningZeroOut`; Flux 2 Klein uses `BasicGuider` with no negative path.
  Recommended values:
  - **Chroma**: `steps=26, cfg=3.5` (workable range 3.5–7), native 1024² or 1152².
  - **Qwen-Image 2512**: `steps=20–50, cfg=4.0`, native 1328² (aspect-ratio
    sweet spots: 1664×928 16:9, 1472×1104 4:3, 1584×1056 3:2).
  - **OpenFLUX.1**: `steps≥20, cfg≈3.5`, native 1024² (any Flux 1 resolution works).
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
$env:COMFY_CHROMA_UNET = "Chroma1-HD-fp8mixed.safetensors"
$env:COMFY_CHROMA_CLIP = "t5xxl_fp16.safetensors"
$env:COMFY_CHROMA_VAE = "ae.safetensors"
$env:COMFY_FLUXEDUP_UNET = "fluxedUpFluxNSFW_40DevFp8.safetensors"
$env:COMFY_QWEN_UNET = "qwen_image_2512_fp8_e4m3fn.safetensors"
$env:COMFY_QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
$env:COMFY_QWEN_VAE = "qwen_image_vae.safetensors"
$env:COMFY_OPENFLUX_UNET = "openflux1-v0.1.0-fp8.safetensors"

# Create the queues + container with `az storage` against Azurite, then:
python -m azure_worker.main
```

Enqueue `sample_request.json` (base64-encoded) onto the inbound queue and
watch the outbound queue for the result.

## LLM pipeline

The worker also drains an LLM request queue (polled before the image queue
every loop iteration). It does NOT run the LLM itself — it makes HTTP calls
to a separately-running Ollama daemon.

Ollama runs natively on Windows. Install + run:

```
winget install Ollama.Ollama        # or download from ollama.com
ollama pull qwen3.6:35b-a3b-q4_K_M  # or any other model you want to serve
# daemon starts automatically (Windows tray); manual: `ollama serve`
```

Then enqueue messages onto `llm-requests`:

```json
{
  "job_id": 12345,
  "name": "summarize-meeting",
  "system_prompt": "You are a concise notes assistant.",
  "user_prompt": "Summarize the following transcript in three bullets: ...",
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "thinking": false,
  "temperature": 0.3,
  "max_tokens": 512
}
```

Results land on `llm-results` as JSON with `completion`, `reasoning`, token
counts, and `finish_reason`. Completions larger than ~50 KiB spill to a JSON
blob with a SAS URL. See `SPEC.md` §13–§14 for the full contract.

The worker uses Ollama's **native `/api/chat`** endpoint rather than the
OpenAI-compatible `/v1/chat/completions`. Reason: only the native endpoint
honors the `think` toggle that controls reasoning on Qwen3-family GGUFs —
the OpenAI-compat layer silently ignores `chat_template_kwargs`, `think`,
and `/no_think` directives, and the model keeps reasoning regardless.

## Out of scope

Multiple workflow templates, ControlNet, img2img, Managed Identity,
dead-letter handling, health endpoint, container manifests — future work.
