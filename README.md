# Krea2 Runpod Worker

Runpod Serverless worker for the custom Krea2 Turbo FP8 ComfyUI checkpoint.

The Docker image does not contain base model weights. The endpoint must be created with:

```bash
--model-reference https://huggingface.co/pakkonen/krea2-base-bundle:main
```

At worker startup `prepare_models.py` waits for Runpod's HF model cache and symlinks:

- `diffusion_models/redcraftKREA2RedMix_krea2Edition.safetensors`
- `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- `vae/qwen_image_vae.safetensors`

The handler accepts a compact JSON generation request and returns PNG images as base64.
Generated images are received from ComfyUI through `SaveImageWebsocket`; the workflow does not use `SaveImage` output files.

The request can select the diffusion checkpoint with:

- `model`: alias from `pakkonen/krea2-model-zoo/models.json`, for example `redcraft` or `civitai-3075206`.
- `model_source` + `model_filename`: override with an HF repo/file, HF file URL, or direct `.safetensors` URL.

LoRA inputs are applied through ComfyUI `LoraLoader` nodes:

- `use_lora`, `lora_source`, `lora_filename`, `lora_strength_model`, `lora_strength_clip`.
- Optional second LoRA: `use_lora_2`, `lora_2_source`, `lora_2_filename`, `lora_2_strength_model`, `lora_2_strength_clip`.

When both are enabled, the worker chains them in order: base model/clip -> LoRA 1 -> LoRA 2.

## Separate identities: regional inpainting (v0.1.14)

The release workflow uses `Dockerfile.release`, layering the handler onto the
digest-pinned v0.1.13 image so this change does not upgrade ComfyUI/CUDA/FP8
dependencies. `Dockerfile` remains the full runtime rebuild recipe.

Runtime placement requirements:

- The pinned runtime's PyTorch uses CUDA 13.0. Set the endpoint's minimum host
  CUDA version to `13.0`; the inherited NVIDIA image banner saying 12.6.3 is
  not the PyTorch requirement. Hosts with a 12.8 driver fail the GPU preflight.
- REST v2 supports a narrow update:
  `PATCH https://api.runpod.io/v2/serverless/<id>` with
  `{"gpu":{"minCudaVersion":"13.0"}}`. This preserves GPU pools and exclusions.
- If private HF preloading stalls before the container starts at
  `initializing model files`, the worker supports direct authenticated downloads
  via `MODEL_REPO_ID` and `HF_TOKEN` after `MODEL_CACHE_WAIT_SECONDS`. Removing
  the endpoint's model references selects this path without deleting the HF
  repository. Preserve the endpoint environment and scaling limits explicitly
  when using legacy GraphQL `saveEndpoint`: omitted fields can reset to defaults.
- A template update does not prove every worker has rolled forward. Check
  `diagnostics.worker_version`, FP8 availability and a real changed-seed job.

Global LoRA chaining does not assign identities to separate people. Use
`mode="regional_inpaint"` with an existing image and non-overlapping face masks:

```json
{
  "mode": "regional_inpaint",
  "image_base64": "<base64 PNG, JPEG or WEBP>",
  "use_lora": true,
  "lora_source": "<first-person-repo>",
  "use_lora_2": true,
  "lora_2_source": "<second-person-repo>",
  "inpaint_regions": [
    {"lora_slot": 1, "prompt": "<first trigger>, adult portrait in scene lighting", "mask_base64": "<base64 grayscale PNG>", "denoise": 0.55},
    {"lora_slot": 2, "prompt": "<second trigger>, adult portrait in scene lighting", "mask_base64": "<base64 grayscale PNG>", "denoise": 0.55}
  ]
}
```

Wrap this object in Runpod's `input`. Existing model, sampler, seed and LoRA
strength controls apply. Each mask must match the source dimensions: white is
editable, black is protected. Masks may not overlap. Each enabled slot may be
used once; one or two regions are supported. This mode requires batch/count 1.
Source dimensions are inferred from the image. A separate prompt is required
for each region; do not put both identity triggers in a region's prompt.

Set `denoise` explicitly. The handler defaults to 0.8 when omitted; in the
paired beach test this sometimes moved faces within their masks. The comparison
runner uses 0.55 to preserve the source pose more closely, at the cost of retaining
more of the source face. Inspect mask boundaries and likeness at full resolution.

Passes run in array order, with seeds `seed`, `seed+1`. Every pass starts from
the unpatched base model/CLIP and applies exactly its selected LoRA. The previous
pass's image, not its model patches, is carried forward. The mask bounds plus
context are enlarged for sampling (default longest side 768), then composited
back. Pixels outside each input mask are preserved exactly, including the other
person's previously edited face. Optional per-region controls: `denoise` (0..1,
excluding 0), `crop_size` (256..1536, multiple of 16), `context` (0..2, default
0.35), `feather` (0..64 source pixels, default 8; inward only).

This isolates LoRA application; likeness still needs visual verification and
depends on the LoRA, mask, source face and denoise strength. The response records
one-LoRA-per-pass provenance, seeds, crop bounds and outside-mask invariance in
`meta.region_passes`. Diagnostics report worker version and available modes.

Downloaded HF checkpoint files are symlinked from the HF cache into ComfyUI's `diffusion_models` directory to avoid duplicating large files. Model manifest entries may include a `revision` field, which is passed to Hugging Face downloads so Runpod model-reference caches for checkpoint-only branches are used instead of downloading `main` at request time.
