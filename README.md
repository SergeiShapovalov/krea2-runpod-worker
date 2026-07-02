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

Downloaded HF checkpoint files are symlinked from the HF cache into ComfyUI's `diffusion_models` directory to avoid duplicating large files. Model manifest entries may include a `revision` field, which is passed to Hugging Face downloads so Runpod model-reference caches for checkpoint-only branches are used instead of downloading `main` at request time.
