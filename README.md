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
