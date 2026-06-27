from __future__ import annotations

import json
import os
import time
from pathlib import Path


MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "pakkonen/krea2-base-bundle")
WAIT_SECONDS = int(os.environ.get("MODEL_CACHE_WAIT_SECONDS", "900"))

COMFY_MODELS = Path("/comfyui/models")
RUNPOD_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

FILES = {
    "diffusion_models/redcraftKREA2RedMix_krea2Edition.safetensors": COMFY_MODELS
    / "diffusion_models"
    / "redcraftKREA2RedMix_krea2Edition.safetensors",
    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors": COMFY_MODELS
    / "text_encoders"
    / "qwen3vl_4b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors": COMFY_MODELS / "vae" / "qwen_image_vae.safetensors",
}


def log(message: str) -> None:
    print(f"krea2-prepare: {message}", flush=True)


def repo_cache_dir(repo_id: str) -> Path:
    return RUNPOD_CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"


def find_snapshot() -> Path | None:
    base = repo_cache_dir(MODEL_REPO_ID)
    snapshots = base / "snapshots"
    if not snapshots.exists():
        return None

    candidates = sorted(
        [p for p in snapshots.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if all((candidate / src).exists() for src in FILES):
            return candidate
    return None


def already_linked() -> bool:
    return all(dest.exists() and dest.stat().st_size > 1_000_000 for dest in FILES.values())


def wait_for_snapshot() -> Path:
    if already_linked():
        log("ComfyUI model files already exist")
        return COMFY_MODELS

    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        snapshot = find_snapshot()
        if snapshot is not None:
            return snapshot
        log(f"waiting for Runpod model cache for {MODEL_REPO_ID}")
        time.sleep(5)

    raise RuntimeError(
        f"Timed out waiting for cached HF model {MODEL_REPO_ID} under {RUNPOD_CACHE_ROOT}"
    )


def link_or_replace(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dest.unlink()
    dest.symlink_to(src)
    log(f"linked {dest} -> {src}")


def main() -> None:
    for folder in ["diffusion_models", "text_encoders", "vae", "loras"]:
        (COMFY_MODELS / folder).mkdir(parents=True, exist_ok=True)

    snapshot = wait_for_snapshot()
    if snapshot == COMFY_MODELS:
        return

    for src_rel, dest in FILES.items():
        link_or_replace(snapshot / src_rel, dest)

    summary = {
        "model_repo_id": MODEL_REPO_ID,
        "snapshot": str(snapshot),
        "files": {src: str(dest) for src, dest in FILES.items()},
    }
    Path("/tmp/krea2_models_prepared.json").write_text(json.dumps(summary, indent=2))
    log("model links ready")


if __name__ == "__main__":
    main()
