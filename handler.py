from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import os
import random
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import runpod
import websocket
from huggingface_hub import hf_hub_download
from PIL import Image


COMFY_URL = "http://127.0.0.1:8188"
COMFY_WS_URL = "ws://127.0.0.1:8188/ws"
COMFY_MODELS = Path("/comfyui/models")
DIFFUSION_DIR = COMFY_MODELS / "diffusion_models"
LORA_DIR = COMFY_MODELS / "loras"

CUSTOM_MODEL_FILENAME = "redcraftKREA2RedMix_krea2Edition.safetensors"
CLIP_FILENAME = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_FILENAME = "qwen_image_vae.safetensors"
DEFAULT_LORA_FILENAME = os.environ.get("DEFAULT_LORA_FILENAME", "pytorch_lora_weights.safetensors")
DEFAULT_MODEL_ALIAS = os.environ.get("DEFAULT_MODEL_ALIAS", "redcraft")
MODEL_ZOO_REPO_ID = os.environ.get("MODEL_ZOO_REPO_ID", "pakkonen/krea2-model-zoo")
MODEL_MANIFEST_FILENAME = os.environ.get("MODEL_MANIFEST_FILENAME", "models.json")
RUNPOD_HF_CACHE = Path(os.environ.get("HF_HUB_CACHE") or "/runpod-volume/huggingface-cache/hub")

BUILTIN_MODELS = {
    "redcraft": {
        "repo_id": os.environ.get("MODEL_REPO_ID", "pakkonen/krea2-base-bundle"),
        "filename": f"diffusion_models/{CUSTOM_MODEL_FILENAME}",
        "local_name": CUSTOM_MODEL_FILENAME,
    },
}
_MODEL_MANIFEST_CACHE: dict | None = None

SAMPLERS = {
    "er_sde",
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "heun",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "dpmpp_sde",
    "uni_pc",
    "deis",
}
SCHEDULERS = {"simple", "beta", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform"}


def log(message: str) -> None:
    print(f"krea2-handler: {message}", flush=True)


def round_up(value: int, multiple: int = 16) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def wait_for_comfy(timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{COMFY_URL}/", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI API did not become reachable.")


def safe_filename(name: str, fallback: str = "custom_lora.safetensors") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._")
    if not cleaned:
        cleaned = fallback
    if not cleaned.endswith(".safetensors"):
        cleaned += ".safetensors"
    return cleaned


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def hf_download(
    repo_id: str,
    filename: str,
    *,
    force_download: bool = False,
    revision: str | None = None,
) -> Path:
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "token": hf_token(),
        "force_download": force_download,
    }
    if revision:
        kwargs["revision"] = revision
    if RUNPOD_HF_CACHE.exists() or RUNPOD_HF_CACHE.parent.exists():
        kwargs["cache_dir"] = str(RUNPOD_HF_CACHE)
    return Path(hf_hub_download(**kwargs))


def link_or_replace(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == src.resolve():
                return dest.name
        except FileNotFoundError:
            pass
        dest.unlink()
    dest.symlink_to(src)
    return dest.name


def infer_lora_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    candidate = unquote(Path(parsed.path).name)
    if not candidate or "." not in candidate:
        candidate = "custom_lora.safetensors"
    return safe_filename(candidate)


def lora_cache_name(source: str, filename: str | None = None) -> str:
    source_hash = hashlib.sha1(f"{source}|{filename or ''}".encode("utf-8")).hexdigest()[:10]
    base_name = filename or infer_lora_name_from_url(source)
    stem = Path(safe_filename(Path(base_name).name)).stem
    if stem == "custom_lora":
        stem = "lora"
    return safe_filename(f"{stem}_{source_hash}.safetensors")


def model_cache_name(
    source: str,
    filename: str | None = None,
    alias: str | None = None,
    revision: str | None = None,
) -> str:
    source_hash = hashlib.sha1(
        f"{source}|{filename or ''}|{revision or ''}".encode("utf-8")
    ).hexdigest()[:10]
    base_name = alias or filename or unquote(Path(urlparse(source).path).name) or "custom_model.safetensors"
    stem = Path(safe_filename(Path(base_name).name, fallback="custom_model.safetensors")).stem
    return safe_filename(f"{stem}_{source_hash}.safetensors", fallback="custom_model.safetensors")


def download_stream(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, stream=True, headers=headers, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
    tmp.replace(dest)


def normalize_hf_blob_url(source: str) -> tuple[str, str] | None:
    parsed = urlparse(source)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
        return "/".join(parts[:2]), "/".join(parts[4:])
    return None


def ensure_lora(lora_source: str, lora_filename: str) -> str:
    source = (lora_source or "").strip()
    filename = (lora_filename or "").strip()
    if not source:
        raise RuntimeError("LoRA source is required when use_lora=true.")

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        hf_blob = normalize_hf_blob_url(source)
        if hf_blob:
            repo_id, file_in_repo = hf_blob
            return ensure_lora(f"hf://{repo_id}", file_in_repo)
        local_name = lora_cache_name(source)
        lora_path = LORA_DIR / local_name
        download_stream(source, lora_path)
        return lora_path.name

    if source.startswith("hf://"):
        source = source[5:]
    if "/" not in source:
        raise RuntimeError("LoRA source must be an HF repo id, HF file URL, or direct safetensors URL.")

    file_in_repo = filename or DEFAULT_LORA_FILENAME
    local_name = lora_cache_name(source, file_in_repo)
    lora_path = LORA_DIR / local_name
    if lora_path.exists() and lora_path.stat().st_size > 1_000_000:
        return lora_path.name

    downloaded = hf_download(source, file_in_repo)
    shutil.copyfile(downloaded, lora_path)
    return lora_path.name


def load_model_manifest() -> dict:
    global _MODEL_MANIFEST_CACHE
    manifest = {"default": DEFAULT_MODEL_ALIAS, "models": dict(BUILTIN_MODELS)}
    try:
        manifest_path = hf_download(MODEL_ZOO_REPO_ID, MODEL_MANIFEST_FILENAME, force_download=True)
        remote = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(remote.get("models"), dict):
            manifest["models"].update(remote["models"])
        if remote.get("default"):
            manifest["default"] = str(remote["default"])
    except Exception as exc:
        if _MODEL_MANIFEST_CACHE is not None:
            return _MODEL_MANIFEST_CACHE
        log(f"model manifest unavailable, using built-ins: {exc}")
    _MODEL_MANIFEST_CACHE = manifest
    return manifest


def ensure_model_from_source(
    source: str,
    filename: str,
    alias: str | None = None,
    revision: str | None = None,
) -> str:
    source = (source or "").strip()
    filename = (filename or "").strip()
    if not source:
        raise RuntimeError("Model source is empty.")

    DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        hf_blob = normalize_hf_blob_url(source)
        if hf_blob:
            repo_id, file_in_repo = hf_blob
            return ensure_model_from_source(f"hf://{repo_id}", file_in_repo, alias=alias)
        local_name = model_cache_name(source, alias=alias, revision=revision)
        model_path = DIFFUSION_DIR / local_name
        download_stream(source, model_path)
        return model_path.name

    if source.startswith("hf://"):
        source = source[5:]
    if "/" not in source:
        local_path = DIFFUSION_DIR / safe_filename(source, fallback="custom_model.safetensors")
        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            return local_path.name
        raise RuntimeError(f"Model source is not an HF repo id, URL, or local model filename: {source}")
    if not filename:
        raise RuntimeError("model_filename is required when model_source is an HF repo id.")

    local_name = model_cache_name(source, filename, alias=alias, revision=revision)
    local_path = DIFFUSION_DIR / local_name
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return local_path.name

    downloaded = hf_download(source, filename, revision=revision)
    return link_or_replace(downloaded, local_path)


def ensure_model(model: str, model_source: str = "", model_filename: str = "") -> tuple[str, str]:
    requested = (model or "").strip() or DEFAULT_MODEL_ALIAS
    source = (model_source or "").strip()
    filename = (model_filename or "").strip()

    if source:
        return ensure_model_from_source(source, filename, alias=requested), requested
    if requested.startswith(("http://", "https://")):
        return ensure_model_from_source(requested, filename, alias=None), requested
    if requested.startswith("hf://") or ("/" in requested and filename):
        return ensure_model_from_source(requested, filename, alias=Path(filename).stem or None), requested

    local_path = DIFFUSION_DIR / safe_filename(requested, fallback="custom_model.safetensors")
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        return local_path.name, requested

    manifest = load_model_manifest()
    alias = manifest.get("default", DEFAULT_MODEL_ALIAS) if requested in {"default", "base"} else requested
    entry = (manifest.get("models") or {}).get(alias)
    if not isinstance(entry, dict):
        available = ", ".join(sorted((manifest.get("models") or {}).keys()))
        raise RuntimeError(f"Unknown model '{requested}'. Available models: {available}")

    local_name = entry.get("local_name")
    if local_name:
        local_path = DIFFUSION_DIR / safe_filename(str(local_name), fallback="custom_model.safetensors")
        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            return local_path.name, alias

    return ensure_model_from_source(
        str(entry.get("repo_id") or MODEL_ZOO_REPO_ID),
        str(entry.get("filename") or ""),
        alias=alias,
        revision=str(entry.get("revision") or ""),
    ), alias


def build_workflow(
    params: dict,
    lora_name: str = "",
    model_name: str = CUSTOM_MODEL_FILENAME,
    lora_specs: list[dict] | None = None,
) -> dict:
    requested_width = int(params["width"])
    requested_height = int(params["height"])
    internal_width = round_up(requested_width, 16)
    internal_height = round_up(requested_height, 16)
    if lora_specs is None:
        lora_specs = []
        if bool(params["use_lora"]) and lora_name:
            lora_specs.append(
                {
                    "name": lora_name,
                    "strength_model": float(params["lora_strength_model"]),
                    "strength_clip": float(params["lora_strength_clip"]),
                }
            )
    model_node = "52"
    clip_node = "53"
    clip_output_index = 0
    negative_node = "55" if params["zero_negative"] else "59"

    workflow = {
        "52": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        "53": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_FILENAME, "type": "krea2", "device": "default"}},
        "58": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILENAME}},
        "57": {"class_type": "EmptyLatentImage", "inputs": {"width": internal_width, "height": internal_height, "batch_size": int(params["batch_size"])}},
        "51": {"class_type": "CLIPTextEncode", "inputs": {"text": params["prompt"], "clip": [clip_node, clip_output_index]}},
        "59": {"class_type": "CLIPTextEncode", "inputs": {"text": params["negative_prompt"], "clip": [clip_node, clip_output_index]}},
        "55": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["51", 0]}},
        "54": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(params["seed"]),
                "steps": int(params["steps"]),
                "cfg": float(params["cfg"]),
                "sampler_name": params["sampler"],
                "scheduler": params["scheduler"],
                "denoise": float(params["denoise"]),
                "model": [model_node, 0],
                "positive": ["51", 0],
                "negative": [negative_node, 0],
                "latent_image": ["57", 0],
            },
        },
        "56": {"class_type": "VAEDecode", "inputs": {"samples": ["54", 0], "vae": ["58", 0]}},
        "30": {"class_type": "SaveImageWebsocket", "inputs": {"images": ["56", 0]}},
    }

    for index, spec in enumerate(lora_specs):
        node_id = str(70 + index)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": [model_node, 0],
                "clip": [clip_node, clip_output_index],
                "lora_name": str(spec["name"]),
                "strength_model": float(spec["strength_model"]),
                "strength_clip": float(spec["strength_clip"]),
            },
        }
        model_node = node_id
        clip_node = node_id
        clip_output_index = 1

    workflow["51"]["inputs"]["clip"] = [clip_node, clip_output_index]
    workflow["59"]["inputs"]["clip"] = [clip_node, clip_output_index]
    workflow["54"]["inputs"]["model"] = [model_node, 0]
    return workflow


def queue_prompt(workflow: dict, client_id: str) -> str:
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    if response.status_code >= 400:
        details = response.text
        try:
            object_info = requests.get(f"{COMFY_URL}/object_info", timeout=10).json()
            interesting = {
                key: object_info.get(key)
                for key in ["UNETLoader", "CLIPLoader", "VAELoader", "LoraLoader", "KSampler", "SaveImageWebsocket"]
                if key in object_info
            }
            details += "\nobject_info=" + json.dumps(interesting, ensure_ascii=False)[:6000]
        except Exception as info_error:
            details += f"\nobject_info_error={info_error}"
        raise RuntimeError(f"ComfyUI prompt error {response.status_code}: {details[:10000]}")
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
    return data["prompt_id"]


def connect_websocket(client_id: str) -> websocket.WebSocket:
    ws = websocket.WebSocket()
    ws.connect(f"{COMFY_WS_URL}?clientId={client_id}", timeout=30)
    return ws


def encode_image(image: Image.Image, filename: str) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "filename": filename,
        "type": "base64",
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "width": image.width,
        "height": image.height,
    }


def normalize_image(image: Image.Image, requested_width: int, requested_height: int) -> Image.Image:
    image = image.convert("RGB")
    if image.size != (requested_width, requested_height):
        left = max(0, (image.width - requested_width) // 2)
        top = max(0, (image.height - requested_height) // 2)
        image = image.crop((left, top, min(left + requested_width, image.width), min(top + requested_height, image.height)))
        if image.size != (requested_width, requested_height):
            image = image.resize((requested_width, requested_height), Image.Resampling.LANCZOS)
    return image


def decode_websocket_image(payload: bytes) -> Image.Image:
    # ComfyUI websocket image frames include an 8-byte binary event header.
    for candidate in (payload[8:], payload):
        try:
            return Image.open(io.BytesIO(candidate))
        except Exception:
            continue
    raise RuntimeError("Could not decode websocket image payload.")


def wait_for_websocket_images(
    ws: websocket.WebSocket,
    prompt_id: str,
    requested_width: int,
    requested_height: int,
    seed: int,
    timeout: int,
) -> list[dict]:
    deadline = time.time() + timeout
    current_node = None
    images = []
    seen = set()

    while time.time() < deadline:
        ws.settimeout(max(1, min(30, int(deadline - time.time()))))
        message = ws.recv()
        if isinstance(message, str):
            event = json.loads(message)
            data = event.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue
            if event.get("type") == "execution_error":
                raise RuntimeError(json.dumps(data, ensure_ascii=False)[:4000])
            if event.get("type") == "executing":
                current_node = data.get("node")
                if current_node is None:
                    break
            continue

        if current_node != "30":
            continue

        image = normalize_image(decode_websocket_image(message), requested_width, requested_height)
        encoded = encode_image(image, f"krea2_websocket_seed{seed}_{len(images) + 1}.png")
        fingerprint = hashlib.sha1(encoded["data"].encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        images.append(encoded)

    if not images:
        raise RuntimeError("No image received from SaveImageWebsocket.")
    return images


def summarize_history(history: dict) -> dict:
    outputs = {}
    for node_id, output in (history.get("outputs") or {}).items():
        node_summary = {}
        for key, value in output.items():
            if key == "images" and isinstance(value, list):
                node_summary[key] = [
                    {
                        "filename": item.get("filename"),
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    }
                    for item in value
                    if isinstance(item, dict)
                ]
            else:
                node_summary[key] = value
        outputs[node_id] = node_summary
    return {
        "outputs": outputs,
        "status": history.get("status"),
    }

def runtime_diagnostics() -> dict:
    diagnostics = {
        "python": sys.version,
        "executable": sys.executable,
    }

    try:
        import torch

        diagnostics["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        diagnostics["torch_error"] = traceback.format_exc(limit=8)

    try:
        diagnostics["comfy_kitchen_distribution"] = importlib.metadata.version("comfy-kitchen")
        import comfy_kitchen

        diagnostics["comfy_kitchen_file"] = getattr(comfy_kitchen, "__file__", None)
        diagnostics["comfy_kitchen_version_attr"] = getattr(comfy_kitchen, "__version__", None)
        diagnostics["comfy_kitchen_backends"] = comfy_kitchen.list_backends()
    except Exception:
        diagnostics["comfy_kitchen_error"] = traceback.format_exc(limit=12)

    try:
        from comfy_kitchen.tensor import TensorCoreFP8Layout, get_layout_class

        diagnostics["comfy_kitchen_tensor"] = {
            "TensorCoreFP8Layout": str(TensorCoreFP8Layout),
            "get_layout_class_fp8": str(get_layout_class("TensorCoreFP8Layout")),
        }
    except Exception:
        diagnostics["comfy_kitchen_tensor_error"] = traceback.format_exc(limit=12)

    try:
        import comfy.quant_ops as quant_ops

        diagnostics["comfy_quant_ops"] = {
            "file": getattr(quant_ops, "__file__", None),
            "_CK_AVAILABLE": getattr(quant_ops, "_CK_AVAILABLE", None),
            "_CK_MXFP8_AVAILABLE": getattr(quant_ops, "_CK_MXFP8_AVAILABLE", None),
            "get_layout_class_fp8": str(quant_ops.get_layout_class("TensorCoreFP8Layout")),
        }
    except Exception:
        diagnostics["comfy_quant_ops_error"] = traceback.format_exc(limit=12)

    try:
        manifest = load_model_manifest()
        diagnostics["model_manifest"] = {
            "repo_id": MODEL_ZOO_REPO_ID,
            "filename": MODEL_MANIFEST_FILENAME,
            "default": manifest.get("default"),
            "aliases": sorted((manifest.get("models") or {}).keys()),
        }
        diagnostics["local_diffusion_models"] = sorted(
            p.name for p in DIFFUSION_DIR.glob("*.safetensors") if p.exists()
        )
    except Exception:
        diagnostics["model_manifest_error"] = traceback.format_exc(limit=8)

    return diagnostics


def normalize_input(job_input: dict) -> dict:
    params = {
        "diagnostics_only": bool(job_input.get("diagnostics_only", False)),
        "model": str(job_input.get("model") or job_input.get("model_alias") or DEFAULT_MODEL_ALIAS).strip(),
        "model_source": str(job_input.get("model_source") or job_input.get("model_repo_id") or ""),
        "model_filename": str(job_input.get("model_filename") or ""),
        "prompt": str(job_input.get("prompt") or "").strip(),
        "negative_prompt": str(job_input.get("negative_prompt") or ""),
        "width": int(job_input.get("width", 1080)),
        "height": int(job_input.get("height", 1920)),
        "steps": int(job_input.get("steps", 12)),
        "cfg": float(job_input.get("cfg", 1.0)),
        "sampler": str(job_input.get("sampler", "euler")),
        "scheduler": str(job_input.get("scheduler", "simple")),
        "denoise": float(job_input.get("denoise", 1.0)),
        "seed": int(job_input.get("seed", random.randint(0, 2**63 - 1))),
        "num_images": int(job_input.get("num_images", 1)),
        "batch_size": int(job_input.get("batch_size", 1)),
        "use_lora": bool(job_input.get("use_lora", True)),
        "lora_source": str(job_input.get("lora_source") or ""),
        "lora_filename": str(job_input.get("lora_filename") or DEFAULT_LORA_FILENAME),
        "lora_strength_model": float(job_input.get("lora_strength_model", 1.0)),
        "lora_strength_clip": float(job_input.get("lora_strength_clip", 1.0)),
        "use_lora_2": bool(job_input.get("use_lora_2", False)),
        "lora_2_source": str(job_input.get("lora_2_source") or ""),
        "lora_2_filename": str(job_input.get("lora_2_filename") or DEFAULT_LORA_FILENAME),
        "lora_2_strength_model": float(job_input.get("lora_2_strength_model", 1.0)),
        "lora_2_strength_clip": float(job_input.get("lora_2_strength_clip", 1.0)),
        "zero_negative": bool(job_input.get("zero_negative", True)),
        "debug": bool(job_input.get("debug", False)),
        "timeout": int(job_input.get("timeout", 900)),
    }
    if params["diagnostics_only"]:
        return params
    if not params["prompt"]:
        raise RuntimeError("prompt is required.")
    if params["sampler"] not in SAMPLERS:
        raise RuntimeError(f"Unsupported sampler: {params['sampler']}")
    if params["scheduler"] not in SCHEDULERS:
        raise RuntimeError(f"Unsupported scheduler: {params['scheduler']}")
    if not 64 <= params["width"] <= 4096 or not 64 <= params["height"] <= 4096:
        raise RuntimeError("width/height must be between 64 and 4096.")
    if not 1 <= params["num_images"] <= 4:
        raise RuntimeError("num_images must be 1..4.")
    if not 1 <= params["batch_size"] <= 2:
        raise RuntimeError("batch_size must be 1..2.")
    if params["use_lora_2"] and not params["use_lora"]:
        raise RuntimeError("use_lora must be true when use_lora_2=true.")
    return params


def handler(job: dict) -> dict:
    started = time.time()
    try:
        wait_for_comfy()
        params = normalize_input(job.get("input") or {})
        if params["diagnostics_only"]:
            return {
                "diagnostics": runtime_diagnostics(),
                "seconds": round(time.time() - started, 3),
            }
        model_name, model_alias = ensure_model(params["model"], params["model_source"], params["model_filename"])
        lora_specs = []
        if params["use_lora"]:
            lora_specs.append(
                {
                    "slot": 1,
                    "source": params["lora_source"],
                    "filename": params["lora_filename"],
                    "name": ensure_lora(params["lora_source"], params["lora_filename"]),
                    "strength_model": params["lora_strength_model"],
                    "strength_clip": params["lora_strength_clip"],
                }
            )
        if params["use_lora_2"]:
            lora_specs.append(
                {
                    "slot": 2,
                    "source": params["lora_2_source"],
                    "filename": params["lora_2_filename"],
                    "name": ensure_lora(params["lora_2_source"], params["lora_2_filename"]),
                    "strength_model": params["lora_2_strength_model"],
                    "strength_clip": params["lora_2_strength_clip"],
                }
            )

        images = []
        seeds = []
        debug_runs = []
        base_seed = int(params["seed"])
        for index in range(params["num_images"]):
            params["seed"] = base_seed + index
            seeds.append(params["seed"])
            workflow = build_workflow(params, model_name=model_name, lora_specs=lora_specs)
            client_id = str(uuid.uuid4())
            ws = connect_websocket(client_id)
            try:
                prompt_id = queue_prompt(workflow, client_id)
                fetched = wait_for_websocket_images(
                    ws,
                    prompt_id,
                    params["width"],
                    params["height"],
                    params["seed"],
                    params["timeout"],
                )
            finally:
                ws.close()
            if params["debug"] or not fetched:
                history = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30).json().get(prompt_id, {})
                debug_runs.append(
                    {
                        "seed": params["seed"],
                        "prompt_id": prompt_id,
                        "history": summarize_history(history),
                    }
                )
            images.extend(fetched)

        result = {
            "images": images,
            "meta": {
                "seconds": round(time.time() - started, 3),
                "seeds": seeds,
                "sampler": params["sampler"],
                "scheduler": params["scheduler"],
                "steps": params["steps"],
                "cfg": params["cfg"],
                "model": model_alias,
                "model_name": model_name,
                "lora_name": lora_specs[0]["name"] if lora_specs else "",
                "loras": lora_specs,
                "model_repo_id": os.environ.get("MODEL_REPO_ID", "pakkonen/krea2-base-bundle"),
                "model_zoo_repo_id": MODEL_ZOO_REPO_ID,
            },
        }
        if debug_runs:
            result["debug"] = debug_runs
        return result
    except Exception as exc:
        log(f"error: {exc}")
        return {"error": str(exc), "seconds": round(time.time() - started, 3)}


runpod.serverless.start({"handler": handler})
