"""Focused payload, graph, lifecycle and diffusers-key regression checks."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import handler
import regional_generate as regional


@pytest.fixture
def payload():
    return {"mode": "regional_generate", "prompt": "Two adult friends on a beach",
            "width": 1080, "height": 1920, "seed": 123,
            "use_lora": True, "use_lora_2": True,
            "lora_source": "owner/a", "lora_2_source": "owner/b",
            "regions": [
                {"lora_slot": 1, "prompt": "PERSON_A, adult woman in teal swimsuit",
                 "box": [0.05, 0.08, 0.62, 0.95]},
                {"lora_slot": 2, "prompt": "PERSON_B, adult woman in red swimsuit",
                 "box": [0.45, 0.1, 0.97, 0.96]},
            ]}


def specs():
    return [{"slot": slot, "name": f"person{slot}.safetensors", "strength_model": 0.85}
            for slot in (1, 2)]


def test_full_scene_routes_both_loras_without_global_loader(payload):
    params = handler.normalize_input(payload)
    graph = regional.build_workflow(handler.build_workflow(params, lora_specs=[]),
                                    params, specs(), ["a.png", "b.png"])
    assert not any(n["class_type"] in {"LoraLoader", "VAEEncode", "SetLatentNoiseMask"} for n in graph.values())
    assert graph["54"]["inputs"]["latent_image"] == ["57", 0]
    assert graph["54"]["inputs"]["model"] == ["150", 0]
    assert graph["54"]["inputs"]["positive"] == ["150", 1]
    assert graph["55"]["inputs"]["conditioning"] == ["150", 1]
    for offset, spec in zip((100, 110), specs()):
        assert graph[str(offset + 3)]["inputs"]["lora_name"] == spec["name"]
        assert graph[str(offset + 2)]["inputs"]["clip"] == ["53", 0]
        assert graph[str(offset + 4)]["inputs"]["loras"] == [str(offset + 3), 0]
    assert graph["114"]["inputs"]["prev_regions"] == ["104", 0]
    assert graph["150"]["inputs"]["exclusive_masks"] is True
    assert graph["150"]["inputs"]["unmaskable_layers"] == "skip"


def test_renderer_pads_masks_and_cleans_temp_files(payload, monkeypatch, tmp_path):
    monkeypatch.setattr(handler, "COMFY_INPUT_DIR", tmp_path)
    params = handler.normalize_input(payload)
    calls = []
    def render(graph, step):
        calls.append(graph)
        for node in ("100", "110"):
            with Image.open(tmp_path / graph[node]["inputs"]["image"]) as mask:
                assert mask.size == (1088, 1920)
                assert mask.getpixel((0, 100)) == (0, 0, 0)
        return [handler.encode_image(Image.new("RGB", (1080, 1920)), "scene.png")], None
    monkeypatch.setattr(handler, "execute_workflow", render)
    images, meta, debug = handler.render_regional_scene(params, "base.safetensors", specs())
    assert len(calls) == len(images) == 1
    assert [r["lora_slot"] for r in meta["regions"]] == [1, 2]
    assert not meta["source_image"] and not meta["clip_lora_applied"]
    assert debug is None and not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["prompt", "duplicate", "missing_slot", "empty", "covered",
                                    "bounds", "nan", "batch", "cfg", "denoise", "source", "options"])
def test_invalid_requests_fail_closed(payload, change):
    if change == "prompt":
        payload["regions"][0]["prompt"] = ""
    elif change == "duplicate":
        payload["regions"][1]["lora_slot"] = 1
    elif change == "missing_slot":
        payload["regions"].pop()
    elif change == "empty":
        payload["regions"][0]["box"] = [0, 0, 0, 1]
    elif change == "covered":
        payload["regions"][1]["box"] = payload["regions"][0]["box"]
    elif change == "bounds":
        payload["regions"][0]["box"][0] = -0.2
    elif change == "nan":
        payload["lora_strength_model"] = float("nan")
    elif change == "batch":
        payload["num_images"] = 2
    elif change == "cfg":
        payload["cfg"] = 2
    elif change == "denoise":
        payload["denoise"] = 0.5
    elif change == "source":
        payload["image_base64"] = "not-used"
    elif change == "options":
        payload["regional_options"] = {"restrict_img_attn": "false"}
    with pytest.raises(ValueError):
        handler.normalize_input(payload)


def test_diffusers_lora_keys_are_mapped_and_unknown_keys_rejected():
    # Execute just the injection function with tiny fakes: no host torch/GPU
    # install. The complete node is exercised on the live GPU separately.
    source = Path("custom_nodes/krea2_regional_api/krea2_regional.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_inject_lora")
    class Layer:
        weight = SimpleNamespace(ndim=2)
        def __call__(self):
            pass
    layer = Layer()
    mapped = {}
    patcher = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(unet_config={})),
        get_model_object=lambda name: SimpleNamespace(named_modules=lambda: [("first", layer)])
            if name == "diffusion_model" else layer,
        add_object_patch=lambda name, obj: mapped.update({name: obj}),
    )
    namespace = {"logging": logging,
        "comfy": SimpleNamespace(utils=SimpleNamespace(krea2_to_diffusers=lambda config: {"img_in.weight": "first.weight"})),
        "_normalize_lora_sd": lambda sd: sd,
        "_patched_linear": lambda obj: SimpleNamespace(regional_adapters={}),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "injection", "exec"), namespace)
    entry = {"type": "lora", "down": SimpleNamespace(shape=(16, 64)), "up": object(), "orig": "transformer.img_in"}
    inject = namespace["_inject_lora"]
    assert inject(patcher, {"img_in": entry}, "a", 0.85) == 1
    assert "a" in mapped["diffusion_model.first"].regional_adapters
    with pytest.raises(RuntimeError, match="zero spatial"):
        inject(patcher, {"unknown": entry}, "b", 0.85)
    with pytest.raises(RuntimeError, match="unmapped"):
        inject(patcher, {"img_in": entry, "unknown": entry}, "b", 0.85)
