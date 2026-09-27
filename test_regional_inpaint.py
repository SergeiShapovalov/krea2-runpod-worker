"""Focused regression tests; GPU inference is verified separately on Runpod."""

import base64
import io

import pytest
from PIL import Image, ImageChops, ImageDraw

import handler
import regional_inpaint as regional


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def payload():
    image = Image.new("RGB", (128, 96), (40, 80, 120))
    regions = []
    for slot, box in [(1, (8, 16, 51, 79)), (2, (76, 16, 119, 79))]:
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        regions.append({"lora_slot": slot, "prompt": f"PERSON{slot}, adult portrait",
                        "mask_base64": encoded(mask), "crop_size": 256})
    return {"mode": "regional_inpaint", "image_base64": encoded(image),
            "inpaint_regions": regions, "seed": 42, "use_lora": True,
            "use_lora_2": True, "lora_source": "owner/first", "lora_2_source": "owner/second"}


def test_single_lora_graph_and_original_generation(payload):
    params = handler.normalize_input(payload)
    specs = [{"slot": slot, "source": f"owner/person{slot}", "name": f"person{slot}.safetensors",
              "strength_model": 0.85, "strength_clip": 0.85} for slot in (1, 2)]
    for spec in specs:
        graph = handler.build_inpaint_workflow(params, "base.safetensors", spec, "crop.png", "mask.png")
        loaders = [node for node in graph.values() if node["class_type"] == "LoraLoader"]
        assert len(loaders) == 1
        assert loaders[0]["inputs"]["lora_name"] == spec["name"]
        assert loaders[0]["inputs"]["model"] == ["52", 0]
        assert loaders[0]["inputs"]["clip"] == ["53", 0]
        assert graph["54"]["inputs"]["latent_image"] == ["64", 0]
        assert graph["64"]["class_type"] == "SetLatentNoiseMask"
        assert graph["62"]["inputs"]["channel"] == "red"
    original = handler.build_workflow(params, lora_specs=specs)
    assert original["71"]["inputs"]["model"] == ["70", 0]
    assert original["57"]["class_type"] == "EmptyLatentImage"


def test_sequential_passes_preserve_other_face_and_background(payload, monkeypatch, tmp_path):
    params = handler.normalize_input(payload)
    monkeypatch.setattr(handler, "COMFY_INPUT_DIR", tmp_path)
    calls, step_images = [], []
    def fake_render(graph, step):
        calls.append((graph, step))
        color = "red" if len(calls) == 1 else "green"
        generated = Image.new("RGB", (step["width"], step["height"]), color)
        return [handler.encode_image(generated, "crop.png")], None
    original_composite = regional.composite_crop
    def remember(*args):
        image = original_composite(*args)
        step_images.append(image.copy())
        return image
    monkeypatch.setattr(handler, "execute_workflow", fake_render)
    monkeypatch.setattr(regional, "composite_crop", remember)
    specs = [{"slot": slot, "source": f"owner/person{slot}", "name": f"person{slot}.safetensors",
              "strength_model": 0.85, "strength_clip": 0.85} for slot in (1, 2)]
    images, passes, _ = handler.render_regions(params, "base.safetensors", specs)
    assert [p["seed"] for p in passes] == [42, 43]
    assert [p["lora_slot"] for p in passes] == [1, 2]
    assert all(p["loras_applied"] == 1 and p["unchanged_outside_mask"] for p in passes)
    assert [call[0]["70"]["inputs"]["lora_name"] for call in calls] == [s["name"] for s in specs]
    assert not any(tmp_path.iterdir())
    final = regional.decode_image(images[0]["data"], "result")
    union = ImageChops.lighter(*(r["mask"] for r in params["inpaint_regions"]))
    assert regional.unchanged_outside_mask(params["input_image"], final, union)
    left = params["inpaint_regions"][0]["mask"].getbbox()
    assert step_images[0].crop(left).tobytes() == final.crop(left).tobytes()
    assert final.tobytes() != params["input_image"].tobytes()
    assert final.size == (128, 96)


@pytest.mark.parametrize("change", ["overlap", "empty", "size", "slot", "duplicate", "prompt", "batch", "denoise", "crop"])
def test_invalid_regions_are_rejected(payload, change):
    region = payload["inpaint_regions"][1]
    if change == "overlap":
        region["mask_base64"] = payload["inpaint_regions"][0]["mask_base64"]
    elif change == "empty":
        region["mask_base64"] = encoded(Image.new("L", (128, 96)))
    elif change == "size":
        region["mask_base64"] = encoded(Image.new("L", (64, 64), 255))
    elif change == "slot":
        region["lora_slot"] = 3
    elif change == "duplicate":
        region["lora_slot"] = 1
    elif change == "prompt":
        region["prompt"] = ""
    elif change == "batch":
        payload["batch_size"] = 2
    elif change == "denoise":
        region["denoise"] = float("nan")
    elif change == "crop":
        region["crop_size"] = 777
    with pytest.raises(ValueError):
        handler.normalize_input(payload)


def test_bad_base64_and_empty_image():
    for value in [None, "not-an-image", "data:image/png,abc"]:
        with pytest.raises(ValueError):
            regional.decode_image(value, "input")


def test_generate_still_requires_prompt():
    with pytest.raises(RuntimeError, match="prompt"):
        handler.normalize_input({"use_lora": False})
