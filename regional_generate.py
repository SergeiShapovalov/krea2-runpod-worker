"""Full-scene generation with spatially routed identity LoRAs, not face pasting."""

import math

from PIL import Image, ImageChops, ImageDraw

from regional_inpaint import decode_image

UPSTREAM_REVISION = "307081f2b954d9f5e683dadafdb53ca971ebdbfd"
NODE_TYPES = ("Krea2LoRA", "Krea2RegionalPrompt", "Krea2ApplyRegional")


def normalize_request(job_input, params, enabled_slots):
    if params["num_images"] != 1 or params["batch_size"] != 1:
        raise ValueError("regional_generate requires num_images=1 and batch_size=1.")
    if params["cfg"] != 1.0 or params["denoise"] != 1.0 or not params["zero_negative"]:
        raise ValueError("regional_generate currently requires cfg=1, denoise=1, zero_negative=true.")
    if job_input.get("image_base64") or job_input.get("inpaint_regions"):
        raise ValueError("regional_generate starts from noise; image/inpaint inputs are not accepted.")
    raw_regions = job_input.get("regions")
    if not isinstance(raw_regions, list) or not 1 <= len(raw_regions) <= 2:
        raise ValueError("regions must contain one or two full-character regions.")
    size = (params["width"], params["height"])
    occupied = Image.new("L", size)
    regions, slots = [], set()
    for raw in raw_regions:
        if not isinstance(raw, dict):
            raise ValueError("Each region must be an object.")
        slot = int(raw.get("lora_slot", 0))
        if slot not in enabled_slots or slot in slots:
            raise ValueError("Each region needs a distinct enabled lora_slot.")
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Each region requires its own character prompt.")
        if ("box" in raw) == ("mask_base64" in raw):
            raise ValueError("Supply exactly one normalized box or mask_base64 per region.")
        if "box" in raw:
            box = raw["box"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("box must be [left, top, right, bottom] normalized to 0..1.")
            box = [float(value) for value in box]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError("box must have positive area inside 0..1.")
            coords = [round(box[i] * size[i % 2]) for i in range(4)]
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                raise ValueError("box must cover at least one pixel.")
            mask = Image.new("L", size)
            ImageDraw.Draw(mask).rectangle((coords[0], coords[1], coords[2] - 1, coords[3] - 1), fill=255)
        else:
            mask = decode_image(raw["mask_base64"], "region mask", "L")
            if mask.size != size:
                raise ValueError("Region mask dimensions must match width/height.")
        support = mask.point(lambda value: 255 if value else 0)
        if not ImageChops.subtract(support, occupied).getbbox():
            raise ValueError("Each region must have nonempty pixels outside earlier regions.")
        occupied = ImageChops.lighter(occupied, support)
        strength = params["lora_strength_model" if slot == 1 else "lora_2_strength_model"]
        if not math.isfinite(strength) or not 0 < strength <= 2:
            raise ValueError("Regional model strengths must be in (0, 2].")
        regions.append({"lora_slot": slot, "prompt": prompt, "mask": mask})
        slots.add(slot)
    if slots != enabled_slots:
        raise ValueError("Every enabled LoRA must be assigned to a region.")
    options = job_input.get("regional_options") or {}
    if not isinstance(options, dict) or set(options) - {"adaptive_masks", "restrict_img_attn", "restrict_end_percent"}:
        raise ValueError("Unsupported regional_options.")
    adaptive = options.get("adaptive_masks", "off")
    if adaptive not in {"off", "refine boxes", "free (ignore boxes)"}:
        raise ValueError("Unsupported adaptive_masks mode.")
    restrict = options.get("restrict_img_attn", False)
    end = float(options.get("restrict_end_percent", 1.0))
    if not isinstance(restrict, bool) or not math.isfinite(end) or not 0.05 <= end <= 1:
        raise ValueError("Invalid attention restriction options.")
    return {"regions": regions, "regional_options": {
        "adaptive_masks": adaptive, "restrict_img_attn": restrict,
        "restrict_end_percent": end,
    }}


def build_workflow(workflow, params, lora_specs, mask_names):
    """Extend a CLEAN base graph: no global LoraLoader or image compositing."""
    if any(node["class_type"] == "LoraLoader" for node in workflow.values()):
        raise ValueError("Regional generation needs an unpatched base graph.")
    if len(mask_names) != len(params["regions"]):
        raise ValueError("A mask filename is required for each region.")
    specs = {spec["slot"]: spec for spec in lora_specs}
    previous = None
    for index, (region, mask_name) in enumerate(zip(params["regions"], mask_names)):
        load, mask, text, lora, combine = [str(100 + index * 10 + n) for n in range(5)]
        spec = specs[region["lora_slot"]]
        workflow[load] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
        workflow[mask] = {"class_type": "ImageToMask", "inputs": {"image": [load, 0], "channel": "red"}}
        workflow[text] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["53", 0], "text": region["prompt"]}}
        workflow[lora] = {"class_type": "Krea2LoRA", "inputs": {
            "lora_name": spec["name"], "strength": spec["strength_model"]}}
        inputs = {"conditioning": [text, 0], "mask": [mask, 0], "loras": [lora, 0]}
        if previous:
            inputs["prev_regions"] = [previous, 0]
        workflow[combine] = {"class_type": "Krea2RegionalPrompt", "inputs": inputs}
        previous = combine
    workflow["150"] = {"class_type": "Krea2ApplyRegional", "inputs": {
        "model": ["52", 0], "conditioning": ["51", 0], "regions": [previous, 0],
        "exclusive_masks": True, "adaptive_steps": 2, "adaptive_threshold": 0.45,
        "base_loras_exclude_regions": False, "region_lock_strength": 0.0,
        "region_lock_start": 0.35, "region_lock_end": 0.85,
        "unmaskable_layers": "skip", **params["regional_options"],
    }}
    workflow["54"]["inputs"].update(model=["150", 0], positive=["150", 1])
    workflow["55"]["inputs"]["conditioning"] = ["150", 1]
    return workflow
