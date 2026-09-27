"""Validated, sequential single-LoRA crops with pixel-exact mask boundaries."""

from __future__ import annotations

import base64
import binascii
import io
import math

from PIL import Image, ImageChops, ImageFilter

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def decode_image(value: str, label: str, mode: str = "RGB") -> Image.Image:
    if not isinstance(value, str) or not value or len(value) > MAX_IMAGE_BYTES * 4 // 3 + 256:
        raise ValueError(f"{label} must be a base64 image of at most 20 MiB.")
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError(f"{label} must use base64 encoding.")
    try:
        raw = base64.b64decode(value, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 20 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("only PNG, JPEG and WEBP are supported")
            if not all(64 <= side <= 4096 for side in image.size):
                raise ValueError("image dimensions must be 64..4096")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated images are not supported")
            image.load()
            return image.convert(mode)
    except (binascii.Error, OSError, ValueError) as exc:
        # Never include the supplied image data in validation errors.
        raise ValueError(f"Invalid {label}: {exc}") from exc


def normalize_request(job_input: dict, enabled_slots: set[int]) -> dict:
    if int(job_input.get("num_images", 1)) != 1 or int(job_input.get("batch_size", 1)) != 1:
        raise ValueError("regional_inpaint requires num_images=1 and batch_size=1.")
    image = decode_image(job_input.get("image_base64"), "image_base64")
    raw_regions = job_input.get("inpaint_regions")
    if not isinstance(raw_regions, list) or not 1 <= len(raw_regions) <= 2:
        raise ValueError("inpaint_regions must contain one or two regions.")
    occupied = Image.new("L", image.size, 0)
    slots = set()
    regions = []
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            raise ValueError("Every inpaint region must be an object.")
        slot = int(raw.get("lora_slot", 0))
        if slot not in enabled_slots or slot in slots:
            raise ValueError("Each region needs a distinct enabled lora_slot (1 or 2).")
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Each region requires its own single-person prompt.")
        mask = decode_image(raw.get("mask_base64"), f"region {index + 1} mask", "L")
        if mask.size != image.size:
            raise ValueError("Every mask must match the input image dimensions.")
        support = mask.point(lambda value: 255 if value else 0)
        if not support.getbbox():
            raise ValueError("Region masks cannot be empty.")
        if ImageChops.multiply(occupied, support).getbbox():
            raise ValueError("Region masks must not overlap; each face needs its own mask.")
        occupied = ImageChops.lighter(occupied, support)
        slots.add(slot)
        denoise = float(raw.get("denoise", 0.8))
        feather = int(raw.get("feather", 8))
        context = float(raw.get("context", 0.35))
        crop_size = int(raw.get("crop_size", 768))
        if not math.isfinite(denoise) or not 0 < denoise <= 1:
            raise ValueError("Region denoise must be greater than 0 and at most 1.")
        if not 0 <= feather <= 64 or not math.isfinite(context) or not 0 <= context <= 2:
            raise ValueError("Region feather must be 0..64 and context must be 0..2.")
        if not 256 <= crop_size <= 1536 or crop_size % 16:
            raise ValueError("Region crop_size must be a multiple of 16 in 256..1536.")
        regions.append({
            "lora_slot": slot, "prompt": prompt, "mask": mask,
            "denoise": denoise, "feather": feather,
            "context": context, "crop_size": crop_size,
        })
    return {"input_image": image, "inpaint_regions": regions,
            "width": image.width, "height": image.height}


def prepare_crop(image: Image.Image, region: dict):
    mask = region["mask"]
    x0, y0, x1, y1 = mask.getbbox()
    padding = round(max(x1 - x0, y1 - y0) * region["context"])
    box = (max(0, x0 - padding), max(0, y0 - padding),
           min(image.width, x1 + padding), min(image.height, y1 + padding))
    crop = image.crop(box)
    scale = region["crop_size"] / max(crop.size)
    size = tuple(max(64, round(side * scale / 16) * 16) for side in crop.size)
    return (crop.resize(size, Image.Resampling.LANCZOS),
            mask.crop(box).resize(size, Image.Resampling.BILINEAR), box)


def composite_crop(image: Image.Image, generated: Image.Image, region: dict, box) -> Image.Image:
    original = image.crop(box)
    mask = region["mask"].crop(box)
    if region["feather"]:
        # Feather inward only: no previously black pixel may become editable.
        mask = ImageChops.darker(mask, mask.filter(ImageFilter.GaussianBlur(region["feather"])))
    generated = generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    blended = Image.composite(generated, original, mask)
    output = image.copy()
    output.paste(blended, box[:2])
    return output


def unchanged_outside_mask(before: Image.Image, after: Image.Image, mask: Image.Image) -> bool:
    outside = mask.point(lambda value: 0 if value else 255)
    difference = ImageChops.difference(before, after)
    return all(ImageChops.multiply(channel, outside).getbbox() is None
               for channel in difference.split())
