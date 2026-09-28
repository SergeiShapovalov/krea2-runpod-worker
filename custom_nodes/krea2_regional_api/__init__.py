"""API-only subset of januspluto/ComfyUI-Krea2-Regional.

Upstream: 307081f2b954d9f5e683dadafdb53ca971ebdbfd (MIT; see LICENSE).
Local changes: native ComfyUI diffusers key aliases and fail-closed LoRA loading.
No UI routes, captioning, downloads, or unrelated nodes are loaded.
"""

from .krea2_regional import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
