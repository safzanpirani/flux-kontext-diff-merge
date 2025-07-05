"""
Flux Kontext Diff Merge - ComfyUI Custom Node

This node preserves image quality by selectively merging only the changed regions
from AI-generated edits back into the original image.
"""

from .flux_kontext_diff_merge import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 