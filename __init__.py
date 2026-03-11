"""Flux Kontext Diff Merge - ComfyUI custom node."""

from pathlib import Path
import sys

try:
    if __package__:
        from .flux_kontext_diff_merge import (
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
    else:
        from flux_kontext_diff_merge import (  # type: ignore
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
except ImportError as exc:
    requirements_path = Path(__file__).resolve().with_name("requirements.txt")
    raise ImportError(
        "Failed to import Flux Kontext Diff Merge dependencies. "
        f"Install them with '{sys.executable} -m pip install -r {requirements_path}'."
    ) from exc

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]