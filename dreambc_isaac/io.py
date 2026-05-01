"""Filesystem and image output helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def save_png(path: Path, image: np.ndarray) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    Image.fromarray(image).save(path)
    return True

