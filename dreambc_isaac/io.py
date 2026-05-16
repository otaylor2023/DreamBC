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


def write_camera_rollout_videos(
    output_dir: Path,
    frames_by_camera: dict[str, list[np.ndarray]],
    *,
    fps: float,
) -> dict[str, str]:
    """Write one H.264 MP4 per camera under ``output_dir/camera_videos``.

    Returns ``{camera_name: absolute_path_str}`` for files written. Empty if
    ``imageio`` is unavailable or every camera had zero frames.
    """
    written: dict[str, str] = {}
    if not frames_by_camera:
        return written
    try:
        import imageio
    except ImportError:
        print("Warning: imageio is not installed; skipping camera rollout videos.")
        return written

    video_dir = output_dir / "camera_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in frames_by_camera.items():
        if not frames:
            continue
        arr_list = [np.asarray(f, dtype=np.uint8) for f in frames]
        if arr_list[0].ndim != 3 or arr_list[0].shape[-1] != 3:
            print(f"Warning: skip video for '{name}': expected (H,W,3) uint8, got {arr_list[0].shape}")
            continue
        out_path = video_dir / f"{name}.mp4"
        try:
            imageio.mimwrite(
                out_path,
                arr_list,
                fps=float(fps),
                codec="libx264",
                quality=8,
            )
        except Exception as exc:
            print(f"Warning: could not write video for camera '{name}': {type(exc).__name__}: {exc}")
            continue
        written[name] = str(out_path.resolve())
    return written

