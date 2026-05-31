"""Discover input MP4s from Isaac rollouts, videos/, or explicit paths."""

from __future__ import annotations

from pathlib import Path


def discover_videos(
    *,
    rollout_dir: Path | None = None,
    videos_dir: Path | None = None,
    camera: str | None = None,
    paths: list[Path] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Return sorted unique MP4 paths from the given sources."""
    found: list[Path] = []

    if paths:
        for p in paths:
            path = Path(p).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            if path.suffix.lower() != ".mp4":
                raise ValueError(f"Expected .mp4 file: {path}")
            found.append(path)

    if rollout_dir is not None:
        rollout = Path(rollout_dir).expanduser().resolve()
        camera_dir = rollout / "camera_videos"
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"No camera_videos/ under rollout dir: {rollout}")
        if camera:
            candidate = camera_dir / f"{camera}.mp4"
            if not candidate.is_file():
                raise FileNotFoundError(f"Camera video not found: {candidate}")
            found.append(candidate)
        else:
            found.extend(sorted(camera_dir.glob("*.mp4")))

    if videos_dir is not None:
        vdir = Path(videos_dir).expanduser().resolve()
        if not vdir.is_dir():
            raise FileNotFoundError(f"Videos directory not found: {vdir}")
        for mp4 in sorted(vdir.glob("*.mp4")):
            if output_dir is not None and output_dir in mp4.parents:
                continue
            found.append(mp4)

    # Preserve order, drop duplicates
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique
