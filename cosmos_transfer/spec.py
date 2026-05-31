"""Inference JSON spec shared by CLI, local GPU, and RunPod worker."""

from pathlib import Path

DEFAULT_PROMPT = (
    "A robot arm performing a precise manipulation task in a photorealistic real-world "
    "environment. High quality video, natural lighting, realistic textures and materials, "
    "detailed scene, no artifacts."
)


def make_spec(
    video_path: Path | str,
    prompt: str,
    *,
    scene_path: Path | str | None = None,
    name: str | None = None,
) -> dict:
    video = Path(video_path).resolve()
    spec: dict = {
        "name": name or video.stem,
        "prompt": prompt,
        "video_path": str(video),
        "guidance": 5 if scene_path else 3,
        "num_steps": 4,
        "edge": {
            "control_weight": 1.0,
        },
    }
    if scene_path is not None:
        spec["image_context_path"] = str(Path(scene_path).resolve())
    return spec
