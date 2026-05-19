#!/usr/bin/env python3
"""
Cosmos Transfer 2.5 2B sim-to-real pipeline.

Reads robot sim videos from videos/, transfers them to photorealistic real-world
appearance using the distilled edge model (~75x speedup vs Transfer1 7B at 50 steps).

The distilled edge model requires COSMOS_EXPERIMENTAL_CHECKPOINTS=1 (set automatically).
Models auto-download from HuggingFace on first run — HF_TOKEN must be in your env.

Usage:
    python sim2real.py                          # process all videos/*.mp4
    python sim2real.py --prompt "..."           # custom realism prompt
    python sim2real.py --gpus 8                 # multi-GPU
    python sim2real.py --videos-dir /path/to/vids --output-dir /path/to/out
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
COSMOS_DIR = REPO_ROOT / "cosmos-transfer2.5"
COSMOS_PYTHON = COSMOS_DIR / ".venv" / "bin" / "python"
VIDEOS_DIR = REPO_ROOT / "videos"
OUTPUT_DIR = VIDEOS_DIR / "output"

DEFAULT_PROMPT = (
    "A robot arm performing a precise manipulation task in a photorealistic real-world "
    "environment. High quality video, natural lighting, realistic textures and materials, "
    "detailed scene, no artifacts."
)


def find_videos(videos_dir: Path, output_dir: Path) -> list[Path]:
    return [v for v in sorted(videos_dir.glob("*.mp4")) if output_dir not in v.parents]


def make_spec(video: Path, prompt: str, scene: Path | None = None) -> dict:
    spec: dict = {
        "name": video.stem,
        "prompt": prompt,
        "video_path": str(video.resolve()),
        "guidance": 5 if scene else 3,
        "num_steps": 4,  # distilled edge: 4 steps (~75x vs Transfer1-7B at 50 steps)
        "edge": {
            # control_path omitted → Canny edge extracted on-the-fly from input video
            "control_weight": 1.0,
        },
    }
    if scene:
        spec["image_context_path"] = str(scene.resolve())
    return spec


def run_inference(spec_file: Path, out_dir: Path, gpus: int, cosmos_dir: Path):
    python = str(COSMOS_PYTHON) if COSMOS_PYTHON.exists() else sys.executable
    inference_script = str(cosmos_dir / "examples" / "inference.py")

    env = os.environ.copy()
    env["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"  # required for distilled edge model

    base_args = [
        "-i", str(spec_file),
        "-o", str(out_dir),
        "--model=edge/distilled",
        "--disable-guardrails",
    ]

    if gpus > 1:
        cmd = ["torchrun", f"--nproc_per_node={gpus}", "--master_port=12341", inference_script] + base_args
    else:
        cmd = [python, inference_script] + base_args

    subprocess.run(cmd, check=True, cwd=cosmos_dir, env=env)


def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer 2.5 2B sim-to-real")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cosmos-dir", type=Path, default=COSMOS_DIR)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--scene", type=Path, default=None, help="Real scene image to transfer into (e.g. scenes/IMG_6962.jpeg)")
    args = parser.parse_args()

    if not args.cosmos_dir.exists():
        print(f"ERROR: cosmos-transfer2.5 not found at {args.cosmos_dir}")
        print("Run:  bash setup_cosmos.sh")
        sys.exit(1)

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set — model download will fail if weights aren't cached.")

    videos = find_videos(args.videos_dir, args.output_dir)
    if not videos:
        print(f"No .mp4 files found in {args.videos_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} video(s)  |  model: edge/distilled (4 steps)")
    print(f"Output → {args.output_dir}\n")

    for video in videos:
        print(f"→ {video.name}")
        spec = make_spec(video, args.prompt, args.scene)
        spec_file = args.output_dir / f"{video.stem}_spec.json"
        spec_file.write_text(json.dumps(spec, indent=2))
        out_dir = args.output_dir / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        run_inference(spec_file, out_dir, args.gpus, args.cosmos_dir)
        print(f"   saved → {out_dir}\n")

    print("Done.")


if __name__ == "__main__":
    main()
