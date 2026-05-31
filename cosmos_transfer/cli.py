#!/usr/bin/env python3
"""Cosmos Transfer 2.5 sim-to-real CLI (RunPod Serverless or local GPU)."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from cosmos_transfer import gcs
from cosmos_transfer.paths import DEFAULT_OUTPUT_DIR, DEFAULT_VIDEOS_DIR, REPO_ROOT
from cosmos_transfer.runpod_client import run_and_wait, run_sync
from cosmos_transfer.sources import discover_videos
from cosmos_transfer.spec import DEFAULT_PROMPT


def _process_remote(
    video: Path,
    *,
    prompt: str,
    scene: Path | None,
    output_dir: Path,
    sync: bool,
) -> None:
    job_id = uuid.uuid4().hex
    print(f"  uploading to GCS (job {job_id})...")
    video_gs = gcs.upload_file(video, kind="inputs", job_id=job_id)
    video_url = gcs.signed_url(video_gs)

    job_input: dict = {
        "video_url": video_url,
        "prompt": prompt,
        "name": video.stem,
    }
    if scene is not None:
        scene_gs = gcs.upload_file(scene, kind="inputs", job_id=job_id)
        job_input["scene_url"] = gcs.signed_url(scene_gs)

    if sync:
        print("  RunPod (/runsync)...")
        result = run_sync(job_input)
    else:
        print("  RunPod (async, waiting)...")
        result = run_and_wait(job_input)

    if result.get("error"):
        raise RuntimeError(result["error"])

    output_url = result.get("output_url")
    output_gs = result.get("output_gs_uri")
    if not output_url and not output_gs:
        raise RuntimeError(f"RunPod job returned no output URL: {result}")

    out_dir = output_dir / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{video.stem}_transferred.mp4"

    if output_gs:
        gcs.download_gs_uri(output_gs, dest)
    elif output_url:
        gcs.download_url(output_url, dest)
    print(f"  saved → {dest}")


def _process_local(
    video: Path,
    *,
    prompt: str,
    scene: Path | None,
    output_dir: Path,
    gpus: int,
    cosmos_dir: Path,
) -> None:
    from cosmos_transfer.local import run_inference

    out_dir = output_dir / video.stem
    run_inference(
        video,
        out_dir,
        prompt=prompt,
        scene=scene,
        gpus=gpus,
        cosmos_dir=cosmos_dir,
    )
    generated = out_dir / f"{video.stem}.mp4"
    if generated.is_file():
        print(f"  saved → {generated}")
    else:
        print(f"  saved → {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cosmos Transfer 2.5 2B sim-to-real (RunPod Serverless or --local GPU)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Explicit .mp4 files to process",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help="Isaac rollout dir containing camera_videos/*.mp4",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Single camera name under camera_videos/ (e.g. exterior_image_1_left)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help=f"Directory of .mp4 files (default: {DEFAULT_VIDEOS_DIR} when no other source)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for transferred outputs",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Real scene image for image context (e.g. scenes/IMG_6962.jpeg)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run inference on local GPU via cosmos-transfer2.5 (requires setup_cosmos.sh)",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Local multi-GPU only")
    parser.add_argument(
        "--cosmos-dir",
        type=Path,
        default=None,
        help="Path to cosmos-transfer2.5 clone",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use RunPod /runsync instead of async /run + poll",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    paths = list(args.paths) if args.paths else None
    videos_dir = args.videos_dir
    if not paths and args.rollout_dir is None and videos_dir is None:
        videos_dir = DEFAULT_VIDEOS_DIR

    try:
        videos = discover_videos(
            rollout_dir=args.rollout_dir,
            videos_dir=videos_dir,
            camera=args.camera,
            paths=paths,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print("No .mp4 files found for the given inputs.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = "local GPU" if args.local else "RunPod Serverless"
    print(f"Found {len(videos)} video(s)  |  mode: {mode}  |  model: edge/distilled (4 steps)")
    print(f"Output → {args.output_dir}\n")

    if not args.local and not os.environ.get("RUNPOD_ENDPOINT_ID"):
        print("ERROR: RUNPOD_ENDPOINT_ID is not set.", file=sys.stderr)
        sys.exit(1)

    if not args.local:
        try:
            gcs._bucket_name()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.local and not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set — model download may fail if weights are not cached.")

    from cosmos_transfer.paths import COSMOS_DIR

    cosmos_dir = args.cosmos_dir or COSMOS_DIR

    for video in videos:
        print(f"→ {video.name}")
        try:
            if args.local:
                _process_local(
                    video,
                    prompt=args.prompt,
                    scene=args.scene,
                    output_dir=args.output_dir,
                    gpus=args.gpus,
                    cosmos_dir=cosmos_dir,
                )
            else:
                _process_remote(
                    video,
                    prompt=args.prompt,
                    scene=args.scene,
                    output_dir=args.output_dir,
                    sync=args.sync,
                )
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            sys.exit(1)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
