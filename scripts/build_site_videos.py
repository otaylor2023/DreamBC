#!/usr/bin/env python3
"""Curate and transcode videos/images for the GitHub Pages site under docs/static/."""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS_STATIC = REPO / "docs" / "static"
VIDEOS = DOCS_STATIC / "videos"
IMAGES = DOCS_STATIC / "images"

TASKS = ("cube", "tomato")
CUBE_PROMPT = "put the orange block in the blue bowl"
TOMATO_PROMPT = "pick the tomato and place it in the blue bowl"

SWEEP_CONFIGS = {
    "cube": ["C06_g8_block", "C16_g5_block", "cube_c16_lora_v1_aug"],
    "tomato": ["TL_g3_red_ball", "tomato_t24_lora_v1_aug"],
}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _ffmpeg_transcode(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000:
        return
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            "scale=-2:480",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "fast",
            "-an",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def _stub_video(dst: Path, label: str = "Coming soon") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000:
        return
    # 2s black video with text via drawtext if available, else plain black
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=640x360:d=2",
                "-vf",
                f"drawtext=text='{label}':fontsize=28:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
                "-c:v",
                "libx264",
                "-t",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ]
        )
    except subprocess.CalledProcessError:
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=#222222:s=640x360:d=2",
                "-c:v",
                "libx264",
                "-t",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ]
        )


def _extract_zip_video(zip_path: Path, member: str, dst: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, path=dst.parent)
    extracted = dst.parent / member
    if extracted != dst:
        extracted.rename(dst)


def _first_frame_png(video: Path, png: Path, caption: str) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video), "-vframes", "1", str(png.with_suffix(".raw.png"))])
  # overlay caption with PIL if available
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(png.with_suffix(".raw.png")).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img.width, 36], fill=(0, 0, 0))
        draw.text((8, 8), caption[:80], fill=(255, 255, 0))
        img.save(png)
        png.with_suffix(".raw.png").unlink(missing_ok=True)
    except ImportError:
        png.with_suffix(".raw.png").rename(png)


def copy_images() -> None:
    src_cv = REPO / "real_robot_checkpoints" / "docs" / "camera_views"
    src_te = REPO / "real_robot_checkpoints" / "docs" / "handcollected_training_examples"
    for sub, dst_name in [(src_cv, "camera_views"), (src_te, "training_examples")]:
        if not sub.is_dir():
            continue
        dst = IMAGES / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(sub, dst)


def build_real_demos() -> None:
    for task, zip_name, prompt in [
        ("cube", "cube.zip", CUBE_PROMPT),
        ("tomato", "tomato.zip", TOMATO_PROMPT),
    ]:
        zp = REPO / zip_name
        if not zp.is_file():
            continue
        out_dir = VIDEOS / "real_demo"
        for demo_id in (1, 2):
            member = f"{task}/video_{demo_id}.mp4"
            tmp = out_dir / f"_tmp_{task}_{demo_id}.mp4"
            dst = out_dir / f"{task}_demo_{demo_id:02d}.mp4"
            try:
                _extract_zip_video(zp, member, tmp)
                _ffmpeg_transcode(tmp, dst)
                if demo_id == 1:
                    _first_frame_png(dst, IMAGES / "first_frame" / f"{task}_first_frame.png", prompt)
            except (KeyError, zipfile.BadZipFile) as e:
                print(f"[skip] {member}: {e}")
            finally:
                tmp.unlink(missing_ok=True)


def _find_dream_mp4(task: str, index: int) -> Path | None:
    candidates: list[Path] = []
    cw = REPO / "Ctrl-World" / "rollouts"
    if cw.is_dir():
        candidates.extend(sorted(cw.glob("rollout_*.mp4")))
    sweep_root = REPO / "rollouts" / "sweep" / task
    if sweep_root.is_dir():
        for mp4 in sorted(sweep_root.rglob("Rollouts_interact_pi/video/*.mp4")):
            candidates.append(mp4)
    legacy = REPO / "rollouts" / f"rollout_{index}.mp4"
    if legacy.is_file():
        candidates.insert(0, legacy)
    if not candidates:
        return None
    # cube: lower indices; tomato: offset
    idx = index if task == "cube" else index + 10
    if idx < len(candidates):
        return candidates[idx]
    return candidates[index % len(candidates)]


def build_dream_rollouts() -> None:
    out_dir = VIDEOS / "dream_rollout"
    for task in TASKS:
        for i in (1, 2):
            src = _find_dream_mp4(task, i)
            dst = out_dir / f"{task}_dream_{i:02d}.mp4"
            if src and src.is_file():
                _ffmpeg_transcode(src, dst)
            else:
                _stub_video(dst, "Dream rollout TBD")


def build_sweep_gallery() -> None:
    for task, configs in SWEEP_CONFIGS.items():
        out_dir = VIDEOS / "dream_sweep" / task
        out_dir.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            pattern = REPO / "rollouts" / "sweep" / task / cfg / "Rollouts_interact_pi" / "video"
            mp4s = sorted(pattern.glob("*.mp4")) if pattern.is_dir() else []
            dst = out_dir / f"{cfg}.mp4"
            if mp4s:
                _ffmpeg_transcode(mp4s[0], dst)
            else:
                _stub_video(dst, cfg)


def build_failure_cases() -> None:
    out_dir = VIDEOS / "failure_cases"
    for task in TASKS:
        sweep_videos = sorted((REPO / "rollouts" / "sweep" / task).rglob("video/*.mp4")) if (
            REPO / "rollouts" / "sweep" / task
        ).is_dir() else []
        # use later files in list as "different" clips for failure panel
        picks = sweep_videos[-2:] if len(sweep_videos) >= 2 else sweep_videos[:2]
        labels = ["drift", "morph"]
        for label, src in zip(labels, picks):
            dst = out_dir / f"{task}_{label}.mp4"
            if src.is_file():
                _ffmpeg_transcode(src, dst)
            else:
                _stub_video(dst, f"failure {label}")
        for label in labels:
            dst = out_dir / f"{task}_{label}.mp4"
            if not dst.is_file():
                _stub_video(dst, f"failure {label}")


def build_stubs() -> None:
    for sub in ("policy_rollout", "teleop_baseline"):
        for task in TASKS:
            for i in (1, 2):
                if sub == "policy_rollout":
                    name = f"{task}_after_{i:02d}.mp4"
                    label = "Trained policy — coming soon"
                else:
                    name = f"{task}_teleop_{i:02d}.mp4"
                    label = "Teleop FT — coming soon"
                _stub_video(VIDEOS / sub / name, label)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--copy-images", action="store_true", help="Copy PNGs and PDF into docs/static/")
    p.add_argument("--videos-only", action="store_true")
    p.add_argument("--images-only", action="store_true")
    args = p.parse_args()

    if args.images_only or args.copy_images:
        copy_images()
        if args.images_only:
            return 0

    if not args.images_only:
        build_real_demos()
        build_dream_rollouts()
        build_sweep_gallery()
        build_failure_cases()
        build_stubs()
        if args.copy_images:
            copy_images()

    print(f"[done] site media under {DOCS_STATIC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
