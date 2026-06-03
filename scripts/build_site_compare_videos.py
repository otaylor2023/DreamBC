#!/usr/bin/env python3
"""Regenerate the three-way comparison videos for docs/writeup.html section 3.5.

- Real demos: extract 3 hand-collected demos per task from cube.zip / tomato.zip,
  stack views in Ctrl-World order [agent_view, exterior_2, wrist], encode to mp4.
- Dream rollouts: re-encode 3 imagined rollouts per task at the same target size.
- Policy rollouts: re-encode 3 Ctrl-World eval LoRA rollouts per task at the
  same target size.

Output: all videos at TARGET_W x TARGET_H (960x320 by default), padded with
black bars when the source aspect differs from the target. The CSS pairs this
with aspect-ratio: 3/1, so every panel renders at the same on-screen size.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VIDEOS = REPO / "docs" / "static" / "videos"

TARGET_W = 960
TARGET_H = 320
HDF5_VIEW_ORDER = ["agent_view", "exterior_2", "wrist"]

REAL_DEMOS = {
    "cube": [1, 2, 3],
    "tomato": [1, 2, 3],
}

DREAM_SOURCES = {
    "cube": REPO / "rollouts" / "C16_g5_block" / "Rollouts_interact_pi" / "video",
    "tomato": REPO / "rollouts" / "TL_g3_red_ball" / "Rollouts_interact_pi" / "video",
}

POLICY_SOURCES = {
    "cube": REPO / "rollouts" / "eval_post_training" / "cube" / "cube_ctrl_world_lora_g5" / "Rollouts_interact_pi" / "video",
    "tomato": REPO / "rollouts" / "eval_post_training" / "tomato" / "tomato_ctrl_world_lora_bowl_g3" / "Rollouts_interact_pi" / "video",
}

# Specific val_ids that we know produced clean rollouts (from user labels).
# These are matched: the SAME val_id is used for the dream column, the policy
# column, and the starting-frame column, so all three columns in the qualitative
# comparison share a starting snapshot per example.
MATCHED_PICKS = {
    "cube": ["0009", "0048"],
    "tomato": ["0003", "0065"],
}

STARTING_FRAMES_DIR = REPO / "docs" / "static" / "images" / "starting_frame"


def _ffmpeg_normalize(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "format=yuv420p"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-crf",
            "26",
            "-preset",
            "fast",
            "-an",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _build_real_demo(task: str, demo_id: int, dst: Path) -> bool:
    zp = REPO / f"{task}.zip"
    if not zp.is_file():
        print(f"[skip] {zp} missing")
        return False
    member = f"{task}/demos_{demo_id}.hdf5"
    try:
        with zipfile.ZipFile(zp) as zf:
            data = zf.read(member)
    except KeyError:
        print(f"[skip] {member} not in zip")
        return False

    with h5py.File(io.BytesIO(data), "r") as f:
        group = f[f"data/demo_{demo_id}/obs"]
        frames = []
        for v in HDF5_VIEW_ORDER:
            if v not in group:
                print(f"[skip] {member} missing view {v}")
                return False
            frames.append(np.asarray(group[v]))

    h = frames[0].shape[1]
    w = frames[0].shape[2]
    stacked = np.concatenate(frames, axis=2)  # (T, h, 3*w, 3)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpf:
        tmp_path = Path(tmpf.name)
    try:
        iio.imwrite(tmp_path, stacked, fps=15, codec="libx264")
        _ffmpeg_normalize(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)
    return True


def _find_traj(src_dir: Path, val_id: str) -> Path | None:
    if not src_dir.is_dir():
        return None
    matches = sorted(src_dir.glob(f"*traj_{val_id}_*.mp4"))
    return matches[0] if matches else None


def _build_dream(task: str, val_id: str, dst: Path) -> bool:
    src = _find_traj(DREAM_SOURCES[task], val_id)
    if src is None:
        print(f"[skip] no dream video for {task} val_id={val_id}")
        return False
    _ffmpeg_normalize(src, dst)
    return True


def _build_policy(task: str, val_id: str, dst: Path) -> bool:
    src = _find_traj(POLICY_SOURCES[task], val_id)
    if src is None:
        print(f"[skip] no policy video for {task} val_id={val_id}")
        return False
    _ffmpeg_normalize(src, dst)
    return True


def _build_starting_frame(task: str, dream_mp4: Path, dst: Path) -> bool:
    """Pull the very first frame of the dream rollout as the starting frame.

    The first frame of a Ctrl-World rollout is the conditioning input (the
    snapshot frame the world model was conditioned off of), so this is exactly
    the starting frame for the trajectory in the next column.
    """
    if not dream_mp4.is_file():
        print(f"[skip] dream mp4 missing for starting frame: {dream_mp4}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(dream_mp4), "-frames:v", "1", str(dst)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> int:
    out_real = VIDEOS / "real_demo"
    out_dream = VIDEOS / "dream_rollout"
    out_policy = VIDEOS / "policy_rollout"

    for task, demo_ids in REAL_DEMOS.items():
        for slot, demo_id in enumerate(demo_ids, start=1):
            dst = out_real / f"{task}_demo_{slot:02d}.mp4"
            ok = _build_real_demo(task, demo_id, dst)
            print(f"{'OK' if ok else '..'} {dst}")

    # Dream + policy + starting frame ALL keyed on the same matched val_id per slot.
    for task, picks in MATCHED_PICKS.items():
        for slot, vid in enumerate(picks, start=1):
            dream_dst = out_dream / f"{task}_dream_{slot:02d}.mp4"
            ok_d = _build_dream(task, vid, dream_dst)
            print(f"{'OK' if ok_d else '..'} {dream_dst} (val_id {vid})")

            policy_dst = out_policy / f"{task}_after_{slot:02d}.mp4"
            ok_p = _build_policy(task, vid, policy_dst)
            print(f"{'OK' if ok_p else '..'} {policy_dst} (val_id {vid})")

            start_dst = STARTING_FRAMES_DIR / f"{task}_start_{slot:02d}.png"
            ok_s = _build_starting_frame(task, dream_dst, start_dst)
            print(f"{'OK' if ok_s else '..'} {start_dst} (val_id {vid})")

    # remove old per-task subdirs from older builds
    for sub in ("cube", "tomato"):
        legacy = out_real / sub
        if legacy.is_dir():
            shutil.rmtree(legacy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
