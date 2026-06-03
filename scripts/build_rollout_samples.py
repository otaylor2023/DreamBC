#!/usr/bin/env python3
"""Transcode curated Ctrl-World rollout examples for the landing-page samples section."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "static" / "videos" / "rollout_samples"

TARGET_W = 960
TARGET_H = 192

# One clear example per task × outcome from dataset-generation rollouts.
PICKS: dict[str, dict[str, Path]] = {
    "cube": {
        "success": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_234704_traj_0009_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        "success2": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_235751_traj_0048_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        "artifact": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_235332_traj_0018_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        "artifact2": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_235123_traj_0016_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        "fail": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_234245_traj_0007_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        "fail2": REPO
        / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260531_000001_traj_0057_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
    },
    "tomato": {
        "success": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_032752_traj_0003_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        "success2": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_035214_traj_0065_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        "artifact": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_033601_traj_0035_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        "artifact2": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_033158_traj_0005_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        "fail": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_034005_traj_0042_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        "fail2": REPO
        / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_040433_traj_0090_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
    },
}


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


def main() -> int:
    for task, outcomes in PICKS.items():
        for outcome, src in outcomes.items():
            dst = OUT / f"{task}_{outcome}.mp4"
            if not src.is_file():
                print(f"[skip] missing source: {src}")
                continue
            _ffmpeg_normalize(src, dst)
            print(f"OK {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
