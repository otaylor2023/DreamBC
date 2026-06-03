#!/usr/bin/env python3
"""Transcode a small set of extra Ctrl-World rollouts to keep in the repo.

These are spare, manually-reviewed examples (clean successes + non-kept fails)
that are not currently wired into the site but are preserved under
docs/static/videos/archive/ for future use. Labels come from the manual
`success` field in each rollout's bc_episodes/episode.json.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "static" / "videos" / "archive"

TARGET_W = 960
TARGET_H = 192

# task -> outcome -> {label: source video}. Trajectory ids chosen from the
# manually-reviewed success labels; not the ones already used on the live site.
PICKS: dict[str, dict[str, dict[str, Path]]] = {
    "cube": {
        "success": {
            "traj0002": REPO / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_234036_traj_0002_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
            "traj0008": REPO / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260530_234454_traj_0008_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        },
        "fail": {
            "traj0001": REPO / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260531_004951_traj_0001_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
            "traj0003": REPO / "rollouts/C16_g5_block/Rollouts_interact_pi/video/pickplace_time_20260531_005200_traj_0003_0_2_put_the_orange_block_in_the_blue_bowl.mp4",
        },
    },
    "tomato": {
        "success": {
            "traj0067": REPO / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_035620_traj_0067_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
            "traj0001": REPO / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_063131_traj_0001_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        },
        "fail": {
            "traj0044": REPO / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_034409_traj_0044_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
            "traj0063": REPO / "rollouts/TL_g3_red_ball/Rollouts_interact_pi/video/pickplace_time_20260531_034812_traj_0063_0_2_put_the_small_red_ball_in_the_blue_bowl.mp4",
        },
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
            "ffmpeg", "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "26", "-preset", "fast",
            "-an", "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    for task, outcomes in PICKS.items():
        for outcome, clips in outcomes.items():
            for label, src in clips.items():
                dst = OUT / f"{task}_{outcome}_{label}.mp4"
                if not src.is_file():
                    print(f"[skip] missing source: {src}")
                    continue
                _ffmpeg_normalize(src, dst)
                print(f"OK {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
