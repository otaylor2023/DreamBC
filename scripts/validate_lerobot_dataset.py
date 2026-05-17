#!/usr/bin/env python3
"""Validate a local LeRobot dataset for SmolVLA pick-cube training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dreambc_isaac.lerobot_dataset import SMOLVLA_CAMERA_MODEL_KEYS, SMOLVLA_PICK_CUBE_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO / "data" / "lerobot" / "dreambc_franka_pick_cube",
    )
    parser.add_argument("--repo-id", default="local/dreambc_franka_pick_cube")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")
    print(f"Dataset root: {args.root}")
    print(f"Episodes: {ds.meta.total_episodes}, frames: {ds.meta.total_frames}, fps: {ds.meta.fps}")
    assert ds.meta.total_episodes > 0, "no episodes recorded"
    assert ds.meta.total_frames > 0, "no frames recorded"

    for key, spec in SMOLVLA_PICK_CUBE_FEATURES.items():
        assert key in ds.meta.features, f"missing feature {key}"
        assert ds.meta.features[key]["shape"] == spec["shape"], f"shape mismatch for {key}"

    sample = ds[0]
    assert sample["task"] == "pick up the cube" or "pick" in str(sample["task"]).lower()
    assert sample["observation.state"].shape[-1] == 6
    assert sample["action"].shape[-1] == 6
    for cam_key in SMOLVLA_CAMERA_MODEL_KEYS:
        assert cam_key in sample, f"missing {cam_key}"
    print("validate_lerobot_dataset: ok")


if __name__ == "__main__":
    main()
