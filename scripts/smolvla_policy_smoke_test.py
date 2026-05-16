#!/usr/bin/env python3
"""Smoke test SmolVLA load + one select_action (no Isaac Sim)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "smolvla_base",
    )
    args = parser.parse_args()

    from dreambc_isaac.smolvla_policy import load_smolvla_runner
    from omegaconf import OmegaConf

    smolvla_cfg = OmegaConf.create(
        {
            "model_path": str(args.model_path),
            "task": "pick up the cube",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "image_size": 256,
            "camera_keys": ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
            "state_arm_indices": [0, 1, 2, 3, 4, 5],
            "action_arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_action_index": 5,
            "gripper_min": 0.02,
            "gripper_max": 0.04,
        }
    )
    runner = load_smolvla_runner(smolvla_cfg)
    obs = {
        "joint_position": np.zeros(7, dtype=np.float32),
        "gripper_position": np.array([0.04], dtype=np.float32),
    }
    for key in smolvla_cfg.camera_keys:
        obs[key] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    action6 = runner.next_action(obs)
    print(f"action shape={action6.shape} dtype={action6.dtype} sample={action6[:3]}")
    assert action6.shape == (6,), f"expected (6,), got {action6.shape}"
    print("smolvla_policy_smoke_test: ok")


if __name__ == "__main__":
    main()
