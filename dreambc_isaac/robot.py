"""Robot articulation helpers."""

from __future__ import annotations

import numpy as np


def safe_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def available_joint_names(robot) -> set[str]:
    return set(getattr(robot, "joint_names", []) or [])


def filter_existing_joint_names(robot, joint_names: list[str]) -> list[str]:
    available = available_joint_names(robot)
    return [name for name in joint_names if name in available]


def get_named_joint_positions(robot, joint_names: list[str]) -> np.ndarray:
    if not joint_names:
        return np.zeros((0,), dtype=np.float32)
    return safe_numpy(robot.get_joint_positions(joint_names=joint_names)).astype(np.float32).reshape(-1)


def lock_named_joints(robot, joint_names: list[str], targets: np.ndarray) -> None:
    if not joint_names:
        return
    target_batch = np.expand_dims(np.asarray(targets, dtype=np.float32), axis=0)
    zero_velocity_batch = np.zeros_like(target_batch)
    robot.set_joint_positions(target_batch, joint_names=joint_names)
    robot.set_joint_velocities(zero_velocity_batch, joint_names=joint_names)

