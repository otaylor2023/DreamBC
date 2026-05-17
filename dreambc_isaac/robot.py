"""Robot articulation helpers (Articulation and SingleArticulation)."""

from __future__ import annotations

import numpy as np


def safe_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def robot_dof_names(robot) -> list[str]:
    names = getattr(robot, "dof_names", None) or getattr(robot, "joint_names", None) or []
    return list(names)


def uses_dof_indices(robot) -> bool:
    return bool(getattr(robot, "dof_names", None)) and not getattr(robot, "joint_names", None)


def available_joint_names(robot) -> set[str]:
    return set(robot_dof_names(robot))


def filter_existing_joint_names(robot, joint_names: list[str]) -> list[str]:
    available = available_joint_names(robot)
    return [name for name in joint_names if name in available]


def joint_indices_for_names(robot, joint_names: list[str]) -> np.ndarray:
    dof_names = robot_dof_names(robot)
    return np.array([dof_names.index(name) for name in joint_names], dtype=np.int64)


def get_named_joint_positions(robot, joint_names: list[str]) -> np.ndarray:
    if not joint_names:
        return np.zeros((0,), dtype=np.float32)
    if uses_dof_indices(robot):
        indices = joint_indices_for_names(robot, joint_names)
        return safe_numpy(robot.get_joint_positions(joint_indices=indices)).astype(np.float32).reshape(-1)
    return safe_numpy(robot.get_joint_positions(joint_names=joint_names)).astype(np.float32).reshape(-1)


def set_named_joint_positions(robot, joint_names: list[str], positions: np.ndarray) -> None:
    if not joint_names:
        return
    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    if uses_dof_indices(robot):
        indices = joint_indices_for_names(robot, joint_names)
        robot.set_joint_positions(pos, joint_indices=indices)
        robot.set_joint_velocities(np.zeros_like(pos), joint_indices=indices)
        return
    batch = np.expand_dims(pos, axis=0)
    robot.set_joint_positions(batch, joint_names=joint_names)
    robot.set_joint_velocities(np.zeros_like(batch), joint_names=joint_names)


def lock_named_joints(robot, joint_names: list[str], targets: np.ndarray) -> None:
    set_named_joint_positions(robot, joint_names, targets)
