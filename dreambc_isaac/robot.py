"""Robot articulation helpers."""

from __future__ import annotations

import numpy as np


def safe_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_1d_float32(values, expected_size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if expected_size == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.size == 1:
        return np.full((expected_size,), float(array[0]), dtype=np.float32)
    if array.size != expected_size:
        raise ValueError(f"{label} expected {expected_size} values, got {array.size}")
    return array.astype(np.float32, copy=False)


def select_joint_values(
    values,
    configured_joint_names: list[str],
    active_joint_names: list[str],
    label: str,
) -> np.ndarray:
    active_count = len(active_joint_names)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if active_count == 0:
        return np.zeros((0,), dtype=np.float32)
    if array.size == 1:
        return np.full((active_count,), float(array[0]), dtype=np.float32)
    if array.size == active_count:
        return array.astype(np.float32, copy=False)
    if configured_joint_names and array.size == len(configured_joint_names):
        value_by_name = {name: float(array[index]) for index, name in enumerate(configured_joint_names)}
        missing = [name for name in active_joint_names if name not in value_by_name]
        if missing:
            raise KeyError(f"{label} missing configured values for joints: {missing}")
        return np.asarray([value_by_name[name] for name in active_joint_names], dtype=np.float32)
    raise ValueError(
        f"{label} expected 1, {active_count}, or {len(configured_joint_names)} values, got {array.size}"
    )


def _batched(values: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.asarray(values, dtype=np.float32).reshape(-1), axis=0)


def available_joint_names(robot) -> set[str]:
    return set(getattr(robot, "joint_names", []) or [])


def filter_existing_joint_names(robot, joint_names: list[str]) -> list[str]:
    available = available_joint_names(robot)
    return [name for name in joint_names if name in available]


def get_named_joint_positions(robot, joint_names: list[str]) -> np.ndarray:
    if not joint_names:
        return np.zeros((0,), dtype=np.float32)
    return safe_numpy(robot.get_joint_positions(joint_names=joint_names)).astype(np.float32).reshape(-1)


def get_named_joint_velocities(robot, joint_names: list[str]) -> np.ndarray:
    if not joint_names:
        return np.zeros((0,), dtype=np.float32)
    return safe_numpy(robot.get_joint_velocities(joint_names=joint_names)).astype(np.float32).reshape(-1)


def get_dof_index_map(robot) -> dict[str, int]:
    dof_names = list(getattr(robot, "dof_names", []) or [])
    return {name: index for index, name in enumerate(dof_names)}


def get_dof_indices(robot, joint_names: list[str]) -> list[int]:
    index_map = get_dof_index_map(robot)
    missing = [name for name in joint_names if name not in index_map]
    if missing:
        raise KeyError(f"Joints not found in articulation DOF list: {missing}")
    return [index_map[name] for name in joint_names]


def lock_named_joints(robot, joint_names: list[str], targets: np.ndarray) -> None:
    if not joint_names:
        return
    set_named_joint_position_targets(robot, joint_names, np.asarray(targets, dtype=np.float32))


def set_named_joint_position_targets(robot, joint_names: list[str], targets: np.ndarray) -> None:
    if not joint_names:
        return
    robot.set_joint_position_targets(_batched(targets), joint_names=joint_names)


def set_named_joint_velocity_targets(robot, joint_names: list[str], targets: np.ndarray) -> None:
    if not joint_names:
        return
    robot.set_joint_velocity_targets(_batched(targets), joint_names=joint_names)


def switch_named_joint_control_mode(robot, joint_names: list[str], mode: str) -> None:
    if not joint_names:
        return
    robot.switch_control_mode(mode=mode, joint_names=joint_names)


def configure_named_joint_drives(
    robot,
    joint_names: list[str],
    stiffness: np.ndarray | list[float] | float,
    damping: np.ndarray | list[float] | float,
    max_effort: np.ndarray | list[float] | float | None = None,
) -> None:
    if not joint_names:
        return
    joint_count = len(joint_names)
    kps = _as_1d_float32(stiffness, joint_count, "stiffness")
    kds = _as_1d_float32(damping, joint_count, "damping")
    robot.set_gains(_batched(kps), _batched(kds), joint_names=joint_names)
    if max_effort is not None:
        efforts = _as_1d_float32(max_effort, joint_count, "max_effort")
        robot.set_max_efforts(_batched(efforts), joint_names=joint_names)


def collect_joint_drive_debug(robot, joint_names: list[str]) -> dict[str, object]:
    if not joint_names:
        return {"joint_names": [], "joint_indices": [], "stiffness": [], "damping": [], "max_effort": []}
    kps, kds = robot.get_gains(joint_names=joint_names)
    max_efforts = robot.get_max_efforts(joint_names=joint_names)
    return {
        "joint_names": list(joint_names),
        "joint_indices": get_dof_indices(robot, joint_names),
        "stiffness": safe_numpy(kps).astype(np.float32).reshape(-1).tolist(),
        "damping": safe_numpy(kds).astype(np.float32).reshape(-1).tolist(),
        "max_effort": safe_numpy(max_efforts).astype(np.float32).reshape(-1).tolist(),
    }
