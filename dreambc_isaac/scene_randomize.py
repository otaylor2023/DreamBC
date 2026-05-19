"""Scene randomization helpers for dataset collection."""

from __future__ import annotations

import math

import numpy as np

from dreambc_isaac.debug import prim_world_position


def randomize_test_cube_xy_yaw(
    cube_object,
    base_position: np.ndarray,
    *,
    xy_m: float,
    yaw_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize cube XY and yaw via Isaac API (avoids invalidating physics tensors)."""
    base = np.asarray(base_position, dtype=np.float64).reshape(3)
    dx = float(rng.uniform(-xy_m, xy_m))
    dy = float(rng.uniform(-xy_m, xy_m))
    yaw = math.radians(float(rng.uniform(-yaw_deg, yaw_deg)))
    pos = base.copy()
    pos[0] += dx
    pos[1] += dy

    half_yaw = yaw * 0.5
    orientation = np.array(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=np.float64,
    )
    if cube_object is not None:
        cube_object.set_world_pose(position=pos, orientation=orientation)
    return pos


def read_test_cube_position(stage, prim_path: str, fallback: np.ndarray) -> np.ndarray:
    pos = prim_world_position(stage, prim_path)
    if pos is None:
        return np.asarray(fallback, dtype=np.float64)
    return pos
