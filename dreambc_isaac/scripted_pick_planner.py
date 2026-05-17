"""Joint-space fallback pick planner when CuRobo is unavailable."""

from __future__ import annotations

import numpy as np

from dreambc_isaac.pick_cube_planner import PickCubePlan, PlannedWaypoint


def _lerp_waypoints(
    start_arm: np.ndarray,
    end_arm: np.ndarray,
    n_steps: int,
    gripper: float,
) -> list[PlannedWaypoint]:
    if n_steps < 1:
        n_steps = 1
    out: list[PlannedWaypoint] = []
    for alpha in np.linspace(0.0, 1.0, n_steps):
        q = (1.0 - alpha) * start_arm + alpha * end_arm
        out.append(PlannedWaypoint(arm_qpos=q.astype(np.float32), gripper=float(gripper)))
    return out


def plan_scripted_pick_cube(
    arm_qpos: np.ndarray,
    cube_position: np.ndarray,
    planner_cfg,
) -> PickCubePlan:
    """Heuristic joint-space pick trajectory toward cube XY on the table."""
    home = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)[:7]
    if home.shape[0] < 7:
        home = np.pad(home, (0, 7 - home.shape[0]))

    cx, cy, _cz = [float(v) for v in np.asarray(cube_position, dtype=np.float64).reshape(3)]
    dx = cx - 0.55
    dy = cy - 0.0

    pre_grasp = home.copy()
    pre_grasp[0] += 0.35 * dx
    pre_grasp[1] += 0.35 * dy
    pre_grasp[2] += 0.15
    pre_grasp[3] += 0.10
    pre_grasp[6] += 0.25 * dx

    grasp = pre_grasp.copy()
    grasp[2] -= 0.08
    grasp[3] -= 0.05

    lift = grasp.copy()
    lift[2] += 0.12

    gripper_open = float(planner_cfg.gripper_open)
    gripper_closed = float(planner_cfg.gripper_closed)
    hold = int(getattr(planner_cfg, "gripper_close_hold_steps", 10))
    hz = int(getattr(planner_cfg, "control_hz", 20))

    waypoints: list[PlannedWaypoint] = []
    waypoints += _lerp_waypoints(home, pre_grasp, hz, gripper_open)
    waypoints += _lerp_waypoints(pre_grasp, grasp, max(hz // 2, 8), gripper_open)
    waypoints += [PlannedWaypoint(arm_qpos=grasp.copy(), gripper=gripper_closed) for _ in range(hold)]
    waypoints += _lerp_waypoints(grasp, lift, hz, gripper_closed)

    return PickCubePlan(waypoints=waypoints, success=True, status="scripted_fallback")
