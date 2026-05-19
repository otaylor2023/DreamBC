"""Top-down pick planner using Isaac Lula inverse kinematics (no CuRobo GPU required)."""

from __future__ import annotations

import numpy as np

from dreambc_isaac.pick_cube_planner import PickCubePlan, PlannedWaypoint

_LULA_SOLVER = None
_EE_FRAME = "right_gripper"


def _get_lula_solver():
    global _LULA_SOLVER
    if _LULA_SOLVER is None:
        import isaacsim.robot_motion.motion_generation as mg
        from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

        kinematics_config = mg.interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        _LULA_SOLVER = LulaKinematicsSolver(**kinematics_config)
    return _LULA_SOLVER


def _downward_ee_orientation() -> np.ndarray:
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    return euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))


def _lerp_waypoints(
    start_arm: np.ndarray,
    end_arm: np.ndarray,
    n_steps: int,
    gripper: float,
    *,
    hold_last: int = 0,
) -> list[PlannedWaypoint]:
    if n_steps < 1:
        n_steps = 1
    out: list[PlannedWaypoint] = []
    for alpha in np.linspace(0.0, 1.0, n_steps):
        q = (1.0 - alpha) * start_arm + alpha * end_arm
        out.append(PlannedWaypoint(arm_qpos=q.astype(np.float32), gripper=float(gripper)))
    if hold_last > 0 and out:
        last = out[-1]
        out.extend([PlannedWaypoint(arm_qpos=last.arm_qpos.copy(), gripper=last.gripper) for _ in range(hold_last)])
    return out


def _solve_ik_chain(
    arm_qpos: np.ndarray,
    positions: list[np.ndarray],
    orientation: np.ndarray,
) -> list[np.ndarray] | None:
    solver = _get_lula_solver()
    warm = np.asarray(arm_qpos, dtype=np.float64).reshape(-1)[:7]
    if warm.shape[0] < 7:
        warm = np.pad(warm, (0, 7 - warm.shape[0]))
    solved: list[np.ndarray] = []
    for pos in positions:
        q, ok = solver.compute_inverse_kinematics(
            _EE_FRAME,
            np.asarray(pos, dtype=np.float64),
            orientation,
            warm_start=warm,
        )
        if not ok:
            return None
        warm = np.asarray(q, dtype=np.float64)
        solved.append(warm.astype(np.float32))
    return solved


def plan_lula_pick_cube(
    arm_qpos: np.ndarray,
    cube_position: np.ndarray,
    planner_cfg,
) -> PickCubePlan:
    """Plan approach, grasp, close, and lift using Lula IK toward the cube."""
    plan = PickCubePlan()
    cube = np.asarray(cube_position, dtype=np.float64).reshape(3)
    half = np.asarray(
        getattr(planner_cfg, "cube_half_extents", [0.025, 0.025, 0.025]),
        dtype=np.float64,
    )
    hz = float(half[2])
    cx, cy, cz = float(cube[0]), float(cube[1]), float(cube[2])
    cube_top_z = cz + hz

    approach_m = float(getattr(planner_cfg, "grasp_approach_offset", 0.12))
    lift_m = float(getattr(planner_cfg, "grasp_lift_offset", 0.12))
    pre_grasp_z = cube_top_z + 0.05
    grasp_z = cube_top_z - 0.012

    ee_targets = [
        np.array([cx, cy, cube_top_z + approach_m]),
        np.array([cx, cy, pre_grasp_z]),
        np.array([cx, cy, grasp_z]),
        np.array([cx, cy, cube_top_z + lift_m]),
    ]
    orientation = _downward_ee_orientation()
    key_q = _solve_ik_chain(arm_qpos, ee_targets, orientation)
    if key_q is None:
        plan.status = "lula_ik_failed"
        return plan

    home = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)[:7]
    if home.shape[0] < 7:
        home = np.pad(home, (0, 7 - home.shape[0]))

    gripper_open = float(planner_cfg.gripper_open)
    gripper_closed = float(planner_cfg.gripper_closed)
    hz_rate = int(getattr(planner_cfg, "control_hz", 20))
    hold_close = int(getattr(planner_cfg, "gripper_close_hold_steps", 20))
    settle = int(getattr(planner_cfg, "waypoint_settle_steps", 6))

    waypoints: list[PlannedWaypoint] = []
    waypoints += _lerp_waypoints(home, key_q[0], hz_rate, gripper_open, hold_last=settle)
    waypoints += _lerp_waypoints(key_q[0], key_q[1], hz_rate, gripper_open, hold_last=settle)
    waypoints += _lerp_waypoints(key_q[1], key_q[2], max(hz_rate, 16), gripper_open, hold_last=settle)
    waypoints += [PlannedWaypoint(arm_qpos=key_q[2].copy(), gripper=gripper_closed) for _ in range(hold_close)]
    waypoints += _lerp_waypoints(key_q[2], key_q[3], hz_rate, gripper_closed, hold_last=settle)

    plan.waypoints = waypoints
    plan.success = True
    plan.status = "lula_ok"
    return plan
