"""Pick-cube motion plans via CuRobo grasp planning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dreambc_isaac.curobo_franka import CuroboFrankaPlanner, PANDA_ARM_JOINTS


@dataclass
class PlannedWaypoint:
    arm_qpos: np.ndarray
    gripper: float


@dataclass
class PickCubePlan:
    waypoints: list[PlannedWaypoint] = field(default_factory=list)
    success: bool = False
    status: str = ""


def _top_down_grasp_goal(
    cube_position: np.ndarray,
    cube_half_extents: np.ndarray,
    device: str,
    tool_frames: list[str],
):
    from curobo.types import GoalToolPose

    cx, cy, cz = [float(v) for v in np.asarray(cube_position, dtype=np.float64).reshape(3)]
    hz = float(cube_half_extents[2]) if cube_half_extents is not None else 0.025
    # Grasp slightly above cube top; tool Z approaches from above.
    grasp_z = cz + hz + 0.01
    positions = torch.tensor(
        [[[[[cx, cy, grasp_z]]]]],
        device=device,
        dtype=torch.float32,
    )
    # wxyz quaternion: point hand frame for top-down (identity-ish; CuRobo selects from goalset)
    quaternions = torch.tensor(
        [[[[[1.0, 0.0, 0.0, 0.0]]]]],
        device=device,
        dtype=torch.float32,
    )
    return GoalToolPose(
        tool_frames=tool_frames,
        position=positions,
        quaternion=quaternions,
    )


def _trajectory_to_arm_waypoints(
    joint_traj,
    gripper_value: float,
    joint_names: list[str],
) -> list[PlannedWaypoint]:
    if joint_traj is None:
        return []
    pos = joint_traj.position
    if isinstance(pos, torch.Tensor):
        pos_np = pos.detach().cpu().numpy()
    else:
        pos_np = np.asarray(pos)
    if pos_np.ndim == 3:
        pos_np = pos_np.reshape(-1, pos_np.shape[-1])
    waypoints: list[PlannedWaypoint] = []
    arm_indices = [joint_names.index(j) for j in PANDA_ARM_JOINTS if j in joint_names]
    for row in pos_np:
        arm = np.array([row[i] for i in arm_indices], dtype=np.float32)
        if arm.shape[0] < 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[: arm.shape[0]] = arm
            arm = padded
        waypoints.append(PlannedWaypoint(arm_qpos=arm[:7], gripper=float(gripper_value)))
    return waypoints


def _resample_waypoints(
    waypoints: list[PlannedWaypoint],
    control_hz: float,
    traj_dt: float,
) -> list[PlannedWaypoint]:
    if not waypoints or control_hz <= 0 or traj_dt <= 0:
        return waypoints
    step_s = 1.0 / float(control_hz)
    duration = max(len(waypoints) - 1, 1) * float(traj_dt)
    n_out = max(int(round(duration / step_s)), 1)
    indices = np.linspace(0, len(waypoints) - 1, n_out)
    out: list[PlannedWaypoint] = []
    for idx in indices:
        out.append(waypoints[int(round(idx))])
    return out


def plan_pick_cube(
    curobo: CuroboFrankaPlanner | None,
    arm_qpos: np.ndarray,
    cube_position: np.ndarray,
    curobo_cfg,
) -> PickCubePlan:
    """Plan approach / grasp / lift and return resampled waypoints at control_hz."""
    if curobo is None:
        from dreambc_isaac.lula_pick_planner import plan_lula_pick_cube

        return plan_lula_pick_cube(arm_qpos, cube_position, curobo_cfg)

    gripper_open = float(curobo_cfg.gripper_open)
    gripper_closed = float(curobo_cfg.gripper_closed)
    control_hz = float(curobo_cfg.control_hz)

    cube_half = np.asarray(
        getattr(curobo_cfg, "cube_half_extents", [0.025, 0.025, 0.025]),
        dtype=np.float64,
    )
    q_start = curobo.joint_state_from_arm_qpos(arm_qpos, gripper_open)
    grasp_poses = _top_down_grasp_goal(
        cube_position,
        cube_half,
        curobo.device,
        curobo.planner.tool_frames,
    )

    result = curobo.planner.plan_grasp(
        current_state=q_start,
        grasp_poses=grasp_poses,
        grasp_approach_offset=float(curobo_cfg.grasp_approach_offset),
        grasp_lift_offset=float(curobo_cfg.grasp_lift_offset),
        grasp_approach_in_tool_frame=bool(curobo_cfg.grasp_approach_in_tool_frame),
        grasp_lift_in_tool_frame=bool(curobo_cfg.grasp_lift_in_tool_frame),
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
    )

    plan = PickCubePlan()
    if result is None or result.success is None or not bool(result.success.any()):
        plan.status = getattr(result, "status", "plan_grasp failed") if result is not None else "no result"
        return plan

    interp_dt = float(curobo.planner.trajopt_solver.config.interpolation_dt)
    joint_names = curobo.joint_names
    segments: list[tuple[object, float]] = [
        (result.approach_interpolated_trajectory, gripper_open),
        (result.grasp_interpolated_trajectory, gripper_open),
    ]
    hold_steps = int(getattr(curobo_cfg, "gripper_close_hold_steps", 8))
    all_wp: list[PlannedWaypoint] = []
    for traj, grip in segments:
        seg_wp = _trajectory_to_arm_waypoints(traj, grip, joint_names)
        all_wp.extend(_resample_waypoints(seg_wp, control_hz, interp_dt))

    if all_wp and hold_steps > 0:
        close_wp = PlannedWaypoint(arm_qpos=all_wp[-1].arm_qpos.copy(), gripper=gripper_closed)
        all_wp.extend([close_wp] * hold_steps)

    lift_wp = _trajectory_to_arm_waypoints(
        result.lift_interpolated_trajectory, gripper_closed, joint_names
    )
    all_wp.extend(_resample_waypoints(lift_wp, control_hz, interp_dt))

    if not all_wp:
        plan.status = "empty trajectory"
        return plan

    plan.waypoints = all_wp
    plan.success = True
    plan.status = "ok"
    return plan


def build_curobo_scene_update(table_position, table_scale, cube_position, cube_dims):
    """Minimal namespace for update_scene."""
    from types import SimpleNamespace

    return SimpleNamespace(
        table=SimpleNamespace(position=table_position, scale=table_scale),
        cube=SimpleNamespace(position=cube_position, dims=cube_dims),
    )
