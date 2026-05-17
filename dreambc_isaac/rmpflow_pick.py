"""Franka pick-and-lift via Isaac RMPFlow + PickPlaceController (works in-process with Isaac Sim)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dreambc_isaac.pick_cube_planner import PickCubePlan, PlannedWaypoint


@dataclass
class RmpflowPickExecutor:
    """Online pick controller; call ``reset`` then ``step`` each sim tick."""

    robot: object
    controller: object
    articulation_controller: object
    arm_joint_names: list[str]
    gripper_joint_names: list[str]
    lift_height_m: float
    stop_event: int

    def reset(self) -> None:
        self.controller.reset()

    def step(self, cube_position: np.ndarray, cube_half_extents: np.ndarray | None = None) -> bool:
        """Apply one PickPlace tick. Returns True while the pick-and-lift sequence is active."""
        if self.controller.get_current_event() >= self.stop_event:
            return False
        cube = np.asarray(cube_position, dtype=np.float64).reshape(3)
        half = np.asarray(
            cube_half_extents if cube_half_extents is not None else [0.025, 0.025, 0.025],
            dtype=np.float64,
        )
        # PickPlace uses object center height for the grasp phase (_h0).
        picking = cube.copy()
        picking[2] = float(cube[2])
        placing = picking + np.array([0.0, 0.0, self.lift_height_m + float(half[2])], dtype=np.float64)
        joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=np.float64).reshape(-1)
        action = self.controller.forward(
            picking_position=picking,
            placing_position=placing,
            current_joint_positions=joint_positions,
            end_effector_offset=np.array([0.0, 0.0, -0.01]),
        )
        self.articulation_controller.apply_action(action)
        return self.controller.get_current_event() < self.stop_event

    def read_arm_gripper(self) -> tuple[np.ndarray, float]:
        from dreambc_isaac.robot import get_named_joint_positions

        arm = get_named_joint_positions(self.robot, self.arm_joint_names)
        grip = get_named_joint_positions(self.robot, self.gripper_joint_names)
        gripper_scalar = float(np.mean(grip)) if grip.size else 0.04
        return arm.reshape(-1), gripper_scalar


def build_rmpflow_pick_executor(
    robot,
    articulation_path: str,
    arm_joint_names: list[str],
    gripper_joint_names: list[str],
    planner_cfg,
) -> RmpflowPickExecutor:
    from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
    from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

    gripper_open = float(planner_cfg.gripper_open)
    gripper_closed = float(planner_cfg.gripper_closed)
    ee_path = f"{articulation_path}/panda_rightfinger"
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=list(gripper_joint_names),
        joint_opened_positions=np.array([gripper_open, gripper_open], dtype=np.float64),
        joint_closed_positions=np.array([gripper_closed, gripper_closed], dtype=np.float64),
    )
    dof_names = list(getattr(robot, "dof_names", None) or getattr(robot, "joint_names", None) or [])
    if not gripper_joint_names:
        raise ValueError("gripper_joint_names is empty; cannot build ParallelGripper")
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=dof_names,
    )
    articulation_controller = robot.get_articulation_controller()
    for dof_idx in gripper.joint_dof_indicies:
        if dof_idx is not None:
            articulation_controller.switch_dof_control_mode(dof_index=int(dof_idx), mode="position")

    events_dt = getattr(planner_cfg, "rmpflow_events_dt", None)
    if events_dt is None:
        events_dt = [0.02, 0.02, 0.08, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

    approach_h = float(getattr(planner_cfg, "approach_height_m", 0.28))
    controller = PickPlaceController(
        name="dreambc_rmpflow_pick",
        gripper=gripper,
        robot_articulation=robot,
        end_effector_initial_height=approach_h,
        events_dt=list(events_dt),
    )
    stop_event = int(getattr(planner_cfg, "rmpflow_stop_after_event", 5))
    lift_m = float(getattr(planner_cfg, "grasp_lift_offset", 0.12))
    return RmpflowPickExecutor(
        robot=robot,
        controller=controller,
        articulation_controller=articulation_controller,
        arm_joint_names=list(arm_joint_names),
        gripper_joint_names=list(gripper_joint_names),
        lift_height_m=lift_m,
        stop_event=stop_event,
    )


def run_rmpflow_pick_episode(
    executor: RmpflowPickExecutor,
    cube_position: np.ndarray,
    *,
    max_controller_steps: int = 2500,
) -> PickCubePlan:
    """Run pick-through-lift and return recorded waypoints (for metadata); execution is online."""
    executor.reset()
    plan = PickCubePlan()
    steps = 0
    while executor.step(cube_position) and steps < max_controller_steps:
        arm_q, grip = executor.read_arm_gripper()
        plan.waypoints.append(PlannedWaypoint(arm_qpos=arm_q.copy(), gripper=grip))
        steps += 1
    if executor.controller.get_current_event() >= 4:
        plan.success = True
        plan.status = "rmpflow_ok"
    else:
        plan.status = f"rmpflow_incomplete_event_{executor.controller.get_current_event()}"
    return plan
