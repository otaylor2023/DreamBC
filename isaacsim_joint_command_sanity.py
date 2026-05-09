"""Measure single-joint command response for Panda arm joints in Isaac Sim."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.io import resolve_project_path
from dreambc_isaac.robot import (
    collect_joint_drive_debug,
    configure_named_joint_drives,
    filter_existing_joint_names,
    get_dof_indices,
    get_named_joint_positions,
    lock_named_joints,
    select_joint_values,
    set_named_joint_position_targets,
    set_named_joint_velocity_targets,
    switch_named_joint_control_mode,
)
from dreambc_isaac.scene import add_simple_scene
from dreambc_isaac.urdf import enable_first_available_urdf_extension, import_urdf


def reset_robot_state(
    robot,
    arm_joint_names: list[str],
    arm_home: np.ndarray,
    gripper_joint_names: list[str],
    gripper_home: float,
    locked_base_joint_names: list[str],
    locked_base_targets: np.ndarray,
    world,
) -> None:
    if arm_joint_names:
        switch_named_joint_control_mode(robot, arm_joint_names, "position")
        robot.set_joint_positions(np.expand_dims(arm_home, axis=0), joint_names=arm_joint_names)
        robot.set_joint_velocities(
            np.zeros((1, len(arm_joint_names)), dtype=np.float32),
            joint_names=arm_joint_names,
        )
        set_named_joint_position_targets(robot, arm_joint_names, arm_home)
    if gripper_joint_names:
        switch_named_joint_control_mode(robot, gripper_joint_names, "position")
        gripper_home_targets = np.full((1, len(gripper_joint_names)), gripper_home, dtype=np.float32)
        robot.set_joint_positions(gripper_home_targets, joint_names=gripper_joint_names)
        robot.set_joint_velocities(
            np.zeros((1, len(gripper_joint_names)), dtype=np.float32),
            joint_names=gripper_joint_names,
        )
        set_named_joint_position_targets(robot, gripper_joint_names, gripper_home_targets.reshape(-1))
    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
    for _ in range(5):
        set_named_joint_position_targets(robot, arm_joint_names, arm_home)
        if gripper_joint_names:
            set_named_joint_position_targets(robot, gripper_joint_names, gripper_home_targets.reshape(-1))
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.extensions import enable_extension

    enabled_extension = enable_first_available_urdf_extension(enable_extension)
    print(f"Enabled URDF importer extension: {enabled_extension}")

    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    gripper = str(cfg.robot.gripper)
    timestamp = datetime.now().strftime(str(cfg.output.timestamp_format))
    output_root = resolve_project_path(project_root, "rollouts/joint_command_sanity")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene)

    urdf_path = resolve_project_path(project_root, cfg.robot.urdfs[gripper])
    robot_root_path = import_urdf(urdf_path, str(cfg.robot.target_path))
    articulation_path = f"{robot_root_path}/{cfg.robot.articulation_child}"
    robot = Articulation(prim_paths_expr=articulation_path, name="panda")
    world.scene.add(robot)

    world.reset()
    locked_base_joint_names = filter_existing_joint_names(robot, list(cfg.robot.locked_base_joint_names))
    arm_joint_names = filter_existing_joint_names(robot, list(cfg.robot.arm_joint_names))
    gripper_joint_names = filter_existing_joint_names(robot, list(cfg.robot.gripper_joint_names[gripper]))
    if not arm_joint_names:
        raise RuntimeError("No configured arm joints found.")

    if locked_base_joint_names:
        base_position_cfg = cfg.robot.initial_locked_base_joint_positions
        configured_base_targets = np.asarray(
            [float(base_position_cfg.get(joint_name, 0.0)) for joint_name in locked_base_joint_names],
            dtype=np.float32,
        )
        robot.set_joint_positions(np.expand_dims(configured_base_targets, axis=0), joint_names=locked_base_joint_names)
        robot.set_joint_velocities(
            np.zeros((1, len(locked_base_joint_names)), dtype=np.float32),
            joint_names=locked_base_joint_names,
        )
    arm_home = select_joint_values(
        cfg.robot.initial_arm_joint_positions,
        list(cfg.robot.arm_joint_names),
        arm_joint_names,
        "initial_arm_joint_positions",
    )
    gripper_home = float(cfg.robot.initial_gripper_position[gripper])
    locked_base_targets = get_named_joint_positions(robot, locked_base_joint_names)

    if bool(cfg.robot.control.enable_runtime_drive_tuning):
        configure_named_joint_drives(
            robot,
            locked_base_joint_names,
            select_joint_values(
                cfg.robot.control.base_position_stiffness,
                list(cfg.robot.locked_base_joint_names),
                locked_base_joint_names,
                "base_position_stiffness",
            ),
            select_joint_values(
                cfg.robot.control.base_position_damping,
                list(cfg.robot.locked_base_joint_names),
                locked_base_joint_names,
                "base_position_damping",
            ),
            select_joint_values(
                cfg.robot.control.base_max_effort,
                list(cfg.robot.locked_base_joint_names),
                locked_base_joint_names,
                "base_max_effort",
            ),
        )
        configure_named_joint_drives(
            robot,
            arm_joint_names,
            select_joint_values(
                cfg.robot.control.arm_position_stiffness,
                list(cfg.robot.arm_joint_names),
                arm_joint_names,
                "arm_position_stiffness",
            ),
            select_joint_values(
                cfg.robot.control.arm_position_damping,
                list(cfg.robot.arm_joint_names),
                arm_joint_names,
                "arm_position_damping",
            ),
            select_joint_values(
                cfg.robot.control.arm_max_effort,
                list(cfg.robot.arm_joint_names),
                arm_joint_names,
                "arm_max_effort",
            ),
        )
        configure_named_joint_drives(
            robot,
            gripper_joint_names,
            select_joint_values(
                cfg.robot.control.gripper_position_stiffness[gripper],
                list(cfg.robot.gripper_joint_names[gripper]),
                gripper_joint_names,
                f"{gripper}_gripper_position_stiffness",
            ),
            select_joint_values(
                cfg.robot.control.gripper_position_damping[gripper],
                list(cfg.robot.gripper_joint_names[gripper]),
                gripper_joint_names,
                f"{gripper}_gripper_position_damping",
            ),
            select_joint_values(
                cfg.robot.control.gripper_max_effort[gripper],
                list(cfg.robot.gripper_joint_names[gripper]),
                gripper_joint_names,
                f"{gripper}_gripper_max_effort",
            ),
        )
    base_drive_debug = collect_joint_drive_debug(robot, locked_base_joint_names)
    arm_drive_debug = collect_joint_drive_debug(robot, arm_joint_names)
    gripper_drive_debug = collect_joint_drive_debug(robot, gripper_joint_names)
    print(f"Base drive debug: {base_drive_debug}")
    print(f"Arm drive debug: {arm_drive_debug}")
    print(f"Gripper drive debug: {gripper_drive_debug}")

    if locked_base_joint_names:
        switch_named_joint_control_mode(robot, locked_base_joint_names, "position")
        set_named_joint_position_targets(robot, locked_base_joint_names, configured_base_targets)

    for _ in range(int(cfg.sim.warmup_steps)):
        set_named_joint_position_targets(robot, arm_joint_names, arm_home)
        if gripper_joint_names:
            set_named_joint_position_targets(
                robot,
                gripper_joint_names,
                np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32),
            )
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    action_hz = float(cfg.policy.pi05_droid_remote.action_hz)
    velocity_magnitude = float(cfg.policy.pi05_droid_remote.max_joint_velocity)
    settle_steps = 15
    command_modes = ("velocity_apply_action", "position_delta_target")
    signs = (-1.0, 1.0)
    results: list[dict[str, object]] = []

    for command_mode in command_modes:
        for joint_index, joint_name in enumerate(arm_joint_names):
            for sign in signs:
                reset_robot_state(
                    robot,
                    arm_joint_names,
                    arm_home,
                    gripper_joint_names,
                    gripper_home,
                    locked_base_joint_names,
                    locked_base_targets,
                    world,
                )
                start_q = get_named_joint_positions(robot, arm_joint_names)
                commanded_velocity = np.zeros((len(arm_joint_names),), dtype=np.float32)
                commanded_velocity[joint_index] = float(sign * velocity_magnitude)
                current_targets = start_q.copy()
                if command_mode == "velocity_apply_action":
                    switch_named_joint_control_mode(robot, arm_joint_names, "velocity")
                else:
                    switch_named_joint_control_mode(robot, arm_joint_names, "position")

                for _ in range(settle_steps):
                    if command_mode == "velocity_apply_action":
                        set_named_joint_velocity_targets(robot, arm_joint_names, commanded_velocity)
                    elif command_mode == "position_delta_target":
                        current_targets = current_targets + commanded_velocity / action_hz
                        set_named_joint_position_targets(robot, arm_joint_names, current_targets)
                    else:
                        raise ValueError(f"Unsupported command_mode: {command_mode}")

                    if gripper_joint_names:
                        set_named_joint_position_targets(
                            robot,
                            gripper_joint_names,
                            np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32),
                        )
                    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
                    world.step(render=True)

                end_q = get_named_joint_positions(robot, arm_joint_names)
                delta_q = end_q - start_q
                expected_direction = int(np.sign(sign))
                observed_direction = int(np.sign(delta_q[joint_index]))
                results.append(
                    {
                        "command_mode": command_mode,
                        "joint_name": joint_name,
                        "joint_index": joint_index,
                        "sign": "positive" if sign > 0 else "negative",
                        "commanded_velocity": commanded_velocity.tolist(),
                        "settle_steps": settle_steps,
                        "start_q": start_q.tolist(),
                        "end_q": end_q.tolist(),
                        "delta_q": delta_q.tolist(),
                        "tested_joint_delta": float(delta_q[joint_index]),
                        "expected_direction": expected_direction,
                        "observed_direction": observed_direction,
                        "direction_matches": bool(expected_direction == observed_direction),
                    }
                )
                print(
                    f"{command_mode} joint={joint_name} sign={sign:+.0f} "
                    f"delta={delta_q[joint_index]:+.5f} direction_match={expected_direction == observed_direction}"
                )

    summary = {
        "script": Path(__file__).name,
        "articulation_path": articulation_path,
        "gripper": gripper,
        "robot_dof_names": list(getattr(robot, "dof_names", []) or []),
        "arm_joint_names": arm_joint_names,
        "arm_joint_indices": get_dof_indices(robot, arm_joint_names),
        "gripper_joint_names": gripper_joint_names,
        "gripper_joint_indices": get_dof_indices(robot, gripper_joint_names) if gripper_joint_names else [],
        "action_hz": action_hz,
        "velocity_magnitude": velocity_magnitude,
        "runtime_drive_tuning_enabled": bool(cfg.robot.control.enable_runtime_drive_tuning),
        "base_drive_debug": base_drive_debug,
        "arm_drive_debug": arm_drive_debug,
        "gripper_drive_debug": gripper_drive_debug,
        "results": results,
    }
    summary_path = output_dir / "joint_command_sanity.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved joint command sanity summary: {summary_path}")

    if not bool(cfg.sim.headless) and bool(cfg.sim.keep_open_after_rollout):
        print("Joint sanity test complete. Isaac Sim UI will stay open; close the window to exit.")
        while sim_app.is_running():
            world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
