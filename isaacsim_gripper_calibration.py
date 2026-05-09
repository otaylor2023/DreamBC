"""Measure Robotiq finger_joint response to commanded position targets in Isaac Sim."""

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
    configure_named_joint_drives,
    filter_existing_joint_names,
    get_named_joint_positions,
    lock_named_joints,
    select_joint_values,
    set_named_joint_position_targets,
    switch_named_joint_control_mode,
)
from dreambc_isaac.scene import add_simple_scene
from dreambc_isaac.urdf import enable_first_available_urdf_extension, import_urdf


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
    output_root = resolve_project_path(project_root, "rollouts/gripper_calibration")
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
    if not gripper_joint_names:
        raise RuntimeError(f"No configured gripper joints found for gripper={gripper}")

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
    else:
        configured_base_targets = np.zeros((0,), dtype=np.float32)

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
    if locked_base_joint_names:
        switch_named_joint_control_mode(robot, locked_base_joint_names, "position")
        set_named_joint_position_targets(robot, locked_base_joint_names, configured_base_targets)

    if arm_joint_names:
        configured_arm_home = select_joint_values(
            cfg.robot.initial_arm_joint_positions,
            list(cfg.robot.arm_joint_names),
            arm_joint_names,
            "initial_arm_joint_positions",
        )
        robot.set_joint_positions(np.expand_dims(configured_arm_home, axis=0), joint_names=arm_joint_names)
        robot.set_joint_velocities(np.zeros((1, len(arm_joint_names)), dtype=np.float32), joint_names=arm_joint_names)
        switch_named_joint_control_mode(robot, arm_joint_names, "position")
        set_named_joint_position_targets(robot, arm_joint_names, configured_arm_home)

    locked_base_targets = get_named_joint_positions(robot, locked_base_joint_names)
    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

    for _ in range(int(cfg.sim.warmup_steps)):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    commanded_targets = np.asarray([0.0, 0.1, 0.2, 0.35, 0.5, 0.625, 0.725], dtype=np.float32)
    settle_steps = 20
    measured_positions: list[dict[str, object]] = []

    for target in commanded_targets:
        target_batch = np.full((1, len(gripper_joint_names)), float(target), dtype=np.float32)
        switch_named_joint_control_mode(robot, gripper_joint_names, "position")
        set_named_joint_position_targets(robot, gripper_joint_names, target_batch.reshape(-1))
        for _ in range(settle_steps):
            lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
            world.step(render=True)
            if arm_joint_names:
                set_named_joint_position_targets(robot, arm_joint_names, configured_arm_home)
            set_named_joint_position_targets(robot, gripper_joint_names, target_batch.reshape(-1))
        measured = get_named_joint_positions(robot, gripper_joint_names)
        measured_positions.append(
            {
                "commanded_target": float(target),
                "measured_joint_positions": measured.tolist(),
                "measured_mean": float(measured.mean()) if measured.size else 0.0,
            }
        )
        print(
            "Measured gripper target "
            f"{target:.4f} -> {np.array2string(measured, precision=4, separator=', ')}"
        )

    measured_means = [entry["measured_mean"] for entry in measured_positions]
    summary = {
        "script": Path(__file__).name,
        "articulation_path": articulation_path,
        "gripper": gripper,
        "gripper_joint_names": gripper_joint_names,
        "commanded_targets": commanded_targets.tolist(),
        "measurements": measured_positions,
        "recommended_closed_target": float(min(measured_means)),
        "recommended_open_target": float(max(measured_means)),
    }
    summary_path = output_dir / "gripper_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved gripper calibration: {summary_path}")

    if not bool(cfg.sim.headless) and bool(cfg.sim.keep_open_after_rollout):
        print("Calibration complete. Isaac Sim UI will stay open; close the window to exit.")
        while sim_app.is_running():
            world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
