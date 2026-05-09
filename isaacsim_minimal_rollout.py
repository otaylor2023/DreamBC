"""
Minimal DreamBC-owned Isaac Sim rollout harness.

Run from an Isaac Sim shell, for example:
  cd ~/.local/share/ov/pkg
  mamba activate dreambc
  source ./setup_conda_env.sh
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  cd /home/wpai/DreamBC
  python isaacsim_minimal_rollout.py sim.headless=true sim.steps=120

This script intentionally does not depend on CDC or OpenPI. It creates the
minimum observation/action loop that a future pi0.5 client can replace:
  Isaac Sim -> images + joints -> fake policy -> joint action -> rollout log.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig, OmegaConf

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.cameras import (
    apply_camera_auto_alignments,
    build_observation,
    camera_observation_keys,
    initialize_cameras,
    make_cameras,
    wait_for_camera_readiness,
)
from dreambc_isaac.debug import (
    add_camera_debug_markers,
    collect_camera_debug,
    collect_grasp_debug,
    collect_prim_world_positions,
    print_camera_world_poses,
    print_prim_world_positions,
    print_stage_tree,
)
from dreambc_isaac.io import resolve_project_path, save_png
from dreambc_isaac.policy import fake_policy_action, make_policy
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


def _as_float32_vector3(values) -> np.ndarray:
    if values is None:
        return np.full((3,), np.nan, dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(3)


def _as_float32_scalar(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _summarize_grasp_debug(rollout: dict[str, list]) -> dict[str, object]:
    distances = np.asarray(rollout["grasp_inner_finger_center_to_cube_distance"], dtype=np.float32)
    deltas = np.asarray(rollout["grasp_inner_finger_center_to_cube_delta"], dtype=np.float32)
    left_distances = np.asarray(rollout["grasp_left_inner_finger_to_cube_distance"], dtype=np.float32)
    right_distances = np.asarray(rollout["grasp_right_inner_finger_to_cube_distance"], dtype=np.float32)
    forced_open_mask = np.asarray(rollout["debug_gripper_forced_open"], dtype=np.bool_)

    summary: dict[str, object] = {
        "recording_timing": "after_step_execution",
        "forced_open_steps_requested": int(np.sum(forced_open_mask)),
        "forced_open_steps_applied": int(np.sum(forced_open_mask)),
    }

    valid = np.isfinite(distances)
    if np.any(valid):
        masked = np.where(valid, distances, np.inf)
        best_step = int(np.argmin(masked))
        summary["min_inner_finger_center_to_cube_distance"] = float(masked[best_step])
        summary["min_inner_finger_center_to_cube_step"] = best_step
        summary["min_inner_finger_center_to_cube_delta"] = deltas[best_step].astype(np.float32).tolist()
    else:
        summary["min_inner_finger_center_to_cube_distance"] = None
        summary["min_inner_finger_center_to_cube_step"] = None
        summary["min_inner_finger_center_to_cube_delta"] = None

    if distances.size > 0:
        summary["final_inner_finger_center_to_cube_distance"] = _as_float32_scalar(distances[-1])
        summary["final_inner_finger_center_to_cube_delta"] = deltas[-1].astype(np.float32).tolist()
        summary["final_left_inner_finger_to_cube_distance"] = _as_float32_scalar(left_distances[-1])
        summary["final_right_inner_finger_to_cube_distance"] = _as_float32_scalar(right_distances[-1])
    else:
        summary["final_inner_finger_center_to_cube_distance"] = None
        summary["final_inner_finger_center_to_cube_delta"] = None
        summary["final_left_inner_finger_to_cube_distance"] = None
        summary["final_right_inner_finger_to_cube_distance"] = None
    return summary


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
    if gripper not in cfg.robot.urdfs:
        raise ValueError(f"Unknown gripper '{gripper}'. Available: {list(cfg.robot.urdfs.keys())}")

    timestamp = datetime.now().strftime(str(cfg.output.timestamp_format))
    output_root = resolve_project_path(project_root, cfg.output.dir)
    output_dir = output_root / timestamp
    samples_dir = output_dir / "camera_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene)

    urdf_path = resolve_project_path(project_root, cfg.robot.urdfs[gripper])
    robot_root_path = import_urdf(urdf_path, str(cfg.robot.target_path))
    articulation_path = f"{robot_root_path}/{cfg.robot.articulation_child}"
    robot = Articulation(prim_paths_expr=articulation_path, name="panda")
    world.scene.add(robot)

    cameras = make_cameras(cfg.cameras)
    require_cameras = bool(getattr(cfg.sim, "require_cameras", True))
    camera_ready = False
    camera_failure_reason = ""
    camera_alignment_summary: dict[str, dict[str, object]] = {}

    world.reset()
    locked_base_joint_names = filter_existing_joint_names(robot, list(cfg.robot.locked_base_joint_names))
    arm_joint_names = filter_existing_joint_names(robot, list(cfg.robot.arm_joint_names))
    gripper_joint_names = filter_existing_joint_names(robot, list(cfg.robot.gripper_joint_names[gripper]))
    missing_locked_base_joints = sorted(set(cfg.robot.locked_base_joint_names) - set(locked_base_joint_names))
    missing_arm_joints = sorted(set(cfg.robot.arm_joint_names) - set(arm_joint_names))
    missing_gripper_joints = sorted(set(cfg.robot.gripper_joint_names[gripper]) - set(gripper_joint_names))
    if missing_locked_base_joints:
        print(f"Warning: configured locked base joints not found in articulation: {missing_locked_base_joints}")
    if missing_arm_joints:
        print(f"Warning: configured arm joints not found in articulation: {missing_arm_joints}")
    if missing_gripper_joints:
        print(f"Warning: configured gripper joints not found in articulation: {missing_gripper_joints}")
    print(f"Hard-locking base joints at initial targets: {locked_base_joint_names}")
    print(f"Controlling arm joints only: {arm_joint_names}")
    print(f"Controlling gripper joints: {gripper_joint_names}")

    if bool(cfg.robot.control.enable_runtime_drive_tuning):
        if locked_base_joint_names:
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
        if arm_joint_names:
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
        if gripper_joint_names:
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
        switch_named_joint_control_mode(robot, locked_base_joint_names, "position")
        set_named_joint_position_targets(robot, locked_base_joint_names, configured_base_targets)
        print(f"Initialized locked base joints to: {dict(zip(locked_base_joint_names, configured_base_targets.tolist()))}")
    else:
        configured_base_targets = np.zeros((0,), dtype=np.float32)

    configured_arm_home = np.zeros((0,), dtype=np.float32)
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
        print(f"Initialized arm joints to tabletop home: {configured_arm_home.tolist()}")
    configured_gripper_home = np.zeros((0,), dtype=np.float32)
    if gripper_joint_names:
        gripper_home = float(cfg.robot.initial_gripper_position[gripper])
        configured_gripper_home = np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32)
        robot.set_joint_positions(np.expand_dims(configured_gripper_home, axis=0), joint_names=gripper_joint_names)
        robot.set_joint_velocities(np.zeros((1, len(gripper_joint_names)), dtype=np.float32), joint_names=gripper_joint_names)
        switch_named_joint_control_mode(robot, gripper_joint_names, "position")
        set_named_joint_position_targets(robot, gripper_joint_names, configured_gripper_home)
        print(f"Initialized gripper joints to: {configured_gripper_home.tolist()}")

    locked_base_targets = get_named_joint_positions(robot, locked_base_joint_names)
    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

    for _ in range(3):
        set_named_joint_position_targets(robot, arm_joint_names, configured_arm_home)
        set_named_joint_position_targets(robot, gripper_joint_names, configured_gripper_home)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
    camera_alignment_summary = apply_camera_auto_alignments(world.stage, cameras, cfg.cameras, cfg.debug.grasp_debug)
    if camera_alignment_summary:
        print(f"Applied camera auto alignment: {json.dumps(camera_alignment_summary, indent=2)}")
    initialize_cameras(cameras, cfg.cameras)

    for _ in range(int(cfg.sim.warmup_steps)):
        set_named_joint_position_targets(robot, arm_joint_names, configured_arm_home)
        set_named_joint_position_targets(robot, gripper_joint_names, configured_gripper_home)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

    if bool(cfg.debug.print_stage_tree):
        print_stage_tree(
            world.stage,
            str(cfg.debug.stage_tree_root),
            int(cfg.debug.stage_tree_max_depth),
        )
    if bool(cfg.debug.print_prim_positions):
        print_prim_world_positions(world.stage, list(cfg.debug.prim_position_paths))
    if bool(cfg.debug.print_camera_poses):
        print_camera_world_poses(cameras, stage=world.stage, camera_cfg=cfg.cameras)
    if bool(cfg.debug.add_camera_markers):
        add_camera_debug_markers(world, cameras, cfg.cameras)
        world.step(render=True)

    try:
        wait_for_camera_readiness(world, robot, cameras, locked_base_joint_names, locked_base_targets)
        camera_ready = True
    except RuntimeError as exc:
        camera_failure_reason = str(exc)
        if require_cameras:
            raise
        print(
            "Warning: camera pipeline not ready, continuing with joint-only rollout "
            f"because sim.require_cameras=false. Reason: {camera_failure_reason}"
        )
        cameras = {}

    policy_kind = str(cfg.policy.kind)
    policy = make_policy(cfg.policy)
    print(f"Using policy kind: {policy_kind}")
    if policy_kind == "pi05_droid_remote" and not bool(cfg.policy.pi05_droid_remote.execute):
        print("pi05_droid_remote execute=false; querying policy but holding current robot targets.")
    active_arm_control_mode = "position"
    force_gripper_open_steps = (
        int(cfg.policy.pi05_droid_remote.force_gripper_open_steps)
        if policy_kind == "pi05_droid_remote"
        else 0
    )
    grasp_debug_enabled = bool(cfg.debug.grasp_debug.enabled)

    rollout = {
        "joint_position": [],
        "gripper_position": [],
        "action": [],
        "arm_command": [],
        "sample_steps": [],
        "debug_gripper_forced_open": [],
    }
    if policy_kind == "pi05_droid_remote":
        rollout["pi05_raw_action"] = []
        rollout["pi05_clipped_action"] = []
        rollout["pi05_policy_gripper_position"] = []
    if grasp_debug_enabled:
        rollout["grasp_cube_position"] = []
        rollout["grasp_gripper_base_position"] = []
        rollout["grasp_left_inner_finger_position"] = []
        rollout["grasp_right_inner_finger_position"] = []
        rollout["grasp_inner_finger_center_position"] = []
        rollout["grasp_inner_finger_center_to_cube_delta"] = []
        rollout["grasp_gripper_base_to_cube_delta"] = []
        rollout["grasp_left_inner_finger_to_cube_distance"] = []
        rollout["grasp_right_inner_finger_to_cube_distance"] = []
        rollout["grasp_inner_finger_center_to_cube_distance"] = []
        rollout["grasp_inner_finger_jaw_gap"] = []

    wrote_png = False
    for step in range(int(cfg.sim.steps)):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)
        current_arm_qpos = get_named_joint_positions(robot, arm_joint_names)
        current_gripper_qpos = get_named_joint_positions(robot, gripper_joint_names)
        if policy_kind == "fake":
            arm_action, gripper_action = fake_policy_action(
                step,
                current_arm_qpos,
                len(gripper_joint_names),
                cfg.fake_policy,
            )
        elif policy_kind == "pi05_droid_remote":
            arm_action, gripper_action = policy.action(
                obs,
                current_arm_qpos,
                current_gripper_qpos,
                len(gripper_joint_names),
            )
        else:
            raise ValueError(f"Unsupported policy.kind: {policy_kind}")
        gripper_forced_open = False
        if (
            policy_kind == "pi05_droid_remote"
            and step < force_gripper_open_steps
            and gripper_joint_names
        ):
            gripper_action = np.full(
                (len(gripper_joint_names),),
                float(cfg.policy.pi05_droid_remote.gripper_open_target),
                dtype=np.float32,
            )
            gripper_forced_open = True
        action = np.concatenate([arm_action, gripper_action]).astype(np.float32)
        action = np.concatenate([locked_base_targets, action]).astype(np.float32)

        if policy_kind == "pi05_droid_remote" and policy.last_arm_command_kind == "velocity":
            if active_arm_control_mode != "velocity":
                switch_named_joint_control_mode(robot, arm_joint_names, "velocity")
                active_arm_control_mode = "velocity"
            set_named_joint_velocity_targets(robot, arm_joint_names, arm_action)
        else:
            if active_arm_control_mode != "position":
                switch_named_joint_control_mode(robot, arm_joint_names, "position")
                active_arm_control_mode = "position"
            set_named_joint_position_targets(robot, arm_joint_names, arm_action)
        set_named_joint_position_targets(robot, gripper_joint_names, gripper_action)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

        step_grasp_debug = None
        if grasp_debug_enabled:
            step_grasp_debug = collect_grasp_debug(world.stage, cfg.debug.grasp_debug)

        rollout["joint_position"].append(obs["joint_position"])
        rollout["gripper_position"].append(obs["gripper_position"])
        rollout["action"].append(action)
        rollout["arm_command"].append(arm_action)
        rollout["debug_gripper_forced_open"].append(gripper_forced_open)
        if policy_kind == "pi05_droid_remote":
            rollout["pi05_raw_action"].append(policy.last_selected_raw_action.copy())
            rollout["pi05_clipped_action"].append(policy.last_selected_clipped_action.copy())
            rollout["pi05_policy_gripper_position"].append(policy.last_policy_gripper_position.copy())
        if step_grasp_debug is not None:
            positions = step_grasp_debug["positions"]
            rollout["grasp_cube_position"].append(_as_float32_vector3(positions.get("cube")))
            rollout["grasp_gripper_base_position"].append(_as_float32_vector3(positions.get("gripper_base")))
            rollout["grasp_left_inner_finger_position"].append(_as_float32_vector3(positions.get("left_inner_finger")))
            rollout["grasp_right_inner_finger_position"].append(_as_float32_vector3(positions.get("right_inner_finger")))
            rollout["grasp_inner_finger_center_position"].append(
                _as_float32_vector3(step_grasp_debug.get("inner_finger_center_position"))
            )
            rollout["grasp_inner_finger_center_to_cube_delta"].append(
                _as_float32_vector3(step_grasp_debug.get("inner_finger_center_to_cube_delta"))
            )
            rollout["grasp_gripper_base_to_cube_delta"].append(
                _as_float32_vector3(step_grasp_debug.get("gripper_base_to_cube_delta"))
            )
            rollout["grasp_left_inner_finger_to_cube_distance"].append(
                _as_float32_scalar(step_grasp_debug.get("left_inner_finger_to_cube_distance"))
            )
            rollout["grasp_right_inner_finger_to_cube_distance"].append(
                _as_float32_scalar(step_grasp_debug.get("right_inner_finger_to_cube_distance"))
            )
            rollout["grasp_inner_finger_center_to_cube_distance"].append(
                _as_float32_scalar(step_grasp_debug.get("inner_finger_center_to_cube_distance"))
            )
            rollout["grasp_inner_finger_jaw_gap"].append(
                _as_float32_scalar(step_grasp_debug.get("inner_finger_jaw_gap"))
            )

        if step % int(cfg.sim.save_every) == 0 or step == int(cfg.sim.steps) - 1:
            rollout["sample_steps"].append(step)
            if cameras:
                sample_npz = samples_dir / f"step_{step:04d}_images.npz"
                np.savez_compressed(
                    sample_npz,
                    **{camera_name: obs[camera_name] for camera_name in cameras},
                )
                for camera_name in cameras:
                    png_path = samples_dir / f"step_{step:04d}_{camera_name}.png"
                    wrote_png = save_png(png_path, obs[camera_name]) or wrote_png

    rollout_npz = output_dir / "rollout.npz"
    rollout_save_data = {
        "joint_position": np.asarray(rollout["joint_position"], dtype=np.float32),
        "gripper_position": np.asarray(rollout["gripper_position"], dtype=np.float32),
        "action": np.asarray(rollout["action"], dtype=np.float32),
        "arm_command": np.asarray(rollout["arm_command"], dtype=np.float32),
        "sample_steps": np.asarray(rollout["sample_steps"], dtype=np.int32),
        "debug_gripper_forced_open": np.asarray(rollout["debug_gripper_forced_open"], dtype=np.bool_),
    }
    if policy_kind == "pi05_droid_remote":
        rollout_save_data["pi05_raw_action"] = np.asarray(rollout["pi05_raw_action"], dtype=np.float32)
        rollout_save_data["pi05_clipped_action"] = np.asarray(rollout["pi05_clipped_action"], dtype=np.float32)
        rollout_save_data["pi05_policy_gripper_position"] = np.asarray(
            rollout["pi05_policy_gripper_position"],
            dtype=np.float32,
        )
    if grasp_debug_enabled:
        rollout_save_data["grasp_cube_position"] = np.asarray(rollout["grasp_cube_position"], dtype=np.float32)
        rollout_save_data["grasp_gripper_base_position"] = np.asarray(
            rollout["grasp_gripper_base_position"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_left_inner_finger_position"] = np.asarray(
            rollout["grasp_left_inner_finger_position"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_right_inner_finger_position"] = np.asarray(
            rollout["grasp_right_inner_finger_position"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_inner_finger_center_position"] = np.asarray(
            rollout["grasp_inner_finger_center_position"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_inner_finger_center_to_cube_delta"] = np.asarray(
            rollout["grasp_inner_finger_center_to_cube_delta"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_gripper_base_to_cube_delta"] = np.asarray(
            rollout["grasp_gripper_base_to_cube_delta"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_left_inner_finger_to_cube_distance"] = np.asarray(
            rollout["grasp_left_inner_finger_to_cube_distance"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_right_inner_finger_to_cube_distance"] = np.asarray(
            rollout["grasp_right_inner_finger_to_cube_distance"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_inner_finger_center_to_cube_distance"] = np.asarray(
            rollout["grasp_inner_finger_center_to_cube_distance"],
            dtype=np.float32,
        )
        rollout_save_data["grasp_inner_finger_jaw_gap"] = np.asarray(
            rollout["grasp_inner_finger_jaw_gap"],
            dtype=np.float32,
        )
    np.savez_compressed(rollout_npz, **rollout_save_data)

    camera_debug = collect_camera_debug(cameras, stage=world.stage, camera_cfg=cfg.cameras) if cameras else {}
    prim_debug = collect_prim_world_positions(world.stage, list(cfg.debug.prim_position_paths))
    grasp_debug_summary = _summarize_grasp_debug(rollout) if grasp_debug_enabled else {}
    debug_summary = {
        "camera_world_poses": camera_debug,
        "prim_world_positions": prim_debug,
        "grasp_debug_summary": grasp_debug_summary,
        "camera_auto_alignment": camera_alignment_summary,
    }
    camera_debug_path = output_dir / "camera_debug.json"
    camera_debug_path.write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")

    metadata = {
        "script": Path(__file__).name,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "gripper": gripper,
        "steps": int(cfg.sim.steps),
        "warmup_steps": int(cfg.sim.warmup_steps),
        "save_every": int(cfg.sim.save_every),
        "require_cameras": require_cameras,
        "camera_ready": camera_ready,
        "camera_failure_reason": camera_failure_reason,
        "articulation_path": articulation_path,
        "robot_dof_names": list(getattr(robot, "dof_names", []) or []),
        "locked_base_joint_names": locked_base_joint_names,
        "locked_base_joint_indices": get_dof_indices(robot, locked_base_joint_names) if locked_base_joint_names else [],
        "arm_joint_names": arm_joint_names,
        "arm_joint_indices": get_dof_indices(robot, arm_joint_names) if arm_joint_names else [],
        "gripper_joint_names": gripper_joint_names,
        "gripper_joint_indices": get_dof_indices(robot, gripper_joint_names) if gripper_joint_names else [],
        "runtime_drive_tuning_enabled": bool(cfg.robot.control.enable_runtime_drive_tuning),
        "base_drive_debug": base_drive_debug,
        "arm_drive_debug": arm_drive_debug,
        "gripper_drive_debug": gripper_drive_debug,
        "grasp_debug_enabled": grasp_debug_enabled,
        "grasp_debug_summary": grasp_debug_summary,
        "camera_auto_alignment": camera_alignment_summary,
        "observation_keys": camera_observation_keys(cameras) + ["joint_position", "gripper_position"],
        "policy_kind": policy_kind,
        "pi05_last_raw_action_shape": (
            list(policy.last_raw_action_shape)
            if policy_kind == "pi05_droid_remote" and policy.last_raw_action_shape is not None
            else None
        ),
        "pi05_arm_command_kind": (
            policy.last_arm_command_kind if policy_kind == "pi05_droid_remote" else "position"
        ),
        "pi05_force_gripper_open_steps": force_gripper_open_steps,
        "action_format": (
            "The saved action array is locked_base targets followed by the commanded arm values and "
            "gripper targets. Arm values are position targets for fake/joint_position/joint_velocity "
            "modes, and velocity targets only when pi05_droid_remote.action_mode=joint_velocity_target."
        ),
        "camera_debug_path": str(camera_debug_path),
        "camera_debug": debug_summary,
        "png_samples_written": wrote_png,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved rollout: {rollout_npz}")
    print(f"Saved camera samples: {samples_dir}")
    print(f"Saved camera debug: {camera_debug_path}")
    if not wrote_png:
        print("PIL was not available; camera samples were saved as compressed npz files only.")

    if not bool(cfg.sim.headless) and bool(cfg.sim.keep_open_after_rollout):
        print("Rollout complete. Isaac Sim UI will stay open; close the window to exit.")
        while sim_app.is_running():
            world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
