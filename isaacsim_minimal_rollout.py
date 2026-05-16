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
minimum observation/action loop:
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
    build_observation,
    camera_observation_keys,
    initialize_cameras,
    make_cameras,
    wait_for_camera_readiness,
)
from dreambc_isaac.debug import (
    add_camera_debug_markers,
    collect_camera_debug,
    collect_prim_world_positions,
    print_camera_world_poses,
    print_prim_world_positions,
    print_stage_tree,
)
from dreambc_isaac.io import resolve_project_path, save_png, write_camera_rollout_videos
from dreambc_isaac.policy import fake_policy_action, load_smolvla_runner_if_configured, smolvla_policy_action
from dreambc_isaac.robot import (
    filter_existing_joint_names,
    get_named_joint_positions,
    lock_named_joints,
)
from dreambc_isaac.robot_mounts import ensure_link7_wrist_camera_mount
from dreambc_isaac.scene import add_simple_scene
from dreambc_isaac.urdf import enable_first_available_urdf_extension, import_urdf


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.types import ArticulationActions

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
    add_simple_scene(world, cfg.scene, project_root)

    urdf_path = resolve_project_path(project_root, cfg.robot.urdfs[gripper])
    robot_root_path = import_urdf(urdf_path, str(cfg.robot.target_path))
    articulation_path = f"{robot_root_path}/{cfg.robot.articulation_child}"
    robot = Articulation(prim_paths_expr=articulation_path, name="panda")
    world.scene.add(robot)

    ensure_link7_wrist_camera_mount(world.stage, cfg.scene)

    cameras = make_cameras(cfg.cameras)
    require_cameras = bool(getattr(cfg.sim, "require_cameras", True))
    camera_ready = False
    camera_failure_reason = ""

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
        print(f"Initialized locked base joints to: {dict(zip(locked_base_joint_names, configured_base_targets.tolist()))}")

    if arm_joint_names:
        configured_arm_home = np.asarray(cfg.robot.initial_arm_joint_positions, dtype=np.float32)[: len(arm_joint_names)]
        robot.set_joint_positions(np.expand_dims(configured_arm_home, axis=0), joint_names=arm_joint_names)
        print(f"Initialized arm joints to tabletop home: {configured_arm_home.tolist()}")
    if gripper_joint_names:
        gripper_home = float(cfg.robot.initial_gripper_position[gripper])
        configured_gripper_home = np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32)
        robot.set_joint_positions(np.expand_dims(configured_gripper_home, axis=0), joint_names=gripper_joint_names)
        print(f"Initialized gripper joints to: {configured_gripper_home.tolist()}")

    locked_base_targets = get_named_joint_positions(robot, locked_base_joint_names)
    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

    for _ in range(3):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
    initialize_cameras(cameras, cfg.cameras)

    for _ in range(int(cfg.sim.warmup_steps)):
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
    if policy_kind not in ("fake", "smolvla"):
        raise ValueError(f"Unsupported policy.kind: {policy_kind} (supported: fake, smolvla)")
    print(f"Using policy kind: {policy_kind}")

    smolvla_runner = None
    if policy_kind == "smolvla":
        if not cameras:
            raise RuntimeError("policy.kind=smolvla requires working cameras (set sim.require_cameras=true).")
        smolvla_runner = load_smolvla_runner_if_configured(cfg)
        print(f"Loaded SmolVLA from {cfg.smolvla.model_path} task={cfg.smolvla.task!r}")

    rollout = {
        "joint_position": [],
        "gripper_position": [],
        "action": [],
        "arm_command": [],
        "sample_steps": [],
    }

    wrote_png = False
    save_camera_videos = bool(cameras) and bool(getattr(cfg.sim, "save_camera_videos", False))
    video_frames: dict[str, list[np.ndarray]] = {name: [] for name in cameras} if save_camera_videos else {}
    video_fps = float(getattr(cfg.sim, "video_fps", 20.0))

    for step in range(int(cfg.sim.steps)):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)
        current_arm_qpos = get_named_joint_positions(robot, arm_joint_names)
        if policy_kind == "fake":
            arm_action, gripper_action = fake_policy_action(
                step,
                current_arm_qpos,
                len(gripper_joint_names),
                cfg.fake_policy,
            )
        else:
            arm_action, gripper_action = smolvla_policy_action(
                smolvla_runner,
                obs,
                current_arm_qpos,
                len(gripper_joint_names),
            )
        action = np.concatenate([arm_action, gripper_action]).astype(np.float32)
        action = np.concatenate([locked_base_targets, action]).astype(np.float32)
        action_joint_names = locked_base_joint_names + arm_joint_names + gripper_joint_names

        robot.apply_action(
            ArticulationActions(
                joint_positions=np.expand_dims(action, axis=0),
                joint_names=action_joint_names,
            )
        )
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

        if save_camera_videos:
            obs_after_step = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)
            for name in cameras:
                video_frames[name].append(np.copy(obs_after_step[name]))

        rollout["joint_position"].append(obs["joint_position"])
        rollout["gripper_position"].append(obs["gripper_position"])
        rollout["action"].append(action)
        rollout["arm_command"].append(arm_action)

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

    camera_videos_written: dict[str, str] = {}
    if save_camera_videos and video_frames:
        camera_videos_written = write_camera_rollout_videos(output_dir, video_frames, fps=video_fps)

    rollout_npz = output_dir / "rollout.npz"
    np.savez_compressed(
        rollout_npz,
        joint_position=np.asarray(rollout["joint_position"], dtype=np.float32),
        gripper_position=np.asarray(rollout["gripper_position"], dtype=np.float32),
        action=np.asarray(rollout["action"], dtype=np.float32),
        arm_command=np.asarray(rollout["arm_command"], dtype=np.float32),
        sample_steps=np.asarray(rollout["sample_steps"], dtype=np.int32),
    )

    camera_debug = collect_camera_debug(cameras, stage=world.stage, camera_cfg=cfg.cameras) if cameras else {}
    prim_debug = collect_prim_world_positions(world.stage, list(cfg.debug.prim_position_paths))
    debug_summary = {
        "camera_world_poses": camera_debug,
        "prim_world_positions": prim_debug,
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
        "locked_base_joint_names": locked_base_joint_names,
        "arm_joint_names": arm_joint_names,
        "gripper_joint_names": gripper_joint_names,
        "observation_keys": camera_observation_keys(cameras) + ["joint_position", "gripper_position"],
        "policy_kind": policy_kind,
        "action_format": (
            "Joint position target: locked_base_joint_names, then arm_joint_names, then gripper_joint_names."
        ),
        "camera_debug_path": str(camera_debug_path),
        "camera_debug": debug_summary,
        "png_samples_written": wrote_png,
        "save_camera_videos": save_camera_videos,
        "video_fps": video_fps if save_camera_videos else None,
        "camera_videos": camera_videos_written,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved rollout: {rollout_npz}")
    print(f"Saved camera samples: {samples_dir}")
    print(f"Saved camera debug: {camera_debug_path}")
    if camera_videos_written:
        print(f"Saved camera videos ({len(camera_videos_written)}): {output_dir / 'camera_videos'}")
    elif save_camera_videos:
        print("Camera video export was enabled but no MP4 files were written (see warnings above).")
    if not wrote_png:
        print("PIL was not available; camera samples were saved as compressed npz files only.")

    if not bool(cfg.sim.headless) and bool(cfg.sim.keep_open_after_rollout):
        print("Rollout complete. Isaac Sim UI will stay open; close the window to exit.")
        while sim_app.is_running():
            world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
