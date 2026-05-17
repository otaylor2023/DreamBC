"""
Collect LeRobot pick-cube demonstrations in Isaac Sim (Lula IK by default).

Example:
  python isaacsim_curobo_collect_dataset.py --config-name=curobo_pick_cube_dataset
  python isaacsim_curobo_collect_dataset.py dataset.num_episodes=5
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
from dreambc_isaac.cameras import build_observation, initialize_cameras, make_cameras, wait_for_camera_readiness
from dreambc_isaac.curobo_franka import load_curobo_franka_planner
from dreambc_isaac.io import resolve_project_path, write_camera_rollout_videos
from dreambc_isaac.lerobot_dataset import (
    SMOLVLA_PICK_CUBE_FEATURES,
    SMOLVLA_TASK_DEFAULT,
    command_to_action6,
    lerobot_frame_from_step,
    preview_mp4_episode_dir,
    should_save_preview_mp4,
)
from dreambc_isaac.pick_cube_planner import build_curobo_scene_update, plan_pick_cube
from dreambc_isaac.rmpflow_pick import build_rmpflow_pick_executor
from dreambc_isaac.robot import filter_existing_joint_names, get_named_joint_positions, lock_named_joints
from dreambc_isaac.robot_mounts import ensure_link7_wrist_camera_mount
from dreambc_isaac.scene import add_simple_scene
from dreambc_isaac.scene_randomize import randomize_test_cube_xy_yaw, read_test_cube_position
from dreambc_isaac.franka_spawn import spawn_franka
from dreambc_isaac.urdf import enable_first_available_urdf_extension


def _init_robot_home(robot, cfg, gripper: str, arm_joint_names, gripper_joint_names) -> None:
    from dreambc_isaac.robot import set_named_joint_positions

    if arm_joint_names:
        configured_arm_home = np.asarray(cfg.robot.initial_arm_joint_positions, dtype=np.float32)[: len(arm_joint_names)]
        set_named_joint_positions(robot, arm_joint_names, configured_arm_home)
    if gripper_joint_names:
        gripper_home = float(cfg.robot.initial_gripper_position[gripper])
        configured_gripper_home = np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32)
        set_named_joint_positions(robot, gripper_joint_names, configured_gripper_home)


@hydra.main(config_path="configs", config_name="curobo_pick_cube_dataset", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation, SingleArticulation
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.types import ArticulationActions

    enable_first_available_urdf_extension(enable_extension)

    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    gripper = str(cfg.robot.gripper)
    dataset_root = resolve_project_path(project_root, cfg.dataset.root)
    log_dir = resolve_project_path(project_root, cfg.output.dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = int(cfg.dataset.num_episodes)
    fps = int(cfg.dataset.fps)
    steps_per_waypoint = max(1, int(getattr(cfg.sim, "steps_per_waypoint", 1)))
    demo_teleport = bool(getattr(cfg.curobo, "demo_teleport_joints", False))
    physics_settle = max(
        steps_per_waypoint,
        int(getattr(cfg.curobo, "physics_settle_steps_per_waypoint", steps_per_waypoint)),
    )
    save_discarded_preview = bool(getattr(cfg.dataset, "preview_mp4_save_discarded", False))
    task = str(getattr(cfg.dataset, "task", SMOLVLA_TASK_DEFAULT))
    policy_camera_keys = list(cfg.smolvla.camera_keys)
    preview_cameras = list(getattr(cfg.dataset, "preview_cameras", policy_camera_keys))
    rng = np.random.default_rng()

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene, project_root)
    test_cube_object = None
    if bool(getattr(cfg.scene.test_cube, "enabled", True)):
        test_cube_object = world.scene.get_object(str(cfg.scene.test_cube.name))
    articulation_path = spawn_franka(world, cfg, project_root, enable_extension)
    planner_backend = str(getattr(cfg.curobo, "planner_backend", "rmpflow")).lower()
    use_rmpflow = planner_backend == "rmpflow"
    if use_rmpflow:
        robot = SingleArticulation(prim_path=articulation_path, name="panda")
    else:
        robot = Articulation(prim_paths_expr=articulation_path, name="panda")
    use_articulation_api = not use_rmpflow
    world.scene.add(robot)
    ensure_link7_wrist_camera_mount(world.stage, cfg.scene)
    cameras = make_cameras(cfg.cameras)

    world.reset()
    if use_rmpflow:
        robot.initialize()
    arm_joint_names = filter_existing_joint_names(robot, list(cfg.robot.arm_joint_names))
    gripper_joint_names = filter_existing_joint_names(robot, list(cfg.robot.gripper_joint_names[gripper]))
    locked_base_joint_names = filter_existing_joint_names(robot, list(cfg.robot.locked_base_joint_names))
    locked_base_targets = get_named_joint_positions(robot, locked_base_joint_names)
    _init_robot_home(robot, cfg, gripper, arm_joint_names, gripper_joint_names)

    for _ in range(int(cfg.sim.warmup_steps)):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    initialize_cameras(cameras, cfg.cameras)
    wait_for_camera_readiness(world, robot, cameras, locked_base_joint_names, locked_base_targets)

    curobo_planner = None
    use_curobo = bool(getattr(cfg.curobo, "enabled", False))
    allow_fallback = bool(
        getattr(cfg.curobo, "allow_lula_fallback", getattr(cfg.curobo, "allow_scripted_fallback", True))
    )
    if use_curobo:
        try:
            curobo_planner = load_curobo_franka_planner(use_cuda=True)
            curobo_planner.warmup()
            print("CuRobo MotionPlanner ready.")
        except Exception as exc:
            print(f"Warning: CuRobo init failed ({exc}).")
            if not allow_fallback:
                raise
            curobo_planner = None
    pick_executor = None
    if use_rmpflow and curobo_planner is None:
        pick_executor = build_rmpflow_pick_executor(
            robot, articulation_path, arm_joint_names, gripper_joint_names, cfg.curobo
        )
        print("Using RMPFlow PickPlace controller for demonstrations.")
    elif curobo_planner is None:
        print("Using Lula IK pick planner for demonstrations.")

    max_pick_steps = int(getattr(cfg.curobo, "max_pick_controller_steps", 2500))

    for _ in range(5):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    tasks_path = meta_dir / "tasks.parquet"
    if info_path.is_file() and tasks_path.is_file():
        print(f"Resuming dataset at {dataset_root}")
        lerobot_ds = LeRobotDataset(
            repo_id=str(cfg.dataset.repo_id),
            root=dataset_root,
            vcodec=str(getattr(cfg.dataset, "vcodec", "h264")),
        )
    else:
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        lerobot_ds = LeRobotDataset.create(
            repo_id=str(cfg.dataset.repo_id),
            fps=fps,
            features=SMOLVLA_PICK_CUBE_FEATURES,
            root=dataset_root,
            robot_type=str(getattr(cfg.dataset, "robot_type", "franka_panda_sim")),
            use_videos=True,
            vcodec=str(getattr(cfg.dataset, "vcodec", "h264")),
        )

    for _ in range(3):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    cube_prim = str(cfg.scene.test_cube.prim_path)
    cube_base = np.asarray(cfg.scene.test_cube.position, dtype=np.float64)
    table_pos = np.asarray(cfg.scene.table.position, dtype=np.float64)
    table_scale = np.asarray(cfg.scene.table.scale, dtype=np.float64)
    rand_xy = float(cfg.dataset.cube_pose_randomization.xy_m)
    rand_yaw = float(cfg.dataset.cube_pose_randomization.yaw_deg)

    stats = {"success": 0, "failed": 0, "episodes": []}
    state_arm_indices = [int(i) for i in cfg.smolvla.state_arm_indices]
    action_arm_indices = [int(i) for i in cfg.smolvla.action_arm_indices]
    gripper_action_index = int(cfg.smolvla.gripper_action_index)

    for episode_idx in range(num_episodes):
        _init_robot_home(robot, cfg, gripper, arm_joint_names, gripper_joint_names)
        for _ in range(8):
            lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
            world.step(render=True)

        cube_pos = randomize_test_cube_xy_yaw(
            test_cube_object,
            cube_base,
            xy_m=rand_xy,
            yaw_deg=rand_yaw,
            rng=rng,
        )
        for _ in range(8):
            world.step(render=True)
        cube_pos = read_test_cube_position(world.stage, cube_prim, cube_pos)
        cube_z_start = float(cube_pos[2])

        if curobo_planner is not None:
            scene_update = build_curobo_scene_update(
                table_pos.tolist(),
                table_scale.tolist(),
                cube_pos.tolist(),
                [float(v) * 2 for v in cfg.curobo.cube_half_extents],
            )
            curobo_planner.update_scene(scene_update)

        record_preview = should_save_preview_mp4(episode_idx, cfg.dataset)
        preview_frames: dict[str, list[np.ndarray]] = {k: [] for k in preview_cameras} if record_preview else {}

        episode_ok = True
        pick_status = "ok"
        num_waypoints = 0

        if pick_executor is not None:
            pick_executor.reset()
            ctrl_steps = 0
            cube_half = np.asarray(cfg.curobo.cube_half_extents, dtype=np.float64)
            while pick_executor.step(cube_pos, cube_half) and ctrl_steps < max_pick_steps:
                for _ in range(physics_settle):
                    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
                    world.step(render=True)
                    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
                arm_q, grip_scalar = pick_executor.read_arm_gripper()
                obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)
                action6 = command_to_action6(
                    arm_q,
                    grip_scalar,
                    state_arm_indices=state_arm_indices,
                    action_arm_indices=action_arm_indices,
                    gripper_action_index=gripper_action_index,
                )
                frame = lerobot_frame_from_step(
                    obs,
                    action6,
                    task=task,
                    policy_camera_keys=policy_camera_keys,
                    image_size=int(cfg.smolvla.image_size),
                )
                try:
                    lerobot_ds.add_frame(frame)
                except Exception as exc:
                    print(f"Episode {episode_idx}: add_frame failed: {exc}")
                    episode_ok = False
                    break
                if record_preview:
                    for cam_name in preview_cameras:
                        if cam_name in obs:
                            preview_frames[cam_name].append(np.copy(obs[cam_name]))
                ctrl_steps += 1
                num_waypoints = ctrl_steps
            pick_status = (
                "rmpflow_ok"
                if pick_executor.controller.get_current_event() >= int(cfg.curobo.rmpflow_stop_after_event) - 1
                else f"rmpflow_event_{pick_executor.controller.get_current_event()}"
            )
        else:
            arm_qpos = get_named_joint_positions(robot, arm_joint_names)
            pick_plan = plan_pick_cube(curobo_planner, arm_qpos, cube_pos, cfg.curobo)
            if not pick_plan.success and curobo_planner is not None and allow_fallback:
                pick_plan = plan_pick_cube(None, arm_qpos, cube_pos, cfg.curobo)
            if not pick_plan.success or not pick_plan.waypoints:
                stats["failed"] += 1
                stats["episodes"].append({"index": episode_idx, "success": False, "status": pick_plan.status})
                print(f"Episode {episode_idx}: planning failed ({pick_plan.status})")
                continue
            pick_status = pick_plan.status
            num_waypoints = len(pick_plan.waypoints)
            for wp in pick_plan.waypoints:
                gripper_action = np.full((len(gripper_joint_names),), wp.gripper, dtype=np.float32)
                action = np.concatenate([wp.arm_qpos[: len(arm_joint_names)], gripper_action]).astype(np.float32)
                action_joint_names = locked_base_joint_names + arm_joint_names + gripper_joint_names
                full_action = np.concatenate([locked_base_targets, action]).astype(np.float32)
                if demo_teleport or use_articulation_api:
                    from dreambc_isaac.robot import set_named_joint_positions

                    set_named_joint_positions(robot, action_joint_names, full_action)
                else:
                    robot.apply_action(
                        ArticulationActions(
                            joint_positions=np.expand_dims(full_action, axis=0),
                            joint_names=action_joint_names,
                        )
                    )
                settle_steps = physics_settle if demo_teleport else steps_per_waypoint
                for _ in range(settle_steps):
                    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
                    world.step(render=True)
                    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
                obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)
                action6 = command_to_action6(
                    wp.arm_qpos,
                    wp.gripper,
                    state_arm_indices=state_arm_indices,
                    action_arm_indices=action_arm_indices,
                    gripper_action_index=gripper_action_index,
                )
                frame = lerobot_frame_from_step(
                    obs,
                    action6,
                    task=task,
                    policy_camera_keys=policy_camera_keys,
                    image_size=int(cfg.smolvla.image_size),
                )
                try:
                    lerobot_ds.add_frame(frame)
                except Exception as exc:
                    print(f"Episode {episode_idx}: add_frame failed: {exc}")
                    episode_ok = False
                    break
                if record_preview:
                    for cam_name in preview_cameras:
                        if cam_name in obs:
                            preview_frames[cam_name].append(np.copy(obs[cam_name]))

        if pick_executor is not None and num_waypoints == 0:
            stats["failed"] += 1
            stats["episodes"].append({"index": episode_idx, "success": False, "status": pick_status})
            print(f"Episode {episode_idx}: no frames recorded ({pick_status})")
            continue

        cube_z_end = float(read_test_cube_position(world.stage, cube_prim, cube_pos)[2])
        min_lift_m = float(getattr(cfg.dataset, "min_cube_lift_m", 0.03))
        pick_lifted = (cube_z_end - cube_z_start) >= min_lift_m

        def _write_preview_mp4(*, success: bool, status: str) -> dict[str, str] | None:
            if not record_preview or not preview_frames:
                return None
            preview_dir = preview_mp4_episode_dir(
                dataset_root, episode_idx, success=success, status=status
            )
            written = write_camera_rollout_videos(preview_dir, preview_frames, fps=float(fps))
            if written:
                print(f"Episode {episode_idx}: preview MP4 -> {preview_dir / 'camera_videos'}")
            return written or None

        if episode_ok and pick_lifted:
            lerobot_ds.save_episode()
            stats["success"] += 1
            ep_info = {
                "index": episode_idx,
                "success": True,
                "steps": num_waypoints,
                "planner": pick_status,
                "cube_lift_m": round(cube_z_end - cube_z_start, 4),
            }
            written = _write_preview_mp4(success=True, status="")
            if written:
                ep_info["preview_mp4"] = written
            stats["episodes"].append(ep_info)
            print(
                f"Episode {episode_idx}: saved ({num_waypoints} steps, "
                f"lift={cube_z_end - cube_z_start:.3f}m)"
            )
        else:
            stats["failed"] += 1
            status = "record_failed" if not episode_ok else "cube_not_lifted"
            ep_fail = {
                "index": episode_idx,
                "success": False,
                "status": status,
                "cube_lift_m": round(cube_z_end - cube_z_start, 4),
            }
            if save_discarded_preview or (episode_ok and not pick_lifted):
                written = _write_preview_mp4(success=False, status=status)
                if written:
                    ep_fail["preview_mp4"] = written
            stats["episodes"].append(ep_fail)
            if lerobot_ds.episode_buffer is not None and lerobot_ds.episode_buffer.get("size", 0) > 0:
                lerobot_ds.clear_episode_buffer(delete_images=True)
            print(
                f"Episode {episode_idx}: discarded ({status}, lift={cube_z_end - cube_z_start:.3f}m)"
            )

    lerobot_ds.finalize()
    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset_root": str(dataset_root),
        "num_episodes_requested": num_episodes,
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = log_dir / f"collect_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Collection complete: {stats['success']} ok, {stats['failed']} failed")
    print(f"Dataset: {dataset_root}")
    print(f"Summary: {summary_path}")

    sim_app.close()


if __name__ == "__main__":
    main()
