"""
Capture one frame per camera for the full minimal-rollout scene (robot + table + cube),
using the same layout and joint homes as isaacsim_minimal_rollout.py. No policy and no rollout steps.

Writes under outputs/minimal_rollout_camera_capture/ (PNG per camera key in configs/minimal_rollout.yaml),
including scene_overview when configured.

  python isaacsim_minimal_rollout_camera_capture.py sim.headless=true
  python isaacsim_minimal_rollout_camera_capture.py sim.headless=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.cameras import (
    build_observation,
    initialize_cameras,
    make_cameras,
    wait_for_camera_readiness,
)
from dreambc_isaac.io import resolve_project_path, save_png
from dreambc_isaac.robot import (
    filter_existing_joint_names,
    get_named_joint_positions,
    lock_named_joints,
)
from dreambc_isaac.robot_mounts import ensure_link7_wrist_camera_mount
from dreambc_isaac.scene import add_simple_scene
from dreambc_isaac.franka_spawn import spawn_franka
from dreambc_isaac.urdf import enable_first_available_urdf_extension


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    out_dir = project_root / "outputs" / "minimal_rollout_camera_capture"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.extensions import enable_extension

    enabled_extension = enable_first_available_urdf_extension(enable_extension)
    print(f"Enabled URDF importer extension: {enabled_extension}")

    gripper = str(cfg.robot.gripper)
    if gripper not in cfg.robot.urdfs:
        raise ValueError(f"Unknown gripper '{gripper}'. Available: {list(cfg.robot.urdfs.keys())}")

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene, project_root)

    urdf_path = resolve_project_path(project_root, cfg.robot.urdfs[gripper])
    articulation_path = spawn_franka(world, cfg, project_root, enable_extension)
    robot = Articulation(prim_paths_expr=articulation_path, name="panda")
    world.scene.add(robot)

    ensure_link7_wrist_camera_mount(world.stage, cfg.scene)

    cameras = make_cameras(cfg.cameras)
    require_cameras = bool(getattr(cfg.sim, "require_cameras", True))

    world.reset()
    locked_base_joint_names = filter_existing_joint_names(robot, list(cfg.robot.locked_base_joint_names))
    arm_joint_names = filter_existing_joint_names(robot, list(cfg.robot.arm_joint_names))
    gripper_joint_names = filter_existing_joint_names(robot, list(cfg.robot.gripper_joint_names[gripper]))

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

    if arm_joint_names:
        configured_arm_home = np.asarray(cfg.robot.initial_arm_joint_positions, dtype=np.float32)[: len(arm_joint_names)]
        robot.set_joint_positions(np.expand_dims(configured_arm_home, axis=0), joint_names=arm_joint_names)
    if gripper_joint_names:
        gripper_home = float(cfg.robot.initial_gripper_position[gripper])
        configured_gripper_home = np.full((len(gripper_joint_names),), gripper_home, dtype=np.float32)
        robot.set_joint_positions(np.expand_dims(configured_gripper_home, axis=0), joint_names=gripper_joint_names)

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

    try:
        wait_for_camera_readiness(world, robot, cameras, locked_base_joint_names, locked_base_targets)
    except RuntimeError as exc:
        if require_cameras:
            raise
        print(f"Warning: camera pipeline not ready (sim.require_cameras=false): {exc}")

    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
    world.step(render=True)
    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)

    obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)

    wrote: list[str] = []
    for name in cameras:
        img = obs[name]
        path = out_dir / f"{name}.png"
        if save_png(path, img):
            wrote.append(str(path))
        else:
            alt = out_dir / f"{name}.npz"
            np.savez_compressed(alt, rgb=img)
            wrote.append(str(alt))

    print(f"Saved {len(wrote)} camera capture(s) under {out_dir}:")
    for p in wrote:
        print(f"  {p}")

    sim_app.close()


if __name__ == "__main__":
    main()
