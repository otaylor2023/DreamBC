"""
Sweep wrist camera orientation fixes at the current (doubled) link7 mount offset.

One fresh Isaac session per variant. Writes:
  outputs/minimal_rollout_camera_capture/wrist_sweep/<name>/scene_overview.png
  outputs/minimal_rollout_camera_capture/wrist_sweep/<name>/wrist_image_left.png

  mamba run -n dreambc python isaacsim_wrist_orientation_sweep.py sim.headless=true
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig, OmegaConf, open_dict

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

WRIST_VARIANTS: list[tuple[str, dict]] = [
    ("01_fwd_up_z", {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, 0.0, 1.0]}),
    ("02_fwd_up_pos_y", {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, 1.0, 0.0]}),
    ("03_fwd_up_neg_y", {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, -1.0, 0.0]}),
    ("04_fwd_up_neg_z", {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, 0.0, -1.0]}),
    (
        "05_fwd_up_z_post_p90",
        {
            "view_forward_xyz": [1.0, 0.0, 0.0],
            "view_up_xyz": [0.0, 0.0, 1.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, 90.0],
        },
    ),
    (
        "06_fwd_up_z_post_m90",
        {
            "view_forward_xyz": [1.0, 0.0, 0.0],
            "view_up_xyz": [0.0, 0.0, 1.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, -90.0],
        },
    ),
    (
        "07_fwd_up_pos_y_post_p90",
        {
            "view_forward_xyz": [1.0, 0.0, 0.0],
            "view_up_xyz": [0.0, 1.0, 0.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, 90.0],
        },
    ),
    (
        "08_fwd_up_pos_y_post_m90",
        {
            "view_forward_xyz": [1.0, 0.0, 0.0],
            "view_up_xyz": [0.0, 1.0, 0.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, -90.0],
        },
    ),
    (
        "09_lookat_usdneg_z",
        {
            "look_at": [0.38, 0.0, 0.0],
            "look_at_up_xyz": [0.0, 0.0, 1.0],
            "look_at_convention": "usd_neg_z",
        },
    ),
    (
        "10_lookat_post_p90",
        {
            "look_at": [0.38, 0.0, 0.0],
            "look_at_up_xyz": [0.0, 0.0, 1.0],
            "look_at_convention": "usd_neg_z",
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, 90.0],
        },
    ),
    (
        "11_lookat_post_m90",
        {
            "look_at": [0.38, 0.0, 0.0],
            "look_at_up_xyz": [0.0, 0.0, 1.0],
            "look_at_convention": "usd_neg_z",
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, -90.0],
        },
    ),
    ("12_euler_y_neg90", {"orientation_euler_xyz_degrees": [0.0, -90.0, 0.0]}),
    (
        "13_euler_y_neg90_post_p90",
        {
            "orientation_euler_xyz_degrees": [0.0, -90.0, 0.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, 90.0],
        },
    ),
    (
        "14_euler_y_neg90_post_m90",
        {
            "orientation_euler_xyz_degrees": [0.0, -90.0, 0.0],
            "orientation_post_euler_xyz_degrees": [0.0, 0.0, -90.0],
        },
    ),
]

WRIST_ORIENTATION_KEYS = frozenset(
    {
        "view_forward_xyz",
        "view_up_xyz",
        "look_at",
        "look_at_up_xyz",
        "look_at_convention",
        "orientation_post_euler_xyz_degrees",
        "orientation_euler_xyz_degrees",
    }
)

CAPTURE_KEYS = ("scene_overview", "wrist_image_left")


def _wrist_cfg_from_variant(base_wrist: DictConfig, variant: dict) -> DictConfig:
    wrist = OmegaConf.create(OmegaConf.to_container(base_wrist, resolve=True))
    for key in WRIST_ORIENTATION_KEYS:
        if key in wrist:
            del wrist[key]
    wrist.update(variant)
    return wrist


def _capture_variant(cfg: DictConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.extensions import enable_extension

    enable_first_available_urdf_extension(enable_extension)

    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    gripper = str(cfg.robot.gripper)
    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene, project_root)

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
    initialize_cameras(cameras, cfg.cameras)

    for _ in range(int(cfg.sim.warmup_steps)):
        lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
        world.step(render=True)

    try:
        wait_for_camera_readiness(world, robot, cameras, locked_base_joint_names, locked_base_targets)
    except RuntimeError as exc:
        if require_cameras:
            raise
        print(f"Warning: camera pipeline not ready: {exc}")

    lock_named_joints(robot, locked_base_joint_names, locked_base_targets)
    world.step(render=True)
    obs = build_observation(robot, cameras, arm_joint_names, gripper_joint_names)

    for key in CAPTURE_KEYS:
        save_png(out_dir / f"{key}.png", obs[key])

    sim_app.close()


def _run_single_variant(cfg: DictConfig, variant_name: str, variant: dict, sweep_root: Path) -> None:
    wrist_cfg = _wrist_cfg_from_variant(cfg.cameras.wrist_image_left, variant)
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(run_cfg.cameras):
        run_cfg.cameras.wrist_image_left = wrist_cfg
    variant_dir = sweep_root / variant_name
    _capture_variant(run_cfg, variant_dir)
    for key in CAPTURE_KEYS:
        print(f"  {variant_dir / f'{key}.png'}")


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    sweep_root = project_root / "outputs" / "minimal_rollout_camera_capture" / "wrist_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    single_name = os.environ.get("WRIST_SWEEP_VARIANT")
    if single_name:
        variant_payload = os.environ.get("WRIST_SWEEP_JSON", "{}")
        variant = json.loads(variant_payload)
        print(f"\n=== {single_name} ===")
        _run_single_variant(cfg, single_name, variant, sweep_root)
        return

    mount_tr = list(cfg.scene.link7_wrist_camera_mount.mount_translation)
    script = Path(__file__).resolve()
    hydra_args = [a for a in sys.argv[1:] if not a.startswith("WRIST_SWEEP")]

    for variant_name, variant in WRIST_VARIANTS:
        print(f"\n=== launching {variant_name} ===")
        env = os.environ.copy()
        env["WRIST_SWEEP_VARIANT"] = variant_name
        env["WRIST_SWEEP_JSON"] = json.dumps(variant)
        cmd = [sys.executable, str(script), *hydra_args]
        subprocess.run(cmd, env=env, check=True)

    index_lines = [f"{name}\t{variant}" for name, variant in WRIST_VARIANTS]
    index_path = sweep_root / "variants.txt"
    index_path.write_text(
        f"# mount_translation (2x): {mount_tr}\n"
        f"# mount_rotate_z_deg: {cfg.scene.link7_wrist_camera_mount.mount_rotate_z_deg}\n\n"
        + "\n".join(index_lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDone: {len(WRIST_VARIANTS)} variants under {sweep_root}")


if __name__ == "__main__":
    main()
