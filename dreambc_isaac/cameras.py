"""Camera creation, initialization, and observation helpers."""

from __future__ import annotations

import numpy as np

from dreambc_isaac.patches import apply_camera_pipeline_patches
from dreambc_isaac.robot import get_named_joint_positions, lock_named_joints


def camera_orientation_from_euler_xyz(degrees_xyz: tuple[float, float, float]):
    import isaacsim.core.utils.numpy.rotations as rot_utils

    return rot_utils.euler_angles_to_quats(np.array(degrees_xyz), degrees=True)


def make_cameras(camera_cfg):
    from isaacsim.core.utils.prims import define_prim
    from isaacsim.sensors.camera import Camera

    define_prim("/World/Cameras", "Xform")
    cameras = {}
    for name, cfg in camera_cfg.items():
        cameras[name] = Camera(
            prim_path=cfg.prim_path,
            name=name,
            position=np.asarray(cfg.position, dtype=np.float64),
            orientation=camera_orientation_from_euler_xyz(tuple(cfg.orientation_euler_xyz_degrees)),
            frequency=int(cfg.frequency),
            resolution=tuple(cfg.resolution),
            annotator_device="cpu",
        )
    return cameras


def initialize_cameras(cameras: dict[str, object]) -> None:
    apply_camera_pipeline_patches()
    for name, camera in cameras.items():
        try:
            print(f"Initializing camera: {name} ({camera.prim_path})")
            camera.initialize()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize camera '{name}' at '{camera.prim_path}'. "
                "This failed while attaching Isaac Sim's rgb annotator. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc


def camera_has_valid_rgba(camera) -> bool:
    try:
        rgba = camera.get_rgba()
    except Exception as exc:
        print(f"Warning: camera '{camera.name}' get_rgba failed: {type(exc).__name__}: {exc}")
        return False
    rgba = None if rgba is None else np.asarray(rgba)
    return rgba is not None and rgba.ndim == 3 and rgba.shape[-1] >= 3 and rgba.shape[0] > 0 and rgba.shape[1] > 0


def wait_for_camera_readiness(
    world,
    robot,
    cameras: dict[str, object],
    locked_base_joint_names: list[str],
    locked_base_targets: np.ndarray,
    max_steps: int = 120,
) -> None:
    for step in range(max_steps):
        lock_named_joints(robot=robot, joint_names=locked_base_joint_names, targets=locked_base_targets)
        world.step(render=True)
        lock_named_joints(robot=robot, joint_names=locked_base_joint_names, targets=locked_base_targets)
        invalid_names = [name for name, camera in cameras.items() if not camera_has_valid_rgba(camera)]
        if not invalid_names:
            print(f"All cameras ready after {step + 1} readiness steps.")
            return
    invalid_names = [name for name, camera in cameras.items() if not camera_has_valid_rgba(camera)]
    raise RuntimeError(
        "Camera pipeline was not ready before rollout. Invalid camera outputs for: "
        f"{invalid_names}. Aborting to avoid writing blank frames."
    )


def blank_camera_image(camera) -> np.ndarray:
    width, height = camera.get_resolution()
    return np.zeros((int(height), int(width), 3), dtype=np.uint8)


def build_observation(
    robot,
    cameras: dict[str, object],
    arm_joint_names: list[str],
    gripper_joint_names: list[str],
) -> dict[str, np.ndarray]:
    arm_qpos = get_named_joint_positions(robot, arm_joint_names)
    gripper_qpos = get_named_joint_positions(robot, gripper_joint_names)
    obs = {
        "joint_position": arm_qpos.copy(),
        "gripper_position": np.array([gripper_qpos.mean() if gripper_qpos.size else 0.0], dtype=np.float32),
    }
    for name, camera in cameras.items():
        try:
            rgba = camera.get_rgba()
        except Exception as exc:
            print(f"Warning: camera '{name}' get_rgba failed: {type(exc).__name__}: {exc}; using a blank frame.")
            rgba = None
        rgba = None if rgba is None else np.asarray(rgba)
        if rgba is None or rgba.ndim != 3 or rgba.shape[-1] < 3:
            print(
                f"Warning: camera '{name}' returned invalid rgba shape "
                f"{None if rgba is None else rgba.shape}; using a blank frame."
            )
            obs[name] = blank_camera_image(camera)
        else:
            obs[name] = np.asarray(rgba[:, :, :3], dtype=np.uint8)
    return obs


def camera_observation_keys(cameras: dict[str, object]) -> list[str]:
    return list(cameras.keys())

