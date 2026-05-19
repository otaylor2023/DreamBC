"""Camera creation, initialization, and observation helpers."""

from __future__ import annotations

import numpy as np

from dreambc_isaac.patches import apply_camera_pipeline_patches
from dreambc_isaac.robot import get_named_joint_positions, lock_named_joints


def camera_orientation_from_euler_xyz(degrees_xyz: tuple[float, float, float]):
    import isaacsim.core.utils.numpy.rotations as rot_utils

    return rot_utils.euler_angles_to_quats(np.array(degrees_xyz), degrees=True)


def camera_orientation_look_at(
    position: np.ndarray,
    target: np.ndarray,
    up_axis: np.ndarray | None = None,
) -> np.ndarray:
    import isaacsim.core.utils.numpy.rotations as rot_utils

    up = np.asarray([0.0, 0.0, 1.0] if up_axis is None else up_axis, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)

    z_axis = up - forward * np.dot(up, forward)
    if np.linalg.norm(z_axis) < 1e-6:
        z_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = z_axis / np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, forward)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(forward, y_axis)

    # Isaac `Camera` + exterior mounts: columns map camera (+X,+Y,+Z) into the frame of `position`/`target`.
    # For world-space `position`+`look_at`, +X as optical axis matches our historical rollout behavior.
    rotation_matrix = np.column_stack([forward, y_axis, z_axis])
    return rot_utils.rot_matrices_to_quats(rotation_matrix)


def camera_orientation_look_at_usd_neg_z(
    position: np.ndarray,
    target: np.ndarray,
    up_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Look-at for prims parented under articulation: USD cameras view along **local -Z**.

    `position` and `target` must be expressed in the **same** frame (typically the camera parent's local frame).
    `up_axis` is in that same frame (default local +Z).
    """
    import isaacsim.core.utils.numpy.rotations as rot_utils

    up = np.asarray([0.0, 0.0, 1.0] if up_axis is None else up_axis, dtype=np.float64)
    view = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    view = view / (np.linalg.norm(view) + 1e-12)
    # Camera +Z points opposite to viewing direction so that -Z looks at `target`.
    z_cam = -view
    x_cam = np.cross(up, z_cam)
    nx = np.linalg.norm(x_cam)
    if nx < 1e-6:
        x_cam = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam = np.cross(x_cam, z_cam)
        nx = np.linalg.norm(x_cam)
    x_cam = x_cam / nx
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / (np.linalg.norm(y_cam) + 1e-12)
    x_cam = np.cross(y_cam, z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-12)
    rotation_matrix = np.column_stack([x_cam, y_cam, z_cam])
    return rot_utils.rot_matrices_to_quats(rotation_matrix)


def camera_orientation_usd_neg_z_forward_up(
    forward: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """USD camera in parent frame: local -Z views along `forward`; local +Y is image up (~`up`)."""
    import isaacsim.core.utils.numpy.rotations as rot_utils

    f = np.asarray(forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.asarray(up, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    z_cam = -f
    y_cam = u - z_cam * np.dot(u, z_cam)
    ny = np.linalg.norm(y_cam)
    if ny < 1e-6:
        y_cam = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        y_cam = y_cam - z_cam * np.dot(y_cam, z_cam)
        ny = np.linalg.norm(y_cam)
    y_cam = y_cam / ny
    x_cam = np.cross(y_cam, z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-12)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / (np.linalg.norm(y_cam) + 1e-12)
    rotation_matrix = np.column_stack([x_cam, y_cam, z_cam])
    return rot_utils.rot_matrices_to_quats(rotation_matrix)


def make_cameras(camera_cfg):
    from isaacsim.core.utils.prims import define_prim, is_prim_path_valid
    from isaacsim.sensors.camera import Camera

    if not is_prim_path_valid("/World/Cameras"):
        define_prim("/World/Cameras", "Xform")
    cameras = {}
    for name, cfg in camera_cfg.items():
        if "translation" in cfg:
            translation = np.asarray(cfg.translation, dtype=np.float64)
            if _cfg_contains(cfg, "view_forward_xyz") and _cfg_contains(cfg, "view_up_xyz"):
                orientation = camera_orientation_usd_neg_z_forward_up(
                    np.asarray(cfg.view_forward_xyz, dtype=np.float64),
                    np.asarray(cfg.view_up_xyz, dtype=np.float64),
                )
                if _cfg_contains(cfg, "orientation_post_euler_xyz_degrees"):
                    import isaacsim.core.utils.numpy.rotations as rot_utils

                    post = np.asarray(tuple(cfg.orientation_post_euler_xyz_degrees), dtype=np.float64)
                    q_post = rot_utils.euler_angles_to_quats(post, degrees=True)
                    r_base = rot_utils.quats_to_rot_matrices(orientation)
                    r_post = rot_utils.quats_to_rot_matrices(q_post)
                    orientation = rot_utils.rot_matrices_to_quats(r_base @ r_post)
            elif "look_at" in cfg:
                # World-space cameras: optical axis +X (`camera_orientation_look_at`).
                # Parented wrist / tool cameras: use `look_at_convention: usd_neg_z` so local -Z
                # looks at the target (matches USD camera / Isaac wrist mounts; avoids inverted image).
                look_conv = str(_cfg_get(cfg, "look_at_convention", "plus_x"))
                target = np.asarray(cfg.look_at, dtype=np.float64)
                up_axis = None
                if _cfg_contains(cfg, "look_at_up_xyz"):
                    up_axis = np.asarray(tuple(cfg.look_at_up_xyz), dtype=np.float64)
                if look_conv == "usd_neg_z":
                    orientation = camera_orientation_look_at_usd_neg_z(translation, target, up_axis=up_axis)
                else:
                    orientation = camera_orientation_look_at(translation, target, up_axis=up_axis)
                if _cfg_contains(cfg, "orientation_post_euler_xyz_degrees"):
                    import isaacsim.core.utils.numpy.rotations as rot_utils

                    post = np.asarray(tuple(cfg.orientation_post_euler_xyz_degrees), dtype=np.float64)
                    q_post = rot_utils.euler_angles_to_quats(post, degrees=True)
                    r_look = rot_utils.quats_to_rot_matrices(orientation)
                    r_post = rot_utils.quats_to_rot_matrices(q_post)
                    orientation = rot_utils.rot_matrices_to_quats(r_look @ r_post)
            else:
                orientation = camera_orientation_from_euler_xyz(tuple(cfg.orientation_euler_xyz_degrees))
                if _cfg_contains(cfg, "orientation_post_euler_xyz_degrees"):
                    import isaacsim.core.utils.numpy.rotations as rot_utils

                    post = np.asarray(tuple(cfg.orientation_post_euler_xyz_degrees), dtype=np.float64)
                    q_post = rot_utils.euler_angles_to_quats(post, degrees=True)
                    r_base = rot_utils.quats_to_rot_matrices(orientation)
                    r_post = rot_utils.quats_to_rot_matrices(q_post)
                    orientation = rot_utils.rot_matrices_to_quats(r_base @ r_post)
            cameras[name] = Camera(
                prim_path=cfg.prim_path,
                name=name,
                translation=translation,
                orientation=orientation,
                frequency=int(cfg.frequency),
                resolution=tuple(cfg.resolution),
                annotator_device="cpu",
            )
        else:
            position = np.asarray(cfg.position, dtype=np.float64)
            if "look_at" in cfg:
                orientation = camera_orientation_look_at(position, np.asarray(cfg.look_at, dtype=np.float64))
            else:
                orientation = camera_orientation_from_euler_xyz(tuple(cfg.orientation_euler_xyz_degrees))
            cameras[name] = Camera(
                prim_path=cfg.prim_path,
                name=name,
                position=position,
                orientation=orientation,
                frequency=int(cfg.frequency),
                resolution=tuple(cfg.resolution),
                annotator_device="cpu",
            )
    return cameras


def _cfg_contains(cfg, key: str) -> bool:
    try:
        return key in cfg
    except TypeError:
        return False


def _cfg_get(cfg, key: str, default=None):
    try:
        return cfg.get(key, default)
    except AttributeError:
        return cfg[key] if key in cfg else default


def apply_camera_lens_settings(camera, cfg) -> None:
    if cfg is None:
        return
    if _cfg_contains(cfg, "focal_length"):
        camera.set_focal_length(float(_cfg_get(cfg, "focal_length")))
    if _cfg_contains(cfg, "horizontal_aperture"):
        camera.set_horizontal_aperture(float(_cfg_get(cfg, "horizontal_aperture")))
    if _cfg_contains(cfg, "vertical_aperture"):
        camera.set_vertical_aperture(float(_cfg_get(cfg, "vertical_aperture")))
    if _cfg_contains(cfg, "clipping_range"):
        near, far = _cfg_get(cfg, "clipping_range")
        camera.set_clipping_range(float(near), float(far))


def initialize_cameras(cameras: dict[str, object], camera_cfg=None) -> None:
    apply_camera_pipeline_patches()
    for name, camera in cameras.items():
        try:
            print(f"Initializing camera: {name} ({camera.prim_path})")
            camera.initialize()
            cfg = _cfg_get(camera_cfg, name) if camera_cfg is not None else None
            apply_camera_lens_settings(camera, cfg)
            if cfg is not None and (
                _cfg_contains(cfg, "focal_length")
                or _cfg_contains(cfg, "horizontal_aperture")
                or _cfg_contains(cfg, "vertical_aperture")
                or _cfg_contains(cfg, "clipping_range")
            ):
                print(
                    f"Applied lens settings for camera '{name}': "
                    f"focal_length={camera.get_focal_length():.4f}, "
                    f"horizontal_aperture={camera.get_horizontal_aperture():.4f}, "
                    f"horizontal_fov_deg={np.degrees(camera.get_horizontal_fov()):.1f}"
                )
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
