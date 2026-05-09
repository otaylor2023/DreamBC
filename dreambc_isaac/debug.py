"""Debug helpers for inspecting Isaac Sim camera and robot placement."""

from __future__ import annotations

import math

import numpy as np


def _parent_prim_path(camera_prim_path: str) -> str:
    p = str(camera_prim_path).rstrip("/")
    if p in ("", "/"):
        return "/World"
    idx = p.rfind("/")
    return "/World" if idx <= 0 else p[:idx]


def _gf_matrix4d_to_numpy(m) -> np.ndarray:
    return np.array(
        [
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [m[3][0], m[3][1], m[3][2], m[3][3]],
        ],
        dtype=np.float64,
    )


def _gf_transform_point(m, p: np.ndarray) -> np.ndarray:
    from pxr import Gf

    point = np.asarray(p, dtype=np.float64).reshape(3)
    transformed = m.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
    return np.asarray([transformed[0], transformed[1], transformed[2]], dtype=np.float64)


def _quat_wxyz_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = (w, x, y, z) in wxyz order."""
    w, x, y, z = [float(c) for c in np.asarray(q, dtype=np.float64).reshape(4)[:4]]
    vx, vy, vz = [float(c) for c in np.asarray(v, dtype=np.float64).reshape(3)[:3]]
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    fx = vx + w * tx + y * tz - z * ty
    fy = vy + w * ty + z * tx - x * tz
    fz = vz + w * tz + x * ty - y * tx
    return np.asarray([fx, fy, fz], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def print_stage_tree(stage, root_path: str, max_depth: int = 4) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"Debug stage tree root not found: {root_path}")
        return

    print(f"Debug stage tree under {root_path}:")

    def visit(prim, depth: int) -> None:
        indent = "  " * depth
        print(f"{indent}{prim.GetPath().pathString} ({prim.GetTypeName()})")
        if depth >= max_depth:
            return
        for child in prim.GetChildren():
            visit(child, depth + 1)

    visit(root, 0)


def prim_world_position(stage, prim_path: str) -> np.ndarray | None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    translation = matrix.ExtractTranslation()
    return np.asarray([translation[0], translation[1], translation[2]], dtype=np.float64)


def print_prim_world_positions(stage, prim_paths: list[str]) -> None:
    print("Debug prim world positions:")
    for prim_path in prim_paths:
        position = prim_world_position(stage, prim_path)
        if position is None:
            print(f"  {prim_path}: MISSING")
        else:
            print(f"  {prim_path}: position={position.tolist()}")


def _camera_cfg_to_dict(camera_cfg) -> dict:
    try:
        from omegaconf import OmegaConf

        cfg_dict = OmegaConf.to_container(camera_cfg, resolve=True)
    except Exception:
        cfg_dict = dict(camera_cfg) if hasattr(camera_cfg, "items") else None
    return cfg_dict if isinstance(cfg_dict, dict) else {}


def collect_prim_world_positions(stage, prim_paths: list[str]) -> dict[str, dict]:
    positions = {}
    for prim_path in prim_paths:
        position = prim_world_position(stage, prim_path)
        positions[str(prim_path)] = {
            "exists": position is not None,
            "position": None if position is None else position.tolist(),
        }
    return positions


def collect_grasp_debug(stage, grasp_cfg) -> dict[str, object]:
    cfg = _camera_cfg_to_dict(grasp_cfg)
    paths = {
        "cube": str(cfg.get("cube_prim_path", "")),
        "gripper_base": str(cfg.get("gripper_base_prim_path", "")),
        "left_outer_finger": str(cfg.get("left_outer_finger_prim_path", "")),
        "right_outer_finger": str(cfg.get("right_outer_finger_prim_path", "")),
        "left_inner_finger": str(cfg.get("left_inner_finger_prim_path", "")),
        "right_inner_finger": str(cfg.get("right_inner_finger_prim_path", "")),
        "wrist_camera": str(cfg.get("wrist_camera_prim_path", "")),
    }
    positions = {
        name: (prim_world_position(stage, path) if path else None)
        for name, path in paths.items()
    }

    def _delta_and_distance(a_name: str, b_name: str, prefix: str, out: dict[str, object]) -> None:
        a = positions.get(a_name)
        b = positions.get(b_name)
        if a is None or b is None:
            out[f"{prefix}_delta"] = None
            out[f"{prefix}_distance"] = None
            return
        delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        out[f"{prefix}_delta"] = delta.tolist()
        out[f"{prefix}_distance"] = float(np.linalg.norm(delta))

    result: dict[str, object] = {
        "paths": paths,
        "positions": {
            name: None if position is None else np.asarray(position, dtype=np.float64).tolist()
            for name, position in positions.items()
        },
    }

    left_inner = positions.get("left_inner_finger")
    right_inner = positions.get("right_inner_finger")
    if left_inner is not None and right_inner is not None:
        inner_center = 0.5 * (np.asarray(left_inner, dtype=np.float64) + np.asarray(right_inner, dtype=np.float64))
        result["inner_finger_center_position"] = inner_center.tolist()
        result["inner_finger_jaw_gap"] = float(
            np.linalg.norm(np.asarray(left_inner, dtype=np.float64) - np.asarray(right_inner, dtype=np.float64))
        )
    else:
        result["inner_finger_center_position"] = None
        result["inner_finger_jaw_gap"] = None

    left_outer = positions.get("left_outer_finger")
    right_outer = positions.get("right_outer_finger")
    if left_outer is not None and right_outer is not None:
        outer_center = 0.5 * (np.asarray(left_outer, dtype=np.float64) + np.asarray(right_outer, dtype=np.float64))
        result["outer_finger_center_position"] = outer_center.tolist()
        result["outer_finger_jaw_gap"] = float(
            np.linalg.norm(np.asarray(left_outer, dtype=np.float64) - np.asarray(right_outer, dtype=np.float64))
        )
    else:
        result["outer_finger_center_position"] = None
        result["outer_finger_jaw_gap"] = None

    cube = positions.get("cube")
    if cube is not None and result["inner_finger_center_position"] is not None:
        delta = np.asarray(result["inner_finger_center_position"], dtype=np.float64) - np.asarray(cube, dtype=np.float64)
        result["inner_finger_center_to_cube_delta"] = delta.tolist()
        result["inner_finger_center_to_cube_distance"] = float(np.linalg.norm(delta))
    else:
        result["inner_finger_center_to_cube_delta"] = None
        result["inner_finger_center_to_cube_distance"] = None

    if cube is not None and result["outer_finger_center_position"] is not None:
        delta = np.asarray(result["outer_finger_center_position"], dtype=np.float64) - np.asarray(cube, dtype=np.float64)
        result["outer_finger_center_to_cube_delta"] = delta.tolist()
        result["outer_finger_center_to_cube_distance"] = float(np.linalg.norm(delta))
    else:
        result["outer_finger_center_to_cube_delta"] = None
        result["outer_finger_center_to_cube_distance"] = None

    _delta_and_distance("gripper_base", "cube", "gripper_base_to_cube", result)
    _delta_and_distance("wrist_camera", "cube", "wrist_camera_to_cube", result)
    _delta_and_distance("left_inner_finger", "cube", "left_inner_finger_to_cube", result)
    _delta_and_distance("right_inner_finger", "cube", "right_inner_finger_to_cube", result)
    _delta_and_distance("left_outer_finger", "cube", "left_outer_finger_to_cube", result)
    _delta_and_distance("right_outer_finger", "cube", "right_outer_finger_to_cube", result)
    return result


def collect_camera_debug(
    cameras: dict[str, object],
    *,
    stage=None,
    camera_cfg=None,
) -> dict[str, dict]:
    cfg_dict = _camera_cfg_to_dict(camera_cfg)
    debug = {}
    usd_geom = None
    if stage is not None:
        from pxr import UsdGeom

        usd_geom = UsdGeom

    for name, camera in cameras.items():
        position, orientation = camera.get_world_pose()
        q = np.asarray(orientation, dtype=np.float64).reshape(4)
        ex = _unit(_quat_wxyz_rotate_vector(q, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
        ey = _unit(_quat_wxyz_rotate_vector(q, np.array([0.0, 1.0, 0.0], dtype=np.float64)))
        ez = _unit(_quat_wxyz_rotate_vector(q, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
        ez_neg = _unit(-ez)

        entry = {
            "prim_path": str(camera.prim_path),
            "parent_prim_path": _parent_prim_path(str(camera.prim_path)),
            "position": np.asarray(position, dtype=np.float64).tolist(),
            "orientation_wxyz": q.tolist(),
            "axes_world": {
                "plus_x": ex.tolist(),
                "plus_y": ey.tolist(),
                "plus_z": ez.tolist(),
                "minus_z": ez_neg.tolist(),
            },
        }
        try:
            horizontal_fov = float(camera.get_horizontal_fov())
            vertical_fov = float(camera.get_vertical_fov())
            entry["lens"] = {
                "focal_length": float(camera.get_focal_length()),
                "horizontal_aperture": float(camera.get_horizontal_aperture()),
                "vertical_aperture": float(camera.get_vertical_aperture()),
                "horizontal_fov_degrees": math.degrees(horizontal_fov),
                "vertical_fov_degrees": math.degrees(vertical_fov),
                "clipping_range": list(camera.get_clipping_range()),
            }
        except Exception as exc:
            entry["lens_error"] = f"{type(exc).__name__}: {exc}"

        cam_cfg = cfg_dict.get(name)
        if isinstance(cam_cfg, dict):
            entry["config"] = cam_cfg
            cfg_ray_world = None
            look_at_world = None
            if "look_at" in cam_cfg:
                la = np.asarray(cam_cfg["look_at"], dtype=np.float64).reshape(3)
                if "translation" in cam_cfg and stage is not None and usd_geom is not None:
                    t_loc = np.asarray(cam_cfg["translation"], dtype=np.float64).reshape(3)
                    parent = stage.GetPrimAtPath(entry["parent_prim_path"])
                    if parent.IsValid():
                        m = usd_geom.Xformable(parent).ComputeLocalToWorldTransform(0.0)
                        origin_world = _gf_transform_point(m, t_loc)
                        look_at_world = _gf_transform_point(m, la)
                        cfg_ray_world = _unit(look_at_world - origin_world)
                        entry["configured_origin_world"] = origin_world.tolist()
                    else:
                        entry["parent_missing"] = True
                elif "position" in cam_cfg:
                    p0 = np.asarray(cam_cfg["position"], dtype=np.float64).reshape(3)
                    cfg_ray_world = _unit(la - p0)
                    look_at_world = la

            if cfg_ray_world is not None:
                entry["configured_look_at_world"] = look_at_world.tolist()
                entry["configured_view_ray_world"] = cfg_ray_world.tolist()
                entry["axis_alignment_with_configured_ray"] = {
                    "dot_plus_x": float(np.dot(ex, cfg_ray_world)),
                    "dot_plus_y": float(np.dot(ey, cfg_ray_world)),
                    "dot_plus_z": float(np.dot(ez, cfg_ray_world)),
                    "dot_minus_z": float(np.dot(ez_neg, cfg_ray_world)),
                }

        debug[name] = entry
    return debug


def print_camera_world_poses(
    cameras: dict[str, object],
    *,
    stage=None,
    camera_cfg=None,
) -> None:
    print("Debug camera world poses:")
    for name, camera in cameras.items():
        position, orientation = camera.get_world_pose()
        print(
            f"  {name}: prim_path={camera.prim_path}, "
            f"position={np.asarray(position).tolist()}, orientation_wxyz={np.asarray(orientation).tolist()}"
        )

    if stage is None or camera_cfg is None:
        return

    cfg_dict = _camera_cfg_to_dict(camera_cfg)
    if not cfg_dict:
        return

    print("Debug camera view axes (world): +X / +Y / -Z expressed in world frame from get_world_pose quaternion.")
    print("  Compare 'cfg_ray_world' (from YAML translation/look_at in parent-local) to these axes;")
    print("  Isaac Camera images should align the rendered optical axis with +X.")

    from pxr import UsdGeom

    for name, camera in cameras.items():
        cam_cfg = cfg_dict.get(name)
        if not isinstance(cam_cfg, dict):
            continue
        q = np.asarray(camera.get_world_pose()[1], dtype=np.float64).reshape(4)
        ex = _quat_wxyz_rotate_vector(q, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        ey = _quat_wxyz_rotate_vector(q, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        ez_neg = _quat_wxyz_rotate_vector(q, np.array([0.0, 0.0, -1.0], dtype=np.float64))

        cfg_ray_world = None
        if "look_at" in cam_cfg:
            la = np.asarray(cam_cfg["look_at"], dtype=np.float64).reshape(3)
            if "translation" in cam_cfg:
                t_loc = np.asarray(cam_cfg["translation"], dtype=np.float64).reshape(3)
                parent_path = _parent_prim_path(str(camera.prim_path))
                parent = stage.GetPrimAtPath(parent_path)
                if parent.IsValid():
                    m = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0.0)
                    ray_local = _unit(la - t_loc)
                    p_w = _gf_transform_point(m, t_loc)
                    la_w = _gf_transform_point(m, t_loc + ray_local)
                    cfg_ray_world = _unit(la_w - p_w)
                else:
                    print(f"  {name}: parent prim missing for view-ray debug: {parent_path}")
            elif "position" in cam_cfg:
                p0 = np.asarray(cam_cfg["position"], dtype=np.float64).reshape(3)
                cfg_ray_world = _unit(la - p0)

        parts = [
            f"+X={np.round(ex, 4).tolist()}",
            f"+Y={np.round(ey, 4).tolist()}",
            f"-Z={np.round(ez_neg, 4).tolist()}",
        ]
        if cfg_ray_world is not None:
            parts.append(f"cfg_ray_world={np.round(cfg_ray_world, 4).tolist()}")
            parts.append(f"dot(+X,cfg)={float(np.dot(ex, cfg_ray_world)):.3f}")
            parts.append(f"dot(-Z,cfg)={float(np.dot(ez_neg, cfg_ray_world)):.3f}")
        print(f"  {name}: " + " ".join(parts))


def add_debug_sphere(world, prim_path: str, position: np.ndarray, radius: float, color: tuple[float, float, float]) -> None:
    from isaacsim.core.api.objects import VisualSphere

    world.scene.add(
        VisualSphere(
            prim_path=prim_path,
            name=prim_path.strip("/").replace("/", "_"),
            position=np.asarray(position, dtype=np.float64),
            radius=float(radius),
            color=np.asarray(color, dtype=np.float64),
        )
    )


def add_camera_debug_markers(world, cameras: dict[str, object], camera_cfg) -> None:
    for name, camera in cameras.items():
        position, _ = camera.get_world_pose()
        safe_name = name.replace("/", "_")
        add_debug_sphere(
            world,
            f"/World/Debug/Cameras/{safe_name}_position",
            np.asarray(position, dtype=np.float64),
            0.025,
            (0.0, 1.0, 0.0),
        )
        cfg = camera_cfg[name]
        if "look_at" not in cfg:
            continue
        la = np.asarray(cfg.look_at, dtype=np.float64)
        if "translation" not in cfg:
            add_debug_sphere(
                world,
                f"/World/Debug/Cameras/{safe_name}_look_at",
                la,
                0.02,
                (1.0, 0.0, 0.0),
            )
        else:
            from pxr import UsdGeom

            parent_path = _parent_prim_path(str(camera.prim_path))
            parent = world.stage.GetPrimAtPath(parent_path)
            if parent.IsValid():
                m = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(0.0)
                t_loc = np.asarray(cfg.translation, dtype=np.float64).reshape(3)
                p_w = _gf_transform_point(m, t_loc)
                la_w = _gf_transform_point(m, la)
                add_debug_sphere(world, f"/World/Debug/Cameras/{safe_name}_cam_origin", p_w, 0.025, (0.0, 1.0, 0.0))
                add_debug_sphere(world, f"/World/Debug/Cameras/{safe_name}_look_at", la_w, 0.02, (1.0, 0.0, 0.0))
