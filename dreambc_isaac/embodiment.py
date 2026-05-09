"""Embodiment alignment helpers for gripper geometry, wrist cameras, and policy state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cfg_to_dict(cfg) -> dict:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        value = dict(cfg) if hasattr(cfg, "items") else {}
    return value if isinstance(value, dict) else {}


def parent_prim_path(prim_path: str) -> str:
    path = str(prim_path).rstrip("/")
    if path in ("", "/"):
        return "/World"
    idx = path.rfind("/")
    return "/World" if idx <= 0 else path[:idx]


def _gf_transform_point(matrix, point: np.ndarray) -> np.ndarray:
    from pxr import Gf

    p = np.asarray(point, dtype=np.float64).reshape(3)
    transformed = matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    return np.asarray([transformed[0], transformed[1], transformed[2]], dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / norm).astype(np.float64)


def prim_world_position(stage, prim_path: str) -> np.ndarray | None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        return None
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    translation = matrix.ExtractTranslation()
    return np.asarray([translation[0], translation[1], translation[2]], dtype=np.float64)


def world_point_to_local(stage, reference_prim_path: str, world_point: np.ndarray) -> np.ndarray | None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(str(reference_prim_path))
    if not prim.IsValid():
        return None
    world_from_local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    local_from_world = world_from_local.GetInverse()
    return _gf_transform_point(local_from_world, world_point)


def prim_local_position(stage, prim_path: str, reference_prim_path: str) -> np.ndarray | None:
    position_world = prim_world_position(stage, prim_path)
    if position_world is None:
        return None
    return world_point_to_local(stage, reference_prim_path, position_world)


def _center(points: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [np.asarray(point, dtype=np.float64).reshape(3) for point in points if point is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def _zero_axes(values: np.ndarray, axes) -> np.ndarray:
    if not axes:
        return np.asarray(values, dtype=np.float64).reshape(3)
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    result = np.asarray(values, dtype=np.float64).reshape(3).copy()
    for axis in axes:
        idx = axis_to_index.get(str(axis).lower())
        if idx is None:
            raise ValueError(f"Unsupported axis '{axis}'. Expected one of: x, y, z.")
        result[idx] = 0.0
    return result


def _ray_metrics(origin: np.ndarray, look_at: np.ndarray, point: np.ndarray) -> dict[str, float]:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    ray = _unit(np.asarray(look_at, dtype=np.float64).reshape(3) - o)
    rel = p - o
    depth = float(np.dot(rel, ray))
    perp = rel - depth * ray
    return {
        "depth_along_view_ray": depth,
        "perpendicular_distance_to_view_ray": float(np.linalg.norm(perp)),
    }


@dataclass
class CameraAutoAlignResult:
    camera_name: str
    camera_prim_path: str
    parent_prim_path: str
    anchor_name: str
    manual_translation_local: np.ndarray
    manual_look_at_local: np.ndarray
    resolved_translation_local: np.ndarray
    resolved_look_at_local: np.ndarray
    anchor_position_local: np.ndarray
    inner_finger_center_local: np.ndarray | None
    outer_finger_center_local: np.ndarray | None
    translation_offset_from_anchor: np.ndarray
    look_at_offset_from_anchor: np.ndarray
    anchor_view_metrics: dict[str, float]
    inner_center_view_metrics: dict[str, float] | None
    outer_center_view_metrics: dict[str, float] | None
    auto_align_cfg: dict

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_name": self.camera_name,
            "camera_prim_path": self.camera_prim_path,
            "parent_prim_path": self.parent_prim_path,
            "anchor_name": self.anchor_name,
            "manual_translation_local": self.manual_translation_local.tolist(),
            "manual_look_at_local": self.manual_look_at_local.tolist(),
            "resolved_translation_local": self.resolved_translation_local.tolist(),
            "resolved_look_at_local": self.resolved_look_at_local.tolist(),
            "anchor_position_local": self.anchor_position_local.tolist(),
            "inner_finger_center_local": None
            if self.inner_finger_center_local is None
            else self.inner_finger_center_local.tolist(),
            "outer_finger_center_local": None
            if self.outer_finger_center_local is None
            else self.outer_finger_center_local.tolist(),
            "translation_offset_from_anchor": self.translation_offset_from_anchor.tolist(),
            "look_at_offset_from_anchor": self.look_at_offset_from_anchor.tolist(),
            "anchor_view_metrics": self.anchor_view_metrics,
            "inner_center_view_metrics": self.inner_center_view_metrics,
            "outer_center_view_metrics": self.outer_center_view_metrics,
            "auto_align_cfg": self.auto_align_cfg,
        }


def resolve_camera_auto_align(stage, camera_name: str, camera_cfg, grasp_cfg) -> CameraAutoAlignResult | None:
    cfg = _cfg_to_dict(camera_cfg)
    auto_align = cfg.get("auto_align") or {}
    if not bool(auto_align.get("enabled", False)):
        return None
    if "translation" not in cfg or "look_at" not in cfg:
        raise ValueError(
            f"Camera '{camera_name}' auto_align requires both translation and look_at in the camera config."
        )

    grasp = _cfg_to_dict(grasp_cfg)
    camera_prim_path = str(cfg["prim_path"])
    parent_path = parent_prim_path(camera_prim_path)
    manual_translation = np.asarray(cfg["translation"], dtype=np.float64).reshape(3)
    manual_look_at = np.asarray(cfg["look_at"], dtype=np.float64).reshape(3)

    inner_center = _center(
        [
            prim_local_position(stage, grasp.get("left_inner_finger_prim_path", ""), parent_path),
            prim_local_position(stage, grasp.get("right_inner_finger_prim_path", ""), parent_path),
        ]
    )
    outer_center = _center(
        [
            prim_local_position(stage, grasp.get("left_outer_finger_prim_path", ""), parent_path),
            prim_local_position(stage, grasp.get("right_outer_finger_prim_path", ""), parent_path),
        ]
    )
    anchors = {
        "inner_finger_center": inner_center,
        "outer_finger_center": outer_center,
    }
    anchor_name = str(auto_align.get("anchor", "inner_finger_center"))
    anchor = anchors.get(anchor_name)
    if anchor is None:
        raise RuntimeError(
            f"Camera '{camera_name}' auto_align anchor '{anchor_name}' is unavailable. "
            f"Available non-empty anchors: {[name for name, value in anchors.items() if value is not None]}"
        )

    if bool(auto_align.get("inherit_translation_offset_from_manual_mount", True)):
        translation_offset = manual_translation - anchor
    else:
        translation_offset = np.asarray(
            auto_align.get("translation_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
    if bool(auto_align.get("inherit_look_at_offset_from_manual_mount", True)):
        look_at_offset = manual_look_at - anchor
    else:
        look_at_offset = np.asarray(
            auto_align.get("look_at_offset", [0.02, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)

    translation_offset = _zero_axes(translation_offset, auto_align.get("zero_translation_offset_axes", []))
    look_at_offset = _zero_axes(look_at_offset, auto_align.get("zero_look_at_offset_axes", []))
    translation_offset += np.asarray(
        auto_align.get("translation_offset_adjustment", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    look_at_offset += np.asarray(
        auto_align.get("look_at_offset_adjustment", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)

    resolved_translation = anchor + translation_offset
    resolved_look_at = anchor + look_at_offset

    return CameraAutoAlignResult(
        camera_name=camera_name,
        camera_prim_path=camera_prim_path,
        parent_prim_path=parent_path,
        anchor_name=anchor_name,
        manual_translation_local=manual_translation,
        manual_look_at_local=manual_look_at,
        resolved_translation_local=resolved_translation,
        resolved_look_at_local=resolved_look_at,
        anchor_position_local=anchor,
        inner_finger_center_local=inner_center,
        outer_finger_center_local=outer_center,
        translation_offset_from_anchor=translation_offset,
        look_at_offset_from_anchor=look_at_offset,
        anchor_view_metrics=_ray_metrics(resolved_translation, resolved_look_at, anchor),
        inner_center_view_metrics=None if inner_center is None else _ray_metrics(resolved_translation, resolved_look_at, inner_center),
        outer_center_view_metrics=None if outer_center is None else _ray_metrics(resolved_translation, resolved_look_at, outer_center),
        auto_align_cfg=auto_align,
    )


def collect_embodiment_alignment(stage, camera_cfg, grasp_cfg) -> dict[str, dict[str, object]]:
    camera_cfg_dict = _cfg_to_dict(camera_cfg)
    summaries: dict[str, dict[str, object]] = {}
    for camera_name, single_camera_cfg in camera_cfg_dict.items():
        try:
            result = resolve_camera_auto_align(stage, str(camera_name), single_camera_cfg, grasp_cfg)
        except Exception as exc:
            summaries[str(camera_name)] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if result is not None:
            summaries[str(camera_name)] = result.to_dict()
    return summaries
