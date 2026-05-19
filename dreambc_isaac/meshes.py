"""Import OBJ/GLB/FBX meshes into the Isaac Sim stage (convert to USD, then reference)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MESH_SUFFIXES = {".obj", ".glb", ".gltf", ".fbx"}
INCHES_TO_METERS = 0.0254

_ASSET_CONVERTER_EXTENSIONS = (
    "omni.kit.asset_converter",
    "omni.kit.tool.asset_importer",
)


def default_usd_cache_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".usd")


def _load_mesh_bounds(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (bounds_min, bounds_max) from the source mesh file."""
    import trimesh

    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        bounds = loaded.bounds
    else:
        bounds = loaded.bounds
    return np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)


def mesh_extent_xyz(mesh_path: Path) -> np.ndarray:
    """Axis-aligned bounds size of the mesh file (meters when authored in meters)."""
    bounds_min, bounds_max = _load_mesh_bounds(mesh_path)
    return bounds_max - bounds_min


def mesh_anchor_point(bounds_min: np.ndarray, bounds_max: np.ndarray, anchor: str) -> np.ndarray:
    """Point in mesh-local space used as the placement handle."""
    center = 0.5 * (bounds_min + bounds_max)
    anchor = anchor.strip().lower()
    if anchor in ("bottom_center", "base", "table_contact"):
        return np.array([center[0], center[1], bounds_min[2]], dtype=np.float64)
    if anchor in ("center", "centroid", "middle"):
        return center
    if anchor in ("origin", "pivot"):
        return np.zeros(3, dtype=np.float64)
    raise ValueError(f"Unknown placement_anchor {anchor!r}")


def uniform_scale_for_target_height(
    mesh_path: Path,
    target_height_m: float,
    *,
    height_axis: str | None = None,
) -> float:
    ext = mesh_extent_xyz(mesh_path)
    axis = (height_axis or "max").strip().lower()
    if axis in ("max", "longest", "largest"):
        native_height = float(np.max(ext))
    else:
        idx = {"x": 0, "y": 1, "z": 2}.get(axis)
        if idx is None:
            raise ValueError(
                f"height_axis must be x, y, z, or max (got {height_axis!r})"
            )
        native_height = float(ext[idx])
    if native_height <= 0.0:
        raise ValueError(f"Mesh has non-positive extent along chosen axis: {mesh_path}")
    return float(target_height_m) / native_height


def resolve_mesh_scale(mesh_path: Path, mesh_cfg) -> object | None:
    """Return uniform float, xyz list, or None from Hydra mesh config."""
    target_height_m = getattr(mesh_cfg, "target_height_m", None)
    height_axis = getattr(mesh_cfg, "height_axis", None)
    scale = getattr(mesh_cfg, "scale", None)

    if target_height_m is not None:
        uniform = uniform_scale_for_target_height(
            mesh_path,
            float(target_height_m),
            height_axis=str(height_axis) if height_axis is not None else None,
        )
        if scale is None:
            return uniform
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        if sc.size == 1:
            return float(sc[0]) * uniform
        if sc.size >= 3:
            return (sc[:3] * uniform).tolist()
        return uniform
    return scale


def ensure_asset_converter_extensions(enable_extension) -> None:
    for extension_name in _ASSET_CONVERTER_EXTENSIONS:
        try:
            enable_extension(extension_name)
        except Exception as exc:
            print(f"Could not enable {extension_name}: {exc}")


async def convert_mesh_to_usd_async(mesh_path: Path, usd_path: Path) -> None:
    import omni.kit.app
    import omni.kit.asset_converter

    mesh_path = mesh_path.resolve()
    usd_path = usd_path.resolve()
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    if usd_path.is_file() and usd_path.stat().st_mtime >= mesh_path.stat().st_mtime:
        return

    context = omni.kit.asset_converter.AssetConverterContext()
    context.use_meter_as_world_unit = True
    # Avoid an extra /World wrapper prim; keep Isaac Sim stage Z-up for GLB/OBJ.
    context.create_world_as_default_root_prim = False
    context.convert_stage_up_z = True

    def progress_callback(progress, total_steps):
        del progress, total_steps

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        str(mesh_path),
        str(usd_path),
        progress_callback,
        context,
    )
    success = await task.wait_until_finished()
    if not success:
        err = task.get_error_message() if hasattr(task, "get_error_message") else ""
        status = task.get_status() if hasattr(task, "get_status") else ""
        raise RuntimeError(f"Failed to convert {mesh_path} to USD (status={status}): {err}")
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()


def _run_async_coroutine(coroutine) -> None:
    """Schedule on Isaac's main loop and block until complete."""
    import omni.kit.app
    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(coroutine)
    app = omni.kit.app.get_app()
    while not task.done():
        app.update()
    if task.cancelled():
        raise RuntimeError("Mesh conversion task was cancelled")
    exc = task.exception()
    if exc is not None:
        raise exc


def convert_mesh_to_usd(mesh_path: Path, usd_path: Path) -> Path:
    _run_async_coroutine(convert_mesh_to_usd_async(mesh_path, usd_path))
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD not created: {usd_path}")
    return usd_path


def _xformable(prim_path: str):
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {prim_path}")
    return stage, UsdGeom.Xformable(prim)


def apply_mesh_xform(
    prim_path: str,
    *,
    pivot_offset=None,
    position=None,
    scale=None,
    rotate_xyz_deg=None,
    clear: bool = True,
) -> None:
    """Set local xform ops (order: pivot, scale, rotate, translate)."""
    from pxr import Gf, UsdGeom

    _, xform = _xformable(prim_path)
    if clear:
        xform.ClearXformOpOrder()

    if pivot_offset is not None:
        pivot = np.asarray(pivot_offset, dtype=np.float64).reshape(-1)
        if pivot.size >= 3 and np.any(np.abs(pivot) > 1e-9):
            op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(float(pivot[0]), float(pivot[1]), float(pivot[2])))

    if scale is not None:
        sc = np.asarray(scale, dtype=np.float64).reshape(-1)
        if sc.size == 1:
            sc = np.repeat(sc, 3)
        if sc.size >= 3 and not np.allclose(sc[:3], 1.0):
            op = xform.AddScaleOp()
            op.Set(Gf.Vec3f(float(sc[0]), float(sc[1]), float(sc[2])))

    if rotate_xyz_deg is not None:
        rot = np.asarray(rotate_xyz_deg, dtype=np.float64).reshape(-1)
        if rot.size == 3 and np.any(np.abs(rot) > 1e-6):
            op = xform.AddRotateXYZOp()
            op.Set(Gf.Vec3f(float(rot[0]), float(rot[1]), float(rot[2])))

    if position is not None:
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.size >= 3:
            op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def world_bounds(prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    stage, xform = _xformable(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedBox()
    return np.asarray(box.GetMin(), dtype=np.float64), np.asarray(box.GetMax(), dtype=np.float64)


def world_anchor_point(prim_path: str, anchor: str) -> np.ndarray:
    bounds_min, bounds_max = world_bounds(prim_path)
    return mesh_anchor_point(bounds_min, bounds_max, anchor)


def _update_stage_for_bounds() -> None:
    import isaacsim.core.utils.stage as stage_utils

    stage_utils.update_stage()


def _read_translate_xyz(prim_path: str) -> np.ndarray | None:
    from pxr import UsdGeom

    _, xform = _xformable(prim_path)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    return None


def snap_world_bottom_to_z(prim_path: str, target_z: float) -> None:
    """Nudge parent translate so the world AABB bottom sits on ``target_z``."""
    for _ in range(6):
        mn, _ = world_bounds(prim_path)
        dz = float(target_z) - float(mn[2])
        if abs(dz) < 1e-4:
            return
        current = _read_translate_xyz(prim_path)
        if current is None:
            current = np.zeros(3, dtype=np.float64)
        apply_mesh_xform(
            prim_path,
            position=current + np.array([0.0, 0.0, dz]),
            clear=True,
        )
        _update_stage_for_bounds()


def spawn_mesh_prop(
    mesh_path: Path,
    prim_path: str,
    *,
    position=None,
    scale=None,
    rotate_xyz_deg=None,
    placement_anchor: str = "bottom_center",
    usd_path: Path | None = None,
) -> str:
    """Spawn mesh under ``prim_path/geo``; ``position`` is the world point for ``placement_anchor``."""
    from pxr import UsdGeom
    from isaacsim.core.utils.stage import add_reference_to_stage

    mesh_path = mesh_path.resolve()
    suffix = mesh_path.suffix.lower()
    if suffix not in MESH_SUFFIXES:
        raise ValueError(f"Unsupported mesh format '{suffix}' for {mesh_path}")

    usd_path = (usd_path or default_usd_cache_path(mesh_path)).resolve()
    convert_mesh_to_usd(mesh_path, usd_path)

    parent_path = prim_path
    geo_path = f"{prim_path}/geo"

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, parent_path)
    UsdGeom.Xform.Define(stage, geo_path)

    add_reference_to_stage(usd_path=str(usd_path), prim_path=geo_path)

    # Scale + rotation on geo; parent is translation-only so world-Z placement/snapping works.
    apply_mesh_xform(
        geo_path,
        scale=scale,
        rotate_xyz_deg=rotate_xyz_deg,
        clear=True,
    )
    apply_mesh_xform(parent_path, clear=True)
    _update_stage_for_bounds()

    if position is not None:
        desired = np.asarray(position, dtype=np.float64).reshape(3)
        current_anchor = world_anchor_point(parent_path, placement_anchor)
        apply_mesh_xform(
            parent_path,
            position=desired - current_anchor,
            clear=True,
        )
        _update_stage_for_bounds()
        if placement_anchor.strip().lower() in ("bottom_center", "base", "table_contact"):
            snap_world_bottom_to_z(parent_path, float(desired[2]))

    return parent_path


def spawn_scene_props(scene_cfg, project_root: Path, *, enable_extension) -> None:
    from dreambc_isaac.io import resolve_project_path

    props_cfg = getattr(scene_cfg, "props", None)
    if props_cfg is None or not bool(getattr(props_cfg, "enabled", True)):
        return

    meshes = getattr(props_cfg, "meshes", None)
    if not meshes:
        return

    ensure_asset_converter_extensions(enable_extension)

    for name, mesh_cfg in meshes.items():
        mesh_path = resolve_project_path(project_root, mesh_cfg.mesh_path)
        usd_cache = None
        if getattr(mesh_cfg, "usd_path", None):
            usd_cache = resolve_project_path(project_root, mesh_cfg.usd_path)

        scale = resolve_mesh_scale(mesh_path, mesh_cfg)
        spawn_mesh_prop(
            mesh_path,
            str(mesh_cfg.prim_path),
            position=getattr(mesh_cfg, "position", None),
            scale=scale,
            rotate_xyz_deg=getattr(mesh_cfg, "rotate_xyz_deg", None),
            placement_anchor=str(getattr(mesh_cfg, "placement_anchor", "bottom_center")),
            usd_path=usd_cache,
        )
        print(f"Spawned scene prop '{name}' -> {mesh_cfg.prim_path} ({mesh_path.name})")
