"""Extra USD mounts parented on imported robot links (not URDF edits; fixed Xforms)."""

from __future__ import annotations

from typing import Any


def _as_vec3(seq: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if seq is None:
        return default
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _set_gprim_display_color(prim, rgb: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    gprim = UsdGeom.Gprim(prim)
    if not gprim:
        return
    gprim.CreateDisplayColorAttr([(Gf.Vec3f(rgb[0], rgb[1], rgb[2]))])


def _ensure_preview_material(stage, mat_path: str, diffuse: tuple[float, float, float], roughness: float = 0.55) -> None:
    """UsdPreviewSurface for RTX / path-traced views (displayColor alone is often not enough)."""
    if stage.GetPrimAtPath(mat_path).IsValid():
        return
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(diffuse[0], diffuse[1], diffuse[2]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")


def _bind_material(stage, gprim_path: str, mat_path: str) -> None:
    from pxr import UsdShade

    gprim = stage.GetPrimAtPath(gprim_path)
    mat = stage.GetPrimAtPath(mat_path)
    if not gprim.IsValid() or not mat.IsValid():
        return
    UsdShade.MaterialBindingAPI.Apply(gprim).Bind(UsdShade.Material(mat))


def _dot_translation_on_face(
    hx: float,
    hy: float,
    hz: float,
    dot_r: float,
    face: str,
) -> tuple[float, float, float]:
    """Center of the chosen box face, offset outward by `dot_r`. Default face is +Z (largest / front panel)."""
    f = face.strip().lower()
    if f in ("x_positive", "+x", "x"):
        return (hx + dot_r, 0.0, 0.0)
    if f in ("y_positive", "+y", "y"):
        return (0.0, hy + dot_r, 0.0)
    if f in ("y_negative", "-y"):
        return (0.0, -(hy + dot_r), 0.0)
    if f in ("z_negative", "-z"):
        return (0.0, 0.0, -(hz + dot_r))
    # z_positive, front, +z
    return (0.0, 0.0, hz + dot_r)


def _ensure_visible_wrist_mount_box(
    stage,
    mount_path: str,
    mount_cfg: Any,
) -> None:
    """Box centered at mount origin; red dot centered on the configured front face."""
    hx, hy, hz = _as_vec3(mount_cfg.get("box_half_extents"), (0.022, 0.016, 0.012))
    dot_r = float(mount_cfg.get("dot_radius", 0.005))
    box_rgb = _as_vec3(mount_cfg.get("box_color_rgb"), (0.48, 0.5, 0.52))
    dot_rgb = _as_vec3(mount_cfg.get("dot_color_rgb"), (0.92, 0.14, 0.1))
    dot_face = str(mount_cfg.get("dot_face", "z_positive"))
    dot_t = _dot_translation_on_face(hx, hy, hz, dot_r, dot_face)

    from pxr import Gf, UsdGeom

    box_path = f"{mount_path}/visual_box/box"
    box_mtl_path = f"{mount_path}/mtls/box_mtl"
    if not stage.GetPrimAtPath(box_path).IsValid():
        mtls_root = f"{mount_path}/mtls"
        if not stage.GetPrimAtPath(mtls_root).IsValid():
            UsdGeom.Scope.Define(stage, mtls_root)
        UsdGeom.Xform.Define(stage, f"{mount_path}/visual_box")
        cube = UsdGeom.Cube.Define(stage, box_path)
        cube.CreateSizeAttr(2.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddScaleOp().Set(Gf.Vec3d(hx, hy, hz))
        _set_gprim_display_color(cube.GetPrim(), box_rgb)
        _ensure_preview_material(stage, box_mtl_path, box_rgb, roughness=0.6)
        _bind_material(stage, box_path, box_mtl_path)

    dot_mount_path = f"{mount_path}/dot_mount"
    if not stage.GetPrimAtPath(dot_mount_path).IsValid():
        UsdGeom.Xform.Define(stage, dot_mount_path)
    dx = UsdGeom.Xformable(stage.GetPrimAtPath(dot_mount_path))
    dx.ClearXformOpOrder()
    dx.AddTranslateOp().Set(Gf.Vec3d(dot_t[0], dot_t[1], dot_t[2]))

    dot_geo_path = f"{dot_mount_path}/dot_marker"
    if not stage.GetPrimAtPath(dot_geo_path).IsValid():
        sph = UsdGeom.Sphere.Define(stage, dot_geo_path)
        sph.CreateRadiusAttr(dot_r)
        _set_gprim_display_color(sph.GetPrim(), dot_rgb)
        dot_mtl = f"{mount_path}/mtls/dot_mtl"
        _ensure_preview_material(stage, dot_mtl, dot_rgb, roughness=0.35)
        _bind_material(stage, dot_geo_path, dot_mtl)


def ensure_link7_wrist_camera_mount(stage, scene_cfg: Any) -> None:
    """Create a fixed Xform under `panda_link7` and optional visible box + dot for the wrist camera.

    This does not add a URDF joint; Isaac already exposes each link as an Xform. The mount
    only applies a constant translation in the parent link's frame (e.g. +X toward the
    band / "front" of link7 in Franka conventions).

    Re-running is safe: missing geometry is filled in under an existing mount Xform.
    """
    try:
        mount = scene_cfg.get("link7_wrist_camera_mount")
    except Exception:
        mount = None
    if mount is None:
        return
    try:
        enabled = bool(mount.get("enabled", True))
    except Exception:
        enabled = True
    if not enabled:
        return

    parent_path = str(mount.get("parent_prim", "/World/Panda/panda_link7")).rstrip("/")
    mount_name = str(mount.get("mount_name", "wrist_camera_mount"))
    mount_path = f"{parent_path}/{mount_name}"

    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        print(f"Warning: link7 wrist camera mount skipped; parent prim missing: {parent_path}")
        return

    from pxr import Gf, UsdGeom

    tr = mount.get("mount_translation", [0.055, 0.0, 0.082])
    tvec = _as_vec3(tr, (0.055, 0.0, 0.082))
    try:
        rz_deg = float(mount.get("mount_rotate_z_deg", 0.0))
    except Exception:
        rz_deg = 0.0

    if not stage.GetPrimAtPath(mount_path).IsValid():
        UsdGeom.Xform.Define(stage, mount_path)

    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(mount_path))
    xformable.ClearXformOpOrder()
    # USD applies ops left-to-right: rotateZ then translate => p_link7 = t + R_z * p_mount.
    if abs(rz_deg) > 1e-9:
        xformable.AddRotateZOp().Set(float(rz_deg))
    xformable.AddTranslateOp().Set(Gf.Vec3d(tvec[0], tvec[1], tvec[2]))

    try:
        show_box = bool(mount.get("visible_box", True))
    except Exception:
        show_box = True
    if show_box:
        _ensure_visible_wrist_mount_box(stage, mount_path, mount)
