"""URDF import helpers shared by DreamBC Isaac Sim scripts."""

from __future__ import annotations

from pathlib import Path


URDF_EXTENSION_CANDIDATES = (
    "isaacsim.asset.importer.urdf",
    "omni.importer.urdf",
)


def enable_first_available_urdf_extension(enable_extension) -> str:
    for extension_name in URDF_EXTENSION_CANDIDATES:
        try:
            enable_extension(extension_name)
            return extension_name
        except Exception as exc:
            print(f"Could not enable {extension_name}: {exc}")
    raise RuntimeError(
        "Could not enable a URDF importer extension. Tried: "
        + ", ".join(URDF_EXTENSION_CANDIDATES)
    )


def import_urdf_bindings():
    try:
        from isaacsim.asset.importer.urdf import _urdf

        return _urdf
    except ModuleNotFoundError:
        from omni.importer.urdf import _urdf

        return _urdf


def world_child_paths(stage) -> set[str]:
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        return set()
    return {child.GetPath().pathString for child in world_prim.GetChildren()}


def top_level_world_path(path: str) -> str:
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "World":
        return f"/World/{parts[1]}"
    return path


def move_imported_prim(imported_path: str, target_path: str, before_paths: set[str]) -> str:
    import omni.kit.commands
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    after_paths = world_child_paths(stage)
    new_paths = sorted(after_paths - before_paths)
    imported_root_path = top_level_world_path(imported_path)

    if stage.GetPrimAtPath(imported_root_path).IsValid():
        source_path = imported_root_path
    elif len(new_paths) == 1:
        source_path = new_paths[0]
    else:
        raise RuntimeError(
            "Could not identify imported root prim. "
            f"imported_path={imported_path}, new_world_paths={new_paths}"
        )

    if stage.GetPrimAtPath(target_path).IsValid():
        result, _ = omni.kit.commands.execute("DeletePrims", paths=[target_path])
        if not result:
            raise RuntimeError(f"Failed to delete existing prim: {target_path}")

    UsdGeom.Xform.Define(stage, target_path)
    child_path = f"{target_path}/Robot"
    result, _ = omni.kit.commands.execute(
        "MovePrim",
        path_from=source_path,
        path_to=child_path,
        keep_world_transform=False,
    )
    if not result:
        raise RuntimeError(f"Failed to move imported prim from {source_path} to {child_path}")
    return target_path


def import_urdf(urdf_path: Path, target_path: str) -> str:
    import omni.kit.commands
    import omni.usd

    urdf = import_urdf_bindings()
    stage = omni.usd.get_context().get_stage()

    import_config = urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    before_paths = world_child_paths(stage)
    result, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return move_imported_prim(imported_path, target_path, before_paths)

