"""
Isaac Sim preview script for Panda variants with different grippers.

Run from Isaac Sim Python, for example:
  ./python.sh isaacsim_dual_robot_preview.py --gripper both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


URDF_EXTENSION_CANDIDATES = (
    "isaacsim.asset.importer.urdf",
    "omni.importer.urdf",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview Panda URDF variants with explicit gripper types in Isaac Sim."
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening UI.")
    parser.add_argument(
        "--gripper",
        choices=["franka", "robotiq", "both"],
        default="both",
        help="Which gripper variant(s) to import into the stage.",
    )
    return parser.parse_args()


def _enable_first_available_extension(enable_extension) -> str:
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


def _import_urdf_bindings():
    try:
        from isaacsim.asset.importer.urdf import _urdf

        return _urdf
    except ModuleNotFoundError:
        from omni.importer.urdf import _urdf

        return _urdf


def _world_child_paths(stage) -> set[str]:
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        return set()
    return {child.GetPath().pathString for child in world_prim.GetChildren()}


def _top_level_world_path(path: str) -> str:
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "World":
        return f"/World/{parts[1]}"
    return path


def _move_imported_prim(imported_path: str, target_path: str, before_paths: set[str]) -> str:
    import omni.kit.commands
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    after_paths = _world_child_paths(stage)
    new_paths = sorted(after_paths - before_paths)
    imported_root_path = _top_level_world_path(imported_path)

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


def _import_urdf(urdf_path: Path, target_path: str):
    import omni.kit.commands
    import omni.usd

    _urdf = _import_urdf_bindings()
    stage = omni.usd.get_context().get_stage()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    if stage.GetPrimAtPath(target_path).IsValid():
        result, _ = omni.kit.commands.execute("DeletePrims", paths=[target_path])
        if not result:
            raise RuntimeError(f"Failed to delete existing prim before import: {target_path}")

    before_paths = _world_child_paths(stage)
    result, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return _move_imported_prim(imported_path, target_path, before_paths)


def main() -> None:
    args = _parse_args()
    sim_app = SimulationApp({"headless": args.headless})

    from pxr import Gf, UsdGeom

    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.extensions import enable_extension
    except ModuleNotFoundError:
        from omni.isaac.core import World
        from omni.isaac.core.utils.extensions import enable_extension

    enabled_extension = _enable_first_available_extension(enable_extension)
    print(f"Enabled URDF importer extension: {enabled_extension}")

    project_root = Path(__file__).resolve().parent
    franka_hand_urdf = project_root / "urdf_models" / "mmp_panda" / "mmp_panda.urdf"
    robotiq_urdf = project_root / "urdf_models" / "mmp_panda" / "mmp_panda_robotiq85.urdf"

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = world.stage

    if args.gripper in {"franka", "both"}:
        franka_hand_path = _import_urdf(franka_hand_urdf, "/World/PandaFrankaHand")
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(franka_hand_path)).SetTranslate(
            Gf.Vec3d(-0.6, 0.0, 0.0)
        )
        print(f"Imported Panda with Franka hand: {franka_hand_path}")

    if args.gripper in {"robotiq", "both"}:
        robotiq_path = _import_urdf(robotiq_urdf, "/World/PandaRobotiq2F85")
        UsdGeom.XformCommonAPI(stage.GetPrimAtPath(robotiq_path)).SetTranslate(
            Gf.Vec3d(0.6, 0.0, 0.0)
        )
        print(f"Imported Panda with Robotiq 2F-85: {robotiq_path}")

    world.reset()
    print("Scene ready. Close Isaac Sim window to exit.")
    while sim_app.is_running():
        world.step(render=True)

    sim_app.close()


if __name__ == "__main__":
    main()
