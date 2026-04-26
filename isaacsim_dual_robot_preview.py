"""
Isaac Sim preview script for Panda variants with different grippers.

Run from Isaac Sim Python, for example:
  ./python.sh isaacsim_dual_robot_preview.py --gripper both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


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


def _import_urdf(urdf_path: Path, dest_path: str):
    import omni.kit.commands
    from omni.importer.urdf import _urdf

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    result, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=dest_path,
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return imported_path


def main() -> None:
    args = _parse_args()
    sim_app = SimulationApp({"headless": args.headless})

    from pxr import Gf, UsdGeom
    from omni.isaac.core import World
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("omni.importer.urdf")

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
