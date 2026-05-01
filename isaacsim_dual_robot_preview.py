"""
Isaac Sim preview script for Panda variants with different grippers.

Run from an Isaac Sim shell, for example:
  python isaacsim_dual_robot_preview.py --gripper both
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.scene import add_basic_lighting
from dreambc_isaac.urdf import enable_first_available_urdf_extension, import_urdf


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


def main() -> None:
    args = _parse_args()
    sim_app = SimulationApp(simulation_app_config(args.headless))

    from pxr import Gf, UsdGeom

    try:
        from isaacsim.core.api import World
        from isaacsim.core.utils.extensions import enable_extension
    except ModuleNotFoundError:
        from omni.isaac.core import World
        from omni.isaac.core.utils.extensions import enable_extension

    enabled_extension = enable_first_available_urdf_extension(enable_extension)
    print(f"Enabled URDF importer extension: {enabled_extension}")

    project_root = Path(__file__).resolve().parent
    franka_hand_urdf = project_root / "urdf_models" / "mmp_panda" / "mmp_panda.urdf"
    robotiq_urdf = project_root / "urdf_models" / "mmp_panda" / "mmp_panda_robotiq85.urdf"

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_basic_lighting(world.stage)

    if args.gripper in {"franka", "both"}:
        franka_hand_path = import_urdf(franka_hand_urdf, "/World/PandaFrankaHand")
        UsdGeom.XformCommonAPI(world.stage.GetPrimAtPath(franka_hand_path)).SetTranslate(
            Gf.Vec3d(-0.6, 0.0, 0.0)
        )
        print(f"Imported Panda with Franka hand: {franka_hand_path}")

    if args.gripper in {"robotiq", "both"}:
        robotiq_path = import_urdf(robotiq_urdf, "/World/PandaRobotiq2F85")
        UsdGeom.XformCommonAPI(world.stage.GetPrimAtPath(robotiq_path)).SetTranslate(
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
