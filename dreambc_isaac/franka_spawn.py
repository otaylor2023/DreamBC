"""Spawn Franka Emika Panda (Isaac Sim asset) or custom URDF."""

from __future__ import annotations

from pathlib import Path

ISAAC_FRANKA_USD_REL = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def robot_source(cfg) -> str:
    """``isaac_franka`` = NVIDIA Franka Emika Panda USD (matches Lula / RMPFlow). ``urdf`` = local import."""
    return str(getattr(cfg.robot, "source", "isaac_franka")).lower()


def articulation_prim_path(cfg, robot_root_path: str | None = None) -> str:
    """USD prim path for the articulation root."""
    if robot_source(cfg) == "isaac_franka":
        return str(cfg.robot.prim_path)
    root = robot_root_path or str(cfg.robot.target_path)
    child = str(getattr(cfg.robot, "articulation_child", "Robot"))
    if child:
        return f"{root}/{child}"
    return root


def spawn_franka(world, cfg, project_root: Path, enable_extension) -> str:
    """Add Franka to the stage; returns articulation prim path."""
    source = robot_source(cfg)
    prim_path = str(getattr(cfg.robot, "prim_path", getattr(cfg.robot, "target_path", "/World/Panda")))

    if source == "isaac_franka":
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError(
                "Isaac Sim assets not found. Run inside Isaac (isaacsim_shell.sh) so "
                "Franka Emika Panda can load from Nucleus."
            )
        usd_path = f"{assets_root}{ISAAC_FRANKA_USD_REL}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        print(f"Spawned Franka Emika Panda from Isaac asset: {usd_path} -> {prim_path}")
        return prim_path

    if source != "urdf":
        raise ValueError(f"Unknown robot.source '{source}' (use isaac_franka or urdf)")

    from dreambc_isaac.io import resolve_project_path
    from dreambc_isaac.urdf import import_urdf

    gripper = str(cfg.robot.gripper)
    urdf_path = resolve_project_path(project_root, cfg.robot.urdfs[gripper])
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    target = str(getattr(cfg.robot, "target_path", prim_path))
    imported = import_urdf(urdf_path, target)
    return articulation_prim_path(cfg, imported)
