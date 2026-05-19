"""Scene construction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def add_basic_lighting(stage, dome_intensity: float = 520.0, distant_intensity: float = 680.0) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(float(dome_intensity))

    key = UsdLux.DistantLight.Define(stage, "/World/DistantKey")
    key.CreateIntensityAttr(float(distant_intensity))
    key.CreateAngleAttr(0.53)
    xform = UsdGeom.Xformable(key)
    xform.ClearXformOpOrder()
    rot = xform.AddRotateXYZOp()
    rot.Set(Gf.Vec3f(-55.0, -35.0, 0.0))


def add_simple_scene(world, scene_cfg, project_root: Path | None = None) -> None:
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.utils.extensions import enable_extension

    from dreambc_isaac.meshes import spawn_scene_props

    if "lighting" in scene_cfg:
        lighting = scene_cfg.lighting
        add_basic_lighting(
            world.stage,
            dome_intensity=float(lighting.dome_intensity),
            distant_intensity=float(lighting.distant_intensity),
        )
    else:
        add_basic_lighting(world.stage)
    world.scene.add_default_ground_plane()
    world.scene.add(
        FixedCuboid(
            prim_path=scene_cfg.table.prim_path,
            name=scene_cfg.table.name,
            position=np.asarray(scene_cfg.table.position, dtype=np.float64),
            scale=np.asarray(scene_cfg.table.scale, dtype=np.float64),
            color=np.asarray(scene_cfg.table.color, dtype=np.float64),
        )
    )

    if project_root is not None:
        spawn_scene_props(scene_cfg, project_root, enable_extension=enable_extension)

    test_cube_cfg = getattr(scene_cfg, "test_cube", None)
    if test_cube_cfg is not None and bool(getattr(test_cube_cfg, "enabled", True)):
        world.scene.add(
            DynamicCuboid(
                prim_path=test_cube_cfg.prim_path,
                name=test_cube_cfg.name,
                position=np.asarray(test_cube_cfg.position, dtype=np.float64),
                scale=np.asarray(test_cube_cfg.scale, dtype=np.float64),
                color=np.asarray(test_cube_cfg.color, dtype=np.float64),
            )
        )

