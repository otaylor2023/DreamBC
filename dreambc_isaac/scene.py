"""Scene construction helpers."""

from __future__ import annotations

import numpy as np


def add_basic_lighting(stage) -> None:
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(800.0)


def add_simple_scene(world, scene_cfg) -> None:
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

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
    world.scene.add(
        DynamicCuboid(
            prim_path=scene_cfg.test_cube.prim_path,
            name=scene_cfg.test_cube.name,
            position=np.asarray(scene_cfg.test_cube.position, dtype=np.float64),
            scale=np.asarray(scene_cfg.test_cube.scale, dtype=np.float64),
            color=np.asarray(scene_cfg.test_cube.color, dtype=np.float64),
        )
    )

