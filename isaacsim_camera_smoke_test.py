"""
Minimal Isaac Sim camera smoke test.

Usage:
  python isaacsim_camera_smoke_test.py
  python isaacsim_camera_smoke_test.py sim.headless=true sim.steps=120
"""

from __future__ import annotations

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.cameras import camera_orientation_from_euler_xyz
from dreambc_isaac.patches import apply_camera_pipeline_patches


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))
    from isaacsim.core.api import World
    from isaacsim.core.utils.prims import define_prim
    from isaacsim.sensors.camera import Camera

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    define_prim("/World/Cameras", "Xform")

    cam_cfg = cfg.cameras.exterior_image_1_left
    camera = Camera(
        prim_path=str(cam_cfg.prim_path),
        name="smoke_camera",
        position=np.asarray(cam_cfg.position, dtype=np.float64),
        orientation=camera_orientation_from_euler_xyz(tuple(cam_cfg.orientation_euler_xyz_degrees)),
        frequency=int(cam_cfg.frequency),
        resolution=tuple(cam_cfg.resolution),
        annotator_device="cpu",
    )

    world.reset()
    apply_camera_pipeline_patches()
    camera.initialize()

    good_frames = 0
    bad_frames = 0
    first_good_step = None
    first_frame_error = None
    total_steps = int(getattr(cfg.sim, "steps", 120))
    for step in range(total_steps):
        world.step(render=True)
        try:
            rgba = camera.get_rgba()
        except Exception as exc:
            rgba = None
            if first_frame_error is None:
                first_frame_error = f"{type(exc).__name__}: {exc}"
        arr = None if rgba is None else np.asarray(rgba)
        valid = arr is not None and arr.ndim == 3 and arr.shape[-1] >= 3 and arr.shape[0] > 0 and arr.shape[1] > 0
        if valid:
            good_frames += 1
            if first_good_step is None:
                first_good_step = step
        else:
            bad_frames += 1

    print(f"Camera smoke test finished. good_frames={good_frames}, bad_frames={bad_frames}, total_steps={total_steps}")
    if first_good_step is not None:
        print(f"First valid frame at step: {first_good_step}")
    else:
        print("No valid frames were produced by the camera pipeline.")
        if first_frame_error is not None:
            print(f"First frame read error: {first_frame_error}")

    sim_app.close()


if __name__ == "__main__":
    main()
