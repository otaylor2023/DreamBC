"""
One-frame Isaac Sim preview: table + test cube + exterior camera, saved as PNG.

No robot, URDF, or policy. Run from the DreamBC Isaac shell after EULA acceptance.

  python isaacsim_scene_preview.py sim.headless=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from isaacsim import SimulationApp
from omegaconf import DictConfig, OmegaConf

from dreambc_isaac.app import simulation_app_config
from dreambc_isaac.cameras import initialize_cameras, make_cameras
from dreambc_isaac.io import resolve_project_path, save_png
from dreambc_isaac.scene import add_simple_scene


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "isaac_scene_preview.png"

    sim_app = SimulationApp(simulation_app_config(bool(cfg.sim.headless)))
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=float(cfg.sim.stage_units_in_meters))
    add_simple_scene(world, cfg.scene, project_root)
    cam_cfg = OmegaConf.create({"exterior_image_1_left": cfg.cameras.exterior_image_1_left})
    cameras = make_cameras(cam_cfg)

    world.reset()
    initialize_cameras(cameras, cam_cfg)

    warmup = max(5, int(cfg.sim.warmup_steps))
    for _ in range(warmup):
        world.step(render=True)

    cam = cameras["exterior_image_1_left"]
    rgba = np.asarray(cam.get_rgba())
    if rgba.ndim != 3 or rgba.shape[-1] < 3:
        raise RuntimeError(f"Unexpected camera rgba shape: {getattr(rgba, 'shape', None)}")
    rgb = rgba[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    if not save_png(png_path, rgb):
        np.savez_compressed(out_dir / "isaac_scene_preview.npz", rgb=rgb)
        print("PIL not available; saved outputs/isaac_scene_preview.npz instead.")
    else:
        print(f"Wrote {png_path}")

    sim_app.close()


if __name__ == "__main__":
    main()
