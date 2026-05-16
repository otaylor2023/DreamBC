"""
Sweep wrist camera orientation: post-rotation around local X (camera / mount frame).

Uses the same doubled link7 mount as minimal_rollout.yaml. Writes:
  outputs/minimal_rollout_camera_capture/wrist_sweep_rot_x/<name>/
    scene_overview.png, wrist_image_left.png

  mamba run -n dreambc python isaacsim_wrist_orientation_sweep_rot_x.py sim.headless=true
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dreambc_isaac.io import resolve_project_path
from isaacsim_wrist_orientation_sweep import (
    CAPTURE_KEYS,
    _capture_variant,
    _wrist_cfg_from_variant,
)

SWEEP_DIR_NAME = "wrist_sweep_rot_x"

_BASE_Z = {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, 0.0, 1.0]}
_BASE_POS_Y = {"view_forward_xyz": [1.0, 0.0, 0.0], "view_up_xyz": [0.0, 1.0, 0.0]}

# Degrees applied as orientation_post_euler_xyz_degrees [X, 0, 0] after forward/up aim.
X_POST_DEGREES = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]


def _build_rot_x_variants() -> list[tuple[str, dict]]:
    variants: list[tuple[str, dict]] = []
    idx = 1
    for base_label, base in (("up_z", _BASE_Z), ("up_pos_y", _BASE_POS_Y)):
        for x_deg in X_POST_DEGREES:
            name = f"{idx:02d}_x_{int(x_deg):+04d}_{base_label}"
            variant = dict(base)
            if x_deg != 0:
                variant["orientation_post_euler_xyz_degrees"] = [float(x_deg), 0.0, 0.0]
            variants.append((name, variant))
            idx += 1
    return variants


WRIST_VARIANTS = _build_rot_x_variants()


def _run_single_variant(cfg: DictConfig, variant_name: str, variant: dict, sweep_root: Path) -> None:
    wrist_cfg = _wrist_cfg_from_variant(cfg.cameras.wrist_image_left, variant)
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(run_cfg.cameras):
        run_cfg.cameras.wrist_image_left = wrist_cfg
    variant_dir = sweep_root / variant_name
    _capture_variant(run_cfg, variant_dir)
    for key in CAPTURE_KEYS:
        print(f"  {variant_dir / f'{key}.png'}")


@hydra.main(config_path="configs", config_name="minimal_rollout", version_base="1.3")
def main(cfg: DictConfig) -> None:
    project_root = resolve_project_path(Path(__file__).resolve().parent, cfg.project_root)
    sweep_root = project_root / "outputs" / "minimal_rollout_camera_capture" / SWEEP_DIR_NAME
    sweep_root.mkdir(parents=True, exist_ok=True)

    single_name = os.environ.get("WRIST_SWEEP_VARIANT")
    if single_name:
        variant = json.loads(os.environ.get("WRIST_SWEEP_JSON", "{}"))
        print(f"\n=== {single_name} ===")
        _run_single_variant(cfg, single_name, variant, sweep_root)
        return

    mount_tr = list(cfg.scene.link7_wrist_camera_mount.mount_translation)
    script = Path(__file__).resolve()
    hydra_args = [a for a in sys.argv[1:] if not a.startswith("WRIST_SWEEP")]

    for variant_name, variant in WRIST_VARIANTS:
        print(f"\n=== launching {variant_name} ===")
        env = os.environ.copy()
        env["WRIST_SWEEP_VARIANT"] = variant_name
        env["WRIST_SWEEP_JSON"] = json.dumps(variant)
        subprocess.run([sys.executable, str(script), *hydra_args], env=env, check=True)

    index_path = sweep_root / "variants.txt"
    index_path.write_text(
        f"# post-rotation: orientation_post_euler_xyz_degrees [X, 0, 0] in camera parent frame\n"
        f"# mount_translation (2x): {mount_tr}\n"
        f"# mount_rotate_z_deg: {cfg.scene.link7_wrist_camera_mount.mount_rotate_z_deg}\n\n"
        + "\n".join(f"{name}\t{variant}" for name, variant in WRIST_VARIANTS)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDone: {len(WRIST_VARIANTS)} variants under {sweep_root}")


if __name__ == "__main__":
    main()
