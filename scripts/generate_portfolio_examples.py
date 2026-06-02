#!/usr/bin/env python3
"""Export example camera-view PNGs from BC bundle npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE_CONFIG = {
    "cube": {
        "bundle_root": Path("rollouts/C16_g5_block"),
        "exterior_png": "cube_exterior.png",
        "wrist_png": "cube_wrist.png",
    },
    "tomato": {
        "bundle_root": Path("rollouts/TL_g3_red_ball"),
        "exterior_png": "tomato_exterior.png",
        "wrist_png": "tomato_wrist.png",
    },
}


def _find_success_episode_npz(bundle_root: Path) -> Path:
    episodes_root = bundle_root / "Rollouts_interact_pi" / "bc_episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"bc_episodes missing under {bundle_root}")

    for episode_dir in sorted(episodes_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        meta_path = episode_dir / "episode.json"
        npz_path = episode_dir / "episode.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("success") is True:
            return npz_path
    raise RuntimeError(f"no success episode found under {episodes_root}")


def _save_frame(array: np.ndarray, path: Path) -> None:
    frame = np.asarray(array)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {frame.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.astype(np.uint8)).save(path)


def write_camera_mapping_md(output_dir: Path) -> None:
    text = """# Camera mapping

Training bundles store observations in `episode.npz`:

| Bundle npz key | Dataset key | Model key (pi05) | Role |
|---|---|---|---|
| `policy_obs_exterior` | `observation/exterior_image_1_left` | `base_0_rgb` | Fixed third-person / exterior view |
| `policy_obs_wrist` | `observation/wrist_image_left` | `left_wrist_0_rgb` | Wrist-mounted camera |
| (padded zeros) | — | `right_wrist_0_rgb` | Unused; mask=False |

- Resolution: **224×224 RGB**, uint8 in bundles
- At inference: same keys and resolution; **no image augmentation**
"""
    (output_dir / "camera_mapping.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("real_robot_checkpoints/docs/camera_views"),
        help="Directory for PNG examples and camera_mapping.md",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Policy decision index within the success episode",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for task, cfg in BUNDLE_CONFIG.items():
        bundle_root = repo_root / cfg["bundle_root"]
        npz_path = _find_success_episode_npz(bundle_root)
        with np.load(npz_path) as data:
            t = min(args.frame_index, int(data["policy_obs_exterior"].shape[0]) - 1)
            exterior = data["policy_obs_exterior"][t]
            wrist = data["policy_obs_wrist"][t]

        _save_frame(exterior, output_dir / cfg["exterior_png"])
        _save_frame(wrist, output_dir / cfg["wrist_png"])
        print(f"[examples] {task}: frame {t} from {npz_path.relative_to(repo_root)}")

    write_camera_mapping_md(output_dir)
    print(f"[examples] wrote camera views to {output_dir}")


if __name__ == "__main__":
    main()
