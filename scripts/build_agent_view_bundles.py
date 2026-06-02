#!/usr/bin/env python3
"""Build DreamBC bundles whose exterior policy image is Control-World agent_view.

The original BC bundles saved three camera streams in ``episode.npz``:

  0. agent_view
  1. exterior_3
  2. wrist

During rollout, policy observations were saved explicitly as
``policy_obs_exterior`` and ``policy_obs_wrist``. Those explicit exterior
observations used camera index 1 (exterior_3). This script derives a sibling
bundle with the same trajectories, labels, actions, and wrist observations, but
replaces ``policy_obs_exterior`` with camera index 0 (agent_view) at the policy
decision cadence.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """Replicate the openpi_client resize_with_pad path used during rollout."""
    cur_height, cur_width = image.shape[:2]
    if (cur_height, cur_width) == (height, width):
        return image.astype(np.uint8)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = Image.fromarray(image.astype(np.uint8)).resize((resized_width, resized_height), Image.BILINEAR)
    out = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    out.paste(resized, (pad_width, pad_height))
    return np.asarray(out, dtype=np.uint8)


def _policy_preprocess_camera_frame(frame: np.ndarray) -> np.ndarray:
    """Match rollout_interact_pi.forward_policy image preprocessing."""
    tensor = torch.from_numpy(frame).to(torch.uint8)
    tensor = torch.nn.functional.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0).float(),
        size=(180, 320),
        mode="bilinear",
        align_corners=False,
    )
    image = tensor.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()
    return _resize_with_pad(image, 224, 224)


def _agent_view_policy_obs(raw: dict[str, np.ndarray]) -> np.ndarray:
    initial_images = raw["initial_images"]
    images = raw["images"]
    policy_obs_exterior = raw["policy_obs_exterior"]

    num_decisions = int(policy_obs_exterior.shape[0])
    if images.shape[0] < 1 or initial_images.shape[0] < 1:
        raise ValueError("episode does not contain camera index 0")
    if images.shape[1] < max(0, (num_decisions - 1) * 4 + 1):
        raise ValueError(
            f"not enough predicted frames: images.shape={images.shape}, decisions={num_decisions}"
        )

    frames = [_policy_preprocess_camera_frame(initial_images[0])]
    for decision_idx in range(1, num_decisions):
        # pred_step=5 and policy_skip_step=2 produced 4 saved frames per policy
        # interaction. The next policy decision aligns with the first frame in
        # the following saved chunk (index 4 * decision_idx).
        frames.append(_policy_preprocess_camera_frame(images[0, 4 * decision_idx]))

    return np.stack(frames, axis=0).astype(np.uint8)


def _write_episode(src_episode_dir: Path, dst_episode_dir: Path, prompt: str | None) -> None:
    dst_episode_dir.mkdir(parents=True, exist_ok=True)

    src_json = src_episode_dir / "episode.json"
    src_npz = src_episode_dir / "episode.npz"
    if not src_json.is_file() or not src_npz.is_file():
        raise FileNotFoundError(f"missing episode.json or episode.npz under {src_episode_dir}")

    meta = json.loads(src_json.read_text())
    if prompt is not None:
        meta["instruction"] = prompt
    meta["policy_obs_exterior_source"] = "agent_view"
    meta["policy_obs_wrist_source"] = "wrist"
    meta["derived_from_bundle"] = str(src_episode_dir)
    meta["notes"] = (meta.get("notes", "") + " Derived agent_view policy_obs_exterior for BC.").strip()

    with np.load(src_npz) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}

    arrays["policy_obs_exterior"] = _agent_view_policy_obs(arrays)

    (dst_episode_dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(dst_episode_dir / "episode.npz", **arrays)


def build_bundle(src_root: Path, dst_root: Path, prompt: str | None, overwrite: bool) -> None:
    src_episodes = sorted((src_root / "Rollouts_interact_pi" / "bc_episodes").glob("*/episode.json"))
    if not src_episodes:
        raise FileNotFoundError(f"no episodes found under {src_root}")

    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {dst_root}")
        shutil.rmtree(dst_root)

    count = 0
    for src_json in src_episodes:
        src_episode_dir = src_json.parent
        rel = src_episode_dir.relative_to(src_root)
        _write_episode(src_episode_dir, dst_root / rel, prompt)
        count += 1

    # Preserve top-level sweep config if present.
    src_sweep = src_root / "sweep_config.json"
    if src_sweep.is_file():
        (dst_root / "sweep_config.json").parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_sweep, dst_root / "sweep_config.json")

    print(f"[agent-view-bundle] wrote {count} episodes -> {dst_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--prompt", default=None, help="Optional replacement instruction for every episode.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_bundle(args.src_root, args.dst_root, args.prompt, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
