#!/usr/bin/env python3
"""Export frame/action training examples from hand-collected BC bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from bc_dataset import DreamBCEpisodeDataset  # noqa: E402

OUT = REPO / "real_robot_checkpoints/docs/handcollected_training_examples"


def save_pair(task: str, ep_dir: Path, t: int, sample: dict) -> dict:
    ext = sample["observation/exterior_image_1_left"]
    wrist = sample["observation/wrist_image_left"]
    actions = sample["actions"]
    joint = sample["observation/joint_position"]
    gripper = sample["observation/gripper_position"]

    canvas = Image.new("RGB", (448, 224), (20, 20, 20))
    canvas.paste(Image.fromarray(ext), (0, 0))
    canvas.paste(Image.fromarray(wrist), (224, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{task} {ep_dir.name} t={t}", fill=(255, 255, 0))
    draw.text((8, 24), "exterior_2 | wrist", fill=(200, 200, 200))
    out_png = OUT / f"{task}_{ep_dir.name}_t{t:03d}.png"
    canvas.save(out_png)

    return {
        "task": task,
        "episode": ep_dir.name,
        "timestep": t,
        "prompt": sample["prompt"],
        "joint_position": joint.tolist(),
        "gripper_position": gripper.tolist(),
        "action_step0": actions[0].tolist(),
        "action_step1": actions[1].tolist(),
        "action_step14": actions[14].tolist(),
        "action_step0_joint_vel": actions[0, :7].tolist(),
        "action_step0_gripper_pos": actions[0, 7:8].tolist(),
        "action_l2_norm_step0": float(np.linalg.norm(actions[0])),
        "action_l2_norm_mean": float(np.linalg.norm(actions, axis=1).mean()),
        "image_png": str(out_png.relative_to(REPO / "real_robot_checkpoints")),
    }


def bundle_stats(root: str, task: str) -> dict:
    ds = DreamBCEpisodeDataset([root], filter_mode="success_true", verbose=False)
    all_actions, all_gripper, all_joint = [], [], []
    for ep in ds.episode_dirs:
        with np.load(ep / "episode.npz") as z:
            all_actions.append(z["policy_action_chunk"].reshape(-1, 8))
            all_gripper.append(z["policy_state_gripper"].reshape(-1))
            all_joint.append(z["policy_state_joint"].reshape(-1, 7))
    actions = np.concatenate(all_actions, axis=0)
    gripper = np.concatenate(all_gripper, axis=0)
    joints = np.concatenate(all_joint, axis=0)
    return {
        "task": task,
        "episodes": ds.num_episodes,
        "samples": len(ds),
        "joint_pos_min": joints.min(axis=0).tolist(),
        "joint_pos_max": joints.max(axis=0).tolist(),
        "gripper_min": float(gripper.min()),
        "gripper_max": float(gripper.max()),
        "joint_vel_min": actions[:, :7].min(axis=0).tolist(),
        "joint_vel_max": actions[:, :7].max(axis=0).tolist(),
        "gripper_action_min": float(actions[:, 7].min()),
        "gripper_action_max": float(actions[:, 7].max()),
        "action_l2_mean": float(np.linalg.norm(actions, axis=1).mean()),
        "action_l2_std": float(np.linalg.norm(actions, axis=1).std()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    for task, root in [
        ("cube", "rollouts/handcollected_cube"),
        ("tomato", "rollouts/handcollected_tomato"),
    ]:
        ds = DreamBCEpisodeDataset([root], filter_mode="success_true", verbose=False)
        ep = Path(root) / "episodes/demo_01"
        meta = json.loads((ep / "episode.json").read_text())
        t_vals = [0, meta["num_policy_decisions"] // 2, meta["num_policy_decisions"] - 1]
        for t in t_vals:
            idx = next(i for i, (ed, tt, _) in enumerate(ds._index) if ed == ep and tt == t)
            examples.append(save_pair(task, ep, t, ds[idx]))

    report = {
        "examples": examples,
        "stats": [
            bundle_stats("rollouts/handcollected_cube", "cube"),
            bundle_stats("rollouts/handcollected_tomato", "tomato"),
        ],
    }
    out_json = OUT / "sample_report.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
