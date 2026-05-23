#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


def _natural_demo_id(path: Path) -> int:
    match = re.search(r"demos_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Unexpected demo filename: {path}")
    return int(match.group(1))


def _load_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open() as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def _adjust_image(frames: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if alpha == 1.0 and beta == 0.0:
        return frames
    adjusted = frames.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _write_video(
    frames: np.ndarray,
    output_path: Path,
    fps: float,
    size: tuple[int, int],
    brightness_alpha: float,
    brightness_beta: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    frames = _adjust_image(frames, brightness_alpha, brightness_beta)
    for frame in frames:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_hdf5_demo(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as h5:
        demo_names = list(h5["data"].keys())
        if len(demo_names) != 1:
            raise ValueError(f"Expected one demo in {path}, found {demo_names}")
        demo_name = demo_names[0]
        demo = h5["data"][demo_name]
        arrays = {
            "ee_pos": demo["obs/EE_POS"][:],
            "ee_euler": demo["obs/EE_EULER"][:],
            "gripper": demo["obs/GRIPPER"][:],
            "joint_pos": demo["obs/JOINT_POS"][:],
            "agent_view": demo["obs/agent_view"][:],
            "wrist": demo["obs/wrist"][:],
            "action_cartesian_position": demo["actions/cartesian_position"][:],
            "action_joint_position": demo["actions/joint_position"][:],
            "action_gripper_position": demo["actions/gripper_position"][:],
            "action_joint_velocity": demo["actions/joint_velocity"][:],
        }
    return demo_name, arrays


def _convert_gripper(gripper: np.ndarray, mode: str, scale: float) -> np.ndarray:
    if mode == "raw":
        converted = gripper
    elif mode == "invert":
        converted = 1.0 - gripper
    elif mode == "invert_scale":
        converted = (1.0 - gripper) * scale
    elif mode == "scale":
        converted = gripper * scale
    else:
        raise ValueError(f"Unknown gripper mode: {mode}")
    return np.clip(converted, 0.0, scale).astype(np.float32)


def convert_vis_comp(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    prompt_map = _load_prompt_map(Path(args.prompts_json) if args.prompts_json else None)
    demo_paths = sorted(source_dir.glob("demos_*.hdf5"), key=_natural_demo_id)
    if args.limit:
        demo_paths = demo_paths[: args.limit]
    if not demo_paths:
        raise RuntimeError(f"No demos_*.hdf5 found in {source_dir}")

    val_ids = []
    for demo_path in demo_paths:
        demo_num = _natural_demo_id(demo_path)
        episode_id = f"{demo_num:04d}"
        try:
            _, data = _read_hdf5_demo(demo_path)
        except OSError as exc:
            if not args.skip_bad:
                raise
            print(f"Skipping unreadable HDF5 {demo_path}: {exc}")
            continue

        gripper = _convert_gripper(data["gripper"], args.gripper_mode, args.gripper_scale)
        states = np.concatenate([data["ee_pos"], data["ee_euler"], gripper], axis=-1)
        joints = np.concatenate([data["joint_pos"], gripper], axis=-1)
        n_frames = min(len(states), len(data["agent_view"]), len(data["wrist"]))
        states = states[:n_frames]
        joints = joints[:n_frames]

        if args.third_view == "agent":
            middle_view = data["agent_view"][:n_frames]
        elif args.third_view == "wrist":
            middle_view = data["wrist"][:n_frames]
        else:
            middle_view = np.zeros_like(data["agent_view"][:n_frames])
        videos = [data["agent_view"][:n_frames], middle_view, data["wrist"][:n_frames]]

        for view_id, frames in enumerate(videos):
            _write_video(
                frames,
                output_dir / "videos" / "val" / episode_id / f"{view_id}.mp4",
                fps=args.fps,
                size=(args.width, args.height),
                brightness_alpha=args.brightness_alpha,
                brightness_beta=args.brightness_beta,
            )

        instruction = prompt_map.get(episode_id) or prompt_map.get(str(demo_num)) or args.default_instruction
        annotation = {
            "texts": [instruction],
            "episode_id": episode_id,
            "success": 0,
            "video_length": int(n_frames),
            "state_length": int(len(states)),
            "raw_length": int(len(states)),
            "videos": [{"video_path": f"videos/val/{episode_id}/{i}.mp4"} for i in range(3)],
            "latent_videos": [{"latent_video_path": f"latent_videos/val/{episode_id}/{i}.pt"} for i in range(3)],
            "states": states.tolist(),
            "joints": joints.tolist(),
            "observation.state.cartesian_position": data["ee_pos"][:n_frames].tolist(),
            "observation.state.joint_position": data["joint_pos"][:n_frames].tolist(),
            "observation.state.gripper_position": gripper[:n_frames].tolist(),
            "action.cartesian_position": data["action_cartesian_position"][:n_frames].tolist(),
            "action.joint_position": data["action_joint_position"][:n_frames].tolist(),
            "action.gripper_position": gripper[:n_frames].tolist(),
            "action.joint_velocity": data["action_joint_velocity"][:n_frames].tolist(),
        }
        annotation_dir = output_dir / "annotation" / "val"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        with (annotation_dir / f"{episode_id}.json").open("w") as f:
            json.dump(annotation, f, indent=2)
        val_ids.append(episode_id)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "val_ids.json").open("w") as f:
        json.dump(val_ids, f, indent=2)
    print(f"Wrote {len(val_ids)} Ctrl-World replay episodes to {output_dir}")
    print("val_ids:", ",".join(val_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="vis_comp")
    parser.add_argument("--output_dir", default="Ctrl-World/dataset_example/vis_comp")
    parser.add_argument("--prompts_json", default=None, help="Optional JSON mapping demo id to instruction.")
    parser.add_argument("--default_instruction", default="", help="Used when prompts_json has no entry.")
    parser.add_argument("--third_view", choices=["agent", "wrist", "black"], default="agent")
    parser.add_argument(
        "--gripper_mode",
        choices=["raw", "invert", "scale", "invert_scale"],
        default="raw",
        help="Convert vis_comp gripper values to Ctrl-World convention.",
    )
    parser.add_argument("--gripper_scale", type=float, default=0.75)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--brightness_alpha", type=float, default=1.0)
    parser.add_argument("--brightness_beta", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_bad", action="store_true", help="Skip unreadable HDF5 files.")
    args = parser.parse_args()
    convert_vis_comp(args)


if __name__ == "__main__":
    main()
