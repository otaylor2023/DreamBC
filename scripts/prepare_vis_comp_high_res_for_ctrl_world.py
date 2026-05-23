#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


def _natural_id(path: Path, prefix: str, suffix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+){re.escape(suffix)}", path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path}")
    return int(match.group(1))


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


def _read_hdf5_annotation(
    path: Path,
    frame_indices: np.ndarray,
    gripper_mode: str,
    gripper_scale: float,
) -> dict:
    with h5py.File(path, "r") as h5:
        demo_names = list(h5["data"].keys())
        if len(demo_names) != 1:
            raise ValueError(f"Expected one demo in {path}, found {demo_names}")
        demo = h5["data"][demo_names[0]]
        ee_pos = demo["obs/EE_POS"][frame_indices]
        ee_euler = demo["obs/EE_EULER"][frame_indices]
        gripper = _convert_gripper(demo["obs/GRIPPER"][frame_indices], gripper_mode, gripper_scale)
        joint_pos = demo["obs/JOINT_POS"][frame_indices]
        states = np.concatenate([ee_pos, ee_euler, gripper], axis=-1)
        joints = np.concatenate([joint_pos, gripper], axis=-1)
        return {
            "states": states.tolist(),
            "joints": joints.tolist(),
            "observation.state.cartesian_position": ee_pos.tolist(),
            "observation.state.joint_position": joint_pos.tolist(),
            "observation.state.gripper_position": gripper.tolist(),
            "action.cartesian_position": demo["actions/cartesian_position"][frame_indices].tolist(),
            "action.joint_position": demo["actions/joint_position"][frame_indices].tolist(),
            "action.gripper_position": gripper.tolist(),
            "action.joint_velocity": demo["actions/joint_velocity"][frame_indices].tolist(),
            "annotation_state_source": "hdf5",
        }


def _placeholder_annotation(n_frames: int) -> dict:
    states = [[0.0] * 7 for _ in range(n_frames)]
    joints = [[0.0] * 8 for _ in range(n_frames)]
    cart = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    joint_pos = [[0.0] * 7 for _ in range(n_frames)]
    gripper = [[0.0] for _ in range(n_frames)]
    return {
        "states": states,
        "joints": joints,
        "observation.state.cartesian_position": cart,
        "observation.state.joint_position": joint_pos,
        "observation.state.gripper_position": gripper,
        "action.cartesian_position": cart,
        "action.joint_position": joint_pos,
        "action.gripper_position": gripper,
        "action.joint_velocity": joint_pos,
        "annotation_state_source": "placeholder_zero_values",
    }


def _write_split_videos(source_path: Path, output_dir: Path, episode_id: str, width: int, height: int, fps: float) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 15.0
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, round(fps_in / fps))
    video_dir = output_dir / "videos" / "val" / episode_id
    video_dir.mkdir(parents=True, exist_ok=True)

    writers = []
    for view_id in range(3):
        writer = cv2.VideoWriter(
            str(video_dir / f"{view_id}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {video_dir / f'{view_id}.mp4'}")
        writers.append(writer)

    written = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            _, frame_width = frame.shape[:2]
            left = frame[:, : frame_width // 2]
            right = frame[:, frame_width // 2 :]
            for writer, crop in zip(writers, (left, left, right)):
                resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(resized)
            written += 1
        frame_idx += 1

    cap.release()
    for writer in writers:
        writer.release()
    return raw_frames, written, sample_every


def convert(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    video_paths = sorted(source_dir.glob("video_*.mp4"), key=lambda p: _natural_id(p, "video", ".mp4"))
    if not video_paths:
        raise RuntimeError(f"No video_*.mp4 found in {source_dir}")

    val_ids = []
    report = []
    for video_path in video_paths:
        demo_id = _natural_id(video_path, "video", ".mp4")
        episode_id = f"{demo_id:04d}"
        raw_frames, video_length, sample_every = _write_split_videos(
            video_path, output_dir, episode_id, args.width, args.height, args.fps
        )

        hdf5_path = source_dir / f"demos_{demo_id}.hdf5"
        hdf5_status = "missing"
        try:
            frame_indices = np.arange(video_length, dtype=np.int64) * sample_every
            ann_arrays = _read_hdf5_annotation(hdf5_path, frame_indices, args.gripper_mode, args.gripper_scale)
            hdf5_status = "ok"
        except Exception as exc:
            if not args.allow_placeholder:
                raise
            ann_arrays = _placeholder_annotation(video_length)
            hdf5_status = f"unreadable: {type(exc).__name__}: {exc}"

        annotation = {
            "texts": [args.default_instruction],
            "episode_id": episode_id,
            "success": 0,
            "video_length": video_length,
            "state_length": video_length,
            "raw_length": raw_frames,
            "videos": [{"video_path": f"videos/val/{episode_id}/{view_id}.mp4"} for view_id in range(3)],
            "latent_videos": [{"latent_video_path": f"latent_videos/val/{episode_id}/{view_id}.pt"} for view_id in range(3)],
            **{key: value for key, value in ann_arrays.items() if key != "annotation_state_source"},
            "metadata": {
                "source_video": str(video_path),
                "source_hdf5": str(hdf5_path),
                "source_hdf5_status": hdf5_status,
                "annotation_state_source": ann_arrays["annotation_state_source"],
                "source_layout": "two horizontal 1280x720 camera views split from 2560x720",
                "middle_view": "duplicate_view_0",
                "sample_every": sample_every,
            },
        }

        annotation_dir = output_dir / "annotation" / "val"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        with (annotation_dir / f"{episode_id}.json").open("w") as f:
            json.dump(annotation, f, indent=2)

        val_ids.append(episode_id)
        report.append(
            {
                "episode_id": episode_id,
                "source_frames": raw_frames,
                "written_frames": video_length,
                "sample_every": sample_every,
                "hdf5_status": hdf5_status,
                "annotation_state_source": ann_arrays["annotation_state_source"],
            }
        )

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "val_ids.json").open("w") as f:
        json.dump(val_ids, f, indent=2)
    with (meta_dir / "conversion_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {len(val_ids)} Ctrl-World episodes to {output_dir}")
    for item in report:
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="vis_comp_high_res")
    parser.add_argument("--output_dir", default="Ctrl-World/dataset_example/vis_comp_high_res")
    parser.add_argument("--default_instruction", default="")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--gripper_mode", choices=["raw", "invert", "scale", "invert_scale"], default="invert")
    parser.add_argument("--gripper_scale", type=float, default=0.75)
    parser.add_argument(
        "--allow_placeholder",
        action="store_true",
        help="Write zero-valued state/action annotations when HDF5 files cannot be read.",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
