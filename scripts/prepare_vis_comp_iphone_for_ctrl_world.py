#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_VIEW_KEYS = ("agent_view", "exterior_2", "wrist")


def _natural_demo_id(path: Path) -> int:
    match = re.fullmatch(r"demos_(\d+)\.hdf5", path.name)
    if not match:
        raise ValueError(f"Unexpected demo filename: {path}")
    return int(match.group(1))


def _load_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open() as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


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


def _write_video(frames: np.ndarray, output_path: Path, width: int, height: int, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    for frame in frames:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    writer.release()


def _video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps


def _read_video_frames(
    path: Path,
    n_frames: int,
    start_frame: int,
    stride: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = start_frame + np.arange(n_frames, dtype=np.int64) * stride
    if len(indices) and indices[-1] >= total_frames:
        raise ValueError(
            f"{path} is too short for requested cam2 sampling: "
            f"last index {int(indices[-1])}, total frames {total_frames}"
        )

    frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read {path} at frame {int(frame_idx)}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def _source_video_path(source_dir: Path, demo_id: int, cam_id: int) -> str | None:
    candidates = [
        source_dir / f"video_{demo_id}_cam{cam_id}.mp4",
        source_dir / f"video_{demo_id}_cam{cam_id}.MP4",
        source_dir / f"video_{demo_id}_cam{cam_id}.mov",
        source_dir / f"video_{demo_id}_cam{cam_id}.MOV",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _required_source_video_path(source_dir: Path, demo_id: int, cam_id: int) -> Path:
    source_path = _source_video_path(source_dir, demo_id, cam_id)
    if source_path is None:
        raise FileNotFoundError(f"Missing video_{demo_id}_cam{cam_id}.mp4/.MOV in {source_dir}")
    return Path(source_path)


def _read_demo(path: Path, view_keys: tuple[str, str, str], gripper_mode: str, gripper_scale: float) -> tuple[str, dict]:
    with h5py.File(path, "r") as h5:
        demo_names = list(h5["data"].keys())
        if len(demo_names) != 1:
            raise ValueError(f"Expected one demo in {path}, found {demo_names}")
        demo_name = demo_names[0]
        demo = h5["data"][demo_name]
        obs = demo["obs"]

        missing = [key for key in view_keys if key not in obs]
        if missing:
            raise KeyError(f"{path} is missing HDF5 observation views: {missing}")

        ee_pos = obs["EE_POS"][:]
        ee_euler = obs["EE_EULER"][:]
        gripper = _convert_gripper(obs["GRIPPER"][:], gripper_mode, gripper_scale)
        joint_pos = obs["JOINT_POS"][:]
        frames = [obs[key][:] for key in view_keys]

        n_frames = min(len(ee_pos), len(ee_euler), len(gripper), len(joint_pos), *(len(view) for view in frames))
        ee_pos = ee_pos[:n_frames]
        ee_euler = ee_euler[:n_frames]
        gripper = gripper[:n_frames]
        joint_pos = joint_pos[:n_frames]
        frames = [view[:n_frames] for view in frames]

        states = np.concatenate([ee_pos, ee_euler, gripper], axis=-1)
        joints = np.concatenate([joint_pos, gripper], axis=-1)

        arrays = {
            "frames": frames,
            "states": states,
            "joints": joints,
            "ee_pos": ee_pos,
            "joint_pos": joint_pos,
            "gripper": gripper,
            "action_cartesian_position": demo["actions/cartesian_position"][:n_frames],
            "action_joint_position": demo["actions/joint_position"][:n_frames],
            "action_joint_velocity": demo["actions/joint_velocity"][:n_frames],
            "n_frames": n_frames,
        }
    return demo_name, arrays


def _write_stat(meta_dir: Path, annotations: list[dict]) -> None:
    actions = []
    for ann in annotations:
        cartesian = np.array(ann["observation.state.cartesian_position"], dtype=np.float32)
        gripper = np.array(ann["observation.state.gripper_position"], dtype=np.float32)
        actions.append(np.concatenate([cartesian, gripper], axis=-1))
    action_all = np.concatenate(actions, axis=0)
    stat = {
        "state_01": np.percentile(action_all, 1, axis=0).tolist(),
        "state_99": np.percentile(action_all, 99, axis=0).tolist(),
    }
    with (meta_dir / "stat.json").open("w") as f:
        json.dump(stat, f, indent=2)


def convert(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    view_keys = tuple(args.view_keys.split(","))
    if len(view_keys) != 3:
        raise ValueError("--view_keys must contain exactly three comma-separated HDF5 observation keys")

    prompt_map = _load_prompt_map(Path(args.prompts_json) if args.prompts_json else None)
    demo_paths = sorted(source_dir.glob("demos_*.hdf5"), key=_natural_demo_id)
    if args.limit:
        demo_paths = demo_paths[: args.limit]
    if not demo_paths:
        raise RuntimeError(f"No demos_*.hdf5 found in {source_dir}")

    annotation_dir = output_dir / "annotation" / "val"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    val_ids = []
    annotations_for_stat = []
    report = []
    for demo_path in demo_paths:
        demo_id = _natural_demo_id(demo_path)
        episode_id = f"{demo_id:04d}"
        _, data = _read_demo(demo_path, view_keys, args.gripper_mode, args.gripper_scale)
        source_videos = [_source_video_path(source_dir, demo_id, cam_id) for cam_id in (1, 2, 3)]

        cam2_sampling = None
        if args.cam2_source == "video":
            cam2_path = _required_source_video_path(source_dir, demo_id, 2)
            cam2_stride = args.cam2_stride
            if cam2_stride is None:
                cam2_stride = max(1, round(_video_fps(cam2_path) / args.source_fps))
            data["frames"][1] = _read_video_frames(
                cam2_path,
                int(data["n_frames"]),
                args.cam2_start_frame,
                cam2_stride,
            )
            cam2_sampling = {
                "path": str(cam2_path),
                "start_frame": args.cam2_start_frame,
                "stride": cam2_stride,
                "source_fps": _video_fps(cam2_path),
                "target_state_fps": args.source_fps,
            }

        for view_id, frames in enumerate(data["frames"]):
            _write_video(
                frames,
                output_dir / "videos" / "val" / episode_id / f"{view_id}.mp4",
                args.width,
                args.height,
                args.fps,
            )

        instruction = prompt_map.get(episode_id) or prompt_map.get(str(demo_id)) or args.default_instruction
        n_frames = int(data["n_frames"])
        annotation = {
            "texts": [instruction],
            "episode_id": episode_id,
            "success": 0,
            "video_length": n_frames,
            "state_length": n_frames,
            "raw_length": n_frames,
            "videos": [{"video_path": f"videos/val/{episode_id}/{view_id}.mp4"} for view_id in range(3)],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/val/{episode_id}/{view_id}.pt"} for view_id in range(3)
            ],
            "states": data["states"].tolist(),
            "joints": data["joints"].tolist(),
            "observation.state.cartesian_position": data["ee_pos"].tolist(),
            "observation.state.joint_position": data["joint_pos"].tolist(),
            "observation.state.gripper_position": data["gripper"].tolist(),
            "action.cartesian_position": data["action_cartesian_position"].tolist(),
            "action.joint_position": data["action_joint_position"].tolist(),
            "action.gripper_position": data["gripper"].tolist(),
            "action.joint_velocity": data["action_joint_velocity"].tolist(),
            "metadata": {
                "source_hdf5": str(demo_path),
                "source_view_keys": list(view_keys),
                "source_videos": source_videos,
                "source_layout": "cam1/cam3 from HDF5-synchronized views; cam2 from iPhone video"
                if args.cam2_source == "video"
                else "three HDF5-synchronized views",
                "cam2_source": args.cam2_source,
                "cam2_sampling": cam2_sampling,
                "sample_every": 1,
            },
        }
        with (annotation_dir / f"{episode_id}.json").open("w") as f:
            json.dump(annotation, f, indent=2)

        val_ids.append(episode_id)
        annotations_for_stat.append(annotation)
        report.append(
            {
                "episode_id": episode_id,
                "written_frames": n_frames,
                "source_hdf5": str(demo_path),
                "source_view_keys": list(view_keys),
                "source_videos": source_videos,
                "cam2_source": args.cam2_source,
                "cam2_sampling": cam2_sampling,
            }
        )

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "val_ids.json").open("w") as f:
        json.dump(val_ids, f, indent=2)
    with (meta_dir / "conversion_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    _write_stat(meta_dir, annotations_for_stat)

    print(f"Wrote {len(val_ids)} Ctrl-World iPhone episodes to {output_dir}")
    for item in report:
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="vis_comp_iphone")
    parser.add_argument("--output_dir", default="Ctrl-World/dataset_example/vis_comp_iphone")
    parser.add_argument("--view_keys", default=",".join(DEFAULT_VIEW_KEYS))
    parser.add_argument("--prompts_json", default=None, help="Optional JSON mapping demo id to instruction.")
    parser.add_argument("--default_instruction", default="")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--gripper_mode", choices=["raw", "invert", "scale", "invert_scale"], default="invert")
    parser.add_argument("--gripper_scale", type=float, default=0.75)
    parser.add_argument("--cam2_source", choices=["video", "hdf5"], default="video")
    parser.add_argument("--cam2_start_frame", type=int, default=0)
    parser.add_argument("--cam2_stride", type=int, default=None)
    parser.add_argument(
        "--source_fps",
        type=float,
        default=15.0,
        help="FPS of the synchronized robot trajectory; used to sample the iPhone cam2 video.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
