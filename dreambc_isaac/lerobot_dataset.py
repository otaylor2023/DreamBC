"""LeRobot dataset schema and frame helpers for SmolVLA Franka pick-cube."""

from __future__ import annotations

import numpy as np


SMOLVLA_TASK_DEFAULT = "pick up the cube"

# LeRobot v3 feature spec aligned with models/smolvla_base/config.json
SMOLVLA_PICK_CUBE_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": None,
    },
    "observation.images.camera1": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera3": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
}

SMOLVLA_CAMERA_MODEL_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


def command_to_action6(
    arm_joint_positions: np.ndarray,
    gripper_scalar: float,
    *,
    state_arm_indices: list[int] | None = None,
    action_arm_indices: list[int] | None = None,
    gripper_action_index: int = 5,
) -> np.ndarray:
    """Build 6-D action label matching smolvla_policy mapping."""
    state_arm_indices = state_arm_indices if state_arm_indices is not None else [0, 1, 2, 3, 4, 5]
    action_arm_indices = action_arm_indices if action_arm_indices is not None else [0, 1, 2, 3, 4, 5]
    arm = np.asarray(arm_joint_positions, dtype=np.float32).reshape(-1)
    action = np.zeros(6, dtype=np.float32)
    for out_i, arm_i in enumerate(action_arm_indices):
        if arm_i < arm.shape[0]:
            action[out_i] = arm[arm_i]
    grip_idx = int(gripper_action_index)
    if grip_idx < 6:
        action[grip_idx] = float(gripper_scalar)
    return action


def observation_state6(arm_joint_positions: np.ndarray, state_arm_indices: list[int] | None = None) -> np.ndarray:
    state_arm_indices = state_arm_indices if state_arm_indices is not None else [0, 1, 2, 3, 4, 5]
    arm = np.asarray(arm_joint_positions, dtype=np.float32).reshape(-1)
    return arm[state_arm_indices].astype(np.float32)


def _resize_rgb_hwc_uint8(image_hwc: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(np.asarray(image_hwc, dtype=np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def lerobot_frame_from_step(
    obs: dict[str, np.ndarray],
    action6: np.ndarray,
    *,
    task: str,
    policy_camera_keys: list[str],
    image_size: int = 256,
) -> dict:
    """Build a dict for LeRobotDataset.add_frame (HWC uint8 images for video features)."""
    frame: dict = {
        "observation.state": observation_state6(obs["joint_position"]),
        "action": np.asarray(action6, dtype=np.float32),
        "task": task,
    }
    for model_key, sim_key in zip(SMOLVLA_CAMERA_MODEL_KEYS, policy_camera_keys, strict=True):
        frame[model_key] = _resize_rgb_hwc_uint8(obs[sim_key], image_size)
    return frame


def should_save_preview_mp4(episode_index: int, dataset_cfg) -> bool:
    if not bool(getattr(dataset_cfg, "save_preview_mp4", False)):
        return False
    if bool(getattr(dataset_cfg, "preview_mp4_all_episodes", False)):
        return True
    every_n = int(getattr(dataset_cfg, "preview_mp4_every_n_episodes", 5))
    max_count = int(getattr(dataset_cfg, "preview_mp4_max_count", 20))
    if episode_index >= max_count * max(every_n, 1):
        return False
    if every_n <= 0:
        return episode_index < max_count
    return (episode_index % every_n) == 0 and (episode_index // every_n) < max_count


def preview_mp4_episode_dir(
    dataset_root,
    episode_index: int,
    *,
    success: bool,
    status: str = "",
) -> "Path":
    from pathlib import Path

    root = Path(dataset_root) / "preview_mp4"
    if success:
        return root / f"episode_{episode_index:04d}"
    suffix = status if status else "failed"
    return root / f"episode_{episode_index:04d}_{suffix}"
