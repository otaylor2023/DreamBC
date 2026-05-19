"""SmolVLA policy adapter for the Isaac minimal rollout harness."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image


def _resize_rgb_to_chw_float(image_hwc: np.ndarray, size: int) -> torch.Tensor:
    """uint8 HWC -> float32 (1, 3, size, size) in [0, 1]."""
    pil = Image.fromarray(np.asarray(image_hwc, dtype=np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def isaac_obs_to_lerobot_frame(obs: dict[str, np.ndarray], task: str, smolvla_cfg) -> dict:
    image_size = int(smolvla_cfg.image_size)
    camera_keys = list(smolvla_cfg.camera_keys)
    if len(camera_keys) != 3:
        raise ValueError(f"smolvla.camera_keys must list exactly 3 cameras, got {len(camera_keys)}")

    state_indices = [int(i) for i in smolvla_cfg.state_arm_indices]
    joint_position = np.asarray(obs["joint_position"], dtype=np.float32).reshape(-1)
    if max(state_indices, default=-1) >= joint_position.shape[0]:
        raise IndexError(
            f"state_arm_indices {state_indices} out of range for joint_position length {joint_position.shape[0]}"
        )
    state = torch.from_numpy(joint_position[state_indices].astype(np.float32)).unsqueeze(0)

    frame = {
        "observation.state": state,
        "task": str(task),
    }
    for model_key, sim_key in zip(
        ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"],
        camera_keys,
        strict=True,
    ):
        if sim_key not in obs:
            raise KeyError(f"SmolVLA camera '{sim_key}' missing from observation (have {sorted(obs)})")
        frame[model_key] = _resize_rgb_to_chw_float(obs[sim_key], image_size)
    return frame


def map_smolvla_action_to_panda(
    action6: np.ndarray,
    current_arm_qpos: np.ndarray,
    gripper_joint_count: int,
    smolvla_cfg,
) -> tuple[np.ndarray, np.ndarray]:
    action6 = np.asarray(action6, dtype=np.float32).reshape(-1)
    if action6.shape[0] < 6:
        raise ValueError(f"SmolVLA action must have at least 6 elements, got shape {action6.shape}")

    arm_indices = [int(i) for i in smolvla_cfg.action_arm_indices]
    arm_action = current_arm_qpos.astype(np.float32).copy()
    for out_i, arm_i in enumerate(arm_indices):
        if arm_i < arm_action.shape[0]:
            arm_action[arm_i] = float(action6[out_i])

    if gripper_joint_count > 0:
        grip_idx = int(smolvla_cfg.gripper_action_index)
        grip_raw = float(action6[grip_idx])
        grip_min = float(smolvla_cfg.gripper_min)
        grip_max = float(smolvla_cfg.gripper_max)
        grip_target = float(np.clip(grip_raw, min(grip_min, grip_max), max(grip_min, grip_max)))
        gripper_action = np.full((gripper_joint_count,), grip_target, dtype=np.float32)
    else:
        gripper_action = np.zeros((0,), dtype=np.float32)
    return arm_action, gripper_action


@dataclass
class SmolVLARunner:
    policy: object
    preprocess: object
    postprocess: object
    device: torch.device
    task: str
    smolvla_cfg: object
    action_queue: list[np.ndarray] = field(default_factory=list)

    def next_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if not self.action_queue:
            frame = isaac_obs_to_lerobot_frame(obs, self.task, self.smolvla_cfg)
            batch = self.preprocess(frame)
            with torch.inference_mode():
                pred = self.policy.select_action(batch)
                pred = self.postprocess(pred)
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred, dtype=np.float32)
            if pred.ndim == 1:
                self.action_queue.append(pred)
            else:
                for row in pred:
                    self.action_queue.append(np.asarray(row, dtype=np.float32).reshape(-1))
        return self.action_queue.pop(0)


def load_smolvla_runner(smolvla_cfg) -> SmolVLARunner:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model_path = str(smolvla_cfg.model_path)
    device_name = str(smolvla_cfg.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: smolvla.device=cuda but CUDA is unavailable; using cpu.")
        device_name = "cpu"
    device = torch.device(device_name)

    policy = SmolVLAPolicy.from_pretrained(model_path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return SmolVLARunner(
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        device=device,
        task=str(smolvla_cfg.task),
        smolvla_cfg=smolvla_cfg,
    )


def smolvla_policy_action(
    runner: SmolVLARunner,
    obs: dict[str, np.ndarray],
    current_arm_qpos: np.ndarray,
    gripper_joint_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    action6 = runner.next_action(obs)
    return map_smolvla_action_to_panda(action6, current_arm_qpos, gripper_joint_count, runner.smolvla_cfg)
