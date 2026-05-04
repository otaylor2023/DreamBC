"""Policy helpers used by the minimal rollout harness."""

from __future__ import annotations

import numpy as np


def fake_policy_action(
    step: int,
    current_arm_qpos: np.ndarray,
    gripper_joint_count: int,
    policy_cfg,
) -> tuple[np.ndarray, np.ndarray]:
    arm_action = current_arm_qpos.astype(np.float32).copy()
    if arm_action.shape[0] > 0:
        phase = step * float(policy_cfg.phase_scale)
        offsets = np.asarray(policy_cfg.arm_offsets, dtype=np.float32)[: arm_action.shape[0]]
        arm_action = arm_action + offsets * np.sin(phase)

    if gripper_joint_count > 0:
        period = int(policy_cfg.gripper_period_steps)
        gripper_target = (
            float(policy_cfg.gripper_closed_target)
            if (step // period) % 2 == 0
            else float(policy_cfg.gripper_open_target)
        )
        gripper_action = np.full((gripper_joint_count,), gripper_target, dtype=np.float32)
    else:
        gripper_action = np.zeros((0,), dtype=np.float32)
    return arm_action, gripper_action


def normalize_gripper_position(raw_position: np.ndarray, open_target: float, closed_target: float) -> np.ndarray:
    if raw_position.size == 0:
        return np.zeros((1,), dtype=np.float32)
    raw_scalar = float(np.asarray(raw_position, dtype=np.float32).mean())
    denominator = float(open_target) - float(closed_target)
    if abs(denominator) < 1e-6:
        raise ValueError("Gripper open_target and closed_target must be different.")
    normalized = (float(open_target) - raw_scalar) / denominator
    return np.asarray([np.clip(normalized, 0.0, 1.0)], dtype=np.float32)


def denormalize_gripper_position(
    normalized_position: float,
    gripper_joint_count: int,
    open_target: float,
    closed_target: float,
) -> np.ndarray:
    if gripper_joint_count == 0:
        return np.zeros((0,), dtype=np.float32)
    normalized = float(np.clip(normalized_position, 0.0, 1.0))
    raw_target = float(open_target) - normalized * (float(open_target) - float(closed_target))
    return np.full((gripper_joint_count,), raw_target, dtype=np.float32)


class Pi05DroidRemotePolicy:
    """OpenPI pi0.5 DROID websocket client with Panda/Robotiq action adaptation."""

    def __init__(self, policy_cfg) -> None:
        try:
            from openpi_client import image_tools
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise ImportError(
                "pi05_droid_remote requires the lightweight OpenPI client. Install it in the Isaac env with: "
                "pip install -e /path/to/openpi/packages/openpi-client"
            ) from exc

        self.image_tools = image_tools
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=str(policy_cfg.host),
            port=int(policy_cfg.port),
        )
        self.prompt = str(policy_cfg.prompt)
        self.exterior_image_key = str(policy_cfg.exterior_image_key)
        self.wrist_image_key = str(policy_cfg.wrist_image_key)
        self.image_size = int(policy_cfg.image_size)
        self.execute = bool(policy_cfg.execute)
        self.action_hz = float(policy_cfg.action_hz)
        self.action_mode = str(policy_cfg.action_mode)
        self.max_joint_velocity = float(policy_cfg.max_joint_velocity)
        self.binarize_gripper = bool(policy_cfg.binarize_gripper)
        self.gripper_open_target = float(policy_cfg.gripper_open_target)
        self.gripper_closed_target = float(policy_cfg.gripper_closed_target)
        self._action_chunk = np.zeros((0, 8), dtype=np.float32)
        self._action_index = 0
        self.last_raw_action_shape: tuple[int, ...] | None = None
        self.last_arm_command_kind = "position"

    def _image_for_policy(self, image: np.ndarray) -> np.ndarray:
        resized = self.image_tools.resize_with_pad(image, self.image_size, self.image_size)
        return self.image_tools.convert_to_uint8(resized)

    def _build_openpi_observation(self, obs: dict[str, np.ndarray], current_gripper_qpos: np.ndarray) -> dict:
        missing_keys = [
            key for key in (self.exterior_image_key, self.wrist_image_key, "joint_position") if key not in obs
        ]
        if missing_keys:
            raise KeyError(f"Missing observation keys required by pi05_droid_remote: {missing_keys}")

        gripper_position = normalize_gripper_position(
            current_gripper_qpos,
            self.gripper_open_target,
            self.gripper_closed_target,
        )
        return {
            "observation/exterior_image_1_left": self._image_for_policy(obs[self.exterior_image_key]),
            "observation/wrist_image_left": self._image_for_policy(obs[self.wrist_image_key]),
            "observation/joint_position": np.asarray(obs["joint_position"], dtype=np.float32),
            "observation/gripper_position": gripper_position,
            "prompt": self.prompt,
        }

    def _ensure_action_chunk(self, obs: dict[str, np.ndarray], current_gripper_qpos: np.ndarray) -> None:
        if self._action_index < len(self._action_chunk):
            return

        openpi_obs = self._build_openpi_observation(obs, current_gripper_qpos)
        result = self.client.infer(openpi_obs)
        if "actions" not in result:
            raise KeyError(f"OpenPI policy result did not contain 'actions'. Available keys: {list(result.keys())}")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[1] < 8:
            raise ValueError(f"Expected pi05_droid actions with shape (horizon, >=8), got {actions.shape}")
        self._action_chunk = actions[:, :8]
        self._action_index = 0
        self.last_raw_action_shape = tuple(actions.shape)
        print(f"Received pi05_droid action chunk shape: {actions.shape}")

    def action(
        self,
        obs: dict[str, np.ndarray],
        current_arm_qpos: np.ndarray,
        current_gripper_qpos: np.ndarray,
        gripper_joint_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_action_chunk(obs, current_gripper_qpos)
        raw_action = self._action_chunk[self._action_index]
        self._action_index += 1

        current_arm_qpos = np.asarray(current_arm_qpos, dtype=np.float32)
        if not self.execute:
            self.last_arm_command_kind = "position"
            return current_arm_qpos.copy(), np.asarray(current_gripper_qpos, dtype=np.float32).copy()

        raw_action = np.clip(raw_action, -1.0, 1.0)
        if self.action_mode == "joint_velocity":
            joint_velocity = np.clip(
                raw_action[: current_arm_qpos.shape[0]],
                -self.max_joint_velocity,
                self.max_joint_velocity,
            )
            arm_action = current_arm_qpos + joint_velocity / self.action_hz
            self.last_arm_command_kind = "position"
        elif self.action_mode == "joint_velocity_target":
            arm_action = np.clip(
                raw_action[: current_arm_qpos.shape[0]],
                -self.max_joint_velocity,
                self.max_joint_velocity,
            )
            self.last_arm_command_kind = "velocity"
        elif self.action_mode == "joint_position":
            arm_action = raw_action[: current_arm_qpos.shape[0]]
            self.last_arm_command_kind = "position"
        else:
            raise ValueError(f"Unsupported pi05_droid action_mode: {self.action_mode}")

        gripper_command = float(raw_action[7])
        if self.binarize_gripper:
            gripper_command = 1.0 if gripper_command > 0.5 else 0.0
        gripper_action = denormalize_gripper_position(
            gripper_command,
            gripper_joint_count,
            self.gripper_open_target,
            self.gripper_closed_target,
        )
        return arm_action.astype(np.float32), gripper_action


def make_policy(policy_cfg):
    kind = str(policy_cfg.kind)
    if kind == "fake":
        return None
    if kind == "pi05_droid_remote":
        return Pi05DroidRemotePolicy(policy_cfg.pi05_droid_remote)
    raise ValueError(f"Unsupported policy.kind: {kind}")
