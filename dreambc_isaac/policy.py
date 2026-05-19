"""Policy helpers used by the minimal rollout harness."""

from __future__ import annotations

import numpy as np


def load_smolvla_runner_if_configured(cfg):
    if str(cfg.policy.kind) != "smolvla":
        return None
    from dreambc_isaac.smolvla_policy import load_smolvla_runner

    return load_smolvla_runner(cfg.smolvla)


def smolvla_policy_action(runner, obs, current_arm_qpos: np.ndarray, gripper_joint_count: int):
    from dreambc_isaac.smolvla_policy import smolvla_policy_action as _smolvla_action

    return _smolvla_action(runner, obs, current_arm_qpos, gripper_joint_count)


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
