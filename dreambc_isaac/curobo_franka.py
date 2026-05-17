"""CuRobo MotionPlanner wrapper for the DreamBC Franka in Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

PANDA_ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


@dataclass
class CuroboFrankaPlanner:
    planner: object
    device: str

    @property
    def joint_names(self) -> list[str]:
        return list(self.planner.joint_names)

    def warmup(self) -> None:
        self.planner.warmup(enable_graph=True, num_warmup_iterations=3)

    def update_scene(self, scene_cfg) -> None:
        from curobo._src.geom.types import Cuboid, SceneCfg

        cuboids = []
        table = scene_cfg.table
        tp = np.asarray(table.position, dtype=np.float64)
        ts = np.asarray(table.scale, dtype=np.float64)
        cuboids = [
            Cuboid(
                name="table",
                pose=[float(tp[0]), float(tp[1]), float(tp[2]), 1.0, 0.0, 0.0, 0.0],
                dims=[float(ts[0]), float(ts[1]), float(ts[2])],
            ),
        ]
        if hasattr(scene_cfg, "cube") and scene_cfg.cube is not None:
            cube = scene_cfg.cube
            cpos = np.asarray(cube.position, dtype=np.float64)
            cscale = np.asarray(cube.dims, dtype=np.float64)
            cuboids.append(
                Cuboid(
                    name="pick_cube",
                    pose=[
                        float(cpos[0]),
                        float(cpos[1]),
                        float(cpos[2]),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dims=[float(cscale[0]), float(cscale[1]), float(cscale[2])],
                )
            )
        world = SceneCfg(cuboid=cuboids)
        self.planner.update_world(world)

    def joint_state_from_arm_qpos(
        self,
        arm_qpos: np.ndarray,
        gripper_open: float,
    ):
        from curobo.types import JointState

        arm_qpos = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
        names = self.joint_names
        full = np.zeros(len(names), dtype=np.float32)
        for i, jn in enumerate(PANDA_ARM_JOINTS):
            if i < arm_qpos.shape[0] and jn in names:
                full[names.index(jn)] = arm_qpos[i]
        for finger_name in ("panda_finger_joint1", "panda_finger_joint2"):
            if finger_name in names:
                full[names.index(finger_name)] = float(gripper_open)
        pos = torch.tensor(full, device=self.device, dtype=torch.float32).unsqueeze(0)
        return JointState.from_position(pos, joint_names=names)


def load_curobo_franka_planner(*, use_cuda: bool = True) -> CuroboFrankaPlanner:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Warning: CuRobo planner running on CPU; planning will be slow.")
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        max_goalset=4,
    )
    planner = MotionPlanner(config)
    return CuroboFrankaPlanner(planner=planner, device=device)
