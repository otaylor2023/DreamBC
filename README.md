# DreamBC

World-model policy improvement pipeline for robot manipulation in simulation and real-robot settings using Ctrl-World imagined rollouts and behavior cloning.

## Overview

DreamBC focuses on generating synthetic robot trajectories in imagination and using them to improve policy performance for both sim and real robots with minimal real-world interaction.

Current repo status:
- Uses Ctrl-World rollout pipelines and checkpoints.
- Includes official Franka Panda + hand assets under `urdf_models/franka_description/` (from Franka Robotics `franka_ros`, Apache-2.0; see package `LICENSE` / `NOTICE`).
- Includes a minimal Isaac Sim rollout harness for DreamBC-owned observation/action loop testing.

## Project Structure

- `urdf_models/franka_description/` - Franka ROS `franka_description` (noetic) with DAE/STL meshes and `panda_arm_hand_isaac.urdf` generated for Isaac Sim.
- `scripts/sync_franka_panda_urdf.py` - Regenerates `panda_arm_hand_isaac.urdf` from xacro (`pip install xacro` required).
- `isaacsim_minimal_rollout.py` - Isaac Sim script that records camera/joint observations while executing a fake policy.
- `isaacsim_camera_smoke_test.py` - Minimal single-camera health check for Replicator/annotator readiness.
- `dreambc_isaac/` - Shared Isaac Sim helpers for app startup, URDF import, camera setup, scene setup, robot joints, and local compatibility patches.
- `scripts/bootstrap_env.sh` - Creates or updates the `dreambc` conda environment and installs the Isaac Sim 5.1-compatible PyTorch wheel set.
- `scripts/isaacsim_shell.sh` - Source this in each terminal before running Isaac Sim scripts.
- `scripts/check_isaacsim_env.py` - Prints the active Python, NumPy, SciPy, and PyTorch environment.
- `environment.yml` - mamba environment definition for local DreamBC tooling.
- `requirements.txt` - lightweight Python requirements for local tooling.

## Quick Start

### 1) Prerequisites

Install these on a new machine before setting up DreamBC:

- NVIDIA driver with a GPU supported by Isaac Sim.
- Isaac Sim 5.1 installed locally. The default expected path is `~/.local/share/ov/pkg`.
- `mamba` or `conda`.

If Isaac Sim is installed somewhere else, set:

```bash
export ISAACSIM_ROOT=/path/to/isaacsim
```

### 2) Create or update the DreamBC environment

```bash
cd /home/wpai/DreamBC
bash scripts/bootstrap_env.sh
```

This creates or updates the `dreambc` environment from `environment.yml`, pins
NumPy to `1.x` for Isaac Sim compatibility, installs SciPy and runtime tooling,
and installs the PyTorch CUDA 12.8 wheels used by this Isaac Sim 5.1 setup.

To use a different conda environment name:

```bash
export DREAMBC_ENV_NAME=my_dreambc_env
bash scripts/bootstrap_env.sh
```

### 3) Prepare each Isaac Sim terminal

Source this once in every new terminal before running DreamBC Isaac Sim scripts:

```bash
cd /home/wpai/DreamBC
source scripts/isaacsim_shell.sh
```

This script activates `dreambc`, sources Isaac Sim's `setup_conda_env.sh`, puts
`$CONDA_PREFIX/lib` first in `LD_LIBRARY_PATH`, and returns you to the DreamBC
repo root.

Quick sanity check:

```bash
python scripts/check_isaacsim_env.py
```

Expected:

- `numpy` is `1.26.x`, not `2.x`.
- `scipy_spatial_transform=ok`.
- `torch` points at `.../envs/dreambc/lib/python3.11/site-packages/torch`.
- `torch_jit=True`.

### 4) Run a minimal Isaac Sim rollout

This script provides the basic loop: camera images + joint state -> fake policy action -> robot step
-> rollout artifacts.

Run the rollout with the UI:

```bash
python isaacsim_minimal_rollout.py sim.headless=false sim.steps=120
```

In UI mode the script keeps Isaac Sim open after the rollout so the stage can be
inspected. Close the Isaac Sim window to exit, or override
`sim.keep_open_after_rollout=false` for the old auto-close behavior.

Run the rollout headless:

```bash
python isaacsim_minimal_rollout.py sim.headless=true sim.steps=120
```

Outputs are written under:

```bash
/home/wpai/DreamBC/rollouts/minimal_rollout/
```

The rollout script uses Hydra config:

```bash
/home/wpai/DreamBC/configs/minimal_rollout.yaml
```

Policy kinds: `fake` (default in `configs/minimal_rollout.yaml`) or `smolvla` (see `configs/smolvla_pick_cube_rollout.yaml`).

```bash
python isaacsim_minimal_rollout.py policy.kind=fake
```

### SmolVLA pick-cube rollout (test cube only, no desk props)

One-time (in the `dreambc` env):

```bash
pip install -r requirements-smolvla.txt
# If numpy was upgraded past 1.x: pip install "numpy>=1.26,<2"
huggingface-cli download lerobot/smolvla_base --local-dir models/smolvla_base
PYTHONPATH=. python scripts/smolvla_policy_smoke_test.py
```

Rollout (Isaac shell):

```bash
source scripts/isaacsim_shell.sh
python isaacsim_minimal_rollout.py --config-name=smolvla_pick_cube_rollout
```

Outputs under `outputs/rollouts/smolvla_pick_cube/<timestamp>/` including `camera_videos/*.mp4` for all four cameras. The base `smolvla_base` checkpoint is not Franka-trained; expect weak zero-shot grasping until finetuned.

### Output layout

| Path | Purpose |
|------|---------|
| `outputs/demos/logs/` | Dataset collection summaries |
| `outputs/demos/test_logs/` | Short test-collect logs |
| `outputs/rollouts/<name>/` | Policy / debug rollout MP4s and metadata |
| `outputs/train/smolvla_pick_cube/` | Fine-tuned checkpoints |
| `data/lerobot/dreambc_franka_pick_cube/` | LeRobot dataset (+ `preview_mp4/` debug videos) |

### Train pick-cube (RMPFlow demos → SmolVLA fine-tune)

Collect demonstrations in Isaac Sim with the **Franka Emika Panda** from Isaac’s robot asset
(`robot.source: isaac_franka`, same model Lula IK / RMPFlow use). **RMPFlow + PickPlace** is the default planner.
CuRobo in-process is disabled (`curobo.enabled: false`) because it conflicts with Isaac’s bundled Warp.

```bash
source scripts/isaacsim_shell.sh
python isaacsim_curobo_collect_dataset.py --config-name=curobo_pick_cube_dataset
# test run with MP4 for every episode (incl. failures): --config-name=curobo_pick_cube_test
```

Dataset: `data/lerobot/dreambc_franka_pick_cube/`. Preview MP4s (subset): `data/lerobot/dreambc_franka_pick_cube/preview_mp4/episode_XXXX/`.

Validate dataset (no Isaac):

```bash
python scripts/validate_lerobot_dataset.py
```

Fine-tune (no Isaac):

```bash
bash scripts/train_smolvla_pick_cube.sh
```

Eval with **required** camera MP4s:

```bash
python isaacsim_minimal_rollout.py --config-name=smolvla_pick_cube_eval \
  smolvla.model_path=outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model
```

Eval videos: `outputs/rollouts/smolvla_pick_cube_eval/<timestamp>/camera_videos/*.mp4`.

Common overrides:

```bash
python isaacsim_minimal_rollout.py robot.gripper=franka sim.steps=240
python isaacsim_minimal_rollout.py output.dir=outputs/rollouts/debug sim.save_every=5
python isaacsim_minimal_rollout.py sim.require_cameras=true
```

Camera requirement mode:

- `sim.require_cameras=true` (strict): rollout aborts if camera pipeline never becomes ready.
- `sim.require_cameras=false` (default for fast integration): if cameras are not ready, rollout continues in joint-only mode and records camera status in `metadata.json`.

Robot assets come from `franka_ros` `franka_description` (vendored under `urdf_models/franka_description/`). The Isaac entry URDF is `panda_arm_hand_isaac.urdf`, produced by `python3 scripts/sync_franka_panda_urdf.py` after updating the vendored tree (requires `pip install xacro`; the script briefly expands `$(find franka_description)` to an absolute path, then restores it). Joint names use the `panda_` prefix (`panda_joint1`…`panda_joint7`, `panda_finger_joint1`, `panda_finger_joint2`).

Do not use Isaac Sim's `./python.sh` for the Hydra rollout script unless that
Python environment also has `hydra-core`, `omegaconf`, `attrs`, and `pillow`
installed. The recommended path is `source scripts/isaacsim_shell.sh`, then
`python ...`.

### 5) Test camera alignment

Verify the images written under `camera_samples/` for each configured camera (exterior views, optional `scene_overview`, and `wrist_image_left`).

Run a one-step camera check:

```bash
python isaacsim_minimal_rollout.py sim.headless=true sim.steps=1 sim.require_cameras=true policy.kind=fake output.dir=outputs/rollouts/camera_alignment_check
```

Inspect the latest PNGs:

```text
outputs/rollouts/camera_alignment_check/<latest>/camera_samples/step_0000_exterior_image_1_left.png
outputs/rollouts/camera_alignment_check/<latest>/camera_samples/step_0000_wrist_image_left.png
outputs/rollouts/camera_alignment_check/<latest>/camera_debug.json
```

For wrist camera debugging, run the UI with stage-tree, prim-pose, camera-pose,
and marker output enabled:

```bash
python isaacsim_minimal_rollout.py sim.headless=false sim.steps=1 sim.require_cameras=true policy.kind=fake output.dir=outputs/rollouts/wrist_debug_ui debug.print_stage_tree=true debug.print_camera_poses=true debug.print_prim_positions=true debug.add_camera_markers=true sim.keep_open_after_rollout=true
```

This prints the USD hierarchy and world poses for key prims such as
`link7` and the finger links, and adds green markers at camera positions in the scene.

### Isaac Sim troubleshooting notes

If the rollout fails at:

```text
Initializing camera: exterior_image_1_left (/World/Cameras/exterior_image_1_left)
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

the scene and URDF import have already progressed far enough to create the first
camera. The failure is inside Isaac Sim's Replicator/SyntheticData graph while
attaching the `rgb` annotator in `Camera.initialize()`. URDF import warnings
such as `getAttributeCount called on non-existent path ...node_STL_BINARY_`
are often benign and are not the direct cause.

When debugging this case, first check the renderer/GPU and Replicator startup
logs above the traceback. The camera path and name are printed before
initialization so it is clear which camera failed first. The rollout script
currently applies a local DreamBC workaround before camera initialization that
casts SyntheticData intergraph dependency handles to `uint64`; this avoids the
`unknown dtype` failure without modifying the Isaac Sim installation.

If the rollout fails while importing SciPy with:

```text
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

the shell is using the system `libstdc++.so.6` instead of the newer one from
the `dreambc` conda environment. Start every Isaac Sim terminal with:

```bash
cd /home/wpai/DreamBC
source scripts/isaacsim_shell.sh
```

If you see:

```text
ModuleNotFoundError: No module named 'osqp'
```

install missing dependency in `dreambc`:

```bash
python -m pip install osqp
```

### 6) Camera smoke test

Use the smoke test to quickly tell whether camera failures are environment-level
or rollout-pipeline-level.

Run:

```bash
python /home/wpai/DreamBC/isaacsim_camera_smoke_test.py
python /home/wpai/DreamBC/isaacsim_camera_smoke_test.py sim.headless=true sim.steps=120
```

What it does:

- Creates a world with one camera (`exterior_image_1_left` config).
- Initializes camera annotator.
- Steps simulation and checks whether `camera.get_rgba()` returns valid image tensors.
- Prints `good_frames`, `bad_frames`, and `first_good_step`.

Interpretation:

- Smoke test fails (no valid frames): likely Isaac Sim/Replicator/environment issue.
- Smoke test passes but minimal rollout fails: likely rollout scene/init sequencing issue.

## Notes

- This repo is currently set up for Isaac Sim workflows.
- For large-scale world-model inference, use a Linux GPU instance with sufficient VRAM.
