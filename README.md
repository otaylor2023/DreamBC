# DreamBC

World-model policy improvement pipeline for robot manipulation in simulation and real-robot settings using Ctrl-World imagined rollouts and behavior cloning.

## Overview

DreamBC focuses on generating synthetic robot trajectories in imagination and using them to improve policy performance for both sim and real robots with minimal real-world interaction.

Current repo status:
- Uses Ctrl-World rollout pipelines and checkpoints.
- Includes Panda + Robotiq URDF assets under `urdf_models/`.
- Includes an Isaac Sim preview script for quickly visualizing gripper variants (Franka hand vs Robotiq 2F-85).
- Includes a minimal Isaac Sim rollout harness for DreamBC-owned observation/action loop testing.

## Project Structure

- `urdf_models/` - robot URDFs and meshes (Panda + Robotiq variants).
- `isaacsim_dual_robot_preview.py` - Isaac Sim script to preview robots side-by-side.
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

### 4) Preview robots in Isaac Sim

```bash
python isaacsim_dual_robot_preview.py --gripper both
```

Options:
- `--gripper franka`
- `--gripper robotiq`
- `--gripper both`
- `--headless`

### 5) Run a minimal Isaac Sim rollout

This script does not load pi0.5 yet. It creates the basic loop that pi0.5 will
eventually plug into: camera images + joint state -> fake policy action -> robot
step -> rollout artifacts.

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

Common overrides:

```bash
python isaacsim_minimal_rollout.py robot.gripper=franka sim.steps=240
python isaacsim_minimal_rollout.py output.dir=rollouts/debug sim.save_every=5
python isaacsim_minimal_rollout.py sim.require_cameras=true
```

Camera requirement mode:

- `sim.require_cameras=true` (strict): rollout aborts if camera pipeline never becomes ready.
- `sim.require_cameras=false` (default for fast integration): if cameras are not ready, rollout continues in joint-only mode and records camera status in `metadata.json`.

The Panda URDFs include mobile-base joints (`base_prismatic_x`,
`base_prismatic_y`, `base_revolute`, `base_prismatic_link0`). The rollout keeps
those joints hard-locked at their initial targets by resetting their joint
positions and clearing their velocities every simulation step. Only the
configured arm and gripper joint names are driven by the fake policy. If the
base still moves, check `robot.locked_base_joint_names` in
`configs/minimal_rollout.yaml` and make sure all mobile-base joints are listed.

Do not use Isaac Sim's `./python.sh` for the Hydra rollout script unless that
Python environment also has `hydra-core`, `omegaconf`, `attrs`, and `pillow`
installed. The recommended path is `source scripts/isaacsim_shell.sh`, then
`python ...`.

### Isaac Sim troubleshooting notes

If the rollout fails at:

```text
Initializing camera: exterior_image_1_left (/World/Cameras/exterior_image_1_left)
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

the scene and URDF import have already progressed far enough to create the first
camera. The failure is inside Isaac Sim's Replicator/SyntheticData graph while
attaching the `rgb` annotator in `Camera.initialize()`. The nearby Robotiq mesh
messages such as `getAttributeCount called on non-existent path ...node_STL_BINARY_`
are warnings and are not the direct cause.

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

### Camera smoke test

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
