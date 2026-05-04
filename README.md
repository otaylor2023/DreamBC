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
- `isaacsim_minimal_rollout.py` - Isaac Sim script that records camera/joint observations while executing a fake policy or a remote pi0.5 DROID policy.
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

This script provides the basic loop used by both the fake policy and the remote
pi0.5 DROID policy: camera images + joint state -> policy action -> robot step
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

Policy selection:

```bash
python isaacsim_minimal_rollout.py policy.kind=fake
python isaacsim_minimal_rollout.py policy.kind=pi05_droid_remote
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

### 6) Run a pi0.5 DROID policy server

Keep OpenPI/pi0.5 in a separate environment from the Isaac Sim `dreambc`
environment. The Isaac Sim environment is sensitive to NumPy, PyTorch, and
library-path versions; the pi0.5 server should run in its own OpenPI `uv`
environment and DreamBC should connect to it over websocket.

Install and sync OpenPI outside conda/Isaac shells:

```bash
cd /home/wpai/DreamBC
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd /home/wpai/DreamBC/openpi
conda deactivate || true
rm -rf .venv
unset CC CXX CFLAGS CPPFLAGS LDFLAGS
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

If `evdev` fails to build with an error such as `KEY_LINK_PHONE undeclared`,
you are probably still using a conda compiler from `(base)`. Leave conda and
retry, or force the system compiler:

```bash
cd /home/wpai/DreamBC/openpi
rm -rf .venv
unset CFLAGS CPPFLAGS LDFLAGS
CC=/usr/bin/gcc CXX=/usr/bin/g++ GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Start the pi0.5 DROID server:

```bash
cd /home/wpai/DreamBC/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The server is ready when it prints:

```text
server listening on 0.0.0.0:8000
```

The first inference call can be slow because JAX compiles and warms up the
model. Later action-chunk calls should be much faster.

### 7) Install the OpenPI client in the Isaac environment

Use a second terminal for DreamBC/Isaac Sim:

```bash
cd /home/wpai/DreamBC
source scripts/isaacsim_shell.sh
cd /home/wpai/DreamBC/openpi/packages/openpi-client
pip install -e .
cd /home/wpai/DreamBC
```

Only install the lightweight `openpi-client` in `dreambc`. Do not install the
full OpenPI model stack into the Isaac Sim environment.

### 8) Test camera alignment

Before executing pi0.5 actions, verify the images written under
`camera_samples/`. The pi0.5 DROID policy expects:

- `observation/exterior_image_1_left`: a third-person view of robot, table, and object.
- `observation/wrist_image_left`: a wrist-mounted camera view fixed to the wrist/gripper frame.
- `observation/joint_position`: 7 Panda/Franka joint positions.
- `observation/gripper_position`: 1 normalized gripper position.

Run a one-step camera check:

```bash
python isaacsim_minimal_rollout.py sim.headless=true sim.steps=1 sim.require_cameras=true policy.kind=fake output.dir=rollouts/camera_alignment_check
```

Inspect the latest PNGs:

```text
rollouts/camera_alignment_check/<latest>/camera_samples/step_0000_exterior_image_1_left.png
rollouts/camera_alignment_check/<latest>/camera_samples/step_0000_wrist_image_left.png
rollouts/camera_alignment_check/<latest>/camera_debug.json
```

For wrist camera debugging, run the UI with stage-tree, prim-pose, camera-pose,
and marker output enabled:

```bash
python isaacsim_minimal_rollout.py sim.headless=false sim.steps=1 sim.require_cameras=true policy.kind=fake output.dir=rollouts/wrist_debug_ui debug.print_stage_tree=true debug.print_camera_poses=true debug.print_prim_positions=true debug.add_camera_markers=true sim.keep_open_after_rollout=true
```

This prints the actual USD hierarchy and world poses for key prims such as
`link7`, `robotiq_85_base_link`, and the finger links. It also adds green
markers at camera positions in the scene. If the wrist camera image is blank or
only shows sky/table, use `camera_debug.json` to compare the wrist camera pose,
view axes, lens settings, gripper prim positions, table position, and test cube
position before running pi0.5.

### 9) Run pi0.5 through Isaac Sim

First run a dry-run that queries pi0.5 but does not execute the returned action.
This verifies websocket IO, image serialization, and action chunk shape:

```bash
python isaacsim_minimal_rollout.py sim.headless=true sim.steps=20 sim.require_cameras=true policy.kind=pi05_droid_remote policy.pi05_droid_remote.execute=false 'policy.pi05_droid_remote.prompt=pick up the red cube'
```

Expected output:

```text
Using policy kind: pi05_droid_remote
Received pi05_droid action chunk shape: (15, 8)
Saved rollout: ...
```

If dry-run works and both policy camera inputs are valid, execute pi0.5 actions
with conservative velocity limits:

```bash
python isaacsim_minimal_rollout.py sim.headless=false sim.steps=120 sim.require_cameras=true policy.kind=pi05_droid_remote policy.pi05_droid_remote.execute=true policy.pi05_droid_remote.max_joint_velocity=0.1 'policy.pi05_droid_remote.prompt=pick up the red cube'
```

The `pi05_droid_remote` adapter follows the OpenPI DROID runtime convention:

- The server returns action chunks shaped like `(horizon, 8)`.
- The first 7 dimensions are Panda/Franka joint velocity commands.
- The 8th dimension is a gripper position command.
- Actions are clipped to `[-1, 1]`.
- The gripper command is binarized by default.
- Isaac Sim executes arm commands with `joint_velocities` and gripper commands
  with joint position targets.

Do not run long `execute=true` rollouts until the wrist camera extrinsic is
reasonable. If the robot twists without moving toward the target, first check
the saved camera samples and `metadata.json`; most failures so far have been
caused by invalid wrist-camera views or mismatched initial robot/camera setup.

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
