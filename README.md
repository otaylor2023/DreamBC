# DreamBC

World-model policy improvement pipeline for robot manipulation in simulation and real-robot settings using Ctrl-World imagined rollouts and behavior cloning.

## Week 6 Checkpoint
### Project Questions / Goals

Our project aims to study whether **Ctrl-World can help improve a VLA policy on robot manipulation tasks where the policy initially fails with π0.5**.

The main questions we want to answer are:

1. Can we identify tasks where a generally competent VLA policy fails?
2. Can Ctrl-World-generated imagined trajectories help the policy improve on those failed tasks?
3. Can the improved policy succeed on the target task without hurting performance on other tasks?

### Evaluation Plan

To evaluate our questions, we would like to apply a straightforward approach including the evaluation of the VLA policy before and after making improvements.

Specifically, the plan includes the following experiments:

1. Evaluate the VLA policy in several Franka manipulation tasks.
2. Find two to three tasks that fail or show poor results.
3. Generate the imaginary rollout trajectories for these tasks.
4. Choose the successful trajectories.
5. Improve the policy based on these data.
6. Re-run the test.

We will consider the project successful if the policy improves from failure to success on at least one target task, or if the success rate clearly increases after using Ctrl-World-generated trajectories.

## Current Progress

So far, we have collected **5 trajectories for 5 different manipulation tasks** using the real Franka robot. For every task, we saved the **task prompt**, **robot actions**, and **video recording**.

Originally, we planned to test **π0.5** directly in PyBullet, but we were not able to run π0.5 successfully in it. Therefore, we shifted our current focus to **real-world task data collection on the Franka arm**.

The simulation code is on `wayne` branch. 

---

## Overview

DreamBC focuses on generating synthetic robot trajectories in imagination and using them to improve policy performance for both sim and real robots with minimal real-world interaction.

Current repo status:
- Uses Ctrl-World rollout pipelines and checkpoints.
- Includes Panda + Robotiq URDF assets under `urdf_models/`.
- Includes an Isaac Sim preview script for quickly visualizing gripper variants (Franka hand vs Robotiq 2F-85).

## Project Structure

- `urdf_models/` - robot URDFs and meshes (Panda + Robotiq variants).
- `isaacsim_dual_robot_preview.py` - Isaac Sim script to preview robots side-by-side.
- `requirements.txt` - lightweight Python requirements for local tooling.

## Quick Start

### 1) Create and activate environment

```bash
conda create -n dreambc_arm python=3.11 -y
conda activate dreambc_arm
pip install -r requirements.txt
```

### 2) Preview robots in Isaac Sim

Run this from Isaac Sim's Python launcher:

```bash
./python.sh isaacsim_dual_robot_preview.py --gripper both
```

Options:
- `--gripper franka`
- `--gripper robotiq`
- `--gripper both`
- `--headless`

## Notes

- This repo is currently set up for Isaac Sim workflows.
- For large-scale world-model inference, use a Linux GPU instance with sufficient VRAM.

---

## Ctrl-World + pi05 imagination sweep

This is the active workstream: run policy-in-the-loop rollouts where pi0.5 acts in the
Ctrl-World imagined environment, starting from single-frame Franka snapshots. Each rollout
writes a BC-ready bundle (raw frames + policy-aligned `(obs, action_chunk)` pairs) under
`rollouts/sweep/<object>/<config_name>/Rollouts_interact_pi/bc_episodes/`.

### Setup on a fresh Linux GPU instance (H100 / A100 80GB recommended)

The repo intentionally does not commit large assets (`Ctrl-World/openpi/`, model
checkpoints, raw snapshot HDF5s, rollout outputs). Fetch them out-of-band:

1) Clone this repo:
   ```bash
   git clone git@github.com:otaylor2023/DreamBC.git
   cd DreamBC
   ```

2) Install openpi inside `Ctrl-World/` (gitignored on purpose):
   ```bash
   cd Ctrl-World
   git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
   cd openpi
   pip install uv
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   cd ../..
   ```

3) Download model checkpoints:
   - Ctrl-World ckpt -> `Ctrl-World/checkpoint/checkpoint-10000.pt` (see the table in
     `Ctrl-World/readme.md`).
   - pi05_droid ckpt: follow openpi instructions; symlink or copy to
     `Ctrl-World/openpi/checkpoint/pi05_droid`.
   - SVD + CLIP: HuggingFace will auto-cache on first use; required snapshot hashes
     are referenced in `scripts/run_sweep.sh` (`SVD_PATH`, `CLIP_PATH`).

4) Place raw snapshot HDF5s under:
   ```
   data/snapshots/cube_snapshots/snapshot_*.hdf5
   data/snapshots/tomato_snapshots/snapshot_*.hdf5
   ```
   (transferred via scp / S3 / `cube_snapshots.zip` and `tomato_snapshots.zip`)

5) Build the Ctrl-World-format datasets used by the sweep configs:
   ```bash
   source Ctrl-World/openpi/.venv/bin/activate

   # Cube: agent_view + exterior_3 + wrist (default v3 layout)
   python scripts/prepare_snapshots_for_ctrl_world.py \
     --source_dir data/snapshots/cube_snapshots \
     --output_dir Ctrl-World/dataset_example/cube_snapshots_v3 \
     --default_instruction "put the orange cube in the blue bowl" \
     --views agent_view,exterior_3,wrist

   # Cube views ablation (exterior_2 + exterior_3 + wrist)
   python scripts/prepare_snapshots_for_ctrl_world.py \
     --source_dir data/snapshots/cube_snapshots \
     --output_dir Ctrl-World/dataset_example/cube_snapshots_v2 \
     --default_instruction "put the orange cube in the blue bowl" \
     --views exterior_2,exterior_3,wrist

   # Tomato v3 (baseline views) and v2 (ablation)
   python scripts/prepare_snapshots_for_ctrl_world.py \
     --source_dir data/snapshots/tomato_snapshots \
     --output_dir Ctrl-World/dataset_example/tomato_snapshots_v3 \
     --default_instruction "put the tomato in the blue bowl" \
     --views agent_view,exterior_3,wrist
   python scripts/prepare_snapshots_for_ctrl_world.py \
     --source_dir data/snapshots/tomato_snapshots \
     --output_dir Ctrl-World/dataset_example/tomato_snapshots_v2 \
     --default_instruction "put the tomato in the blue bowl" \
     --views exterior_2,exterior_3,wrist
   ```

### Launching a sweep (detached so it survives SSH/Cursor disconnect)

Full 26-config sweep (cube + tomato, ~9.7 h on a single H100):
```bash
mkdir -p rollouts/sweep/_logs
setsid nohup bash scripts/run_sweep.sh \
  > rollouts/sweep/_logs/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
disown
```

Focused tomato-only run (edit the config list in `scripts/run_sweep.sh` to keep just the
`T*` block, or use the smaller helpers as templates):
- `scripts/run_cube_prompt_extra.sh` - 4-config cube prompt x guidance grid (10 shared
  random IDs per config).
- `scripts/run_c16_fill.sh` - fills the remaining 50 cube IDs for a single config.

Both helpers follow the same pattern: edit `VAL_IDS`, `INTERACT_NUM`, `Z_MIN`, the
`run_config` lines, then launch with the same `setsid nohup ... & disown` invocation.

Sanity-check the launch is fully detached:
```bash
ps -p $(cat rollouts/sweep/_logs/<your_pid_file>) -o pid,ppid,sid,cmd
# ppid should be 1 within a few seconds
```

### Output layout per run

```
rollouts/sweep/
  sweep_index.json                              # rolled-up over all configs
  _logs/
    _master.log                                 # one line per config start/end
    <config_name>.log                           # per-config stdout/stderr
    <config_name>.done                          # marker on successful exit
  cube/<config_name>/
    sweep_config.json                           # all knobs for this run
    Rollouts_interact_pi/
      video/...mp4                              # predicted-only 3-cam concat
      info/...json                              # high-frequency policy info
      bc_episodes/
        <task>_time_<ts>_traj_<id>_..._<text>/
          episode.npz                           # initial+predicted frames, joint_*,
                                                # state_fk, policy_obs_exterior,
                                                # policy_obs_wrist, policy_state_*,
                                                # policy_action_chunk
          episode.json                          # instruction, success=null, all knobs
          camera_{0,1,2}.mp4                    # per-camera predicted videos
  tomato/<config_name>/...
```

### Rebuilding the rollup index

After (or during) a sweep, regenerate `rollouts/sweep/sweep_index.json`:
```bash
python scripts/build_sweep_index.py --sweep_root rollouts/sweep
```

### Multi-instance collection

`rollouts/` is gitignored, so each GPU machine collects independently. To merge sweeps
from multiple instances onto one host for BC training, rsync the relevant
`rollouts/sweep/<object>/<config_name>/` dirs across, then re-run
`scripts/build_sweep_index.py`. Configs running on the same machine must serialize on the
GPU (the runner scripts loop one config at a time).
