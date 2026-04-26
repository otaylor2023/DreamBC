# DreamBC

World-model policy improvement pipeline for robot manipulation, based on Ctrl-World-style imagined rollouts and behavior cloning.

## Overview

DreamBC focuses on generating synthetic robot trajectories in imagination and using them to improve policy performance with minimal real-world interaction.

Current repo status:
- Includes `Ctrl-World/` codebase for world-model rollouts.
- Includes Panda + Robotiq URDF assets under `urdf_models/`.
- Includes an Isaac Sim preview script for quickly visualizing gripper variants (Franka hand vs Robotiq 2F-85).

## Project Structure

- `Ctrl-World/` - upstream Ctrl-World training/inference code.
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
- Ctrl-World inference/training dependencies and checkpoints are managed inside `Ctrl-World/`.
- For large-scale world-model inference, use a Linux GPU instance with sufficient VRAM.
