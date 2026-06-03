# DreamBC

World-model policy improvement pipeline for robot manipulation in simulation and real-robot settings using Ctrl-World imagined rollouts and behavior cloning.

## Final Report
https://docs.google.com/document/d/1odM9GF-Pmqk3XJ_2HiZ-iAMGKYLHIHZ-EvoGepL-4_8/edit?usp=sharing

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
