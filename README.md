# DreamBC

**DreamBC: Robot Policy Improvement via World Model Rollouts and Behavior Cloning** — Stanford CS348K (Bhat, Chu, Taylor).

We use **Ctrl-World** imagined rollouts as behavior-cloning data to adapt **pi0.5** (`pi05_droid`) on real-Franka pick-and-place tasks, reducing reliance on new human teleoperation for every task.

| Resource | Link |
|----------|------|
| **Project site** | https://otaylor2023.github.io/DreamBC/ |
| **Full report** | [docs/report.html](docs/report.html) |
| **Checkpoint portfolio (metadata)** | [real_robot_checkpoints/](real_robot_checkpoints/) |
| **Curated media** | [assets/](assets/) → built into [docs/static/](docs/static/) |
| **Results index** | [results/](results/) |

## Headline results (real Franka)

| Task | Base pi0.5 | Teleop FT | Ctrl-World FT (ours) |
|------|------------|-----------|----------------------|
| Pick up cube | 90% | 100% | 90% (smoother motion) |
| Pick up tomato | 60% | 70% | **100%** |

Rollout videos and interpretation are on the [full report](https://otaylor2023.github.io/DreamBC/report.html).

## Pipeline

1. **Snapshots** — single-frame starts from the real Franka setup (HDF5).
2. **Imagined rollouts** — pi0.5 acts in the loop; Ctrl-World renders multi-camera video. Successful trajectories are hand-curated into BC bundles (`scripts/bc_dataset.py`, `Ctrl-World/scripts/rollout_interact_pi.py`).
3. **Fine-tune pi0.5** — LoRA on curated synthetic (or teleop) bundles; EMA-loss thresholds tag comparable checkpoints across tasks (`scripts/train_dreambc.py`, `scripts/run_with_loss_threshold.py`).
4. **Deploy** — selected LoRA adapters evaluated on the physical Franka; site videos use the four checkpoints listed below.

Full-FT checkpoints exist in the portfolio for completeness but were **not** used in the final real-robot runs.

## Final deployed checkpoints

These four LoRA adapters produced the real-robot results on the site (Ctrl-World vs teleop baseline, cube + tomato). Packaged copy + provenance: [`final_used_checkpoints/`](final_used_checkpoints/) (weights gitignored; upload via Drive).

| Policy | Task | Path | Deploy prompt |
|--------|------|------|---------------|
| Ctrl-World (ours) | Cube | `cube/lora_aug_thr004_step200` | `put the orange block in the blue bowl` |
| Teleop baseline | Cube | `cube/lora_aug_thr004_step225_hand` | `put the orange block in the blue bowl` |
| Ctrl-World (ours) | Tomato | `tomato/lora_aug_thr004_step200_bowl` | `pick the tomato and place it in the blue bowl` |
| Teleop baseline | Tomato | `tomato/lora_aug_thr004_step200_hand_bowl` | `pick the tomato and place it in the blue bowl` |

Training data: **30** curated cube + **24** tomato Ctrl-World rollouts (guidance g=5 / g=3 with reworded prompts). Teleop baselines use matched hand-collected demo counts and the same LoRA recipe.

## Model training (overview)

**BC bundle format.** Each policy decision stores exterior + wrist RGB (224×224), joint/gripper state, and a 15-step action chunk in DROID layout. Bundles live under `rollouts/<dataset>/Rollouts_interact_pi/bc_episodes/`.

**LoRA fine-tuning.** Default recipe: `gemma_2b_lora` + `gemma_300m_lora`, base pi05_droid frozen, optional mild image augmentation on exterior + wrist. Training runs log EMA loss; checkpoints are tagged at thresholds **0.10 / 0.04 / 0.015** so cube and tomato runs can be compared at similar training progress.

**Choosing a checkpoint.** For each task we picked the aug + 0.04-threshold LoRA step used in post-training Ctrl-World eval (matched seeds, same world-model settings as dataset generation). See `real_robot_checkpoints/README.md` and `MANIFEST.json` for the full 42-checkpoint portfolio.

**Teleop baseline.** Same training stack on hand-collected HDF5 demos (`scripts/prepare_handcollected_for_ctrl_world.py` → BC bundles → identical LoRA hyperparameters).

## Repo layout

```
DreamBC/
  docs/                    # GitHub Pages site (index + report)
  scripts/                 # training, rollout helpers, site media builders
  final_used_checkpoints/  # staged copy of the 4 deployed adapters (gitignored weights)
  real_robot_checkpoints/  # portfolio README, MANIFEST, meta.json (weights ignored)
  vendor_overrides/        # patches overlaid on upstream Ctrl-World
  success_lists/           # curated rollout / demo IDs
  assets/                  # source media before build_site_videos.py
  results/                 # evaluation summaries
  urdf_models/             # Panda + Robotiq assets
```

`Ctrl-World/` is **not** committed — clone upstream and overlay `vendor_overrides/Ctrl-World/`. Large dirs are gitignored: `rollouts/`, `training_runs/`, checkpoint `params/`, raw snapshot zips.

## Quick commands

```bash
# LoRA fine-tune on a BC bundle root
./scripts/run_train_lora.sh rollouts/C16_g5_block my_exp --enable-image-aug

# Threshold-based train + early-stop tagging
python scripts/run_with_loss_threshold.py \
  --variant lora --bundles-root rollouts/handcollected_tomato \
  --exp-name handcollected_tomato_lora_aug --enable-image-aug \
  --thresholds 0.10,0.04 --num-train-steps 400 --log-file logs/hand.log

# Package checkpoints for the real-robot portfolio
python scripts/package_real_robot_checkpoints.py  # see script --help

# Eval a checkpoint in Ctrl-World (matched-seed comparison)
./scripts/eval_dreambc_rollout.sh \
  real_robot_checkpoints/tomato/lora_aug_thr004_step200_bowl lora \
  rollouts/eval_post_training/tomato

# Build / refresh GitHub Pages videos
python3 scripts/build_site_videos.py --copy-images
python3 scripts/build_rollout_samples.py
python3 scripts/build_archive_clips.py

# Pre-push size check
python3 scripts/check_repo_ready.py
```

## Enable GitHub Pages

**Settings → Pages → Deploy from branch → `main` → `/docs`** → https://otaylor2023.github.io/DreamBC/

See [docs/README.md](docs/README.md) for media curation.

## Ctrl-World setup (upstream)

`Ctrl-World/` is upstream [Ctrl-World](https://github.com/Robotic-AI-Lab/Ctrl-World). DreamBC patches live in [`vendor_overrides/Ctrl-World/`](vendor_overrides/Ctrl-World/) (rollout script with pi0.5 LoRA loading + BC bundle outputs, config guidance scale, etc.).

**Minimal bootstrap on a GPU machine:**

```bash
git clone git@github.com:otaylor2023/DreamBC.git && cd DreamBC

git clone git@github.com:Robotic-AI-Lab/Ctrl-World.git Ctrl-World
rsync -a vendor_overrides/Ctrl-World/ Ctrl-World/

cd Ctrl-World
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi && pip install uv && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../..

# Checkpoints (see Ctrl-World/readme.md + openpi docs):
#   Ctrl-World/checkpoint/checkpoint-10000.pt
#   Ctrl-World/openpi/checkpoint/pi05_droid
# Snapshots:
#   data/snapshots/cube_snapshots/snapshot_*.hdf5
#   data/snapshots/tomato_snapshots/snapshot_*.hdf5

source Ctrl-World/openpi/.venv/bin/activate
python scripts/prepare_snapshots_for_ctrl_world.py \
  --source_dir data/snapshots/cube_snapshots \
  --output_dir Ctrl-World/dataset_example/cube_snapshots_v3 \
  --default_instruction "put the orange block in the blue bowl" \
  --views agent_view,exterior_3,wrist
```

After rollouts are collected and curated, convert to BC training roots with `scripts/bc_dataset.py` and launch fine-tuning as above. Generation configs used for the final datasets (prompt + guidance) are documented in the [full report](docs/report.html) §2.2.
