# DreamBC

**Ctrl-World for Real-Time SimToReal Policy Generalization** — Stanford CS348K (Bhat, Chu, Taylor).

We use **Ctrl-World** imagined rollouts as behavior-cloning data to adapt **pi0.5** (`pi05_droid`) on real-Franka pick-and-place tasks, reducing reliance on new human teleoperation for every task.

| Resource | Link |
|----------|------|
| **Project site** | https://otaylor2023.github.io/DreamBC/ |
| **Full write-up** | [docs/writeup.html](docs/writeup.html) |
| **Checkpoints (metadata)** | [real_robot_checkpoints/](real_robot_checkpoints/) |
| **Curated media** | [assets/](assets/) → built into [docs/static/](docs/static/) |
| **Results index** | [results/](results/) |

## Headline results (real Franka)

| Task | Base pi0.5 | Ctrl-World FT | Notes |
|------|------------|---------------|--------|
| Pick up cube | 90% | 90% | Smoother trajectories; success saturated |
| Pick up tomato | 60% | 100% | Large gain from synthetic-rollout FT |

Teleoperation-FT baselines and additional rollout videos are documented on the [project site](https://otaylor2023.github.io/DreamBC/writeup.html).

## Pipeline (short)

1. **Snapshots** — real Franka HDF5 / single-frame starts.
2. **Imagined rollouts** — pi0.5 acts in the loop; Ctrl-World predicts multi-camera video (`Ctrl-World/scripts/rollout_interact_pi.py`, `scripts/run_sweep.sh`).
3. **BC bundles** — `(exterior, wrist, joint, gripper) → 15-step action chunk` per policy decision (`scripts/bc_dataset.py`).
4. **Fine-tune** — LoRA (image-aug optional) with EMA-loss threshold early stop (`scripts/train_dreambc.py`, `scripts/run_with_loss_threshold.py`). Full-FT checkpoints exist in the portfolio but were **not real-robot evaluated**.
5. **Deploy** — 42 packaged checkpoints under `real_robot_checkpoints/` (weights gitignored; `meta.json` tracked).

## Repo layout

```
DreamBC/
  docs/                    # GitHub Pages site (index + writeup)
  scripts/                 # training, sweep, portfolio, site media helpers
  real_robot_checkpoints/  # portfolio README, MANIFEST, meta.json, docs/ (weights ignored)
  vendor_overrides/        # our patches to upstream third-party repos
    Ctrl-World/            #   files to overlay on a fresh Ctrl-World clone
  success_lists/           # curated demo IDs
  assets/                  # where to put source media before build_site_videos.py
  results/                 # pointer to evaluation artifacts
  urdf_models/             # Panda + Robotiq assets
```

`Ctrl-World/` is **not** committed to this repo — clone it yourself (see below).
Other large artifacts are gitignored: `rollouts/`, `training_runs/`, checkpoint `params/`, `cube.zip`, `tomato.zip`.

## Quick commands

```bash
# Fine-tune on a BC bundle root (LoRA)
./scripts/run_train_lora.sh rollouts/C16_g5_block my_exp --enable-image-aug

# Threshold-based train + early stop
python scripts/run_with_loss_threshold.py \
  --variant lora --bundles-root rollouts/handcollected_tomato \
  --exp-name handcollected_tomato_lora_aug --enable-image-aug \
  --thresholds 0.10,0.04 --num-train-steps 400 --log-file logs/hand.log

# Eval a packaged checkpoint in sim rollout
./scripts/eval_dreambc_rollout.sh \
  real_robot_checkpoints/tomato/lora_aug_thr004_step200_hand_bowl lora \
  rollouts/eval/smoke_tomato

# Build / refresh GitHub Pages videos (~17 MB under docs/static/videos/)
python3 scripts/build_site_videos.py --copy-images

# Pre-push size check (no commit)
python3 scripts/check_repo_ready.py
```

## Enable GitHub Pages

**Settings → Pages → Deploy from branch → `main` → `/docs`** → https://otaylor2023.github.io/DreamBC/

See [docs/README.md](docs/README.md) for media curation details.

## Note on `Ctrl-World/` (upstream code)

`Ctrl-World/` is the upstream [Ctrl-World](https://github.com/Robotic-AI-Lab/Ctrl-World) world-model codebase. It was briefly vendored into this repo but has since been **removed from version control** (the entire `Ctrl-World/` directory is gitignored). The handful of files DreamBC needs to modify on top of upstream live in [`vendor_overrides/Ctrl-World/`](vendor_overrides/Ctrl-World/), which mirrors the upstream layout so they can be copied in directly.

Files we override (full details in [`vendor_overrides/Ctrl-World/README.md`](vendor_overrides/Ctrl-World/README.md)):

- `scripts/rollout_interact_pi.py` — DreamBC LoRA/full pi0.5 checkpoint loading + per-decision BC bundle outputs (`policy_obs_*`, `policy_state_*`, `policy_action_chunk`).
- `scripts/rollout_replay_traj.py`, `scripts/rollout_key_board.py` — extra CLI flags so the rollouts can be driven from sweep shell scripts.
- `config.py` — bumps `wm_args.guidance_scale` from `1.0` to `7.5` (the value all our checkpoints and result tables were produced with).

To run the DreamBC pipeline, follow the "Setup on a fresh Linux GPU instance" section below, which clones upstream Ctrl-World and then overlays `vendor_overrides/Ctrl-World/` on top.

## Citation

```bibtex
@misc{dreambc2026,
  title={Ctrl-World for Real-Time SimToReal Policy Generalization},
  author={Bhat, Abhijnya and Chu, Wayne and Taylor, Olivia},
  year={2026},
  howpublished={\url{https://otaylor2023.github.io/DreamBC/}}
}
```

---

## Ctrl-World + pi05 imagination sweep

This is the active workstream: run policy-in-the-loop rollouts where pi0.5 acts in the
Ctrl-World imagined environment, starting from single-frame Franka snapshots. Each rollout
writes a BC-ready bundle (raw frames + policy-aligned `(obs, action_chunk)` pairs) under
`rollouts/sweep/<object>/<config_name>/Rollouts_interact_pi/bc_episodes/`.

### Setup on a fresh Linux GPU instance (H100 / A100 80GB recommended)

The repo intentionally does not commit large assets (`Ctrl-World/` itself,
`Ctrl-World/openpi/`, model checkpoints, raw snapshot HDF5s, rollout outputs).
Fetch them out-of-band:

1) Clone this repo:
   ```bash
   git clone git@github.com:otaylor2023/DreamBC.git
   cd DreamBC
   ```

2) Clone upstream Ctrl-World and apply our overrides:
   ```bash
   git clone git@github.com:Robotic-AI-Lab/Ctrl-World.git Ctrl-World
   # overlay our patched files (rollout_interact_pi.py, rollout_replay_traj.py,
   # rollout_key_board.py, config.py); see vendor_overrides/Ctrl-World/README.md
   rsync -a vendor_overrides/Ctrl-World/ Ctrl-World/
   ```

3) Install openpi inside `Ctrl-World/`:
   ```bash
   cd Ctrl-World
   git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
   cd openpi
   pip install uv
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   cd ../..
   ```

4) Download model checkpoints:
   - Ctrl-World ckpt -> `Ctrl-World/checkpoint/checkpoint-10000.pt` (see the table in
     `Ctrl-World/readme.md`).
   - pi05_droid ckpt: follow openpi instructions; symlink or copy to
     `Ctrl-World/openpi/checkpoint/pi05_droid`.
   - SVD + CLIP: HuggingFace will auto-cache on first use; required snapshot hashes
     are referenced in `scripts/run_sweep.sh` (`SVD_PATH`, `CLIP_PATH`).

5) Place raw snapshot HDF5s under:
   ```
   data/snapshots/cube_snapshots/snapshot_*.hdf5
   data/snapshots/tomato_snapshots/snapshot_*.hdf5
   ```
   (transferred via scp / S3 / `cube_snapshots.zip` and `tomato_snapshots.zip`)

6) Build the Ctrl-World-format datasets used by the sweep configs:
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
