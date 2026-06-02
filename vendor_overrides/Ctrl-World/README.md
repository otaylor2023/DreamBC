# Ctrl-World local overrides

This directory mirrors the upstream [Ctrl-World](https://github.com/Robotic-AI-Lab/Ctrl-World)
layout and contains the **only files DreamBC modifies** on top of upstream. The upstream
repo is no longer vendored into this project; clone it yourself and drop these files in.

## What's in here

| Path | Purpose |
|------|---------|
| `scripts/rollout_interact_pi.py` | Required by DreamBC. Adds (a) loading of fine-tuned `dreambc_lora` / `dreambc_full` pi0.5 checkpoints via a custom `TrainConfig`, and (b) per-decision BC bundle outputs (`policy_obs_exterior`, `policy_obs_wrist`, `policy_state_joint`, `policy_state_gripper`, `policy_action_chunk`) written under `<save_dir>/<task>/bc_episodes/...`. Consumed by `scripts/bc_dataset.py` and every `scripts/run_*.sh` sweep. |
| `scripts/rollout_replay_traj.py` | Adds CLI flags (`--val_dataset_dir`, `--val_ids`, `--start_idxs`, `--save_dir`, `--data_stat_path`, `--interact_num`, `--pred_step`, `--output_fps`) so replay rollouts can be parameterised from a shell script instead of editing `config.py`. |
| `scripts/rollout_key_board.py` | Same CLI-flag additions as the replay script, for keyboard-driven rollouts. |
| `config.py` | Single-line behaviour change: `wm_args.guidance_scale = 7.5` (upstream default is `1.0`). This is the value all our checkpoints and result tables were produced with. |

## How to set up Ctrl-World after cloning DreamBC

`Ctrl-World/` is gitignored in this repo, so you'll have an empty path after cloning.
Do the following on a GPU machine:

```bash
# from the DreamBC repo root
git clone git@github.com:Robotic-AI-Lab/Ctrl-World.git Ctrl-World

# overlay our local edits on top (overwrites the 4 files listed above)
rsync -a vendor_overrides/Ctrl-World/ Ctrl-World/

# then continue with upstream setup: install openpi inside Ctrl-World/openpi,
# fetch checkpoint-10000.pt and pi05_droid, etc. See the top-level README.md
# "Setup on a fresh Linux GPU instance" section.
```

If you'd rather see the deltas as a patch before applying, run
`diff -ru Ctrl-World/ vendor_overrides/Ctrl-World/` after the clone.

## Re-pulling upstream changes later

If upstream Ctrl-World updates one of these files, `git pull` inside `Ctrl-World/`
will conflict with our overrides. Resolve by hand and then refresh this directory:

```bash
cp Ctrl-World/scripts/rollout_interact_pi.py vendor_overrides/Ctrl-World/scripts/
cp Ctrl-World/scripts/rollout_replay_traj.py vendor_overrides/Ctrl-World/scripts/
cp Ctrl-World/scripts/rollout_key_board.py   vendor_overrides/Ctrl-World/scripts/
cp Ctrl-World/config.py                      vendor_overrides/Ctrl-World/
```
