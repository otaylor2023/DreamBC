# DreamBC Real-Robot Checkpoint Portfolio

Fine-tuned **pi05_droid** checkpoints for one-shot real-robot evaluation on two pick-and-place tasks trained from sim rollouts in the same lab setup.

Generated: 2026-06-01 14:46 UTC

## Overview

- **20 cube** checkpoints + **22 tomato** checkpoints
- Methods: LoRA (~9 GB each) and full fine-tune (~30 GB each)
- Loss-threshold tagging at EMA 0.10 / 0.04 / 0.015 (plus cube LoRA v0 baseline at step 500)
- See `MANIFEST.json` for machine-readable metadata

## Camera views

| Training bundle key | Dataset key | Model key (pi05) | Role |
|---|---|---|---|
| `policy_obs_exterior` | `observation/exterior_image_1_left` | `base_0_rgb` | Fixed third-person / exterior view |
| `policy_obs_wrist` | `observation/wrist_image_left` | `left_wrist_0_rgb` | Wrist-mounted camera |
| (padded zeros) | — | `right_wrist_0_rgb` | Unused; mask=False |

- Resolution: **224×224 RGB**, uint8 in bundles
- Example frames: [`docs/camera_views/`](docs/camera_views/) (`cube_exterior.png`, `cube_wrist.png`, `tomato_exterior.png`, `tomato_wrist.png`)
- **No image augmentation at inference**

## Architecture

- Base model: **pi0.5 (pi05)** from `gs://openpi-assets/checkpoints/pi05_droid`
- **LoRA**: `gemma_2b_lora` + `gemma_300m_lora`, base weights frozen
- **Full FT**: all parameters trainable
- Action space: DROID 8-D (7 joints + gripper), padded to 32 for pi05
- **Action horizon**: 15 steps per policy query
- Norm stats: `assets/droid/norm_stats.json` inside each checkpoint (from pi05_droid)
- Training augmentation (where noted): mild crop + brightness/contrast on exterior + wrist only

## Prompts at deploy

| Task | Prompt |
|---|---|
| Cube | `put the orange block in the blue bowl` |
| Tomato (most checkpoints) | `Pick the tomato` |
| Tomato (recommended #4 / #6 `_bowl` retrain) | `pick the tomato and place it in the blue bowl` |

## Recommended 6-per-task (one-shot eval)

### Cube

| # | Path | Method | Aug | Loss thr | Rationale |
|---|------|--------|-----|----------|-----------|
| 1 | `real_robot_checkpoints/cube/lora_v0_step500` | lora | noaug | 0.012 | reference baseline |
| 2 | `real_robot_checkpoints/cube/lora_noaug_thr004_step200` | lora | noaug | 0.04 | early-checkpoint hypothesis |
| 3 | `real_robot_checkpoints/cube/lora_aug_thr010_step175` | lora | aug | 0.10 | max generalization |
| 4 | `real_robot_checkpoints/cube/lora_aug_thr004_step200` | lora | aug | 0.04 | likely best overall |
| 5 | `real_robot_checkpoints/cube/full_noaug_thr004_step225` | full | noaug | 0.04 | isolates aug effect |
| 6 | `real_robot_checkpoints/cube/full_aug_thr004_step250` | full | aug | 0.04 | extra capacity + regularized |
| 7 | `real_robot_checkpoints/cube/lora_agent_view_noaug_thr004_step200` | lora | noaug | 0.04 | agent_view exterior (LoRA sweet spot) |
| 8 | `real_robot_checkpoints/cube/lora_agent_view_noaug_thr0015_step400` | lora | noaug | 0.015 | agent_view exterior (near-converged LoRA) |
| 9 | `real_robot_checkpoints/cube/full_agent_view_noaug_thr004_step225` | full | noaug | 0.04 | agent_view exterior (full FT sweet spot) |
| 10 | `real_robot_checkpoints/cube/full_agent_view_noaug_thr0015_step625` | full | noaug | 0.015 | agent_view exterior (near-converged full FT) |

### Tomato

| # | Path | Method | Aug | Loss thr | Rationale |
|---|------|--------|-----|----------|-----------|
| 1 | `real_robot_checkpoints/tomato/lora_noaug_thr004_step200` | lora | noaug | 0.04 | early-checkpoint hypothesis |
| 2 | `real_robot_checkpoints/tomato/lora_aug_thr010_step175` | lora | aug | 0.10 | max generalization |
| 3 | `real_robot_checkpoints/tomato/full_noaug_thr004_step225` | full | noaug | 0.04 | isolates aug effect |
| 4 | `real_robot_checkpoints/tomato/full_aug_thr004_step225_bowl` | full | aug | 0.04 | recommended #4: full FT + aug (bowl prompt) |
| 5 | `real_robot_checkpoints/tomato/lora_noaug_thr0015_step350_bowl` | lora | noaug | 0.015 | recommended #6: near-converged LoRA (bowl prompt) |
| 6 | `real_robot_checkpoints/tomato/lora_agent_view_noaug_thr004_step200_bowl` | lora | noaug | 0.04 | agent_view exterior (LoRA sweet spot, bowl prompt) |
| 7 | `real_robot_checkpoints/tomato/lora_agent_view_noaug_thr0015_step375_bowl` | lora | noaug | 0.015 | agent_view exterior (near-converged LoRA, bowl prompt) |
| 8 | `real_robot_checkpoints/tomato/full_agent_view_noaug_thr004_step225_bowl` | full | noaug | 0.04 | agent_view exterior (full FT sweet spot, bowl prompt) |
| 9 | `real_robot_checkpoints/tomato/full_agent_view_noaug_thr0015_step625_bowl` | full | noaug | 0.015 | agent_view exterior (near-converged full FT, bowl prompt) |

## All checkpoints

### Cube (20)

| Path | Method | Aug | Loss thr | Step | EMA loss | Recommended | Rationale |
|------|--------|-----|----------|------|----------|-------------|-----------|
| `real_robot_checkpoints/cube/lora_v0_step500` | lora | noaug | 0.012 | 500 | — | yes | reference baseline |
| `real_robot_checkpoints/cube/lora_noaug_thr010_step175` | lora | noaug | 0.10 | 175 | — |  |  |
| `real_robot_checkpoints/cube/lora_noaug_thr004_step200` | lora | noaug | 0.04 | 200 | — | yes | early-checkpoint hypothesis |
| `real_robot_checkpoints/cube/lora_noaug_thr0015_step400` | lora | noaug | 0.015 | 400 | — |  |  |
| `real_robot_checkpoints/cube/lora_aug_thr010_step175` | lora | aug | 0.10 | 175 | — | yes | max generalization |
| `real_robot_checkpoints/cube/lora_aug_thr004_step200` | lora | aug | 0.04 | 200 | — | yes | likely best overall |
| `real_robot_checkpoints/cube/lora_aug_thr0015_step475` | lora | aug | 0.015 | 475 | — |  | LoRA late + aug: bracket right side |
| `real_robot_checkpoints/cube/full_noaug_thr010_step175` | full | noaug | 0.10 | 175 | — |  |  |
| `real_robot_checkpoints/cube/full_noaug_thr004_step225` | full | noaug | 0.04 | 225 | — | yes | isolates aug effect |
| `real_robot_checkpoints/cube/full_noaug_thr0015_step625` | full | noaug | 0.015 | 625 | — |  |  |
| `real_robot_checkpoints/cube/full_aug_thr010_step175` | full | aug | 0.10 | 175 | — |  | full FT early + aug: full FT early stop |
| `real_robot_checkpoints/cube/full_aug_thr004_step250` | full | aug | 0.04 | 250 | — | yes | extra capacity + regularized |
| `real_robot_checkpoints/cube/full_aug_thr0015_step650` | full | aug | 0.015 | 650 | — |  |  |
| `real_robot_checkpoints/cube/lora_agent_view_noaug_thr010_step175` | lora | noaug | 0.10 | 175 | 0.056519999999999994 |  | agent_view exterior (LoRA, thr=0.10) |
| `real_robot_checkpoints/cube/lora_agent_view_noaug_thr004_step200` | lora | noaug | 0.04 | 200 | 0.03278 | yes | agent_view exterior (LoRA sweet spot) |
| `real_robot_checkpoints/cube/lora_agent_view_noaug_thr0015_step400` | lora | noaug | 0.015 | 400 | 0.014480000000000002 | yes | agent_view exterior (near-converged LoRA) |
| `real_robot_checkpoints/cube/full_agent_view_noaug_thr010_step175` | full | noaug | 0.10 | 175 | 0.07688 |  | agent_view exterior (full FT, thr=0.10) |
| `real_robot_checkpoints/cube/full_agent_view_noaug_thr004_step225` | full | noaug | 0.04 | 225 | 0.0397 | yes | agent_view exterior (full FT sweet spot) |
| `real_robot_checkpoints/cube/full_agent_view_noaug_thr0015_step625` | full | noaug | 0.015 | 625 | 0.014799999999999999 | yes | agent_view exterior (near-converged full FT) |
| `real_robot_checkpoints/cube/lora_aug_thr004_step225_hand` | lora | aug | 0.04 | 225 | 0.03427999999999999 |  | hand-collected demos; exterior_2 + wrist |

### Tomato (22)

| Path | Method | Aug | Loss thr | Step | EMA loss | Recommended | Rationale |
|------|--------|-----|----------|------|----------|-------------|-----------|
| `real_robot_checkpoints/tomato/lora_noaug_thr010_step175` | lora | noaug | 0.10 | 175 | — |  |  |
| `real_robot_checkpoints/tomato/lora_noaug_thr004_step200` | lora | noaug | 0.04 | 200 | — | yes | early-checkpoint hypothesis |
| `real_robot_checkpoints/tomato/lora_noaug_thr0015_step350` | lora | noaug | 0.015 | 350 | — |  | near-converged baseline (superseded by _bowl retrain) |
| `real_robot_checkpoints/tomato/lora_aug_thr010_step175` | lora | aug | 0.10 | 175 | — | yes | max generalization |
| `real_robot_checkpoints/tomato/lora_aug_thr004_step200` | lora | aug | 0.04 | 200 | — |  | likely best overall (superseded by _bowl retrain) |
| `real_robot_checkpoints/tomato/lora_aug_thr0015_step425` | lora | aug | 0.015 | 425 | — |  | LoRA late + aug: bracket right side |
| `real_robot_checkpoints/tomato/full_noaug_thr010_step175` | full | noaug | 0.10 | 175 | — |  |  |
| `real_robot_checkpoints/tomato/full_noaug_thr004_step225` | full | noaug | 0.04 | 225 | — | yes | isolates aug effect |
| `real_robot_checkpoints/tomato/full_noaug_thr0015_step600` | full | noaug | 0.015 | 600 | — |  |  |
| `real_robot_checkpoints/tomato/full_aug_thr010_step175` | full | aug | 0.10 | 175 | — |  | full FT early + aug: full FT early stop |
| `real_robot_checkpoints/tomato/full_aug_thr004_step225` | full | aug | 0.04 | 225 | — |  | extra capacity + regularized (superseded by _bowl retrain) |
| `real_robot_checkpoints/tomato/full_aug_thr0015_step650` | full | aug | 0.015 | 650 | — |  |  |
| `real_robot_checkpoints/tomato/lora_aug_thr004_step200_bowl` | lora | aug | 0.04 | 200 | 0.032619999999999996 |  | likely best overall (bowl prompt; optional vs #3) |
| `real_robot_checkpoints/tomato/full_aug_thr004_step225_bowl` | full | aug | 0.04 | 225 | 0.038 | yes | recommended #4: full FT + aug (bowl prompt) |
| `real_robot_checkpoints/tomato/lora_noaug_thr0015_step350_bowl` | lora | noaug | 0.015 | 350 | 0.01446 | yes | recommended #6: near-converged LoRA (bowl prompt) |
| `real_robot_checkpoints/tomato/lora_agent_view_noaug_thr010_step175_bowl` | lora | noaug | 0.10 | 175 | 0.06674 |  | agent_view exterior (LoRA, bowl prompt, thr=0.10) |
| `real_robot_checkpoints/tomato/lora_agent_view_noaug_thr004_step200_bowl` | lora | noaug | 0.04 | 200 | 0.032119999999999996 | yes | agent_view exterior (LoRA sweet spot, bowl prompt) |
| `real_robot_checkpoints/tomato/lora_agent_view_noaug_thr0015_step375_bowl` | lora | noaug | 0.015 | 375 | 0.013919999999999998 | yes | agent_view exterior (near-converged LoRA, bowl prompt) |
| `real_robot_checkpoints/tomato/full_agent_view_noaug_thr010_step175_bowl` | full | noaug | 0.10 | 175 | 0.06731999999999999 |  | agent_view exterior (full FT, bowl prompt, thr=0.10) |
| `real_robot_checkpoints/tomato/full_agent_view_noaug_thr004_step225_bowl` | full | noaug | 0.04 | 225 | 0.03752 | yes | agent_view exterior (full FT sweet spot, bowl prompt) |
| `real_robot_checkpoints/tomato/full_agent_view_noaug_thr0015_step625_bowl` | full | noaug | 0.015 | 625 | 0.01472 | yes | agent_view exterior (near-converged full FT, bowl prompt) |
| `real_robot_checkpoints/tomato/lora_aug_thr004_step200_hand_bowl` | lora | aug | 0.04 | 200 | 0.03828 |  | hand-collected demos; exterior_2 + wrist; bowl prompt |

## Training data

| Task | Bundle | Success demos | Notes |
|------|--------|---------------|-------|
| Cube | `rollouts/C16_g5_block` | 16 | Orange block → blue bowl |
| Tomato | `rollouts/TL_g3_red_ball` | 24 | Curated clean-success list (`success_lists/tomato_v1_clean.txt`) |

Threshold-based early stop: tag checkpoints when EMA loss crosses 0.10 / 0.04 / 0.015, then prune extras.

## How to load / eval

Each checkpoint directory contains `params/` and `assets/droid/`. Use the matching train config variant:

```bash
# Sim rollout eval (from repo root)
./scripts/eval_dreambc_rollout.sh \
  real_robot_checkpoints/cube/lora_noaug_thr004_step200 lora \
  rollouts/eval/smoke_cube

./scripts/eval_dreambc_rollout.sh \
  real_robot_checkpoints/tomato/lora_aug_thr004_step200 lora \
  rollouts/eval/smoke_tomato
```

Replace `lora` with `full` for full fine-tune checkpoints. Per-checkpoint `meta.json` includes `variant` and `eval_train_config`.

## Additional docs

- [`docs/REAL_ROBOT_PORTFOLIO_cube.md`](docs/REAL_ROBOT_PORTFOLIO_cube.md)
- [`docs/REAL_ROBOT_PORTFOLIO_tomato.md`](docs/REAL_ROBOT_PORTFOLIO_tomato.md)
- [`docs/camera_views/camera_mapping.md`](docs/camera_views/camera_mapping.md)
