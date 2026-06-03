# Real-Robot Portfolio v1

Recommended 6-checkpoint selection for one-shot eval:

| # | Checkpoint path | Method | Aug | Loss thr | Rationale |
|---|-----------------|--------|-----|----------|-----------|
| 1 | `training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v0/500` | lora | noaug | ~0.012 | reference |
| 2 | `training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_noaug/200` | lora | noaug | 0.04 | early-checkpoint hypothesis |
| 3 | `training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/175` | lora | aug | 0.10 | max generalization |
| 4 | `training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/200` | lora | aug | 0.04 | likely best overall |
| 5 | `training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_aug/250` | full | aug | 0.04 | extra capacity + regularized |
| 6 | `training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_noaug/225` | full | noaug | 0.04 | isolates aug effect |

## Optional 7th/8th slots

- (7) `training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/475` — LoRA late + aug: bracket right side
- (8) `training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_aug/175` — full FT early + aug: full FT early stop

## All tagged checkpoints per run

### cube_c16_full_v0
- (no manifest yet)
### cube_c16_full_v1_aug
- thr=0.1 step=175 ema=0.0836 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_aug/175`
- thr=0.04 step=250 ema=0.0365 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_aug/250`
- thr=0.015 step=650 ema=0.0148 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_aug/650`

### cube_c16_full_v1_noaug
- thr=0.1 step=175 ema=0.0813 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_noaug/175`
- thr=0.04 step=225 ema=0.0395 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_noaug/225`
- thr=0.015 step=625 ema=0.0145 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_full_v1_noaug/625`

### cube_c16_lora_v0
- (no manifest yet)
### cube_c16_lora_v1_aug
- thr=0.1 step=175 ema=0.0741 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/175`
- thr=0.04 step=200 ema=0.0361 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/200`
- thr=0.015 step=475 ema=0.0150 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_aug/475`

### cube_c16_lora_v1_noaug
- thr=0.1 step=175 ema=0.0759 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_noaug/175`
- thr=0.04 step=200 ema=0.0342 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_noaug/200`
- thr=0.015 step=400 ema=0.0143 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v1_noaug/400`

