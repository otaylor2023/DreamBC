# Real-Robot Portfolio Tomato v1

Recommended 6-checkpoint selection for one-shot eval:

| # | Checkpoint path | Method | Aug | Loss thr | Rationale |
|---|-----------------|--------|-----|----------|-----------|
| 1 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_noaug/200` | lora | noaug | 0.04 | early-checkpoint hypothesis |
| 2 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/175` | lora | aug | 0.10 | max generalization |
| 3 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/200` | lora | aug | 0.04 | likely best overall |
| 4 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_aug/225` | full | aug | 0.04 | extra capacity + regularized |
| 5 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_noaug/225` | full | noaug | 0.04 | isolates aug effect |
| 6 | `training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_noaug/350` | lora | noaug | 0.015 | near-converged baseline |

## Optional 7th/8th slots

- (7) `training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/425` - LoRA late + aug: bracket right side
- (8) `training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_aug/175` - full FT early + aug: full FT early stop

## All tagged checkpoints per run

### tomato_t24_full_v1_aug
- thr=0.1 step=175 ema=0.0706 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_aug/175`
- thr=0.04 step=225 ema=0.0381 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_aug/225`
- thr=0.015 step=650 ema=0.0144 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_aug/650`

### tomato_t24_full_v1_noaug
- thr=0.1 step=175 ema=0.0900 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_noaug/175`
- thr=0.04 step=225 ema=0.0393 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_noaug/225`
- thr=0.015 step=600 ema=0.0149 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_full_v1_noaug/600`

### tomato_t24_lora_v1_aug
- thr=0.1 step=175 ema=0.0741 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/175`
- thr=0.04 step=200 ema=0.0350 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/200`
- thr=0.015 step=425 ema=0.0150 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_aug/425`

### tomato_t24_lora_v1_noaug
- thr=0.1 step=175 ema=0.0707 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_noaug/175`
- thr=0.04 step=200 ema=0.0319 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_noaug/200`
- thr=0.015 step=350 ema=0.0150 -> `/home/ubuntu/DreamBC/training_runs/checkpoints/pi05_dreambc/tomato_t24_lora_v1_noaug/350`

