# Dataset generation timing

Timing is measured from the matched post-training evaluation runs and extrapolated to the dataset sizes used for training.

| Task | Dataset | Measured configs | Approx seconds/video | Extrapolated total |
| --- | --- | ---: | ---: | ---: |
| cube | C16_g5_block (60 videos) | 4 | 135.6 | 2.3 hr |
| tomato | TL_g3_red_ball (100 videos) | 3 | 135.5 | 3.8 hr |

Per-config timings:

- cube_ctrl_world_lora_g5: 10 videos in 22.7 min (136.4 sec/video), rc=0
- cube_teleop_lora_g5: 10 videos in 22.6 min (135.4 sec/video), rc=0
- tomato_base_pi05_bowl_g3: 10 videos in 22.5 min (135.0 sec/video), rc=0
- tomato_ctrl_world_lora_bowl_g3: 10 videos in 22.6 min (135.6 sec/video), rc=0
- tomato_teleop_lora_bowl_g3: 10 videos in 22.6 min (135.8 sec/video), rc=0
- cube_base_pi05_baseline_g3: 10 videos in 22.6 min (135.4 sec/video), rc=0
- cube_base_pi05_baseline_g5: 10 videos in 22.5 min (135.1 sec/video), rc=0
