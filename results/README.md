# Results

Machine-readable and human-readable **evaluation outputs** (not multi-GB checkpoint weights):

| Artifact | Path |
|----------|------|
| Checkpoint portfolio (42 variants) | [`real_robot_checkpoints/`](../real_robot_checkpoints/) |
| Per-checkpoint metadata | `real_robot_checkpoints/{cube,tomato}/*/meta.json` |
| Portfolio manifest | `real_robot_checkpoints/MANIFEST.json` |
| Training example exports | `real_robot_checkpoints/docs/handcollected_training_examples/` |
| Public site | https://otaylor2023.github.io/DreamBC/ |
| Full writeup (rendered) | https://otaylor2023.github.io/DreamBC/writeup.html |

Checkpoint **weights** (`params/`, `train_state/`) stay on the training machine and are gitignored.
