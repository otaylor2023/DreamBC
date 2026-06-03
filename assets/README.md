# Assets (curated media)

Large binary inputs live outside git (zips, rollouts, checkpoints). **Curated** clips and images for the project site are built into [`docs/static/`](../docs/static/) via:

```bash
python scripts/build_site_videos.py
```

| Source | Location on disk | Site output |
|--------|------------------|-------------|
| Hand-collected teleop | `cube.zip`, `tomato.zip` | `docs/static/videos/real_demo/` |
| Ctrl-World imagined | `Ctrl-World/rollouts/`, `rollouts/sweep/` | `docs/static/videos/dream_rollout/`, `dream_sweep/` |
| Finetuned policy (you add) | — | `docs/static/videos/policy_rollout/` |
| Teleop-FT baseline (you add) | — | `docs/static/videos/teleop_baseline/` |

PNG figures: copy from `real_robot_checkpoints/docs/` into `docs/static/images/` (see `build_site_videos.py --copy-images`).

The published site lives at <https://otaylor2023.github.io/DreamBC/>.
