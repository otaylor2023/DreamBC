# DreamBC project site (GitHub Pages)

Static site for the CS348K final project **Ctrl-World for Real-Time SimToReal Policy Generalization**.

| Page | URL (after enabling Pages) |
|------|----------------------------|
| Landing | https://otaylor2023.github.io/DreamBC/ |
| Full writeup | https://otaylor2023.github.io/DreamBC/writeup.html |

## Enable GitHub Pages

1. Push this repo to GitHub (`otaylor2023/DreamBC`).
2. **Settings → Pages → Build and deployment**
3. Source: **Deploy from a branch**
4. Branch: **`main`** (or your default branch), folder **`/docs`**
5. Save. The site updates within a few minutes.

## Regenerate media

Large videos are not stored in git sources; curate them into `docs/static/`:

```bash
# From repo root (requires ffmpeg)
python3 scripts/build_site_videos.py --copy-images
```

Replace placeholders when you have real robot rollouts:

- `docs/static/videos/policy_rollout/<task>_after_01.mp4` — Ctrl-World finetuned policy
- `docs/static/videos/teleop_baseline/<task>_teleop_01.mp4` — teleop-FT policy

Then re-run `build_site_videos.py` or overwrite files directly and commit.

## Pre-push check

```bash
python3 scripts/check_repo_ready.py
```
