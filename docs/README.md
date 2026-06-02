# DreamBC project site (GitHub Pages)

Static site for the CS348K final project **DreamBC: World Models for Robot Policy Improvement via Imagined Rollouts**.

| Page | URL (after enabling Pages) |
|------|----------------------------|
| Landing | https://otaylor2023.github.io/DreamBC/ |
| Full report | https://otaylor2023.github.io/DreamBC/writeup.html |

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

### Trajectory comparison (Ex 3 slots)

The landing page and full report use per-column **Ex 1 / Ex 2 / Ex 3** selectors. Example 3 currently shows `static/images/placeholder.svg` until you add:

| Folder | Files to add |
|--------|----------------|
| `real_demo/` | `cube_demo_03.mp4`, `tomato_demo_03.mp4` |
| `dream_rollout/` | `cube_dream_03.mp4`, `tomato_dream_03.mp4` |
| `policy_rollout/` | `cube_after_03.mp4`, `tomato_after_03.mp4` |

After uploading a `_03.mp4`, swap the matching `<img data-ex="3">` for a `<video data-ex="3" src="..." muted loop playsinline hidden>` in `index.html` and `writeup.html` (or keep the same path and only change the tag).

**Note:** Existing `policy_rollout/*_after_01.mp4` and `*_02.mp4` are ~6 KB stubs. Re-render all four on the GPU instance when you have real finetuned rollouts.

Then re-run `build_site_videos.py` or overwrite files directly and commit.

## Pre-push check

```bash
python3 scripts/check_repo_ready.py
```
