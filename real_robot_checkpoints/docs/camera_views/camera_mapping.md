# Camera mapping

Training bundles store observations in `episode.npz`:

| Bundle npz key | Dataset key | Model key (pi05) | Role |
|---|---|---|---|
| `policy_obs_exterior` | `observation/exterior_image_1_left` | `base_0_rgb` | Fixed third-person / exterior view |
| `policy_obs_wrist` | `observation/wrist_image_left` | `left_wrist_0_rgb` | Wrist-mounted camera |
| (padded zeros) | — | `right_wrist_0_rgb` | Unused; mask=False |

- Resolution: **224×224 RGB**, uint8 in bundles
- At inference: same keys and resolution; **no image augmentation**
