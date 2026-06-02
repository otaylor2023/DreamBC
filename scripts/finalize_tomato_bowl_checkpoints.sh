#!/usr/bin/env bash
# Move tomato bowl-prompt retrain checkpoints into real_robot_checkpoints/ and refresh docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEST="$REPO_ROOT/real_robot_checkpoints/tomato"
SRC_ROOT="$REPO_ROOT/training_runs/checkpoints/pi05_dreambc"

declare -A MOVES=(
  ["tomato_t24_lora_bowl_aug/200"]="lora_aug_thr004_step200_bowl"
  ["tomato_t24_full_bowl_aug/225"]="full_aug_thr004_step225_bowl"
)

for rel in "${!MOVES[@]}"; do
  src="$SRC_ROOT/$rel"
  dst="$DEST/${MOVES[$rel]}"
  if [[ ! -d "$src/params" ]]; then
    echo "error: missing checkpoint $src" >&2
    exit 1
  fi
  if [[ -e "$dst" ]]; then
    echo "error: destination exists: $dst" >&2
    exit 1
  fi
  echo "[finalize] mv $src -> $dst"
  mv "$src" "$dst"
done

python3 "$REPO_ROOT/scripts/build_portfolio_manifest.py" \
  --repo-root "$REPO_ROOT" \
  --dest-root "real_robot_checkpoints"

python3 "$REPO_ROOT/scripts/set_bundle_instruction.py" \
  --bundles-root rollouts/TL_g3_red_ball \
  --prompt "Pick the tomato"

echo "[finalize] restored bundle prompts to 'Pick the tomato'"
echo "[finalize] updated MANIFEST.json and README.md"
