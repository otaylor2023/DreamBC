#!/usr/bin/env bash
# Smoke-test the LoRA and (optionally) full fine-tune launchers.
#
# Trains for 2 steps on a tiny subset of the supplied bundles directory.
# Exits non-zero if either run fails to take a step.
#
# Usage:
#   ./scripts/smoke_train.sh [bundles_root]
#
# Defaults bundles_root to rollouts/C16_g5_block.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLES_ROOT="${1:-$REPO_ROOT/rollouts/C16_g5_block}"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"

if [[ ! -d "$BUNDLES_ROOT" ]]; then
  echo "error: bundles root does not exist: $BUNDLES_ROOT" >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: openpi venv not found at $VENV" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

SMOKE_BASE="$REPO_ROOT/training_runs/smoke"
mkdir -p "$SMOKE_BASE"

run_smoke() {
  local variant="$1"
  local exp="smoke_${variant}_$(date +%Y%m%d_%H%M%S)"
  echo
  echo "======================================================================"
  echo "[smoke] running variant=$variant exp=$exp"
  echo "======================================================================"
  python "$REPO_ROOT/scripts/train_dreambc.py" \
    --variant "$variant" \
    --bundles-root "$BUNDLES_ROOT" \
    --exp-name "$exp" \
    --filter-mode all \
    --max-episodes 2 \
    --num-train-steps 2 \
    --batch-size 1 \
    --num-workers 0 \
    --log-interval 1 \
    --save-interval 1 \
    --keep-period 0 \
    --warmup-steps 1 \
    --peak-lr 1e-5 \
    --decay-lr 1e-5 \
    --checkpoint-base-dir "$SMOKE_BASE/checkpoints" \
    --assets-base-dir "$SMOKE_BASE/assets" \
    --no-wandb --overwrite
}

if [[ "${SMOKE_VARIANT:-both}" == "both" ]]; then
  run_smoke lora
  run_smoke full
elif [[ "${SMOKE_VARIANT}" == "lora" || "${SMOKE_VARIANT}" == "full" ]]; then
  run_smoke "$SMOKE_VARIANT"
else
  echo "error: SMOKE_VARIANT must be lora|full|both (got $SMOKE_VARIANT)" >&2
  exit 2
fi

echo
echo "[smoke] OK"
