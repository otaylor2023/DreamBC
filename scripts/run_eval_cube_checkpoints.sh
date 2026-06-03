#!/usr/bin/env bash
# Roll out multiple training checkpoints in Ctrl-World for comparison.
#
# Usage:
#   ./scripts/run_eval_cube_checkpoints.sh <exp_name> <variant: lora|full> [steps] [val_ids]
#
# Examples:
#   ./scripts/run_eval_cube_checkpoints.sh cube_c16_lora_v0 lora 500,1000,1500
#   ./scripts/run_eval_cube_checkpoints.sh cube_c16_full_v0 full 500,1000,1500,2000,2500
#   ./scripts/run_eval_cube_checkpoints.sh cube_c16_lora_v0 lora 1500 0001,0002,0003
#
# Checkpoints are read from:
#   training_runs/checkpoints/pi05_dreambc/<exp_name>/<step>/
# Eval outputs land at:
#   rollouts/eval/<exp_name>_step<step>/
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <exp_name> <variant: lora|full> [steps] [val_ids]" >&2
  exit 2
fi

EXP_NAME="$1"
VARIANT="$2"
STEPS="${3:-500,1000,1500}"
VAL_IDS="${4:-0001,0002,0003,0004,0005,0006,0007,0008,0009,0010}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_BASE="$REPO_ROOT/training_runs/checkpoints/pi05_dreambc/$EXP_NAME"

if [[ ! -d "$CKPT_BASE" ]]; then
  echo "error: checkpoint dir not found: $CKPT_BASE" >&2
  exit 1
fi

IFS=',' read -ra STEP_LIST <<< "$STEPS"
for STEP in "${STEP_LIST[@]}"; do
  STEP="$(echo "$STEP" | tr -d ' ')"
  CKPT_DIR="$CKPT_BASE/$STEP"
  SAVE_DIR="$REPO_ROOT/rollouts/eval/${EXP_NAME}_step${STEP}"
  if [[ ! -d "$CKPT_DIR/params" ]]; then
    echo "[eval] skip step $STEP — no checkpoint at $CKPT_DIR/params"
    continue
  fi
  echo "[eval $(date -Is)] exp=$EXP_NAME variant=$VARIANT step=$STEP -> $SAVE_DIR"
  "$REPO_ROOT/scripts/eval_dreambc_rollout.sh" \
    "$CKPT_DIR" "$VARIANT" "$SAVE_DIR" "$VAL_IDS"
done

echo "[eval $(date -Is)] done exp=$EXP_NAME steps=$STEPS"
