#!/usr/bin/env bash
# Detached chain: LoRA-aug fine-tunes on hand-collected cube (30) and tomato (24) bundles.
#
# Matches tomato bowl LoRA settings: thresholds 0.10,0.04; max 400 steps; image aug on.
#
# Usage:
#   ./scripts/run_handcollected_lora_chain.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_TAG="handcollected_lora_${TIMESTAMP}"
MASTER_LOG="$REPO_ROOT/logs/${CHAIN_TAG}.log"
MARKER_DIR="$REPO_ROOT/training_runs/chain_markers"

CUBE_BUNDLES="$REPO_ROOT/rollouts/handcollected_cube"
TOMATO_BUNDLES="$REPO_ROOT/rollouts/handcollected_tomato"

mkdir -p "$REPO_ROOT/logs" "$MARKER_DIR"

if [[ "${DREAMBC_HANDCOLLECTED_DETACHED:-}" != "1" ]]; then
  export DREAMBC_HANDCOLLECTED_DETACHED=1
  echo "[hand-lora-chain] starting detached orchestrator -> $MASTER_LOG"
  setsid nohup "$0" > "$MASTER_LOG" 2>&1 < /dev/null &
  disown
  echo "[hand-lora-chain] pid=$! master_log=$MASTER_LOG"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_env.sh"
load_env_file "$REPO_ROOT/.env"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[hand-lora-chain $(date -Is)] error: openpi venv not found at $VENV" >&2
  exit 1
fi

THRESHOLDS="0.10,0.04"
START_MARK="$MARKER_DIR/${CHAIN_TAG}.start"
DONE_MARK="$MARKER_DIR/${CHAIN_TAG}.done"
FAIL_MARK="$MARKER_DIR/${CHAIN_TAG}.failed"
date -Is > "$START_MARK"
echo "[hand-lora-chain $(date -Is)] START chain=$CHAIN_TAG"

run_one() {
  local variant="$1"
  local exp_name="$2"
  local bundles_root="$3"
  local max_steps="$4"
  local done_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.done"
  local fail_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.failed"
  local run_log="$REPO_ROOT/logs/${exp_name}.log"

  echo "[hand-lora-chain $(date -Is)] START $exp_name bundles=$bundles_root"

  "$VENV/bin/python" -u "$REPO_ROOT/scripts/run_with_loss_threshold.py" \
    --variant "$variant" \
    --bundles-root "$bundles_root" \
    --exp-name "$exp_name" \
    --thresholds "$THRESHOLDS" \
    --num-train-steps "$max_steps" \
    --save-interval 25 \
    --keep-period 25 \
    --log-interval 25 \
    --log-file "$run_log" \
    --enable-image-aug \
    --overwrite
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    date -Is > "$fail_mark"
    echo "[hand-lora-chain $(date -Is)] FAILED $exp_name rc=$rc (see $run_log)" | tee -a "$FAIL_MARK"
    return "$rc"
  fi

  date -Is > "$done_mark"
  echo "[hand-lora-chain $(date -Is)] DONE $exp_name"
  return 0
}

run_one lora handcollected_cube_lora_aug "$CUBE_BUNDLES" 400 || exit $?
run_one lora handcollected_tomato_lora_aug "$TOMATO_BUNDLES" 400 || exit $?

date -Is > "$DONE_MARK"
echo "[hand-lora-chain $(date -Is)] CHAIN COMPLETE. master_log=$MASTER_LOG"
