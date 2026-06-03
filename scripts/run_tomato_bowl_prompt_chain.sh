#!/usr/bin/env bash
# Detached chain to retrain tomato recommended #4 (LoRA aug) and #6 (full FT aug)
# under the longer prompt "pick the tomato and place it in the blue bowl".
#
# Bundles must already have the new prompt set via:
#   python scripts/set_bundle_instruction.py \
#       --bundles-root rollouts/TL_g3_red_ball \
#       --prompt "pick the tomato and place it in the blue bowl"
#
# Stops at EMA loss 0.04 (the threshold used for #4 and #6).
#
# Usage:
#   ./scripts/run_tomato_bowl_prompt_chain.sh [bundles_root]
set -uo pipefail

BUNDLES_ROOT="${1:-rollouts/TL_g3_red_ball}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_TAG="tomato_bowl_prompt_${TIMESTAMP}"
MASTER_LOG="$REPO_ROOT/logs/${CHAIN_TAG}.log"
MARKER_DIR="$REPO_ROOT/training_runs/chain_markers"

mkdir -p "$REPO_ROOT/logs" "$MARKER_DIR"

if [[ "${DREAMBC_TOMATO_BOWL_DETACHED:-}" != "1" ]]; then
  export DREAMBC_TOMATO_BOWL_DETACHED=1
  echo "[bowl-chain] starting detached orchestrator -> $MASTER_LOG"
  setsid nohup "$0" "$BUNDLES_ROOT" > "$MASTER_LOG" 2>&1 < /dev/null &
  disown
  echo "[bowl-chain] pid=$! master_log=$MASTER_LOG"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_env.sh"
load_env_file "$REPO_ROOT/.env"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[bowl-chain $(date -Is)] error: openpi venv not found at $VENV" >&2
  exit 1
fi

THRESHOLDS="0.10,0.04"
START_MARK="$MARKER_DIR/${CHAIN_TAG}.start"
DONE_MARK="$MARKER_DIR/${CHAIN_TAG}.done"
FAIL_MARK="$MARKER_DIR/${CHAIN_TAG}.failed"
date -Is > "$START_MARK"
echo "[bowl-chain $(date -Is)] START chain=$CHAIN_TAG bundles=$BUNDLES_ROOT"

run_one() {
  local variant="$1"
  local exp_name="$2"
  local max_steps="$3"
  local enable_aug="$4"
  local done_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.done"
  local fail_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.failed"
  local run_log="$REPO_ROOT/logs/${exp_name}.log"

  echo "[bowl-chain $(date -Is)] START $exp_name (variant=$variant max_steps=$max_steps aug=$enable_aug)"

  local -a extra_args=(--overwrite)
  if [[ "$enable_aug" == "true" ]]; then
    extra_args+=(--enable-image-aug)
  fi

  "$VENV/bin/python" -u "$REPO_ROOT/scripts/run_with_loss_threshold.py" \
    --variant "$variant" \
    --bundles-root "$BUNDLES_ROOT" \
    --exp-name "$exp_name" \
    --thresholds "$THRESHOLDS" \
    --num-train-steps "$max_steps" \
    --save-interval 25 \
    --keep-period 25 \
    --log-interval 25 \
    --log-file "$run_log" \
    "${extra_args[@]}"
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    date -Is > "$fail_mark"
    echo "[bowl-chain $(date -Is)] FAILED $exp_name rc=$rc (see $run_log)" | tee -a "$FAIL_MARK"
    return "$rc"
  fi

  date -Is > "$done_mark"
  echo "[bowl-chain $(date -Is)] DONE $exp_name"
  return 0
}

run_one lora tomato_t24_lora_bowl_aug 400 true || exit $?
run_one full tomato_t24_full_bowl_aug 400 true || exit $?

date -Is > "$DONE_MARK"
echo "[bowl-chain $(date -Is)] CHAIN COMPLETE. master_log=$MASTER_LOG"
