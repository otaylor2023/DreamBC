#!/usr/bin/env bash
# Run LoRA training, and on success run full fine-tune immediately after.
#
# Usage:
#   ./scripts/run_train_lora_then_full.sh <bundles_root> <lora_exp_name> <full_exp_name>
#
# Detached overnight launch (recommended):
#   setsid nohup ./scripts/run_train_lora_then_full.sh \
#       rollouts/C16_g5_block cube_c16_lora_v0 cube_c16_full_v0 \
#       > logs/chain_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <bundles_root> <lora_exp_name> <full_exp_name>" >&2
  exit 2
fi

BUNDLES_ROOT="$1"
LORA_EXP="$2"
FULL_EXP="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/training_runs/chain_markers"

CHAIN_TAG="${LORA_EXP}__then__${FULL_EXP}"
LORA_LOG="$REPO_ROOT/logs/${LORA_EXP}.log"
FULL_LOG="$REPO_ROOT/logs/${FULL_EXP}.log"
START_MARK="$REPO_ROOT/training_runs/chain_markers/${CHAIN_TAG}.start"
LORA_DONE="$REPO_ROOT/training_runs/chain_markers/${CHAIN_TAG}.lora_done"
FULL_DONE="$REPO_ROOT/training_runs/chain_markers/${CHAIN_TAG}.full_done"
LORA_FAIL="$REPO_ROOT/training_runs/chain_markers/${CHAIN_TAG}.lora_failed"

date -Is > "$START_MARK"
echo "[chain $(date -Is)] START LoRA: $LORA_EXP (bundles=$BUNDLES_ROOT)"

"$REPO_ROOT/scripts/run_train_lora.sh" \
  "$BUNDLES_ROOT" "$LORA_EXP" \
  > "$LORA_LOG" 2>&1
LORA_RC=$?

if [[ $LORA_RC -ne 0 ]]; then
  echo "[chain $(date -Is)] LoRA FAILED with rc=$LORA_RC. NOT starting full FT. See $LORA_LOG" | tee -a "$LORA_FAIL"
  exit "$LORA_RC"
fi

date -Is > "$LORA_DONE"
echo "[chain $(date -Is)] LoRA done. Starting full FT: $FULL_EXP"

"$REPO_ROOT/scripts/run_train_full.sh" \
  "$BUNDLES_ROOT" "$FULL_EXP" \
  > "$FULL_LOG" 2>&1
FULL_RC=$?

if [[ $FULL_RC -ne 0 ]]; then
  echo "[chain $(date -Is)] Full FT FAILED with rc=$FULL_RC. See $FULL_LOG" >&2
  exit "$FULL_RC"
fi

date -Is > "$FULL_DONE"
echo "[chain $(date -Is)] CHAIN COMPLETE. LoRA -> $LORA_LOG, full -> $FULL_LOG"

# Run eval rollouts for all saved checkpoints from both runs.
if [[ -x "$REPO_ROOT/scripts/run_eval_cube_checkpoints.sh" ]]; then
  echo "[chain $(date -Is)] Starting post-training eval rollouts..."
  "$REPO_ROOT/scripts/run_eval_cube_checkpoints.sh" \
    "$LORA_EXP" lora 500,1000,1500 \
    >> "$REPO_ROOT/logs/${LORA_EXP}_eval.log" 2>&1 || true
  "$REPO_ROOT/scripts/run_eval_cube_checkpoints.sh" \
    "$FULL_EXP" full 500,1000,1500,2000,2500 \
    >> "$REPO_ROOT/logs/${FULL_EXP}_eval.log" 2>&1 || true
  echo "[chain $(date -Is)] Eval rollouts finished (see logs/*_eval.log)"
fi
