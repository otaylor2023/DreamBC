#!/usr/bin/env bash
# Sequential agent_view + wrist training chain.
#
# Builds derived BC bundles where policy_obs_exterior comes from Control-World
# agent_view instead of exterior_3, then trains:
#   1. cube LoRA
#   2. tomato LoRA (bowl prompt)
#   3. cube full fine-tune
#   4. tomato full fine-tune (bowl prompt)
#
# Usage:
#   ./scripts/run_agent_view_portfolio_chain.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_TAG="agent_view_portfolio_${TIMESTAMP}"
MASTER_LOG="$REPO_ROOT/logs/${CHAIN_TAG}.log"
MARKER_DIR="$REPO_ROOT/training_runs/chain_markers"

mkdir -p "$REPO_ROOT/logs" "$MARKER_DIR"

if [[ "${DREAMBC_AGENT_VIEW_DETACHED:-}" != "1" ]]; then
  export DREAMBC_AGENT_VIEW_DETACHED=1
  echo "[agent-view] starting detached orchestrator -> $MASTER_LOG"
  setsid nohup "$0" > "$MASTER_LOG" 2>&1 < /dev/null &
  disown
  echo "[agent-view] pid=$! master_log=$MASTER_LOG"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_env.sh"
load_env_file "$REPO_ROOT/.env"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[agent-view $(date -Is)] error: openpi venv not found at $VENV" >&2
  exit 1
fi

date -Is > "$MARKER_DIR/${CHAIN_TAG}.start"
echo "[agent-view $(date -Is)] START chain=$CHAIN_TAG"

echo "[agent-view $(date -Is)] building derived agent_view bundles"
"$VENV/bin/python" "$REPO_ROOT/scripts/build_agent_view_bundles.py" \
  --src-root "$REPO_ROOT/rollouts/C16_g5_block" \
  --dst-root "$REPO_ROOT/rollouts/C16_g5_block_agent_view" \
  --overwrite || exit $?

"$VENV/bin/python" "$REPO_ROOT/scripts/build_agent_view_bundles.py" \
  --src-root "$REPO_ROOT/rollouts/TL_g3_red_ball" \
  --dst-root "$REPO_ROOT/rollouts/TL_g3_red_ball_agent_view" \
  --prompt "pick the tomato and place it in the blue bowl" \
  --overwrite || exit $?

run_one() {
  local variant="$1"
  local bundles_root="$2"
  local exp_name="$3"
  local max_steps="$4"
  local warmup="$5"
  local todo_label="$6"
  local run_log="$REPO_ROOT/logs/${exp_name}.log"
  local done_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.done"
  local fail_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.failed"

  echo "[agent-view $(date -Is)] START $todo_label: $exp_name variant=$variant"

  "$VENV/bin/python" -u "$REPO_ROOT/scripts/run_with_loss_threshold.py" \
    --variant "$variant" \
    --bundles-root "$bundles_root" \
    --exp-name "$exp_name" \
    --thresholds "0.10,0.04,0.015" \
    --num-train-steps "$max_steps" \
    --save-interval 25 \
    --keep-period 25 \
    --log-interval 25 \
    --warmup-steps "$warmup" \
    --log-file "$run_log" \
    --overwrite
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    date -Is > "$fail_mark"
    echo "[agent-view $(date -Is)] FAILED $exp_name rc=$rc (see $run_log)"
    return "$rc"
  fi

  date -Is > "$done_mark"
  echo "[agent-view $(date -Is)] DONE $exp_name"
}

run_one lora "$REPO_ROOT/rollouts/C16_g5_block_agent_view" \
  cube_c16_lora_agent_view_noaug 600 40 "cube LoRA" || exit $?

run_one lora "$REPO_ROOT/rollouts/TL_g3_red_ball_agent_view" \
  tomato_t24_lora_agent_view_noaug 800 40 "tomato LoRA" || exit $?

run_one full "$REPO_ROOT/rollouts/C16_g5_block_agent_view" \
  cube_c16_full_agent_view_noaug 800 80 "cube full FT" || exit $?

run_one full "$REPO_ROOT/rollouts/TL_g3_red_ball_agent_view" \
  tomato_t24_full_agent_view_noaug 1000 80 "tomato full FT" || exit $?

date -Is > "$MARKER_DIR/${CHAIN_TAG}.done"
echo "[agent-view $(date -Is)] CHAIN COMPLETE. master_log=$MASTER_LOG"
