#!/usr/bin/env bash
# Fine-tune SmolVLA in a detached tmux session (collect pipeline calls this after demos).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${DREAMBC_TRAIN_TMUX_SESSION:-dreambc_train}"
TMUX_SOCK="${DREAMBC_TMUX_SOCKET:-${ROOT}/.tmux/dreambc.sock}"
mkdir -p "$(dirname "${TMUX_SOCK}")"

if tmux -S "${TMUX_SOCK}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux train session already running: ${SESSION}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/outputs/train/logs/smolvla_pick_cube_${STAMP}.log"
mkdir -p "${ROOT}/outputs/train/logs"

tmux -S "${TMUX_SOCK}" new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && DATASET_ROOT='${ROOT}/data/lerobot/dreambc_franka_pick_cube' \
   REPO_ID='local/dreambc_franka_pick_cube' \
   OUTPUT_DIR='${ROOT}/outputs/train/smolvla_pick_cube' \
   STEPS='${STEPS:-20000}' BATCH_SIZE='${BATCH_SIZE:-8}' SAVE_FREQ='${SAVE_FREQ:-5000}' \
   mamba run -n dreambc bash scripts/train_smolvla_pick_cube.sh 2>&1 | tee '${LOG}'"

echo "Training tmux: ${SESSION} (socket ${TMUX_SOCK})"
echo "  tmux -S '${TMUX_SOCK}' attach -t ${SESSION}"
echo "  tail -f '${LOG}'"
