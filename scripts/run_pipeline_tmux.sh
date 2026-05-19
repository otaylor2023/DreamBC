#!/usr/bin/env bash
# Start full collect -> train pipeline in a detached tmux session.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${DREAMBC_TMUX_SESSION:-dreambc_pipeline}"
TMUX_SOCK="${DREAMBC_TMUX_SOCKET:-${ROOT}/.tmux/dreambc.sock}"
mkdir -p "$(dirname "${TMUX_SOCK}")"
TMUX=(tmux -S "${TMUX_SOCK}")

if "${TMUX[@]}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "  tmux attach -t ${SESSION}" >&2
  exit 1
fi

chmod +x "${ROOT}/scripts/pipeline_collect_and_train.sh"
"${TMUX[@]}" new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && exec bash scripts/pipeline_collect_and_train.sh"

echo "Started tmux session: ${SESSION}"
echo "  tmux -S '${TMUX_SOCK}' attach -t ${SESSION}"
echo "  tail -f outputs/demos/logs/pipeline_status.txt"
