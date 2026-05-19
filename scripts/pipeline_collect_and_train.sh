#!/usr/bin/env bash
# Collect 100 RMPFlow pick demos, validate, then fine-tune SmolVLA (run inside tmux).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${ROOT}/outputs/demos/logs/pipeline_${STAMP}.log"
COLLECT_LOG="${ROOT}/outputs/demos/logs/collect_100ep_${STAMP}.log"
TRAIN_LOG="${ROOT}/outputs/train/logs/smolvla_pick_cube_${STAMP}.log"
STATUS_FILE="${ROOT}/outputs/demos/logs/pipeline_status.txt"

mkdir -p "${ROOT}/outputs/demos/logs" "${ROOT}/outputs/train/logs"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${PIPELINE_LOG}" "${STATUS_FILE}"
}

log "phase=starting"

log "phase=collect"
log "Collecting 100 episodes -> data/lerobot/dreambc_franka_pick_cube"
if [[ -f "${ROOT}/scripts/isaacsim_shell.sh" ]] && [[ -d "${ISAACSIM_ROOT:-${HOME}/.local/share/ov/pkg}" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/isaacsim_shell.sh"
  COLLECT_PY=(python)
else
  # Pip-installed isaacsim in dreambc conda env (no OV kit path).
  COLLECT_PY=(mamba run -n dreambc python)
fi
"${COLLECT_PY[@]}" "${ROOT}/isaacsim_curobo_collect_dataset.py" --config-name=curobo_pick_cube_dataset \
  2>&1 | tee -a "${COLLECT_LOG}" "${PIPELINE_LOG}"
log "phase=collect_done"

log "phase=validate"
mamba run -n dreambc python "${ROOT}/scripts/validate_lerobot_dataset.py" \
  --root "${ROOT}/data/lerobot/dreambc_franka_pick_cube" \
  --repo-id local/dreambc_franka_pick_cube \
  2>&1 | tee -a "${TRAIN_LOG}" "${PIPELINE_LOG}"
log "phase=validate_done"

log "phase=train"
export DATASET_ROOT="${ROOT}/data/lerobot/dreambc_franka_pick_cube"
export REPO_ID="local/dreambc_franka_pick_cube"
export OUTPUT_DIR="${ROOT}/outputs/train/smolvla_pick_cube"
export STEPS="${STEPS:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
mamba run -n dreambc bash "${ROOT}/scripts/train_smolvla_pick_cube.sh" \
  2>&1 | tee -a "${TRAIN_LOG}" "${PIPELINE_LOG}"
log "phase=train_done"
