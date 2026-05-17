#!/usr/bin/env bash
# Fine-tune SmolVLA on RMPFlow/Lula-collected Franka pick-cube demos (no Isaac Sim required).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/lerobot/dreambc_franka_pick_cube}"
REPO_ID="${REPO_ID:-local/dreambc_franka_pick_cube}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/train/smolvla_pick_cube}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"

if [[ ! -f "${ROOT}/models/smolvla_base/model.safetensors" ]]; then
  echo "Missing weights. Run:" >&2
  echo "  huggingface-cli download lerobot/smolvla_base --local-dir ${ROOT}/models/smolvla_base" >&2
  exit 1
fi

# LeRobot refuses to start if output_dir exists (unless --resume=true). Do not mkdir here.
if [[ -d "${OUTPUT_DIR}" ]] && [[ "${RESUME:-false}" != "true" ]]; then
  if [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
    echo "Output dir exists: ${OUTPUT_DIR}. Set RESUME=true or remove it." >&2
    exit 1
  fi
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

RESUME_ARGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
  RESUME_ARGS=(--resume=true)
fi

lerobot-train \
  "${RESUME_ARGS[@]}" \
  --policy.path="${POLICY_PATH}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=smolvla_pick_cube \
  --policy.device=cuda \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}"

echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
