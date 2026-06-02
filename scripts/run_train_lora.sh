#!/usr/bin/env bash
# Launch a LoRA fine-tune of pi05_droid on a DreamBC bundle directory.
#
# Usage:
#   ./scripts/run_train_lora.sh <bundles_root> <exp_name> [extra args to train_dreambc.py ...]
#
# Examples:
#   ./scripts/run_train_lora.sh rollouts/C16_g5_block cube_c16_lora_v0
#   ./scripts/run_train_lora.sh rollouts/C16_g5_block cube_c16_lora_v0 \
#       --num-train-steps 2000 --batch-size 32
#
# Defaults to filter_mode=success_true so it only trains on episodes with
# success=true (run scripts/apply_success_list.py first to label them).
# Metrics log to the DreamBC wandb project when WANDB_API_KEY is in .env.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <bundles_root> <exp_name> [extra args...]" >&2
  exit 2
fi

BUNDLES_ROOT="$1"; shift
EXP_NAME="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"

# Pick up WANDB_API_KEY and friends if .env exists.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_env.sh"
load_env_file "$REPO_ROOT/.env"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: openpi venv not found at $VENV" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Make sure jax/cuda can see the H100.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

exec python "$REPO_ROOT/scripts/train_dreambc.py" \
  --variant lora \
  --bundles-root "$BUNDLES_ROOT" \
  --exp-name "$EXP_NAME" \
  --num-train-steps 1500 \
  --batch-size 32 \
  --num-workers 4 \
  --warmup-steps 150 \
  --peak-lr 5e-5 \
  --decay-lr 5e-6 \
  --log-interval 25 \
  --save-interval 250 \
  --keep-period 500 \
  --filter-mode success_true \
  "$@"
