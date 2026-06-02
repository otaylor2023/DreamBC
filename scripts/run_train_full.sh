#!/usr/bin/env bash
# Launch a *full* fine-tune of pi05_droid on a DreamBC bundle directory.
#
# Usage:
#   ./scripts/run_train_full.sh <bundles_root> <exp_name> [extra args to train_dreambc.py ...]
#
# Examples:
#   ./scripts/run_train_full.sh rollouts/C16_g5_block cube_c16_full_v0
#   ./scripts/run_train_full.sh rollouts/C16_g5_block cube_c16_full_v0 \
#       --batch-size 16 --num-train-steps 3000
#
# Full fine-tuning consumes substantially more GPU memory than LoRA. The
# default batch-size of 16 fits in a single H100-80GB with FSDP off; lower
# it if you OOM on a smaller card.
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

export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

exec python "$REPO_ROOT/scripts/train_dreambc.py" \
  --variant full \
  --bundles-root "$BUNDLES_ROOT" \
  --exp-name "$EXP_NAME" \
  --num-train-steps 2500 \
  --batch-size 16 \
  --num-workers 4 \
  --warmup-steps 250 \
  --peak-lr 2.5e-5 \
  --decay-lr 2.5e-6 \
  --log-interval 25 \
  --save-interval 500 \
  --keep-period 1000 \
  --filter-mode success_true \
  "$@"
