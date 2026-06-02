#!/usr/bin/env bash
# Roll out a fine-tuned DreamBC checkpoint in Ctrl-World.
#
# Usage:
#   ./scripts/eval_dreambc_rollout.sh <ckpt_dir> <variant> <save_dir> [val_ids] [extra args to rollout_interact_pi.py ...]
#
#   <ckpt_dir>  Path to a single training step directory, e.g.
#               training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v0/4000
#               (must contain params/ and assets/droid/).
#   <variant>   "lora" or "full" — must match what was used at train time.
#   <save_dir>  Directory where bc_episodes/info/video go.
#   [val_ids]   Comma-separated cube snapshot IDs (default: 0001,0002).
#
# Examples:
#   ./scripts/eval_dreambc_rollout.sh \
#       training_runs/checkpoints/pi05_dreambc/cube_c16_lora_v0/4000 lora \
#       rollouts/eval/cube_c16_lora_v0_step4000
#
#   ./scripts/eval_dreambc_rollout.sh \
#       training_runs/checkpoints/pi05_dreambc/cube_c16_full_v0/2000 full \
#       rollouts/eval/cube_c16_full_v0_step2000 0001,0002,0003,0004,0005
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <ckpt_dir> <variant: lora|full> <save_dir> [val_ids] [extra args...]" >&2
  exit 2
fi

CKPT_DIR="$1"; shift
VARIANT="$1"; shift
SAVE_DIR="$1"; shift
VAL_IDS="${1:-0001,0002}"
if [[ $# -ge 1 ]]; then shift; fi

case "$VARIANT" in
  lora) PI_TRAIN_CONFIG="dreambc_lora" ;;
  full) PI_TRAIN_CONFIG="dreambc_full" ;;
  *) echo "error: variant must be lora|full (got $VARIANT)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR_ABS="$(cd "$CKPT_DIR" && pwd)"

if [[ ! -d "$CKPT_DIR_ABS/params" ]]; then
  echo "error: $CKPT_DIR_ABS/params missing — pass a step dir like .../<exp_name>/<step>" >&2
  exit 1
fi
if [[ ! -d "$CKPT_DIR_ABS/assets" ]]; then
  echo "error: $CKPT_DIR_ABS/assets missing — checkpoint must contain assets/<asset_id>/norm_stats.json" >&2
  exit 1
fi

VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "error: openpi venv not found at $VENV" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

SVD_PATH="${SVD_PATH:-/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e}"
CLIP_PATH="${CLIP_PATH:-/home/ubuntu/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}"
WM_CKPT="${WM_CKPT:-$REPO_ROOT/Ctrl-World/checkpoint/checkpoint-10000.pt}"
CUBE_V3="${CUBE_V3:-$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots_v3}"

INTERACT_NUM="${INTERACT_NUM:-22}"
Z_MIN="${Z_MIN:-0.17}"
GUIDANCE="${GUIDANCE:-5}"
INSTRUCTION="${INSTRUCTION:-put the orange block in the blue bowl}"

mkdir -p "$SAVE_DIR"
# rollout_interact_pi.py's resolve_paths() silently prefixes any RELATIVE
# --save_dir with the Ctrl-World/ root, which makes outputs land in a
# confusing place. Pass an absolute path to avoid that rewrite.
SAVE_DIR_ABS="$(cd "$SAVE_DIR" && pwd)"

# Pad start_idxs with the same length as VAL_IDS.
COUNT=$(awk -F, '{print NF}' <<<"$VAL_IDS")
START_IDXS=$(python -c "print(','.join(['0']*$COUNT))")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}" \
  python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
    --task_type pickplace \
    --val_dataset_dir "$CUBE_V3" \
    --val_ids "$VAL_IDS" \
    --start_idxs "$START_IDXS" \
    --instructions "$INSTRUCTION" \
    --dataset_root_path "$REPO_ROOT/Ctrl-World/dataset_example" \
    --dataset_meta_info_path "$REPO_ROOT/Ctrl-World/dataset_meta_info" \
    --dataset_names droid_subset \
    --svd_model_path "$SVD_PATH" \
    --clip_model_path "$CLIP_PATH" \
    --ckpt_path "$WM_CKPT" \
    --pi_ckpt "$CKPT_DIR_ABS" \
    --pi_train_config "$PI_TRAIN_CONFIG" \
    --save_dir "$SAVE_DIR_ABS" \
    --guidance_scale "$GUIDANCE" \
    --z_min "$Z_MIN" \
    --interact_num "$INTERACT_NUM" \
    "$@"
