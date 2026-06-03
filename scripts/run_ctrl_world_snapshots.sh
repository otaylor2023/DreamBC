#!/usr/bin/env bash
# Run Ctrl-World + pi0.5 policy-in-the-loop rollouts on snapshot datasets.
#
# Smoke mode (default): one snapshot from --dataset, fast end-to-end check.
# Sweep mode: every val_id in --dataset (or both datasets when --dataset all).
#
# Usage:
#   scripts/run_ctrl_world_snapshots.sh                              # smoke on cube/0001
#   scripts/run_ctrl_world_snapshots.sh --dataset tomato             # smoke on tomato/0001
#   scripts/run_ctrl_world_snapshots.sh --mode sweep                 # sweep cube
#   scripts/run_ctrl_world_snapshots.sh --mode sweep --dataset all   # sweep cube + tomato
#   scripts/run_ctrl_world_snapshots.sh --dataset cube --ids 0001,0002 --mode subset
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="smoke"          # smoke | sweep | subset
DATASET="cube"        # cube | tomato | all
IDS=""                # only for --mode subset
GRIPPER_MODE="invert" # only affects re-conversion hint; runtime reads existing dataset
SAVE_ROOT="$REPO_ROOT/rollouts/snapshots"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

if [[ ! -d "$REPO_ROOT/Ctrl-World/openpi/.venv" ]]; then
  echo "ERROR: openpi venv missing at Ctrl-World/openpi/.venv. Run env setup first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/Ctrl-World/openpi/.venv/bin/activate"

SVD_PATH="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP_PATH="/home/ubuntu/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WM_CKPT="$REPO_ROOT/Ctrl-World/checkpoint/checkpoint-10000.pt"
PI_CKPT="$REPO_ROOT/Ctrl-World/openpi/checkpoint/pi05_droid"

for p in "$SVD_PATH" "$CLIP_PATH" "$WM_CKPT" "$PI_CKPT"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing required asset: $p" >&2
    exit 1
  fi
done

CUBE_DIR="$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots"
TOMATO_DIR="$REPO_ROOT/Ctrl-World/dataset_example/tomato_snapshots"
CUBE_INSTR="put the orange cube in the blue bowl"
TOMATO_INSTR="put the tomato in the blue bowl"

run_one_dataset() {
  local name="$1"        # cube | tomato
  local val_ids_csv="$2" # comma-separated 4-digit ids
  local val_dir instr save_dir
  case "$name" in
    cube)
      val_dir="$CUBE_DIR"
      instr="$CUBE_INSTR"
      ;;
    tomato)
      val_dir="$TOMATO_DIR"
      instr="$TOMATO_INSTR"
      ;;
    *) echo "unknown dataset: $name" >&2; exit 2 ;;
  esac
  save_dir="$SAVE_ROOT/$name"
  mkdir -p "$save_dir"
  # start_idxs must match val_ids count; broadcast single 0
  local count start_idxs
  count=$(awk -F, '{print NF}' <<<"$val_ids_csv")
  start_idxs=$(python -c "print(','.join(['0']*$count))")

  echo "=========================================="
  echo "Dataset:       $name"
  echo "val_dataset:   $val_dir"
  echo "snapshot ids:  $val_ids_csv"
  echo "instruction:   $instr"
  echo "save_dir:      $save_dir"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4} \
  python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
    --task_type pickplace \
    --val_dataset_dir "$val_dir" \
    --val_ids "$val_ids_csv" \
    --start_idxs "$start_idxs" \
    --instructions "$instr" \
    --dataset_root_path "$REPO_ROOT/Ctrl-World/dataset_example" \
    --dataset_meta_info_path "$REPO_ROOT/Ctrl-World/dataset_meta_info" \
    --dataset_names droid_subset \
    --svd_model_path "$SVD_PATH" \
    --clip_model_path "$CLIP_PATH" \
    --ckpt_path "$WM_CKPT" \
    --pi_ckpt "$PI_CKPT" \
    --save_dir "$save_dir"
}

ids_from_meta() {
  local name="$1" dir
  case "$name" in
    cube) dir="$CUBE_DIR" ;;
    tomato) dir="$TOMATO_DIR" ;;
    *) echo "unknown dataset: $name" >&2; exit 2 ;;
  esac
  python -c "import json; print(','.join(json.load(open('$dir/meta/val_ids.json'))))"
}

case "$MODE" in
  smoke)
    case "$DATASET" in
      cube|tomato) run_one_dataset "$DATASET" "0001" ;;
      all)
        run_one_dataset cube "0001"
        run_one_dataset tomato "0001"
        ;;
      *) echo "smoke requires --dataset cube|tomato|all" >&2; exit 2 ;;
    esac
    ;;
  subset)
    if [[ -z "$IDS" ]]; then
      echo "--mode subset requires --ids 0001,0002,..." >&2; exit 2
    fi
    case "$DATASET" in
      cube|tomato) run_one_dataset "$DATASET" "$IDS" ;;
      *) echo "subset requires --dataset cube|tomato" >&2; exit 2 ;;
    esac
    ;;
  sweep)
    case "$DATASET" in
      cube)   run_one_dataset cube   "$(ids_from_meta cube)" ;;
      tomato) run_one_dataset tomato "$(ids_from_meta tomato)" ;;
      all)
        run_one_dataset cube   "$(ids_from_meta cube)"
        run_one_dataset tomato "$(ids_from_meta tomato)"
        ;;
      *) echo "sweep requires --dataset cube|tomato|all" >&2; exit 2 ;;
    esac
    ;;
  *)
    echo "Unknown --mode: $MODE (expected smoke|sweep|subset)" >&2
    exit 2 ;;
esac
