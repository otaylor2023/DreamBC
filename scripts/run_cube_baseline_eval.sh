#!/usr/bin/env bash
# Run base pi05_droid on the matched 10 cube val_ids at guidance 3 and 5
# with the cube baseline prompt. Appends to the same timing/log structure
# as scripts/run_post_training_eval.sh so the summarizer picks it up.
#
# Detached launch:
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_cube_baseline_eval.sh \
#     > rollouts/eval_post_training/_logs/cube_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="$REPO_ROOT/rollouts/eval_post_training"
RESULTS_ROOT="$REPO_ROOT/results/post_training_eval"
LOG_DIR="$OUT_ROOT/_logs"
MASTER_LOG="$LOG_DIR/_master.log"
TIMING_JSON="$RESULTS_ROOT/timing.json"

INTERACT_NUM="${INTERACT_NUM:-22}"
Z_MIN="${Z_MIN:-0.17}"
SVD_PATH="${SVD_PATH:-/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e}"
CLIP_PATH="${CLIP_PATH:-/home/ubuntu/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268}"
WM_CKPT="${WM_CKPT:-$REPO_ROOT/Ctrl-World/checkpoint/checkpoint-10000.pt}"
BASE_PI_CKPT="${BASE_PI_CKPT:-$REPO_ROOT/Ctrl-World/openpi/checkpoint/pi05_droid}"

CUBE_V3="$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots_v3"
CUBE_IDS="0002,0007,0008,0009,0015,0016,0018,0041,0048,0057"
CUBE_BASELINE_PROMPT="put the orange cube in the blue bowl"

mkdir -p "$LOG_DIR" "$RESULTS_ROOT"

if [[ ! -x "$REPO_ROOT/Ctrl-World/openpi/.venv/bin/python" ]]; then
  echo "ERROR: openpi venv missing at Ctrl-World/openpi/.venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/Ctrl-World/openpi/.venv/bin/activate"

if [[ ! -f "$TIMING_JSON" ]]; then
  printf '[]\n' > "$TIMING_JSON"
fi

log_master() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

append_timing() {
  python3 - "$TIMING_JSON" "$@" <<'PY'
import json
import sys
from pathlib import Path

(
    timing_path,
    task,
    config_name,
    checkpoint,
    pi_train_config,
    val_ids,
    guidance,
    instruction,
    save_dir,
    log_file,
    start_iso,
    end_iso,
    start_epoch,
    end_epoch,
    rc,
) = sys.argv[1:]

ids = [v for v in val_ids.split(",") if v]
elapsed = int(end_epoch) - int(start_epoch)
entry = {
    "task": task,
    "config_name": config_name,
    "checkpoint": checkpoint,
    "pi_train_config": pi_train_config,
    "val_ids": ids,
    "num_rollouts": len(ids),
    "guidance_scale": float(guidance),
    "instruction": instruction,
    "save_dir": save_dir,
    "log_file": log_file,
    "start_time": start_iso,
    "end_time": end_iso,
    "elapsed_seconds": elapsed,
    "seconds_per_video": elapsed / len(ids) if ids else None,
    "return_code": int(rc),
}

path = Path(timing_path)
try:
    data = json.loads(path.read_text())
except FileNotFoundError:
    data = []
data.append(entry)
path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

run_config() {
  local task="$1"
  local config_name="$2"
  local dataset_dir="$3"
  local val_ids="$4"
  local guidance="$5"
  local instruction="$6"
  local pi_ckpt="$7"
  local pi_train_config="$8"

  local save_dir="$OUT_ROOT/$task/$config_name"
  local config_log="$LOG_DIR/${config_name}.log"
  local config_done="$LOG_DIR/${config_name}.done"
  local count start_idxs start_iso end_iso start_epoch end_epoch status rc

  mkdir -p "$save_dir"
  count=$(awk -F, '{print NF}' <<<"$val_ids")
  start_idxs=$(python3 -c "print(','.join(['0']*$count))")

  start_iso="$(date -Is)"
  start_epoch="$(date +%s)"
  log_master "START task=$task config=$config_name guidance=$guidance ckpt=$pi_ckpt"

  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}" \
    python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
      --task_type pickplace \
      --val_dataset_dir "$dataset_dir" \
      --val_ids "$val_ids" \
      --start_idxs "$start_idxs" \
      --instructions "$instruction" \
      --dataset_root_path "$REPO_ROOT/Ctrl-World/dataset_example" \
      --dataset_meta_info_path "$REPO_ROOT/Ctrl-World/dataset_meta_info" \
      --dataset_names droid_subset \
      --svd_model_path "$SVD_PATH" \
      --clip_model_path "$CLIP_PATH" \
      --ckpt_path "$WM_CKPT" \
      --pi_ckpt "$pi_ckpt" \
      --pi_train_config "$pi_train_config" \
      --save_dir "$save_dir" \
      --guidance_scale "$guidance" \
      --z_min "$Z_MIN" \
      --interact_num "$INTERACT_NUM" \
    > "$config_log" 2>&1
  rc=$?
  set -e

  end_iso="$(date -Is)"
  end_epoch="$(date +%s)"
  if [[ $rc -eq 0 ]]; then
    status="ok"
    date -Is > "$config_done"
  else
    status="failed"
  fi

  append_timing "$task" "$config_name" "$pi_ckpt" "$pi_train_config" "$val_ids" \
    "$guidance" "$instruction" "$save_dir" "$config_log" "$start_iso" "$end_iso" \
    "$start_epoch" "$end_epoch" "$rc"
  log_master "END config=$config_name STATUS=$status rc=$rc start=$start_iso end=$end_iso log=$config_log"
}

run_config cube cube_base_pi05_baseline_g3 "$CUBE_V3" "$CUBE_IDS" 3 "$CUBE_BASELINE_PROMPT" \
  "$BASE_PI_CKPT" default

run_config cube cube_base_pi05_baseline_g5 "$CUBE_V3" "$CUBE_IDS" 5 "$CUBE_BASELINE_PROMPT" \
  "$BASE_PI_CKPT" default

log_master "CUBE BASELINE EVAL COMPLETE pid=$$"
