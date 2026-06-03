#!/usr/bin/env bash
# Tomato low-guidance prompt sweep: guidance 3 and 5 x 3 prompts, 10 shared random IDs.
#
# Launch detached (survives SSH/Cursor close):
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_tomato_lowg_sweep.sh \
#     > rollouts/sweep/_logs/tomato_lowg_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ROOT="$REPO_ROOT/rollouts/sweep"
LOG_DIR="$SWEEP_ROOT/_logs"
MASTER_LOG="$LOG_DIR/_master_tomato_lowg.log"
PID_FILE="$LOG_DIR/tomato_lowg.pid"
START_MARKER="$LOG_DIR/tomato_lowg.start"
DONE_MARKER="$LOG_DIR/tomato_lowg.done"

# Shared random 10 IDs (seed=20260530 from tomato_snapshots_v3/meta/val_ids.json)
VAL_IDS="0003,0005,0035,0042,0044,0063,0065,0067,0085,0090"
INTERACT_NUM=22
Z_MIN=0.17

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"
date -Is > "$START_MARKER"

if [[ ! -d "$REPO_ROOT/Ctrl-World/openpi/.venv" ]]; then
  echo "ERROR: openpi venv missing at Ctrl-World/openpi/.venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/Ctrl-World/openpi/.venv/bin/activate"

SVD_PATH="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP_PATH="/home/ubuntu/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WM_CKPT="$REPO_ROOT/Ctrl-World/checkpoint/checkpoint-10000.pt"
PI_CKPT="$REPO_ROOT/Ctrl-World/openpi/checkpoint/pi05_droid"

TOMATO_V3="$REPO_ROOT/Ctrl-World/dataset_example/tomato_snapshots_v3"

TOMATO_BASELINE="put the tomato in the blue bowl"
TOMATO_RED_BALL="put the small red ball in the blue bowl"
TOMATO_RUBBER="put the small rubber tomato in the blue bowl"

for p in "$SVD_PATH" "$CLIP_PATH" "$WM_CKPT" "$PI_CKPT"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing required asset: $p" >&2
    exit 1
  fi
done

log_master() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

run_config() {
  local object="$1"
  local config_name="$2"
  local dataset_dir="$3"
  local guidance="$4"
  local instruction="$5"

  local save_dir="$SWEEP_ROOT/$object/$config_name"
  local config_log="$LOG_DIR/${config_name}.log"
  local config_done="$LOG_DIR/${config_name}.done"
  local start_ts end_ts status

  mkdir -p "$save_dir"
  start_ts="$(date -Is)"
  log_master "START config=$config_name object=$object guidance=$guidance dataset=$dataset_dir"

  local count start_idxs
  count=$(awk -F, '{print NF}' <<<"$VAL_IDS")
  start_idxs=$(python -c "print(','.join(['0']*$count))")

  set +e
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4} \
    python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
      --task_type pickplace \
      --val_dataset_dir "$dataset_dir" \
      --val_ids "$VAL_IDS" \
      --start_idxs "$start_idxs" \
      --instructions "$instruction" \
      --dataset_root_path "$REPO_ROOT/Ctrl-World/dataset_example" \
      --dataset_meta_info_path "$REPO_ROOT/Ctrl-World/dataset_meta_info" \
      --dataset_names droid_subset \
      --svd_model_path "$SVD_PATH" \
      --clip_model_path "$CLIP_PATH" \
      --ckpt_path "$WM_CKPT" \
      --pi_ckpt "$PI_CKPT" \
      --save_dir "$save_dir" \
      --guidance_scale "$guidance" \
      --z_min "$Z_MIN" \
      --interact_num "$INTERACT_NUM" \
    > "$config_log" 2>&1
  local rc=$?
  set -e

  end_ts="$(date -Is)"
  if [[ $rc -eq 0 ]]; then
    status="ok"
    date -Is > "$config_done"
  else
    status="failed"
  fi
  log_master "END config=$config_name STATUS=$status rc=$rc start=$start_ts end=$end_ts log=$config_log"
}

log_master "TOMATO LOW-GUIDANCE SWEEP START pid=$$ val_ids=$VAL_IDS"

run_config tomato TL_g3_baseline  "$TOMATO_V3" 3 "$TOMATO_BASELINE"
run_config tomato TL_g5_baseline  "$TOMATO_V3" 5 "$TOMATO_BASELINE"
run_config tomato TL_g3_red_ball  "$TOMATO_V3" 3 "$TOMATO_RED_BALL"
run_config tomato TL_g5_red_ball  "$TOMATO_V3" 5 "$TOMATO_RED_BALL"
run_config tomato TL_g3_rubber     "$TOMATO_V3" 3 "$TOMATO_RUBBER"
run_config tomato TL_g5_rubber     "$TOMATO_V3" 5 "$TOMATO_RUBBER"

log_master "Running build_sweep_index.py"
python "$REPO_ROOT/scripts/build_sweep_index.py" --sweep_root "$SWEEP_ROOT" >> "$MASTER_LOG" 2>&1

date -Is > "$DONE_MARKER"
log_master "TOMATO LOW-GUIDANCE SWEEP COMPLETE pid=$$ sweep_root=$SWEEP_ROOT"
