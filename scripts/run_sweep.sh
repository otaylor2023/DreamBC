#!/usr/bin/env bash
# Overnight Ctrl-World + pi0.5 sweep: 26 configs (13 cube + 13 tomato), 10 trajectories each.
#
# Launch detached (survives SSH/Cursor close):
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_sweep.sh \
#     > rollouts/sweep/_logs/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ROOT="$REPO_ROOT/rollouts/sweep"
LOG_DIR="$SWEEP_ROOT/_logs"
MASTER_LOG="$LOG_DIR/_master.log"
PID_FILE="$LOG_DIR/sweep.pid"
START_MARKER="$LOG_DIR/sweep.start"
DONE_MARKER="$LOG_DIR/sweep.done"

VAL_IDS="0001,0002,0003,0004,0005,0006,0007,0008,0009,0010"
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

CUBE_V3="$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots_v3"
CUBE_V2="$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots_v2"
TOMATO_V3="$REPO_ROOT/Ctrl-World/dataset_example/tomato_snapshots_v3"
TOMATO_V2="$REPO_ROOT/Ctrl-World/dataset_example/tomato_snapshots_v2"

CUBE_BASELINE="put the orange cube in the blue bowl"
TOMATO_BASELINE="put the tomato in the blue bowl"

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

# ---- Cube configs (13) ----
run_config cube C01_g8_baseline      "$CUBE_V3" 8    "$CUBE_BASELINE"
run_config cube C02_g5_baseline      "$CUBE_V3" 5    "$CUBE_BASELINE"
run_config cube C03_g3_baseline      "$CUBE_V3" 3    "$CUBE_BASELINE"
run_config cube C04_g10_baseline     "$CUBE_V3" 10   "$CUBE_BASELINE"
run_config cube C05_g12_baseline     "$CUBE_V3" 12   "$CUBE_BASELINE"

run_config cube C06_g8_block         "$CUBE_V3" 8    "put the orange block in the blue bowl"
run_config cube C07_g8_pick_place    "$CUBE_V3" 8    "pick up the orange cube and place it in the blue bowl"
run_config cube C08_g8_wooden        "$CUBE_V3" 8    "put the small wooden orange cube in the blue bowl"

run_config cube C09_views_ext2_ext3  "$CUBE_V2" 8    "$CUBE_BASELINE"

run_config cube C10_g10_block        "$CUBE_V3" 10   "put the orange block in the blue bowl"
run_config cube C11_g12_block        "$CUBE_V3" 12   "put the orange block in the blue bowl"
run_config cube C12_g10_pick_place   "$CUBE_V3" 10   "pick up the orange cube and place it in the blue bowl"
run_config cube C13_g10_wooden       "$CUBE_V3" 10   "put the small wooden orange cube in the blue bowl"

# ---- Tomato configs (13) ----
run_config tomato T01_g8_baseline    "$TOMATO_V3" 8    "$TOMATO_BASELINE"
run_config tomato T02_g5_baseline    "$TOMATO_V3" 5    "$TOMATO_BASELINE"
run_config tomato T03_g3_baseline    "$TOMATO_V3" 3    "$TOMATO_BASELINE"
run_config tomato T04_g10_baseline   "$TOMATO_V3" 10   "$TOMATO_BASELINE"
run_config tomato T05_g12_baseline   "$TOMATO_V3" 12   "$TOMATO_BASELINE"

run_config tomato T06_g8_red_ball    "$TOMATO_V3" 8    "put the small red ball in the blue bowl"
run_config tomato T07_g8_rubber      "$TOMATO_V3" 8    "put the small rubber tomato in the blue bowl"
run_config tomato T08_g8_red_sphere  "$TOMATO_V3" 8    "put the red sphere in the blue bowl"
run_config tomato T09_g8_pick_place  "$TOMATO_V3" 8    "pick up the tomato and place it in the blue bowl"

run_config tomato T10_g10_red_ball   "$TOMATO_V3" 10   "put the small red ball in the blue bowl"
run_config tomato T11_g12_red_ball   "$TOMATO_V3" 12   "put the small red ball in the blue bowl"
run_config tomato T12_g10_rubber     "$TOMATO_V3" 10   "put the small rubber tomato in the blue bowl"

run_config tomato T13_views_ext2_ext3 "$TOMATO_V2" 8    "$TOMATO_BASELINE"

log_master "Running build_sweep_index.py"
python "$REPO_ROOT/scripts/build_sweep_index.py" --sweep_root "$SWEEP_ROOT" >> "$MASTER_LOG" 2>&1

date -Is > "$DONE_MARKER"
log_master "SWEEP COMPLETE pid=$$ sweep_root=$SWEEP_ROOT"
