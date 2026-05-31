#!/usr/bin/env bash
# Extra cube prompt sweep at guidance 3 and 5 with two non-baseline prompts.
# Uses the same 10 random (seed=42) snapshot IDs across all 4 configs so the
# (prompt x guidance) cells are directly comparable.
#
# Launch detached:
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_cube_prompt_extra.sh \
#     > rollouts/sweep/_logs/cube_prompt_extra_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ROOT="$REPO_ROOT/rollouts/sweep"
LOG_DIR="$SWEEP_ROOT/_logs"
MASTER_LOG="$LOG_DIR/_master.log"
PID_FILE="$LOG_DIR/cube_prompt_extra.pid"
START_MARKER="$LOG_DIR/cube_prompt_extra.start"
DONE_MARKER="$LOG_DIR/cube_prompt_extra.done"

# Same random IDs (seed=42, sample 10 of 1..60), reused across all configs
VAL_IDS="0002,0007,0008,0009,0015,0016,0018,0041,0048,0057"
INTERACT_NUM=22
Z_MIN=0.17

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_FILE"
date -Is > "$START_MARKER"

# shellcheck disable=SC1091
source "$REPO_ROOT/Ctrl-World/openpi/.venv/bin/activate"

SVD_PATH="/home/ubuntu/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
CLIP_PATH="/home/ubuntu/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
WM_CKPT="$REPO_ROOT/Ctrl-World/checkpoint/checkpoint-10000.pt"
PI_CKPT="$REPO_ROOT/Ctrl-World/openpi/checkpoint/pi05_droid"

CUBE_V3="$REPO_ROOT/Ctrl-World/dataset_example/cube_snapshots_v3"

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

# ---- Cube prompt x low-guidance grid (4 configs) ----
run_config cube C14_g3_block     "$CUBE_V3" 3  "put the orange block in the blue bowl"
run_config cube C15_g3_wooden    "$CUBE_V3" 3  "put the small wooden orange cube in the blue bowl"
run_config cube C16_g5_block     "$CUBE_V3" 5  "put the orange block in the blue bowl"
run_config cube C17_g5_wooden    "$CUBE_V3" 5  "put the small wooden orange cube in the blue bowl"

log_master "Running build_sweep_index.py (cube_prompt_extra)"
python "$REPO_ROOT/scripts/build_sweep_index.py" --sweep_root "$SWEEP_ROOT" >> "$MASTER_LOG" 2>&1

date -Is > "$DONE_MARKER"
log_master "CUBE_PROMPT_EXTRA COMPLETE pid=$$ sweep_root=$SWEEP_ROOT"
