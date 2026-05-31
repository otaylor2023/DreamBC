#!/usr/bin/env bash
# Fill C16_g5_block to all 60 cube_snapshots_v3 IDs.
# The original C16 run covered the 10 seeded-random IDs:
#   0002,0007,0008,0009,0015,0016,0018,0041,0048,0057
# This fill covers the remaining 50 IDs, writing into the SAME save_dir so
# bc_episodes/ accumulates and build_sweep_index.py picks them up.
#
# Launch detached:
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_c16_fill.sh \
#     > rollouts/sweep/_logs/c16_fill_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_ROOT="$REPO_ROOT/rollouts/sweep"
LOG_DIR="$SWEEP_ROOT/_logs"
MASTER_LOG="$LOG_DIR/_master.log"
PID_FILE="$LOG_DIR/c16_fill.pid"
START_MARKER="$LOG_DIR/c16_fill.start"
DONE_MARKER="$LOG_DIR/c16_fill.done"

VAL_IDS="0001,0003,0004,0005,0006,0010,0011,0012,0013,0014,0017,0019,0020,0021,0022,0023,0024,0025,0026,0027,0028,0029,0030,0031,0032,0033,0034,0035,0036,0037,0038,0039,0040,0042,0043,0044,0045,0046,0047,0049,0050,0051,0052,0053,0054,0055,0056,0058,0059,0060"
INTERACT_NUM=22
Z_MIN=0.17
GUIDANCE=5
INSTRUCTION="put the orange block in the blue bowl"

SAVE_DIR="$SWEEP_ROOT/cube/C16_g5_block"

mkdir -p "$LOG_DIR" "$SAVE_DIR"
echo "$$" > "$PID_FILE"
date -Is > "$START_MARKER"

# Preserve the original 10-ID sweep_config before the rollout overwrites it.
if [[ -f "$SAVE_DIR/sweep_config.json" && ! -f "$SAVE_DIR/sweep_config_part1.json" ]]; then
  cp "$SAVE_DIR/sweep_config.json" "$SAVE_DIR/sweep_config_part1.json"
fi

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

CONFIG_NAME="C16_g5_block_fill"
CONFIG_LOG="$LOG_DIR/${CONFIG_NAME}.log"
CONFIG_DONE="$LOG_DIR/${CONFIG_NAME}.done"

start_ts="$(date -Is)"
log_master "START config=$CONFIG_NAME object=cube guidance=$GUIDANCE dataset=$CUBE_V3 n_ids=50 save_dir=$SAVE_DIR"

count=$(awk -F, '{print NF}' <<<"$VAL_IDS")
start_idxs=$(python -c "print(','.join(['0']*$count))")

set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4} \
  python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
    --task_type pickplace \
    --val_dataset_dir "$CUBE_V3" \
    --val_ids "$VAL_IDS" \
    --start_idxs "$start_idxs" \
    --instructions "$INSTRUCTION" \
    --dataset_root_path "$REPO_ROOT/Ctrl-World/dataset_example" \
    --dataset_meta_info_path "$REPO_ROOT/Ctrl-World/dataset_meta_info" \
    --dataset_names droid_subset \
    --svd_model_path "$SVD_PATH" \
    --clip_model_path "$CLIP_PATH" \
    --ckpt_path "$WM_CKPT" \
    --pi_ckpt "$PI_CKPT" \
    --save_dir "$SAVE_DIR" \
    --guidance_scale "$GUIDANCE" \
    --z_min "$Z_MIN" \
    --interact_num "$INTERACT_NUM" \
  > "$CONFIG_LOG" 2>&1
rc=$?
set -e

end_ts="$(date -Is)"
if [[ $rc -eq 0 ]]; then
  status="ok"
  date -Is > "$CONFIG_DONE"
  # Rename the just-written sweep_config.json so both runs are preserved.
  if [[ -f "$SAVE_DIR/sweep_config.json" ]]; then
    cp "$SAVE_DIR/sweep_config.json" "$SAVE_DIR/sweep_config_part2.json"
  fi
else
  status="failed"
fi
log_master "END config=$CONFIG_NAME STATUS=$status rc=$rc start=$start_ts end=$end_ts log=$CONFIG_LOG"

log_master "Running build_sweep_index.py (c16_fill)"
python "$REPO_ROOT/scripts/build_sweep_index.py" --sweep_root "$SWEEP_ROOT" >> "$MASTER_LOG" 2>&1

date -Is > "$DONE_MARKER"
log_master "C16_FILL COMPLETE pid=$$"
