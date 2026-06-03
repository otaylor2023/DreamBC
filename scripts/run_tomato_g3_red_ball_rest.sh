#!/usr/bin/env bash
# Finish g3_red_ball coverage: remaining 90 tomato snapshot IDs at guidance 3.
#
# Launch detached (survives SSH/Cursor close):
#   cd /home/ubuntu/DreamBC
#   setsid nohup bash scripts/run_tomato_g3_red_ball_rest.sh \
#     > rollouts/sweep/_logs/g3_red_ball_rest_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
#   disown
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/rollouts/sweep/_logs"
MASTER_LOG="$LOG_DIR/_master_g3_red_ball_rest.log"
PID_FILE="$LOG_DIR/g3_red_ball_rest.pid"
START_MARKER="$LOG_DIR/g3_red_ball_rest.start"
DONE_MARKER="$LOG_DIR/g3_red_ball_rest.done"

# Remaining 90 IDs (0001..0100 minus the original 10 random IDs), ascending
VAL_IDS="0001,0002,0004,0006,0007,0008,0009,0010,0011,0012,0013,0014,0015,0016,0017,0018,0019,0020,0021,0022,0023,0024,0025,0026,0027,0028,0029,0030,0031,0032,0033,0034,0036,0037,0038,0039,0040,0041,0043,0045,0046,0047,0048,0049,0050,0051,0052,0053,0054,0055,0056,0057,0058,0059,0060,0061,0062,0064,0066,0068,0069,0070,0071,0072,0073,0074,0075,0076,0077,0078,0079,0080,0081,0082,0083,0084,0086,0087,0088,0089,0091,0092,0093,0094,0095,0096,0097,0098,0099,0100"
INTERACT_NUM=22
Z_MIN=0.17
GUIDANCE=3
INSTRUCTION="put the small red ball in the blue bowl"

SAVE_DIR="$REPO_ROOT/rollouts/TL_g3_red_ball"
CONFIG_LOG="$LOG_DIR/TL_g3_red_ball_rest.log"
CONFIG_DONE="$LOG_DIR/TL_g3_red_ball_rest.done"

mkdir -p "$LOG_DIR" "$SAVE_DIR"
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

for p in "$SVD_PATH" "$CLIP_PATH" "$WM_CKPT" "$PI_CKPT"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing required asset: $p" >&2
    exit 1
  fi
done

log_master() {
  echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"
}

start_ts="$(date -Is)"
log_master "G3 RED BALL REST START pid=$$ val_ids_count=$(awk -F, '{print NF}' <<<"$VAL_IDS") save_dir=$SAVE_DIR"

count=$(awk -F, '{print NF}' <<<"$VAL_IDS")
start_idxs=$(python -c "print(','.join(['0']*$count))")

set +e
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4} \
  python "$REPO_ROOT/Ctrl-World/scripts/rollout_interact_pi.py" \
    --task_type pickplace \
    --val_dataset_dir "$TOMATO_V3" \
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
else
  status="failed"
fi
log_master "G3 RED BALL REST END STATUS=$status rc=$rc start=$start_ts end=$end_ts log=$CONFIG_LOG"

date -Is > "$DONE_MARKER"
log_master "G3 RED BALL REST COMPLETE pid=$$ save_dir=$SAVE_DIR"
