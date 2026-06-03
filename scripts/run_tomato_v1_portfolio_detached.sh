#!/usr/bin/env bash
# Detached sequential launcher for the v1 real-robot tomato portfolio.
#
# Mirrors run_v1_portfolio_detached.sh but for tomato/red-ball bundles.
# Runs four threshold-wrapped training jobs back-to-back:
#   1. tomato_t24_lora_v1_noaug
#   2. tomato_t24_lora_v1_aug
#   3. tomato_t24_full_v1_noaug
#   4. tomato_t24_full_v1_aug
#
# Fire-and-forget launch:
#   ./scripts/run_tomato_v1_portfolio_detached.sh <bundles_root>
#
# Example:
#   ./scripts/run_tomato_v1_portfolio_detached.sh rollouts/TL_g3_red_ball
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <bundles_root>" >&2
  exit 2
fi

BUNDLES_ROOT="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/Ctrl-World/openpi/.venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_TAG="tomato_t24_v1_portfolio_${TIMESTAMP}"
MASTER_LOG="$REPO_ROOT/logs/${CHAIN_TAG}.log"
MARKER_DIR="$REPO_ROOT/training_runs/chain_markers"

mkdir -p "$REPO_ROOT/logs" "$MARKER_DIR"

if [[ "${DREAMBC_TOMATO_V1_PORTFOLIO_DETACHED:-}" != "1" ]]; then
  export DREAMBC_TOMATO_V1_PORTFOLIO_DETACHED=1
  echo "[tomato-portfolio] starting detached orchestrator -> $MASTER_LOG"
  setsid nohup "$0" "$BUNDLES_ROOT" > "$MASTER_LOG" 2>&1 < /dev/null &
  disown
  echo "[tomato-portfolio] pid=$! master_log=$MASTER_LOG"
  exit 0
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/load_env.sh"
load_env_file "$REPO_ROOT/.env"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[tomato-portfolio $(date -Is)] error: openpi venv not found at $VENV" >&2
  exit 1
fi

THRESHOLDS="0.10,0.04,0.015"
START_MARK="$MARKER_DIR/${CHAIN_TAG}.start"
DONE_MARK="$MARKER_DIR/${CHAIN_TAG}.done"
FAIL_MARK="$MARKER_DIR/${CHAIN_TAG}.failed"

date -Is > "$START_MARK"
echo "[tomato-portfolio $(date -Is)] START chain=$CHAIN_TAG bundles=$BUNDLES_ROOT"

run_one() {
  local variant="$1"
  local exp_name="$2"
  local max_steps="$3"
  local enable_aug="$4"
  local done_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.done"
  local fail_mark="$MARKER_DIR/${CHAIN_TAG}.${exp_name}.failed"
  local run_log="$REPO_ROOT/logs/${exp_name}.log"

  echo "[tomato-portfolio $(date -Is)] START $exp_name (variant=$variant max_steps=$max_steps aug=$enable_aug)"

  local -a extra_args=(--overwrite)
  if [[ "$enable_aug" == "true" ]]; then
    extra_args+=(--enable-image-aug)
  fi

  "$VENV/bin/python" -u "$REPO_ROOT/scripts/run_with_loss_threshold.py" \
    --variant "$variant" \
    --bundles-root "$BUNDLES_ROOT" \
    --exp-name "$exp_name" \
    --thresholds "$THRESHOLDS" \
    --num-train-steps "$max_steps" \
    --save-interval 25 \
    --keep-period 25 \
    --log-interval 25 \
    --log-file "$run_log" \
    "${extra_args[@]}"
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    date -Is > "$fail_mark"
    echo "[tomato-portfolio $(date -Is)] FAILED $exp_name rc=$rc (see $run_log)" | tee -a "$FAIL_MARK"
    return "$rc"
  fi

  date -Is > "$done_mark"
  echo "[tomato-portfolio $(date -Is)] DONE $exp_name"
  return 0
}

# 24 demos vs cube's 16 -> bump full-FT cap to 1000 so we don't miss the 0.015 crossing.
run_one lora tomato_t24_lora_v1_noaug 800  false || exit $?
run_one lora tomato_t24_lora_v1_aug   800  true  || exit $?
run_one full tomato_t24_full_v1_noaug 1000 false || exit $?
run_one full tomato_t24_full_v1_aug   1000 true  || exit $?

date -Is > "$DONE_MARK"
echo "[tomato-portfolio $(date -Is)] CHAIN COMPLETE. master_log=$MASTER_LOG"

# Write portfolio summary doc.
PORTFOLIO_DOC="$REPO_ROOT/training_runs/checkpoints/pi05_dreambc/REAL_ROBOT_PORTFOLIO_tomato_v1.md"
"$VENV/bin/python" - "$PORTFOLIO_DOC" "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
repo = Path(sys.argv[2])
ckpt_root = out_path.parent

runs = [
    ("tomato_t24_lora_v1_noaug", "LoRA sweet spot", "lora", "noaug", "0.04", "early-checkpoint hypothesis"),
    ("tomato_t24_lora_v1_aug",   "LoRA very early + aug", "lora", "aug",   "0.10", "max generalization"),
    ("tomato_t24_lora_v1_aug",   "LoRA sweet + aug",      "lora", "aug",   "0.04", "likely best overall"),
    ("tomato_t24_full_v1_aug",   "full FT sweet + aug",   "full", "aug",   "0.04", "extra capacity + regularized"),
    ("tomato_t24_full_v1_noaug", "full FT sweet, no aug", "full", "noaug", "0.04", "isolates aug effect"),
    ("tomato_t24_lora_v1_noaug", "LoRA late, no aug",     "lora", "noaug", "0.015", "near-converged baseline"),
]
optional = [
    ("tomato_t24_lora_v1_aug",  "LoRA late + aug", "lora", "aug", "0.015", "bracket right side"),
    ("tomato_t24_full_v1_aug",  "full FT early + aug", "full", "aug", "0.10", "full FT early stop"),
]

lines = [
    "# Real-Robot Portfolio Tomato v1",
    "",
    "Recommended 6-checkpoint selection for one-shot eval:",
    "",
    "| # | Checkpoint path | Method | Aug | Loss thr | Rationale |",
    "|---|-----------------|--------|-----|----------|-----------|",
]

def resolve_path(run_name: str, thr: str) -> str:
    manifest = ckpt_root / run_name / "loss_threshold_manifest.json"
    if not manifest.is_file():
        return f"training_runs/checkpoints/pi05_dreambc/{run_name}/<pending>"
    data = json.loads(manifest.read_text())
    for entry in data.get("thresholds", []):
        if abs(float(entry["threshold"]) - float(thr)) < 1e-9:
            ckpt = entry.get("checkpoint_dir")
            if ckpt:
                try:
                    return str(Path(ckpt).relative_to(repo))
                except ValueError:
                    return ckpt
    return f"training_runs/checkpoints/pi05_dreambc/{run_name}/<thr_{thr}_missing>"

for i, (run, label, method, aug, thr, why) in enumerate(runs, 1):
    path = resolve_path(run, thr)
    lines.append(f"| {i} | `{path}` | {method} | {aug} | {thr} | {why} |")

lines.extend(["", "## Optional 7th/8th slots", ""])
for i, (run, label, method, aug, thr, why) in enumerate(optional, 7):
    path = resolve_path(run, thr)
    lines.append(f"- ({i}) `{path}` - {label}: {why}")

lines.extend(["", "## All tagged checkpoints per run", ""])
for exp_dir in sorted(ckpt_root.glob("tomato_t24_*_v1_*")):
    manifest = exp_dir / "loss_threshold_manifest.json"
    lines.append(f"### {exp_dir.name}")
    if not manifest.is_file():
        lines.append("- (no manifest yet)")
        continue
    data = json.loads(manifest.read_text())
    for entry in data.get("thresholds", []):
        ckpt = entry.get("checkpoint_dir", "?")
        lines.append(
            f"- thr={entry['threshold']} step={entry.get('checkpoint_step')} ema={entry.get('ema_loss'):.4f} -> `{ckpt}`"
        )
    lines.append("")

out_path.write_text("\n".join(lines) + "\n")
print(f"[tomato-portfolio] wrote {out_path}")
PY

echo "[tomato-portfolio $(date -Is)] portfolio doc -> $PORTFOLIO_DOC"
