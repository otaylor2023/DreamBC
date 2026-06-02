#!/usr/bin/env bash
# Move all v1 real-robot checkpoints into real_robot_checkpoints/ with docs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEST="$REPO_ROOT/real_robot_checkpoints"
LOG="${LOG:-$REPO_ROOT/logs/organize_real_robot_checkpoints.log}"
mkdir -p "$(dirname "$LOG")" "$DEST"

exec > >(tee -a "$LOG") 2>&1

echo "[organize] $(date -Is) starting at $REPO_ROOT"
echo "[organize] destination: $DEST"

if [[ -d "$DEST/cube" ]] && [[ -n "$(ls -A "$DEST/cube" 2>/dev/null || true)" ]]; then
  echo "[organize] error: $DEST/cube already has checkpoints — aborting to avoid overwrite" >&2
  exit 1
fi
if [[ -d "$DEST/tomato" ]] && [[ -n "$(ls -A "$DEST/tomato" 2>/dev/null || true)" ]]; then
  echo "[organize] error: $DEST/tomato already has checkpoints — aborting to avoid overwrite" >&2
  exit 1
fi

echo "[organize] generating camera view examples..."
python3 "$REPO_ROOT/scripts/generate_portfolio_examples.py" \
  --repo-root "$REPO_ROOT" \
  --output-dir "real_robot_checkpoints/docs/camera_views"

echo "[organize] moving checkpoints and writing manifest..."
python3 "$REPO_ROOT/scripts/build_portfolio_manifest.py" \
  --repo-root "$REPO_ROOT" \
  --dest-root "real_robot_checkpoints" \
  --move

echo "[organize] verifying structure..."
python3 - <<'PY'
from pathlib import Path
import json
import sys

repo = Path(".").resolve()
dest = repo / "real_robot_checkpoints"
manifest = json.loads((dest / "MANIFEST.json").read_text())
expected = manifest["checkpoint_count"]
errors = []

for ckpt in manifest["checkpoints"]:
    p = repo / ckpt["path"]
    if not (p / "params").is_dir():
        errors.append(f"missing params: {p}")
    if not (p / "assets" / "droid").is_dir():
        errors.append(f"missing assets/droid: {p}")
    if not (p / "meta.json").is_file():
        errors.append(f"missing meta.json: {p}")

for name in ["README.md", "MANIFEST.json"]:
    if not (dest / name).is_file():
        errors.append(f"missing {name}")

cam = dest / "docs" / "camera_views"
for png in ["cube_exterior.png", "cube_wrist.png", "tomato_exterior.png", "tomato_wrist.png"]:
    if not (cam / png).is_file():
        errors.append(f"missing {png}")

actual = len(list((dest / "cube").iterdir())) + len(list((dest / "tomato").iterdir()))
if actual != expected:
    errors.append(f"count mismatch: expected {expected}, found {actual} checkpoint dirs")

if errors:
    print("[verify] FAILED:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)

total_bytes = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
print(f"[verify] OK: {expected} checkpoints, {total_bytes / 1e9:.1f} GB total under {dest}")
PY

echo "[organize] $(date -Is) complete — log: $LOG"
