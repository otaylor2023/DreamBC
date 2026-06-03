#!/usr/bin/env bash
# Wait for TL_g3_red_ball.zip upload to finish, then unzip into rollouts/ and
# write a summary so the agent can decide on success-list + training launch.
set -uo pipefail

REPO_ROOT="/home/ubuntu/DreamBC"
ZIP_PATH="$REPO_ROOT/TL_g3_red_ball.zip"
DEST_DIR="$REPO_ROOT/rollouts"
LOG="$REPO_ROOT/logs/wait_and_unzip_tl.log"
SUMMARY="$REPO_ROOT/logs/tl_bundle_summary.txt"
STABLE_SECONDS=30
POLL_SECONDS=10

mkdir -p "$REPO_ROOT/logs"
exec >> "$LOG" 2>&1

echo "[$(date -Is)] starting watcher for $ZIP_PATH"
echo "[$(date -Is)] will wait for size stable for ${STABLE_SECONDS}s"

last_size=-1
stable_since=0
while true; do
  if [[ ! -f "$ZIP_PATH" ]]; then
    echo "[$(date -Is)] file missing, retrying in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi
  size=$(stat -c %s "$ZIP_PATH")
  now=$(date +%s)
  if [[ "$size" == "$last_size" ]]; then
    elapsed=$((now - stable_since))
    echo "[$(date -Is)] size stable at $size bytes for ${elapsed}s"
    if [[ "$elapsed" -ge "$STABLE_SECONDS" ]]; then
      echo "[$(date -Is)] STABLE -> proceeding to unzip"
      break
    fi
  else
    echo "[$(date -Is)] size=$size (was $last_size); resetting stability timer"
    last_size="$size"
    stable_since="$now"
  fi
  sleep "$POLL_SECONDS"
done

cd "$DEST_DIR"
echo "[$(date -Is)] unzipping into $DEST_DIR"
if ! command -v unzip > /dev/null; then
  echo "[$(date -Is)] unzip not installed; trying python -m zipfile"
  python3 -m zipfile -e "$ZIP_PATH" "$DEST_DIR"
else
  unzip -o -q "$ZIP_PATH" -d "$DEST_DIR"
fi
echo "[$(date -Is)] unzip done"

# Find the bundle root: prefer a single new top-level dir created by the unzip.
TL_DIR=""
for d in "$DEST_DIR"/TL_g3_red_ball*; do
  if [[ -d "$d" ]]; then
    TL_DIR="$d"
    break
  fi
done
if [[ -z "$TL_DIR" ]]; then
  for d in "$DEST_DIR"/T*; do
    [[ -d "$d" ]] && TL_DIR="$d" && break
  done
fi

{
  echo "=== TL bundle summary $(date -Is) ==="
  echo "zip_size=$(stat -c %s "$ZIP_PATH")"
  echo "bundle_root_guess=$TL_DIR"
  echo
  echo "--- top of $DEST_DIR ---"
  ls -la "$DEST_DIR" | head -40
  echo
  if [[ -d "$TL_DIR" ]]; then
    echo "--- top of $TL_DIR ---"
    ls -la "$TL_DIR" | head -40
    echo
    echo "--- recursive depth-3 view ---"
    find "$TL_DIR" -maxdepth 3 -type d | head -40
    echo
    echo "--- episode.json count ---"
    find "$TL_DIR" -name episode.json | wc -l
    echo
    echo "--- first 5 episode.json paths ---"
    find "$TL_DIR" -name episode.json | head -5
    echo
    echo "--- sample episode.json content (first one) ---"
    first_ep=$(find "$TL_DIR" -name episode.json | head -1)
    [[ -n "$first_ep" ]] && cat "$first_ep"
  fi
} > "$SUMMARY" 2>&1

echo "[$(date -Is)] wrote summary -> $SUMMARY"
echo "[$(date -Is)] watcher complete"
