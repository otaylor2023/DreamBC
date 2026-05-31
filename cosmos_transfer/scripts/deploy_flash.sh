#!/usr/bin/env bash
# Deploy cosmos_transfer stub worker via RunPod Flash (no Docker Hub required).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLASH_DIR="${REPO_ROOT}/cosmos_transfer/flash_deploy"
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(grep -E '^api_key' "${HOME}/.runpod/config.toml" | sed 's/.*= *"\?\([^"]*\)"\?/\1/')}"

if [ -f "${REPO_ROOT}/.env" ]; then
  # shellcheck disable=SC1090
  set -a && source "${REPO_ROOT}/.env" && set +a
fi

export GCS_BUCKET="${GCS_BUCKET:-dreambc_videos}"
export GCS_PREFIX="${GCS_PREFIX:-cosmos-transfer}"

# Embed SA JSON for worker (Flash env vars).
if [ -f "${GOOGLE_APPLICATION_CREDENTIALS:-${REPO_ROOT}/dreambc-e3a16d04725e.json}" ]; then
  export GCP_SA_JSON="$(cat "${GOOGLE_APPLICATION_CREDENTIALS:-${REPO_ROOT}/dreambc-e3a16d04725e.json}")"
fi

cd "${FLASH_DIR}"
echo "=== Flash deploy from ${FLASH_DIR} ==="
flash deploy --app dreambc-cosmos-flash 2>&1
