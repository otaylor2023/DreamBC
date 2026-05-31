#!/usr/bin/env bash
# Build, push, and deploy cosmos_transfer RunPod Serverless endpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dreambc-cosmos-transfer}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
DOCKER_USER="${DOCKER_USER:-otaylor2023}"
FULL_IMAGE="docker.io/${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

export RUNPOD_API_KEY="${RUNPOD_API_KEY:-$(grep -E '^api_key' "${HOME}/.runpod/config.toml" | sed 's/.*= *"\?\([^"]*\)"\?/\1/')}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${REPO_ROOT}/.env" ]; then
  # shellcheck disable=SC1090
  set -a && source "${REPO_ROOT}/.env" && set +a
fi

echo "=== Building ${FULL_IMAGE} (linux/amd64) ==="
docker build --platform linux/amd64 \
  -f "${SCRIPT_DIR}/runpod/Dockerfile" \
  --build-arg HF_TOKEN="${HF_TOKEN:-}" \
  -t "${FULL_IMAGE}" \
  "${REPO_ROOT}"

echo "=== Pushing ${FULL_IMAGE} ==="
docker push "${FULL_IMAGE}"

echo "=== Creating RunPod template + endpoint ==="
export GCS_BUCKET="${GCS_BUCKET:-dreambc_videos}"
export GCS_PREFIX="${GCS_PREFIX:-cosmos-transfer}"
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1

python3 "${SCRIPT_DIR}/deploy_endpoint.py" --image "${FULL_IMAGE}"

echo "Done. Set RUNPOD_ENDPOINT_ID in .env from output above."
