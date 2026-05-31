#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COSMOS_DIR="${REPO_ROOT}/cosmos-transfer2.5"

echo "=== Cosmos Transfer 2.5 Setup ==="

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

if [ ! -d "${COSMOS_DIR}" ]; then
    echo "Cloning cosmos-transfer2.5..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "${COSMOS_DIR}"
fi

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

cd "${COSMOS_DIR}"

echo "Installing Python dependencies (Python 3.10, CUDA 12.8)..."
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra=cu128 --python=3.10

mkdir -p "${REPO_ROOT}/videos"

echo ""
echo "Pre-downloading distilled edge checkpoint..."
cd "${REPO_ROOT}"
UV_CACHE_DIR=/tmp/uv-cache "${COSMOS_DIR}/.venv/bin/python" -m cosmos_transfer.download_model

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run inference (RunPod — default):"
echo "  export RUNPOD_ENDPOINT_ID=your-endpoint-id"
echo "  export GCS_BUCKET=dreambc_videos"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=${REPO_ROOT}/dreambc-e3a16d04725e.json"
echo "  python -m cosmos_transfer --rollout-dir outputs/rollouts/.../<timestamp>"
echo ""
echo "Local GPU (--local):"
echo "  source ${COSMOS_DIR}/.venv/bin/activate"
echo "  python -m cosmos_transfer --local --videos-dir ${REPO_ROOT}/videos"
