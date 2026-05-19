#!/usr/bin/env bash
set -euo pipefail

echo "=== Cosmos Transfer 2.5 Setup ==="

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running:"
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

# Clone — skip LFS example assets (we use our own videos)
if [ ! -d "/DreamBC/cosmos-transfer2.5" ]; then
    echo "Cloning cosmos-transfer2.5..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /DreamBC/cosmos-transfer2.5
fi

# Install uv if missing
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

cd /DreamBC/cosmos-transfer2.5

echo "Installing Python dependencies (Python 3.10, CUDA 12.8)..."
# flash-attn wheels are only built for cp310
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra=cu128 --python=3.10

mkdir -p /DreamBC/videos

echo ""
echo "Pre-downloading distilled edge checkpoint..."
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/python /DreamBC/download_model.py

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run inference:"
echo "  source /DreamBC/cosmos-transfer2.5/.venv/bin/activate"
echo "  # Drop your sim robot .mp4 files into /DreamBC/videos/"
echo "  python /DreamBC/sim2real.py"
