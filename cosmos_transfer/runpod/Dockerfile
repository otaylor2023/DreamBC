# Cosmos Transfer 2.5 distilled-edge RunPod Serverless worker.
#
# Build from repo root:
#   docker build -f cosmos_transfer/runpod/Dockerfile \
#     --build-arg HF_TOKEN="$HF_TOKEN" \
#     -t dreambc-cosmos-transfer .
#
# Requires HF_TOKEN at build time to bake distilled edge weights (optional but recommended).

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV COSMOS_EXPERIMENTAL_CHECKPOINTS=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg git git-lfs wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy

WORKDIR /workspace

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Clone and install cosmos-transfer2.5 (same as upstream Dockerfile / setup_cosmos.sh).
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-transfer2.5.git . \
    && uv sync --locked --extra=cu128 --python=3.10

ENV PATH="/workspace/.venv/bin:$PATH"

# Pre-download distilled edge checkpoint when HF_TOKEN is provided at build time.
RUN if [ -n "${HF_TOKEN}" ]; then \
      COSMOS_EXPERIMENTAL_CHECKPOINTS=1 python -c "\
from cosmos_oss.checkpoints_transfer2 import register_checkpoints; \
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint; \
register_checkpoints(); \
download_checkpoint('41f07f13-f2e4-4e34-ba4c-86f595acbc20'); \
print('checkpoint cached')"; \
    else echo "HF_TOKEN not set at build — weights download on first job"; fi

# Worker app (spec + gcs are copied flat into /app for simple imports).
WORKDIR /app
RUN pip install --no-cache-dir runpod google-cloud-storage

COPY cosmos_transfer/spec.py cosmos_transfer/gcs.py /app/
COPY cosmos_transfer/runpod/handler.py /app/handler.py
COPY cosmos_transfer/runpod/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "/app/handler.py"]
