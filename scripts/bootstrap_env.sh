#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${DREAMBC_ENV_NAME:-dreambc}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v mamba >/dev/null 2>&1; then
  CONDA_FRONTEND="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_FRONTEND="conda"
else
  echo "Could not find mamba or conda on PATH." >&2
  exit 1
fi

if "${CONDA_FRONTEND}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_FRONTEND}" env update -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml" --prune
else
  "${CONDA_FRONTEND}" env create -n "${ENV_NAME}" -f "${REPO_ROOT}/environment.yml"
fi

eval "$("${CONDA_FRONTEND}" shell hook --shell bash)"
"${CONDA_FRONTEND}" activate "${ENV_NAME}"

python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.7.0+cu128" \
  "torchvision==0.22.0+cu128" \
  "torchaudio==2.7.0+cu128" \
  "coverage==7.4.4"

python - <<'PY'
import numpy as np
import scipy.spatial.transform
import torch

print(f"numpy={np.__version__}")
print("scipy=ok")
print(f"torch={torch.__version__}")
print(f"torch_jit={hasattr(torch, 'jit')}")
PY

echo "Environment '${ENV_NAME}' is ready."
