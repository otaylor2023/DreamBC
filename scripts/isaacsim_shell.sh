#!/usr/bin/env bash
# Source this file before running DreamBC Isaac Sim scripts:
#   source scripts/isaacsim_shell.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source scripts/isaacsim_shell.sh" >&2
  exit 1
fi

_dreambc_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DREAMBC_ROOT="$(cd "${_dreambc_script_dir}/.." && pwd)"
export DREAMBC_ENV_NAME="${DREAMBC_ENV_NAME:-dreambc}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-${HOME}/.local/share/ov/pkg}"

if [[ ! -d "${ISAACSIM_ROOT}" ]]; then
  echo "ISAACSIM_ROOT does not exist: ${ISAACSIM_ROOT}" >&2
  echo "Set it first, for example:" >&2
  echo "  export ISAACSIM_ROOT=/path/to/isaacsim" >&2
  return 1
fi

if [[ ! -f "${ISAACSIM_ROOT}/setup_conda_env.sh" ]]; then
  echo "Missing Isaac Sim setup script: ${ISAACSIM_ROOT}/setup_conda_env.sh" >&2
  return 1
fi

if command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate "${DREAMBC_ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${DREAMBC_ENV_NAME}"
else
  echo "Could not find mamba or conda on PATH." >&2
  return 1
fi

source "${ISAACSIM_ROOT}/setup_conda_env.sh"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${DREAMBC_ROOT}"

python - <<'PY'
import numpy as np
import scipy.spatial.transform

assert np.lib.NumpyVersion(np.__version__) < "2.0.0", np.__version__
print(f"DreamBC Isaac shell ready. numpy={np.__version__}")
PY
