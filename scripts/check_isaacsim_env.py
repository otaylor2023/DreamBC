"""Quick import check for a DreamBC Isaac Sim shell."""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.spatial.transform


def main() -> None:
    print(f"python={sys.executable}")
    print(f"conda_prefix={os.environ.get('CONDA_PREFIX', '')}")
    print(f"numpy={np.__version__} ({np.__file__})")
    print("scipy_spatial_transform=ok")
    try:
        import torch

        print(f"torch={torch.__version__} ({torch.__file__})")
        print(f"torch_jit={hasattr(torch, 'jit')}")
    except Exception as exc:
        print(f"torch=failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
