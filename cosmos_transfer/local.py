"""Local GPU inference via cosmos-transfer2.5 subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from cosmos_transfer.paths import COSMOS_DIR, COSMOS_PYTHON
from cosmos_transfer.spec import make_spec


def run_inference(
    video: Path,
    out_dir: Path,
    *,
    prompt: str,
    scene: Path | None = None,
    gpus: int = 1,
    cosmos_dir: Path | None = None,
) -> Path:
    """Run distilled edge inference locally; return path to output directory."""
    cosmos_dir = cosmos_dir or COSMOS_DIR
    if not cosmos_dir.exists():
        raise FileNotFoundError(
            f"cosmos-transfer2.5 not found at {cosmos_dir}. Run: bash cosmos_transfer/setup_cosmos.sh"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = make_spec(video, prompt, scene_path=scene)
    spec_file = out_dir / f"{video.stem}_spec.json"
    spec_file.write_text(json.dumps(spec, indent=2))

    python = str(COSMOS_PYTHON) if COSMOS_PYTHON.exists() else sys.executable
    inference_script = str(cosmos_dir / "examples" / "inference.py")

    env = os.environ.copy()
    env["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"

    base_args = [
        "-i",
        str(spec_file),
        "-o",
        str(out_dir),
        "--model=edge/distilled",
        "--disable-guardrails",
    ]

    if gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpus}",
            "--master_port=12341",
            inference_script,
            *base_args,
        ]
    else:
        cmd = [python, inference_script, *base_args]

    subprocess.run(cmd, check=True, cwd=cosmos_dir, env=env)
    return out_dir
