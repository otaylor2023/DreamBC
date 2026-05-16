#!/usr/bin/env python3
"""
Expand Franka ROS `franka_description` Panda + hand xacro into a plain URDF for Isaac Sim.

Prerequisite: vendor tree at `urdf_models/franka_description/` (copy from frankarobotics/franka_ros
`noetic-devel` branch: `franka_ros/franka_description/`).

Temporarily replaces `$(find franka_description)/` with this directory's absolute path, runs `xacro`,
strips `package://franka_description/` from mesh paths, writes `panda_arm_hand_isaac.urdf`, then
restores `$(find franka_description)/` so the vendored xacro stays machine-independent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

NEEDLE = "$(find franka_description)/"


def patch_tree(fd: Path, old: str, new: str) -> None:
    for path in fd.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".xacro", ".xml"}:
            continue
        text = path.read_text(encoding="utf-8")
        if old not in text:
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    fd = repo / "urdf_models" / "franka_description"
    xacro_in = fd / "robots" / "panda" / "panda.urdf.xacro"
    out = fd / "panda_arm_hand_isaac.urdf"
    if not xacro_in.is_file():
        print(
            "Expected franka_description at urdf_models/franka_description/ "
            "(copy from https://github.com/frankarobotics/franka_ros noetic-devel: franka_description/).",
            file=sys.stderr,
        )
        sys.exit(1)
    fd_abs = str(fd.resolve()) + "/"
    xacro_bin = shutil.which("xacro")
    if not xacro_bin:
        print("Install the `xacro` PyPI package (pip install xacro).", file=sys.stderr)
        sys.exit(1)
    patch_tree(fd, NEEDLE, fd_abs)
    try:
        proc = subprocess.run(
            [xacro_bin, str(xacro_in), "hand:=true"],
            cwd=str(fd),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        patch_tree(fd, fd_abs, NEEDLE)
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        raise
    patch_tree(fd, fd_abs, NEEDLE)
    urdf = proc.stdout.replace("package://franka_description/", "")
    out.write_text(urdf, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
