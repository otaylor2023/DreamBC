#!/usr/bin/env python3
"""Print a pre-push checklist for DreamBC (does not run git add/commit)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_FILE_MB = 50
MAX_TOTAL_MB = 100


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def main() -> int:
    print("=== DreamBC pre-push checklist ===\n")
    branch = _git("branch", "--show-current")
    print(f"Branch: {branch}")
    print("Pages URL (after enabling): https://otaylor2023.github.io/DreamBC/\n")

    # What would be added if user runs git add -A (respecting gitignore)
    try:
        out = _git("ls-files", "--others", "--exclude-standard")
        untracked = [REPO / line for line in out.splitlines() if line]
    except subprocess.CalledProcessError:
        untracked = []

    sizes: list[tuple[int, Path]] = []
    for p in untracked:
        if p.is_file():
            sizes.append((p.stat().st_size, p))

    sizes.sort(reverse=True)
    total = sum(s for s, _ in sizes)
    print(f"Untracked files (would be addable): {len(sizes)}")
    print(f"Untracked total size: {total / (1024**2):.1f} MB\n")
    print("Largest untracked files:")
    for sz, p in sizes[:15]:
        rel = p.relative_to(REPO)
        flag = " *** TOO LARGE" if sz > MAX_FILE_MB * 1024**2 else ""
        print(f"  {sz / (1024**2):.2f} MB  {rel}{flag}")

    bad = [p for sz, p in sizes if sz > MAX_FILE_MB * 1024**2]
    if total > MAX_TOTAL_MB * 1024**2:
        print(f"\nWARN: untracked total > {MAX_TOTAL_MB} MB")
    if bad:
        print("\nFAIL: files exceed GitHub soft limit (50 MB)")
        return 1

    for forbidden in [
        "real_robot_checkpoints/cube/lora_aug_thr004_step225_hand/params",
        "training_runs/",
    ]:
        hits = [p for _, p in sizes if forbidden in str(p.relative_to(REPO))]
        if hits:
            print(f"\nFAIL: should not commit {forbidden}")
            return 1

    readme = (REPO / "README.md").read_text(encoding="utf-8", errors="replace")
    if "otaylor2023.github.io/DreamBC" not in readme:
        print("\nWARN: README.md missing Pages URL")

    docs_readme = REPO / "docs" / "README.md"
    if not docs_readme.is_file():
        print("\nWARN: docs/README.md missing")

    print("\nEnable GitHub Pages:")
    print("  Settings -> Pages -> Deploy from branch -> main -> /docs")
    print("\nOK: no oversized untracked files detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
