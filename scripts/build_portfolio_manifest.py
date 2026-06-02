#!/usr/bin/env python3
"""Build MANIFEST.json, per-checkpoint meta.json, and README for real-robot portfolio."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_robot_checkpoint_registry import (
    CHECKPOINT_ENTRIES,
    V1_SOURCE_EXPS,
    CheckpointEntry,
)

DEFAULT_CKPT_SRC = Path("training_runs/checkpoints/pi05_dreambc")
DEFAULT_DEST = Path("real_robot_checkpoints")


def _load_threshold_lookup(source_root: Path, source_exp: str) -> dict[int, dict[str, Any]]:
    manifest_path = source_root / source_exp / "loss_threshold_manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text())
    return {int(item["step"]): item for item in data.get("thresholds", [])}


def _ema_loss_for_entry(source_root: Path, entry: CheckpointEntry) -> float | None:
    lookup = _load_threshold_lookup(source_root, entry.source_exp)
    item = lookup.get(entry.source_step)
    if item is None:
        return None
    return float(item["ema_loss"])


def _meta_dict(
    entry: CheckpointEntry,
    *,
    dest_path: Path,
    repo_root: Path,
    ema_loss: float | None,
) -> dict[str, Any]:
    return {
        "task": entry.task,
        "path": str(dest_path.relative_to(repo_root)),
        "dest_name": entry.dest_name,
        "source_exp": entry.source_exp,
        "source_step": entry.source_step,
        "method": entry.method,
        "variant": entry.method,
        "augmentation": entry.augmentation,
        "loss_threshold": entry.loss_threshold,
        "ema_loss": ema_loss,
        "train_step": entry.source_step,
        "prompt": entry.prompt,
        "recommended_for_robot": entry.recommended_for_robot,
        "rationale": entry.rationale,
        "eval_train_config": f"dreambc_{entry.method}",
    }


def write_meta_json(
    entry: CheckpointEntry,
    *,
    dest_ckpt_dir: Path,
    repo_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    ema_loss = _ema_loss_for_entry(source_root, entry)
    meta = _meta_dict(entry, dest_path=dest_ckpt_dir, repo_root=repo_root, ema_loss=ema_loss)
    (dest_ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_manifest(dest_root: Path, repo_root: Path, metas: list[dict[str, Any]]) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(dest_root.relative_to(repo_root)),
        "checkpoint_count": len(metas),
        "checkpoints": metas,
    }
    (dest_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _thr_label(threshold: float | None) -> str:
    if threshold is None:
        return "—"
    if threshold == 0.10:
        return "0.10"
    if threshold == 0.04:
        return "0.04"
    if threshold == 0.015:
        return "0.015"
    return f"{threshold:g}"


def _aug_label(aug: bool) -> str:
    return "aug" if aug else "noaug"


def write_readme(dest_root: Path, repo_root: Path, metas: list[dict[str, Any]]) -> None:
    rel = dest_root.relative_to(repo_root)

    def table_rows(task: str) -> str:
        rows = []
        for m in metas:
            if m["task"] != task:
                continue
            rec = "yes" if m["recommended_for_robot"] else ""
            rows.append(
                f"| `{m['path']}` | {m['method']} | {_aug_label(m['augmentation'])} | "
                f"{_thr_label(m['loss_threshold'])} | {m['train_step']} | "
                f"{m.get('ema_loss') or '—'} | {rec} | {m['rationale']} |"
            )
        return "\n".join(rows)

    def recommended_rows(task: str) -> str:
        rows = []
        for i, m in enumerate([x for x in metas if x["task"] == task and x["recommended_for_robot"]], 1):
            rows.append(
                f"| {i} | `{m['path']}` | {m['method']} | {_aug_label(m['augmentation'])} | "
                f"{_thr_label(m['loss_threshold'])} | {m['rationale']} |"
            )
        return "\n".join(rows)

    readme = f"""# DreamBC Real-Robot Checkpoint Portfolio

Fine-tuned **pi05_droid** checkpoints for one-shot real-robot evaluation on two pick-and-place tasks trained from sim rollouts in the same lab setup.

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Overview

- **{len([m for m in metas if m['task'] == 'cube'])} cube** checkpoints + **{len([m for m in metas if m['task'] == 'tomato'])} tomato** checkpoints
- Methods: LoRA (~9 GB each) and full fine-tune (~30 GB each)
- Loss-threshold tagging at EMA 0.10 / 0.04 / 0.015 (plus cube LoRA v0 baseline at step 500)
- See `MANIFEST.json` for machine-readable metadata

## Camera views

| Training bundle key | Dataset key | Model key (pi05) | Role |
|---|---|---|---|
| `policy_obs_exterior` | `observation/exterior_image_1_left` | `base_0_rgb` | Fixed third-person / exterior view |
| `policy_obs_wrist` | `observation/wrist_image_left` | `left_wrist_0_rgb` | Wrist-mounted camera |
| (padded zeros) | — | `right_wrist_0_rgb` | Unused; mask=False |

- Resolution: **224×224 RGB**, uint8 in bundles
- Example frames: [`docs/camera_views/`](docs/camera_views/) (`cube_exterior.png`, `cube_wrist.png`, `tomato_exterior.png`, `tomato_wrist.png`)
- **No image augmentation at inference**

## Architecture

- Base model: **pi0.5 (pi05)** from `gs://openpi-assets/checkpoints/pi05_droid`
- **LoRA**: `gemma_2b_lora` + `gemma_300m_lora`, base weights frozen
- **Full FT**: all parameters trainable
- Action space: DROID 8-D (7 joints + gripper), padded to 32 for pi05
- **Action horizon**: 15 steps per policy query
- Norm stats: `assets/droid/norm_stats.json` inside each checkpoint (from pi05_droid)
- Training augmentation (where noted): mild crop + brightness/contrast on exterior + wrist only

## Prompts at deploy

| Task | Prompt |
|---|---|
| Cube | `put the orange block in the blue bowl` |
| Tomato (most checkpoints) | `Pick the tomato` |
| Tomato (recommended #4 / #6 `_bowl` retrain) | `pick the tomato and place it in the blue bowl` |

## Recommended 6-per-task (one-shot eval)

### Cube

| # | Path | Method | Aug | Loss thr | Rationale |
|---|------|--------|-----|----------|-----------|
{recommended_rows("cube")}

### Tomato

| # | Path | Method | Aug | Loss thr | Rationale |
|---|------|--------|-----|----------|-----------|
{recommended_rows("tomato")}

## All checkpoints

### Cube ({len([m for m in metas if m['task'] == 'cube'])})

| Path | Method | Aug | Loss thr | Step | EMA loss | Recommended | Rationale |
|------|--------|-----|----------|------|----------|-------------|-----------|
{table_rows("cube")}

### Tomato ({len([m for m in metas if m['task'] == 'tomato'])})

| Path | Method | Aug | Loss thr | Step | EMA loss | Recommended | Rationale |
|------|--------|-----|----------|------|----------|-------------|-----------|
{table_rows("tomato")}

## Training data

| Task | Bundle | Success demos | Notes |
|------|--------|---------------|-------|
| Cube | `rollouts/C16_g5_block` | 16 | Orange block → blue bowl |
| Tomato | `rollouts/TL_g3_red_ball` | 24 | Curated clean-success list (`success_lists/tomato_v1_clean.txt`) |

Threshold-based early stop: tag checkpoints when EMA loss crosses 0.10 / 0.04 / 0.015, then prune extras.

## How to load / eval

Each checkpoint directory contains `params/` and `assets/droid/`. Use the matching train config variant:

```bash
# Sim rollout eval (from repo root)
./scripts/eval_dreambc_rollout.sh \\
  {rel}/cube/lora_noaug_thr004_step200 lora \\
  rollouts/eval/smoke_cube

./scripts/eval_dreambc_rollout.sh \\
  {rel}/tomato/lora_aug_thr004_step200 lora \\
  rollouts/eval/smoke_tomato
```

Replace `lora` with `full` for full fine-tune checkpoints. Per-checkpoint `meta.json` includes `variant` and `eval_train_config`.

## Additional docs

- [`docs/REAL_ROBOT_PORTFOLIO_cube.md`](docs/REAL_ROBOT_PORTFOLIO_cube.md)
- [`docs/REAL_ROBOT_PORTFOLIO_tomato.md`](docs/REAL_ROBOT_PORTFOLIO_tomato.md)
- [`docs/camera_views/camera_mapping.md`](docs/camera_views/camera_mapping.md)
"""
    (dest_root / "README.md").write_text(readme)


def move_checkpoints(
    *,
    repo_root: Path,
    source_root: Path,
    dest_root: Path,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    for entry in CHECKPOINT_ENTRIES:
        src = source_root / entry.source_exp / str(entry.source_step)
        dst = dest_root / entry.task / entry.dest_name
        if not src.is_dir():
            raise FileNotFoundError(f"missing checkpoint: {src}")
        if not (src / "params").is_dir():
            raise FileNotFoundError(f"missing params/: {src}")
        if not (src / "assets").is_dir():
            raise FileNotFoundError(f"missing assets/: {src}")
        if dst.exists():
            raise FileExistsError(f"destination already exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[move] {src.relative_to(repo_root)} -> {dst.relative_to(repo_root)}")
        if not dry_run:
            shutil.move(str(src), str(dst))
        moved.append((src, dst))
    return moved


def cleanup_source_exps(source_root: Path, *, dry_run: bool = False) -> None:
    for exp in sorted(V1_SOURCE_EXPS):
        exp_dir = source_root / exp
        if not exp_dir.is_dir():
            continue
        remaining = [p.name for p in exp_dir.iterdir()]
        print(f"[cleanup] removing {exp_dir} (remaining: {remaining})")
        if not dry_run:
            shutil.rmtree(exp_dir)


def copy_portfolio_docs(repo_root: Path, dest_root: Path, *, dry_run: bool = False) -> None:
    docs_dir = dest_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        repo_root / DEFAULT_CKPT_SRC / "REAL_ROBOT_PORTFOLIO_v1.md": docs_dir / "REAL_ROBOT_PORTFOLIO_cube.md",
        repo_root / DEFAULT_CKPT_SRC / "REAL_ROBOT_PORTFOLIO_tomato_v1.md": docs_dir / "REAL_ROBOT_PORTFOLIO_tomato.md",
    }
    for src, dst in copies.items():
        if not src.is_file():
            raise FileNotFoundError(f"missing portfolio doc: {src}")
        print(f"[copy] {src.relative_to(repo_root)} -> {dst.relative_to(repo_root)}")
        if not dry_run:
            shutil.copy2(src, dst)


def build_all(
    *,
    repo_root: Path,
    source_root: Path,
    dest_root: Path,
    move: bool,
    dry_run: bool,
) -> None:
    if move:
        move_checkpoints(repo_root=repo_root, source_root=source_root, dest_root=dest_root, dry_run=dry_run)
        copy_portfolio_docs(repo_root, dest_root, dry_run=dry_run)
        if not dry_run:
            cleanup_source_exps(source_root)

    metas: list[dict[str, Any]] = []
    for entry in CHECKPOINT_ENTRIES:
        dest_ckpt_dir = dest_root / entry.task / entry.dest_name
        if not dest_ckpt_dir.is_dir() and not dry_run:
            raise FileNotFoundError(f"checkpoint not at destination: {dest_ckpt_dir}")
        if dry_run:
            ema_loss = _ema_loss_for_entry(source_root, entry)
            meta = _meta_dict(entry, dest_path=dest_ckpt_dir, repo_root=repo_root, ema_loss=ema_loss)
        else:
            meta = write_meta_json(
                entry,
                dest_ckpt_dir=dest_ckpt_dir,
                repo_root=repo_root,
                source_root=source_root,
            )
        metas.append(meta)

    if not dry_run:
        write_manifest(dest_root, repo_root, metas)
        write_readme(dest_root, repo_root, metas)
        print(f"[manifest] wrote {dest_root / 'MANIFEST.json'}")
        print(f"[readme] wrote {dest_root / 'README.md'}")
        print(f"[done] {len(metas)} checkpoints organized under {dest_root.relative_to(repo_root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_CKPT_SRC)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--move", action="store_true", help="Move checkpoints from training_runs into dest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = (repo_root / args.source_root).resolve()
    dest_root = (repo_root / args.dest_root).resolve()

    build_all(
        repo_root=repo_root,
        source_root=source_root,
        dest_root=dest_root,
        move=args.move,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
