#!/usr/bin/env python3
"""Run one DreamBC training job with loss-threshold checkpoint tagging and auto-stop.

Monitors the training log, maintains an EMA of training loss, tags the latest
checkpoint whenever a threshold is crossed from above, SIGTERMs the trainer once
the lowest threshold is hit (after one save-interval grace), then prunes all
checkpoints except the tagged ones.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

STEP_LOSS_RE = re.compile(r"Step (\d+): .*?loss=([\d.]+)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("lora", "full"), required=True)
    p.add_argument("--bundles-root", type=Path, required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument(
        "--thresholds",
        default="0.10,0.04,0.015",
        help="Comma-separated loss thresholds, highest to lowest (default: 0.10,0.04,0.015).",
    )
    p.add_argument("--num-train-steps", type=int, required=True, help="Hard cap on training steps.")
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--keep-period", type=int, default=25)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--warmup-steps", type=int, default=None, help="Override warmup; default depends on variant.")
    p.add_argument("--log-file", type=Path, required=True)
    p.add_argument("--checkpoint-base-dir", type=Path, default=None)
    p.add_argument("--config-name", default="pi05_dreambc")
    p.add_argument("--enable-image-aug", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min-free-gb", type=float, default=200.0)
    p.add_argument("--ema-window", type=int, default=5)
    p.add_argument("--poll-seconds", type=float, default=2.0)
    p.add_argument("--stop-grace-seconds", type=float, default=120.0)
    return p.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_disk(path: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise SystemExit(
            f"error: only {free_gb:.1f} GB free on {path}; need at least {min_free_gb:.0f} GB"
        )
    print(f"[threshold] disk ok: {free_gb:.1f} GB free on {path}")


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    base = args.checkpoint_base_dir or (_repo_root() / "training_runs" / "checkpoints")
    return base / args.config_name / args.exp_name


def _latest_numeric_checkpoint(checkpoint_dir: Path) -> tuple[int, Path] | None:
    if not checkpoint_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            step = int(child.name)
            if best is None or step > best[0]:
                best = (step, child)
    return best


def _first_checkpoint_at_or_after(checkpoint_dir: Path, step: int) -> tuple[int, Path] | None:
    if not checkpoint_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            checkpoint_step = int(child.name)
            if checkpoint_step >= step and (best is None or checkpoint_step < best[0]):
                best = (checkpoint_step, child)
    return best


def _build_train_cmd(args: argparse.Namespace) -> list[str]:
    repo = _repo_root()
    launcher = repo / ("scripts/run_train_lora.sh" if args.variant == "lora" else "scripts/run_train_full.sh")
    if not launcher.is_file():
        raise SystemExit(f"error: launcher not found: {launcher}")

    warmup = args.warmup_steps
    if warmup is None:
        warmup = 40 if args.variant == "lora" else 80

    if args.variant == "lora":
        cmd = [
            str(launcher),
            str(args.bundles_root.resolve()),
            args.exp_name,
            "--num-train-steps",
            str(args.num_train_steps),
            "--save-interval",
            str(args.save_interval),
            "--keep-period",
            str(args.keep_period),
            "--log-interval",
            str(args.log_interval),
            "--warmup-steps",
            str(warmup),
        ]
    else:
        # Full FT OOMs at bs=16 when checkpointing every 25 steps; bs=8 is safer.
        cmd = [
            str(launcher),
            str(args.bundles_root.resolve()),
            args.exp_name,
            "--num-train-steps",
            str(args.num_train_steps),
            "--save-interval",
            str(args.save_interval),
            "--keep-period",
            str(args.keep_period),
            "--log-interval",
            str(args.log_interval),
            "--warmup-steps",
            str(warmup),
            "--batch-size",
            "8",
        ]
    if args.enable_image_aug:
        cmd.append("--enable-image-aug")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _update_ema(values: list[float], window: int) -> float | None:
    if not values:
        return None
    tail = values[-window:]
    return sum(tail) / len(tail)


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"thresholds": entries}, indent=2) + "\n")


def _prune_checkpoints(checkpoint_dir: Path, keep_paths: set[Path]) -> None:
    if not checkpoint_dir.is_dir():
        return
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if child.resolve() not in keep_paths:
            print(f"[threshold] pruning {child}")
            shutil.rmtree(child)


def main() -> int:
    args = _parse_args()
    thresholds = sorted({float(x.strip()) for x in args.thresholds.split(",") if x.strip()}, reverse=True)
    if not thresholds:
        raise SystemExit("error: --thresholds must contain at least one value")

    repo = _repo_root()
    checkpoint_dir = _checkpoint_dir(args)
    manifest_path = checkpoint_dir / "loss_threshold_manifest.json"
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    _check_disk(checkpoint_dir.parent, args.min_free_gb)

    cmd = _build_train_cmd(args)
    print(f"[threshold] launching: {' '.join(cmd)}")
    print(f"[threshold] log -> {args.log_file}")
    print(f"[threshold] checkpoints -> {checkpoint_dir}")
    print(f"[threshold] thresholds (high->low): {thresholds}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with args.log_file.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            cwd=str(repo),
            start_new_session=True,
            env=env,
        )

    losses: list[float] = []
    tagged: dict[float, dict] = {}
    pending: dict[float, dict] = {}
    lowest = thresholds[-1]
    stop_requested = False
    stop_request_time: float | None = None
    last_step = -1

    try:
        while True:
            if proc.poll() is not None:
                print(f"[threshold] trainer exited with rc={proc.returncode}")
                break

            if args.log_file.is_file():
                text = args.log_file.read_text(encoding="utf-8", errors="replace")
                for match in STEP_LOSS_RE.finditer(text):
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    if step <= last_step:
                        continue
                    last_step = step
                    losses.append(loss)
                    ema = _update_ema(losses, args.ema_window)
                    if ema is None:
                        continue

                    prev_ema = _update_ema(losses[:-1], args.ema_window)
                    for thr in thresholds:
                        if thr in tagged or thr in pending:
                            continue
                        crossed = prev_ema is not None and prev_ema > thr >= ema
                        first_below = prev_ema is None and ema <= thr
                        if crossed or first_below:
                            pending[thr] = {
                                "threshold": thr,
                                "step": step,
                                "ema_loss": ema,
                                "raw_loss": loss,
                                "checkpoint_dir": None,
                                "checkpoint_step": None,
                            }
                            print(
                                f"[threshold] crossed loss<={thr}: step={step} ema={ema:.4f}; "
                                "waiting for checkpoint at/after crossing"
                            )

                    if not stop_requested and ema <= lowest:
                        stop_requested = True
                        stop_request_time = time.time()
                        print(
                            f"[threshold] lowest threshold {lowest} crossed at step={step} "
                            f"(ema={ema:.4f}); waiting for post-cross checkpoint before SIGTERM"
                        )

            for thr, entry in list(pending.items()):
                found = _first_checkpoint_at_or_after(checkpoint_dir, int(entry["step"]))
                if found is None:
                    continue
                entry["checkpoint_dir"] = str(found[1])
                entry["checkpoint_step"] = found[0]
                tagged[thr] = entry
                del pending[thr]
                _write_manifest(manifest_path, list(tagged.values()))
                print(
                    f"[threshold] tagged loss<={thr}: crossing_step={entry['step']} "
                    f"ema={entry['ema_loss']:.4f} ckpt={entry['checkpoint_dir']}"
                )

            if stop_requested and stop_request_time is not None:
                latest = _latest_numeric_checkpoint(checkpoint_dir)
                elapsed = time.time() - stop_request_time
                have_post_cross_ckpt = latest is not None and latest[0] >= last_step
                if have_post_cross_ckpt or elapsed >= args.stop_grace_seconds:
                    print(f"[threshold] sending SIGTERM to trainer pid={proc.pid}")
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    stop_request_time = None

            time.sleep(args.poll_seconds)

        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("[threshold] trainer did not exit after SIGTERM; sending SIGKILL")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=60)

    except KeyboardInterrupt:
        print("[threshold] interrupted; terminating trainer")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=120)
        raise

    # Backfill checkpoint paths for entries tagged before the first save landed.
    for thr, entry in list(tagged.items()):
        if entry.get("checkpoint_dir"):
            continue
        cross_step = entry.get("step", 0)
        best: tuple[int, Path] | None = None
        if checkpoint_dir.is_dir():
            for child in checkpoint_dir.iterdir():
                if child.is_dir() and child.name.isdigit():
                    step = int(child.name)
                    if step >= cross_step and (best is None or step < best[0]):
                        best = (step, child)
        if best is None:
            best = _latest_numeric_checkpoint(checkpoint_dir)
        if best is not None:
            entry["checkpoint_dir"] = str(best[1])
            entry["checkpoint_step"] = best[0]
    if tagged:
        _write_manifest(manifest_path, list(tagged.values()))

    keep_paths = {
        Path(entry["checkpoint_dir"]).resolve()
        for entry in tagged.values()
        if entry.get("checkpoint_dir")
    }
    _prune_checkpoints(checkpoint_dir, keep_paths)

    if proc.returncode not in (0, -signal.SIGTERM, 128 + signal.SIGTERM, 143):
        print(f"[threshold] warning: trainer rc={proc.returncode}")

    if lowest not in tagged:
        print(
            f"[threshold] warning: lowest threshold {lowest} was never crossed; "
            f"kept {len(tagged)} tagged checkpoint(s)"
        )

    print(f"[threshold] done. manifest -> {manifest_path}")
    print(f"[threshold] kept {len(keep_paths)} checkpoint(s)")
    return 0 if proc.returncode in (0, -signal.SIGTERM, 128 + signal.SIGTERM, 143) else proc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
