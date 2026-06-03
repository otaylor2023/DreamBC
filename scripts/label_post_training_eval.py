#!/usr/bin/env python3
"""Create/apply manual labels for post-training Ctrl-World eval rollouts.

Default usage after rollouts finish:
  python scripts/label_post_training_eval.py template

Then edit results/post_training_eval/labels.tsv, filling the success column with
true/false (or yes/no, 1/0). Apply labels:
  python scripts/label_post_training_eval.py apply --labels results/post_training_eval/labels.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_ROOT = REPO / "rollouts" / "eval_post_training"
DEFAULT_TEMPLATE = REPO / "results" / "post_training_eval" / "labels.tsv"


def _parse_bool(text: str) -> bool | None:
    t = text.strip().lower()
    if t in ("", "null", "none", "unlabeled", "tbd"):
        return None
    if t in ("true", "1", "yes", "y", "success", "pass", "ok"):
        return True
    if t in ("false", "0", "no", "n", "fail", "failure"):
        return False
    raise ValueError(f"unrecognized success label {text!r}")


def _discover_episode_rows(rollout_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for episode_json in sorted(rollout_root.rglob("episode.json")):
        meta = json.loads(episode_json.read_text())
        rel_parts = episode_json.relative_to(rollout_root).parts
        if len(rel_parts) < 5:
            continue
        task = rel_parts[0]
        config_name = rel_parts[1]
        row = {
            "task": task,
            "config_name": config_name,
            "val_id": str(meta.get("episode_id", "")),
            "success": "" if meta.get("success") is None else str(bool(meta["success"])).lower(),
            "label_note": str(meta.get("label_note", "")),
            "concat_video_path": str(meta.get("concat_video_path", "")),
            "episode_dir": str(episode_json.parent),
        }
        rows.append(row)
    return rows


def write_template(rollout_root: Path, out_path: Path) -> int:
    rows = _discover_episode_rows(rollout_root)
    if not rows:
        raise SystemExit(f"no episode.json files found under {rollout_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "config_name",
        "val_id",
        "success",
        "label_note",
        "concat_video_path",
        "episode_dir",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


def apply_labels(labels_path: Path, dry_run: bool) -> int:
    with labels_path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"labels file is empty: {labels_path}")

    changed = 0
    skipped = 0
    for row in rows:
        success_raw = row.get("success", "")
        success = _parse_bool(success_raw)
        if success is None:
            skipped += 1
            continue

        episode_dir = Path(row["episode_dir"])
        if not episode_dir.is_absolute():
            episode_dir = REPO / episode_dir
        episode_json = episode_dir / "episode.json"
        if not episode_json.is_file():
            raise SystemExit(f"missing episode.json for row: {episode_json}")

        meta = json.loads(episode_json.read_text())
        old_success = meta.get("success")
        old_note = meta.get("label_note")
        note = row.get("label_note", "").strip()
        meta["success"] = success
        if note:
            meta["label_note"] = note
        elif "label_note" in meta:
            meta.pop("label_note")

        if old_success != success or old_note != meta.get("label_note"):
            changed += 1
            if not dry_run:
                episode_json.write_text(json.dumps(meta, indent=2) + "\n")

    suffix = " (dry-run)" if dry_run else ""
    print(f"applied labels from {labels_path}{suffix}")
    print(f"  changed: {changed}")
    print(f"  skipped unlabeled: {skipped}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_template = sub.add_parser("template", help="write labels TSV template from rollout outputs")
    p_template.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    p_template.add_argument("--out", type=Path, default=DEFAULT_TEMPLATE)

    p_apply = sub.add_parser("apply", help="apply filled labels TSV to episode.json files")
    p_apply.add_argument("--labels", type=Path, default=DEFAULT_TEMPLATE)
    p_apply.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "template":
        return write_template(args.rollout_root, args.out)
    if args.cmd == "apply":
        return apply_labels(args.labels, args.dry_run)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
