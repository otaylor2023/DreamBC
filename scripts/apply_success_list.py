#!/usr/bin/env python3
"""Apply a success label list to BC bundles.

A success-list file is a plain text file with one entry per line. Each entry
may be:
  * an absolute or relative episode directory path
    (e.g. ``rollouts/C16_g5_block/.../bc_episodes/pickplace_..._traj_0002_...``)
  * a basename of an episode directory
    (e.g. ``pickplace_time_20260530_234036_traj_0002_..._block_in_the_blue_bowl``)
  * a bare episode_id (4-digit string, e.g. ``0002``) — matches by
    ``episode.json`` ``episode_id`` field
  * a snapshot index integer (e.g. ``2``) — matched against ``int(episode_id)``

Lines beginning with ``#`` and blank lines are ignored.

Every episode under ``--bundles-root`` is scanned. For each match the
``episode.json`` ``success`` field is set according to ``--label``
(default ``true``). Episodes not present in the list are left as-is unless
``--default-others`` is given, in which case unmatched episodes get the
specified label (typically ``false``).

Outputs a short summary, and writes ``--report-out`` (JSON) if requested.

Examples:
  Label 20 cube trajectories as successful, leave the rest untouched:
    python scripts/apply_success_list.py \
        --bundles-root rollouts/C16_g5_block \
        --list cube_success_ids.txt \
        --label true

  Same, but also explicitly mark the rest as failures:
    python scripts/apply_success_list.py \
        --bundles-root rollouts/C16_g5_block \
        --list cube_success_ids.txt \
        --label true --default-others false
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _coerce_label(text: str) -> bool | None:
    t = text.strip().lower()
    if t in ("true", "1", "yes", "y", "success", "ok"):
        return True
    if t in ("false", "0", "no", "n", "fail", "failure"):
        return False
    if t in ("null", "none", "unlabeled", ""):
        return None
    raise argparse.ArgumentTypeError(f"unrecognised label: {text!r}")


def _load_list(path: Path) -> set[str]:
    entries: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _candidate_keys(episode_dir: Path, meta: dict) -> set[str]:
    cand = {
        str(episode_dir),
        str(episode_dir.resolve()),
        episode_dir.name,
    }
    ep_id = meta.get("episode_id")
    if ep_id is not None:
        cand.add(str(ep_id))
        try:
            cand.add(str(int(ep_id)))
        except (TypeError, ValueError):
            pass
    return cand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundles-root", required=True, type=Path,
                        help="Directory containing one or more BC bundles (recursively searched for episode.json).")
    parser.add_argument("--list", required=True, type=Path,
                        help="Text file with one episode identifier per line.")
    parser.add_argument("--label", default="true", type=_coerce_label,
                        help="Label to apply to matched episodes. true|false|null. Default: true.")
    parser.add_argument("--default-others", default=None, type=_coerce_label,
                        help="If set, write this label to every episode NOT matched in --list. Use false to mark unlabeled bundles as failures explicitly.")
    parser.add_argument("--dry-run", action="store_true", help="Compute matches but do not write episode.json files.")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="Optional JSON file summarising matched, unmatched, and unknown ids.")
    args = parser.parse_args()

    if not args.bundles_root.is_dir():
        print(f"error: --bundles-root does not exist or is not a directory: {args.bundles_root}", file=sys.stderr)
        return 2
    if not args.list.is_file():
        print(f"error: --list does not exist: {args.list}", file=sys.stderr)
        return 2

    wanted = _load_list(args.list)
    if not wanted:
        print(f"error: list file is empty after stripping blanks/comments: {args.list}", file=sys.stderr)
        return 2

    episode_jsons = sorted(Path(args.bundles_root).rglob("episode.json"))
    if not episode_jsons:
        print(f"error: no episode.json files under {args.bundles_root}", file=sys.stderr)
        return 2

    matched: list[tuple[Path, set[str]]] = []
    unmatched: list[Path] = []
    seen_keys: set[str] = set()

    for ej in episode_jsons:
        try:
            meta = json.loads(ej.read_text())
        except json.JSONDecodeError as exc:
            print(f"warn: skipping unreadable episode.json {ej}: {exc}", file=sys.stderr)
            continue
        keys = _candidate_keys(ej.parent, meta)
        hits = keys & wanted
        if hits:
            matched.append((ej, hits))
            seen_keys |= hits
        else:
            unmatched.append(ej)

    unknown = sorted(wanted - seen_keys)

    changed = 0
    for ej, _hits in matched:
        meta = json.loads(ej.read_text())
        if meta.get("success") != args.label:
            meta["success"] = args.label
            if not args.dry_run:
                ej.write_text(json.dumps(meta, indent=2))
            changed += 1

    cleared = 0
    if args.default_others is not None:
        for ej in unmatched:
            meta = json.loads(ej.read_text())
            if meta.get("success") != args.default_others:
                meta["success"] = args.default_others
                if not args.dry_run:
                    ej.write_text(json.dumps(meta, indent=2))
                cleared += 1

    print(f"scanned {len(episode_jsons)} episodes under {args.bundles_root}")
    print(f"matched {len(matched)} episodes -> success={args.label}{' (dry-run)' if args.dry_run else ''}")
    print(f"  wrote {changed} updated episode.json files")
    if args.default_others is not None:
        print(f"applied default --default-others={args.default_others} to {len(unmatched)} unmatched episodes")
        print(f"  wrote {cleared} updated episode.json files")
    if unknown:
        print(f"warning: {len(unknown)} list entries did not match any episode:")
        for k in unknown[:25]:
            print(f"  - {k}")
        if len(unknown) > 25:
            print(f"  ... and {len(unknown) - 25} more")

    if args.report_out is not None:
        report = {
            "bundles_root": str(args.bundles_root.resolve()),
            "list_file": str(args.list.resolve()),
            "label": args.label,
            "default_others": args.default_others,
            "dry_run": args.dry_run,
            "matched": [{"episode_json": str(ej), "matched_keys": sorted(hits)} for ej, hits in matched],
            "unmatched": [str(ej) for ej in unmatched],
            "unknown_list_entries": unknown,
        }
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2))
        print(f"wrote report -> {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
