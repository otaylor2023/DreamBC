#!/usr/bin/env python3
"""Rewrite the ``instruction`` field of every ``episode.json`` under a bundles root.

The DreamBC dataset (``scripts/bc_dataset.py``) maps ``episode.json['instruction']``
to the model's ``prompt`` field, so this is the canonical knob for changing
training-time language conditioning. Inference-time prompts are independent
(passed via ``--instructions`` to ``rollout_interact_pi.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", required=True, type=Path)
    parser.add_argument("--prompt", required=True, help="New instruction text.")
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Only rewrite episodes with success=true (default: rewrite all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bundles_root.is_dir():
        raise SystemExit(f"error: --bundles-root not found: {args.bundles_root}")

    episode_jsons = sorted(args.bundles_root.rglob("episode.json"))
    if not episode_jsons:
        raise SystemExit(f"error: no episode.json under {args.bundles_root}")

    changed = 0
    skipped = 0
    unchanged = 0
    seen_prior: dict[str | None, int] = {}

    for ej in episode_jsons:
        meta = json.loads(ej.read_text())
        if args.only_success and meta.get("success") is not True:
            skipped += 1
            continue

        prior = meta.get("instruction")
        seen_prior[prior] = seen_prior.get(prior, 0) + 1

        if prior == args.prompt:
            unchanged += 1
            continue

        meta["instruction"] = args.prompt
        if not args.dry_run:
            ej.write_text(json.dumps(meta, indent=2))
        changed += 1

    action = "would change" if args.dry_run else "changed"
    print(f"{action} {changed} / {len(episode_jsons)} episode.json files")
    if args.only_success:
        print(f"skipped (success != true): {skipped}")
    print(f"unchanged (already matched new prompt): {unchanged}")
    print("prior instruction histogram:")
    for prior, count in sorted(seen_prior.items(), key=lambda x: -x[1]):
        print(f"  [{count:>4}] {prior!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
