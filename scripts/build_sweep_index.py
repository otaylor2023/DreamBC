#!/usr/bin/env python3
"""Build rollouts/sweep/sweep_index.json from completed sweep episode bundles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _traj_id_from_dirname(name: str) -> str | None:
    match = re.search(r"_traj_(\d{4})_", name)
    return match.group(1) if match else None


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def build_index(sweep_root: Path) -> list[dict]:
    entries: list[dict] = []
    for object_dir in sorted(sweep_root.iterdir()):
        if not object_dir.is_dir() or object_dir.name.startswith("_"):
            continue
        object_name = object_dir.name
        for config_dir in sorted(object_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name
            sweep_config = _load_json(config_dir / "sweep_config.json") or {}
            bc_root = config_dir / "Rollouts_interact_pi" / "bc_episodes"
            if not bc_root.is_dir():
                continue
            for episode_dir in sorted(bc_root.iterdir()):
                if not episode_dir.is_dir():
                    continue
                episode_json_path = episode_dir / "episode.json"
                episode_meta = _load_json(episode_json_path) or {}
                val_id = episode_meta.get("episode_id") or _traj_id_from_dirname(episode_dir.name)
                views = episode_meta.get("source_view_keys") or sweep_config.get("views")
                concat_video = episode_meta.get("concat_video_path")
                entries.append(
                    {
                        "object": object_name,
                        "config": config_name,
                        "val_id": val_id,
                        "episode_dir": str(episode_dir.resolve()),
                        "instruction": episode_meta.get("instruction") or sweep_config.get("instructions", [None])[0],
                        "guidance_scale": episode_meta.get("guidance_scale", sweep_config.get("guidance_scale")),
                        "interact_num": episode_meta.get("interact_num", sweep_config.get("interact_num")),
                        "views": views,
                        "num_predicted_frames": episode_meta.get("num_predicted_frames"),
                        "num_policy_decisions": episode_meta.get("num_policy_decisions"),
                        "concat_video": concat_video,
                        "success": episode_meta.get("success"),
                    }
                )
    entries.sort(key=lambda e: (e["object"], e["config"], e["val_id"] or ""))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sweep_index.json for overnight Ctrl-World BC sweep.")
    parser.add_argument(
        "--sweep_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rollouts" / "sweep",
        help="Root directory containing cube/ and tomato/ config subdirs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <sweep_root>/sweep_index.json).",
    )
    args = parser.parse_args()
    sweep_root = args.sweep_root.resolve()
    output_path = (args.output or sweep_root / "sweep_index.json").resolve()

    entries = build_index(sweep_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(entries, f, indent=2)

    by_object: dict[str, int] = {}
    by_config: dict[str, int] = {}
    for e in entries:
        by_object[e["object"]] = by_object.get(e["object"], 0) + 1
        key = f"{e['object']}/{e['config']}"
        by_config[key] = by_config.get(key, 0) + 1

    print(f"Wrote {len(entries)} entries to {output_path}")
    print(f"  by object: {dict(sorted(by_object.items()))}")
    print(f"  configs with episodes: {len(by_config)}")


if __name__ == "__main__":
    main()
