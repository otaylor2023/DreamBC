#!/usr/bin/env python3
"""Summarize matched-seed post-training Ctrl-World eval labels and timing."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_ROOT = REPO / "rollouts" / "eval_post_training"
DEFAULT_RESULTS_ROOT = REPO / "results" / "post_training_eval"

CONFIG_COLUMNS = {
    "cube_base_pi05_baseline_g3": "base_g3_baseline",
    "cube_base_pi05_baseline_g5": "base_g5_baseline",
    "cube_ctrl_world_lora_g5": "ctrl_world_lora",
    "cube_teleop_lora_g5": "teleop_lora",
    "tomato_base_pi05_bowl_g3": "base",
    "tomato_ctrl_world_lora_bowl_g3": "ctrl_world_lora",
    "tomato_teleop_lora_bowl_g3": "teleop_lora",
}

TASK_DATASET_TOTALS = {
    "cube": {
        "name": "C16_g5_block",
        "rollouts": 60,
        "note": "10 initial labeled seeds plus 50 continuation seeds",
    },
    "tomato": {
        "name": "TL_g3_red_ball",
        "rollouts": 100,
        "note": "10 initial labeled seeds plus 90 continuation seeds",
    },
}


def _fmt_bool(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "TBD"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} sec"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    return f"{hours:.1f} hr"


def _load_cube_before(results_root: Path) -> dict[str, dict]:
    path = results_root / "cube_before_labels.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {row["val_id"]: row for row in data.get("labels", [])}


def _episode_rows(rollout_root: Path) -> list[dict]:
    rows: list[dict] = []
    for episode_json in sorted(rollout_root.rglob("episode.json")):
        meta = json.loads(episode_json.read_text())
        rel_parts = episode_json.relative_to(rollout_root).parts
        if len(rel_parts) < 5:
            continue
        task = rel_parts[0]
        config_name = rel_parts[1]
        column = CONFIG_COLUMNS.get(config_name, config_name)
        rows.append(
            {
                "task": task,
                "config_name": config_name,
                "column": column,
                "val_id": str(meta.get("episode_id", "")),
                "success": meta.get("success"),
                "label_note": meta.get("label_note"),
                "concat_video_path": meta.get("concat_video_path"),
                "episode_dir": str(episode_json.parent),
            }
        )
    return rows


def _build_table(rows: list[dict], cube_before: dict[str, dict]) -> dict:
    by_task: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

    for val_id, label in cube_before.items():
        by_task["cube"][val_id]["base_strict"] = label.get("strict_success")
        by_task["cube"][val_id]["base_lax"] = label.get("lax_success")
        by_task["cube"][val_id]["base_label"] = label.get("raw_label")

    for row in rows:
        by_task[row["task"]][row["val_id"]][row["column"]] = row

    summary: dict[str, dict] = {}
    for task, val_map in sorted(by_task.items()):
        task_rows = []
        totals = defaultdict(lambda: {"success": 0, "labeled": 0})
        for val_id, cells in sorted(val_map.items()):
            base_value = None
            base_display = "TBD"
            if task == "cube":
                strict = cells.get("base_strict")
                lax = cells.get("base_lax")
                base_display = f"{_fmt_bool(strict)} strict / {_fmt_bool(lax)} lax"
                for key, value in (("base_strict", strict), ("base_lax", lax)):
                    if value is not None:
                        totals[key]["labeled"] += 1
                        totals[key]["success"] += int(bool(value))
                for col_key in ("base_g3_baseline", "base_g5_baseline"):
                    row_entry = cells.get(col_key)
                    val = None if row_entry is None else row_entry.get("success")
                    if val is not None:
                        totals[col_key]["labeled"] += 1
                        totals[col_key]["success"] += int(bool(val))
            else:
                base_row = cells.get("base")
                base_value = None if base_row is None else base_row.get("success")
                base_display = _fmt_bool(base_value)
                if base_value is not None:
                    totals["base"]["labeled"] += 1
                    totals["base"]["success"] += int(bool(base_value))

            ctrl_row = cells.get("ctrl_world_lora")
            teleop_row = cells.get("teleop_lora")
            ctrl_value = None if ctrl_row is None else ctrl_row.get("success")
            teleop_value = None if teleop_row is None else teleop_row.get("success")

            for key, value in (("ctrl_world_lora", ctrl_value), ("teleop_lora", teleop_value)):
                if value is not None:
                    totals[key]["labeled"] += 1
                    totals[key]["success"] += int(bool(value))

            row_entry = {
                "val_id": val_id,
                "base": base_display,
                "base_label": cells.get("base_label"),
                "base_success": base_value,
                "ctrl_world_lora": _fmt_bool(ctrl_value),
                "teleop_lora": _fmt_bool(teleop_value),
            }
            if task == "cube":
                for col_key in ("base_g3_baseline", "base_g5_baseline"):
                    cell = cells.get(col_key)
                    row_entry[col_key] = _fmt_bool(None if cell is None else cell.get("success"))
            task_rows.append(row_entry)

        summary[task] = {"rows": task_rows, "totals": dict(totals)}
    return summary


def _count_text(totals: dict, key: str) -> str:
    t = totals.get(key, {"success": 0, "labeled": 0})
    if t["labeled"] == 0:
        return "TBD"
    return f"{t['success']}/{t['labeled']}"


def _write_summary_md(summary: dict, out_path: Path) -> None:
    lines = [
        "# Post-training Ctrl-World evaluation",
        "",
        "Matched-seed comparison using the same snapshots, prompts, guidance, and rollout length within each task. Only the policy checkpoint changes.",
        "",
    ]
    for task in ("cube", "tomato"):
        task_summary = summary.get(task)
        if not task_summary:
            continue
        totals = task_summary["totals"]
        lines.extend([f"## {task.title()}", ""])
        if task == "cube":
            lines.append(
                f"Totals: base strict {_count_text(totals, 'base_strict')}; "
                f"base lax {_count_text(totals, 'base_lax')}; "
                f"base g3 baseline {_count_text(totals, 'base_g3_baseline')}; "
                f"base g5 baseline {_count_text(totals, 'base_g5_baseline')}; "
                f"Ctrl-World LoRA {_count_text(totals, 'ctrl_world_lora')}; "
                f"teleop LoRA {_count_text(totals, 'teleop_lora')}."
            )
            lines.extend(
                [
                    "",
                    "| val_id | base (chat: strict/lax) | base g3 baseline | base g5 baseline | Ctrl-World LoRA | teleop LoRA |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in task_summary["rows"]:
                lines.append(
                    f"| {row['val_id']} | {row['base']} | {row.get('base_g3_baseline','TBD')} | "
                    f"{row.get('base_g5_baseline','TBD')} | {row['ctrl_world_lora']} | {row['teleop_lora']} |"
                )
        else:
            lines.append(
                f"Totals: base {_count_text(totals, 'base')}; "
                f"Ctrl-World LoRA {_count_text(totals, 'ctrl_world_lora')}; "
                f"teleop LoRA {_count_text(totals, 'teleop_lora')}."
            )
            lines.extend(
                [
                    "",
                    "| val_id | base | Ctrl-World LoRA | teleop LoRA |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for row in task_summary["rows"]:
                lines.append(
                    f"| {row['val_id']} | {row['base']} | {row['ctrl_world_lora']} | {row['teleop_lora']} |"
                )
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n")


def _timing_report(timing_path: Path) -> tuple[list[dict], str]:
    if not timing_path.is_file():
        return [], "No timing data found yet.\n"
    entries = json.loads(timing_path.read_text())
    ok_entries = [e for e in entries if e.get("return_code") == 0 and e.get("seconds_per_video")]

    lines = [
        "# Dataset generation timing",
        "",
        "Timing is measured from the matched post-training evaluation runs and extrapolated to the dataset sizes used for training.",
        "",
        "| Task | Dataset | Measured configs | Approx seconds/video | Extrapolated total |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for task in ("cube", "tomato"):
        task_entries = [e for e in ok_entries if e.get("task") == task]
        if not task_entries:
            info = TASK_DATASET_TOTALS[task]
            lines.append(f"| {task} | {info['name']} ({info['rollouts']} videos) | 0 | TBD | TBD |")
            continue
        sec_per_video = mean(float(e["seconds_per_video"]) for e in task_entries)
        info = TASK_DATASET_TOTALS[task]
        total_seconds = sec_per_video * info["rollouts"]
        lines.append(
            f"| {task} | {info['name']} ({info['rollouts']} videos) | {len(task_entries)} | "
            f"{sec_per_video:.1f} | {_duration(total_seconds)} |"
        )

    lines.extend(["", "Per-config timings:", ""])
    for e in entries:
        lines.append(
            f"- {e.get('config_name')}: {e.get('num_rollouts')} videos in "
            f"{_duration(float(e.get('elapsed_seconds', 0)))} "
            f"({float(e.get('seconds_per_video') or 0):.1f} sec/video), rc={e.get('return_code')}"
        )
    return entries, "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)
    rows = _episode_rows(args.rollout_root)
    cube_before = _load_cube_before(args.results_root)
    summary = _build_table(rows, cube_before)
    timing_entries, timing_md = _timing_report(args.results_root / "timing.json")

    output = {
        "rollout_root": str(args.rollout_root),
        "num_episode_rows": len(rows),
        "summary": summary,
        "timing": timing_entries,
    }
    (args.results_root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    _write_summary_md(summary, args.results_root / "summary.md")
    (args.results_root / "timing_report.md").write_text(timing_md)

    print(f"wrote {(args.results_root / 'summary.json')}")
    print(f"wrote {(args.results_root / 'summary.md')}")
    print(f"wrote {(args.results_root / 'timing_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
