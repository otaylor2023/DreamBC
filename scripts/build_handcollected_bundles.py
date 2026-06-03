#!/usr/bin/env python3
"""Convert hand-collected hdf5 demos (in a zip) into DreamBC BC bundles.

Each zip contains ``<task>/demos_N.hdf5`` with groups::

  data/demo_N/obs/{exterior_2, wrist, ...}
  data/demo_N/obs/JOINT_POS, GRIPPER
  data/demo_N/actions/joint_velocity, gripper_position

We emit ``episode.npz`` + ``episode.json`` trees compatible with
``scripts/bc_dataset.DreamBCEpisodeDataset``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

ACTION_HORIZON = 15


def _parse_demo_ids(spec: str | None, max_demos: int | None) -> list[int]:
    if spec:
        ids: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ".." in part:
                lo_s, hi_s = part.split("..", 1)
                lo, hi = int(lo_s), int(hi_s)
                ids.extend(range(lo, hi + 1))
            else:
                ids.append(int(part))
        return sorted(set(ids))
    if max_demos is not None:
        return list(range(1, max_demos + 1))
    raise ValueError("provide --demo-ids or --max-demos")


def _demo_group_name(hdf5: h5py.File, demo_id: int) -> str:
    """Return the single ``data/demo_*`` group name for this demo id."""
    prefix = f"data/demo_{demo_id}"
    if prefix in hdf5:
        return prefix
    # Some files may use demo_1 inside demos_1.hdf5 only.
    candidates = [k for k in hdf5.keys() if k.startswith("data/")]
    for name in sorted(candidates):
        if name == prefix or name.endswith(f"/demo_{demo_id}"):
            return name
    raise KeyError(f"no demo group for id {demo_id} in {list(hdf5.keys())}")


def _build_action_chunks(joint_vel: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
    """(T, 15, 8) float32 action chunks; pad tail by repeating last row."""
    actions = np.concatenate(
        [joint_vel.astype(np.float32), gripper_pos.astype(np.float32)],
        axis=-1,
    )
    t = actions.shape[0]
    chunks = np.zeros((t, ACTION_HORIZON, 8), dtype=np.float32)
    for i in range(t):
        window = actions[i : i + ACTION_HORIZON]
        if window.shape[0] < ACTION_HORIZON:
            pad = np.repeat(actions[-1:], ACTION_HORIZON - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        chunks[i] = window
    return chunks


def _convert_one_demo(
    zf: zipfile.ZipFile,
    hdf5_name: str,
    demo_id: int,
    dst_episode_dir: Path,
    *,
    prompt: str,
    source_zip: str,
) -> None:
    data = zf.read(hdf5_name)
    with h5py.File(io.BytesIO(data), "r") as f:
        group = _demo_group_name(f, demo_id)
        g = f[group]
        obs = g["obs"]
        acts = g["actions"]

        exterior = np.asarray(obs["exterior_2"], dtype=np.uint8)
        wrist = np.asarray(obs["wrist"], dtype=np.uint8)
        joint_pos = np.asarray(obs["JOINT_POS"], dtype=np.float32)
        gripper = np.asarray(obs["GRIPPER"], dtype=np.float32)
        joint_vel = np.asarray(acts["joint_velocity"], dtype=np.float32)
        gripper_pos = np.asarray(acts["gripper_position"], dtype=np.float32)

    t = int(exterior.shape[0])
    if wrist.shape[0] != t or joint_pos.shape[0] != t:
        raise ValueError(
            f"length mismatch in {hdf5_name} demo {demo_id}: "
            f"exterior={exterior.shape[0]} wrist={wrist.shape[0]} joints={joint_pos.shape[0]}"
        )

    policy_action_chunk = _build_action_chunks(joint_vel, gripper_pos)

    dst_episode_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst_episode_dir / "episode.npz",
        policy_obs_exterior=exterior,
        policy_obs_wrist=wrist,
        policy_state_joint=joint_pos,
        policy_state_gripper=gripper.astype(np.float32),
        policy_action_chunk=policy_action_chunk,
    )

    meta = {
        "episode_id": demo_id,
        "success": True,
        "instruction": prompt,
        "num_policy_decisions": t,
        "policy_action_horizon": ACTION_HORIZON,
        "policy_obs_exterior_source": "exterior_2",
        "policy_obs_wrist_source": "wrist",
        "source_zip": source_zip,
        "source_hdf5": hdf5_name,
        "notes": "Hand-collected trajectory; exterior_2 + wrist for BC.",
    }
    (dst_episode_dir / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")


def _hdf5_member_for_demo(zf: zipfile.ZipFile, demo_id: int) -> str:
    pattern = re.compile(rf"(?:^|/)demos_{demo_id}\.hdf5$")
    matches = [n for n in zf.namelist() if pattern.search(n)]
    if not matches:
        raise FileNotFoundError(f"demos_{demo_id}.hdf5 not found in zip")
    if len(matches) > 1:
        # Prefer shortest path (e.g. cube/demos_1.hdf5)
        matches.sort(key=len)
    return matches[0]


def build_bundles(
    zip_path: Path,
    dst_root: Path,
    *,
    prompt: str,
    demo_ids: list[int],
    overwrite: bool,
) -> int:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {dst_root} (pass --overwrite)")
        import shutil

        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for demo_id in demo_ids:
            hdf5_name = _hdf5_member_for_demo(zf, demo_id)
            episode_dir = dst_root / "episodes" / f"demo_{demo_id:02d}"
            _convert_one_demo(
                zf,
                hdf5_name,
                demo_id,
                episode_dir,
                prompt=prompt,
                source_zip=str(zip_path.resolve()),
            )
            count += 1
            print(f"[handcollected] wrote {episode_dir} from {hdf5_name}")

    print(f"[handcollected] {count} episodes -> {dst_root}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--demo-ids", default=None, help='e.g. "1..30" or "1,2,5"')
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    demo_ids = _parse_demo_ids(args.demo_ids, args.max_demos)
    build_bundles(
        args.zip.resolve(),
        args.dst_root.resolve(),
        prompt=args.prompt,
        demo_ids=demo_ids,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
