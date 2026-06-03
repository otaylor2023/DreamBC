"""Torch ``Dataset`` over Ctrl-World+pi05 BC bundles.

Each bundle directory under ``bundles_root`` contains:

  bundle_root/
    ...
      bc_episodes/
        <episode_dir>/
          episode.json     # metadata (success label, instruction, ...)
          episode.npz      # arrays incl. policy_obs_{exterior,wrist},
                           # policy_state_{joint,gripper}, policy_action_chunk

``policy_obs_{exterior,wrist}``  shape: (T, 224, 224, 3)  uint8
``policy_state_joint``           shape: (T, 7)            float32
``policy_state_gripper``         shape: (T, 1)            float32
``policy_action_chunk``          shape: (T, action_horizon, 8)  float32

where T == ``num_policy_decisions``. Each (episode, t) pair becomes one
training sample so the dataset emits exactly the obs/action pairs pi0.5
actually saw at decision time, with no LeRobot ``delta_timestamps`` chunk
assembly.

The emitted sample dict matches the keys consumed by
``openpi.policies.droid_policy.DroidInputs``:

  observation/exterior_image_1_left : (224, 224, 3) uint8
  observation/wrist_image_left       : (224, 224, 3) uint8
  observation/joint_position         : (7,)          float32
  observation/gripper_position       : (1,)          float32
  actions                            : (action_horizon, 8) float32
  prompt                             : str
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from torch.utils.data import Dataset


# Keys we eagerly materialise from episode.npz into the per-episode cache.
# Decompression is what costs us, not memory; once we touch one frame from
# policy_obs_exterior the cheapest thing is to read the whole array.
_CACHED_KEYS: tuple[str, ...] = (
    "policy_obs_exterior",
    "policy_obs_wrist",
    "policy_state_joint",
    "policy_state_gripper",
    "policy_action_chunk",
)


def _episode_passes(meta: dict, episode_dir: Path, filter_mode: str, include_set: set[str]) -> bool:
    if filter_mode == "all":
        return True
    if filter_mode == "success_true":
        return meta.get("success") is True
    if filter_mode == "include_list":
        cand = {
            str(episode_dir),
            str(episode_dir.resolve()),
            episode_dir.name,
            str(meta.get("episode_id")),
        }
        try:
            cand.add(str(int(meta.get("episode_id"))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        return bool(cand & include_set)
    raise ValueError(f"unknown filter_mode={filter_mode!r}")


class DreamBCEpisodeDataset(Dataset):
    """Map ``(episode, decision_step)`` to a DroidInputs-ready sample."""

    def __init__(
        self,
        bundle_dirs: Sequence[str | Path],
        *,
        filter_mode: str = "success_true",
        include_list: Iterable[str] | None = None,
        action_horizon: int = 15,
        max_episodes: int | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.filter_mode = filter_mode
        self.include_set = set(include_list or [])
        self.action_horizon = action_horizon

        roots = [Path(p) for p in bundle_dirs]
        if not roots:
            raise ValueError("bundle_dirs must not be empty")

        self._index: list[tuple[Path, int, dict]] = []
        kept_episodes = 0
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"bundle dir does not exist: {root}")
            for ej in sorted(root.rglob("episode.json")):
                ed = ej.parent
                npz = ed / "episode.npz"
                if not npz.is_file():
                    continue
                try:
                    meta = json.loads(ej.read_text())
                except json.JSONDecodeError:
                    continue
                if not _episode_passes(meta, ed, self.filter_mode, self.include_set):
                    continue

                T = int(meta.get("num_policy_decisions") or 0)
                if T <= 0:
                    with np.load(npz) as z:
                        T = int(z["policy_obs_exterior"].shape[0])

                saved_horizon = int(meta.get("policy_action_horizon") or 0)
                if saved_horizon and saved_horizon != self.action_horizon:
                    raise ValueError(
                        f"episode {ed} was saved with policy_action_horizon={saved_horizon} "
                        f"which differs from dataset action_horizon={self.action_horizon}"
                    )

                for t in range(T):
                    self._index.append((ed, t, meta))
                kept_episodes += 1
                if max_episodes is not None and kept_episodes >= max_episodes:
                    break
            if max_episodes is not None and kept_episodes >= max_episodes:
                break

        if not self._index:
            raise ValueError(
                f"no episodes survived filter_mode={self.filter_mode!r} under bundle_dirs={[str(r) for r in roots]}"
            )

        # Per-episode array cache. Populated lazily in worker processes.
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

        if verbose:
            print(
                f"[DreamBCEpisodeDataset] {kept_episodes} episodes, "
                f"{len(self._index)} (episode, decision) samples "
                f"(filter_mode={self.filter_mode}, action_horizon={self.action_horizon})"
            )

    # Avoid sending the cache across the pickling boundary to workers.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = {}
        state.pop("_cache_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    @property
    def episode_dirs(self) -> list[Path]:
        seen: dict[Path, None] = {}
        for ed, _, _ in self._index:
            seen.setdefault(ed)
        return list(seen.keys())

    @property
    def num_episodes(self) -> int:
        return len(self.episode_dirs)

    def __len__(self) -> int:
        return len(self._index)

    def _load_episode(self, episode_dir: Path) -> dict[str, np.ndarray]:
        with self._cache_lock:
            cached = self._cache.get(episode_dir)
            if cached is not None:
                return cached
        # Load outside the lock; cost is O(decompress) once per episode per worker.
        with np.load(episode_dir / "episode.npz") as raw:
            arrays = {k: np.asarray(raw[k]) for k in _CACHED_KEYS}
        with self._cache_lock:
            self._cache.setdefault(episode_dir, arrays)
            return self._cache[episode_dir]

    def __getitem__(self, idx: int) -> dict:
        ed, t, meta = self._index[idx]
        a = self._load_episode(ed)
        sample = {
            "observation/exterior_image_1_left": np.ascontiguousarray(a["policy_obs_exterior"][t]),
            "observation/wrist_image_left": np.ascontiguousarray(a["policy_obs_wrist"][t]),
            "observation/joint_position": np.asarray(a["policy_state_joint"][t], dtype=np.float32),
            "observation/gripper_position": np.asarray(a["policy_state_gripper"][t], dtype=np.float32),
            "actions": np.asarray(a["policy_action_chunk"][t], dtype=np.float32),
            "prompt": meta["instruction"],
        }
        return sample
