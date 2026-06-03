#!/usr/bin/env python3
"""Train pi0.5 on DreamBC bundles via openpi.

This is a thin wrapper around ``openpi.scripts.train.main`` that:

1. Monkey-patches ``openpi.training.data_loader.create_torch_dataset`` so that
   when our :class:`DreamBCDataConfig` is in use (``repo_id == "dreambc_local"``)
   we return a :class:`DreamBCEpisodeDataset` built straight from local
   ``episode.npz``/``episode.json`` bundles instead of going through LeRobot.
2. Builds a :class:`openpi.training.config.TrainConfig` for either LoRA or
   full fine-tuning of ``pi05_droid``, reusing the original DROID norm stats.
3. Hands that config to ``openpi.scripts.train.main``.

Example (LoRA, smoke):

  python scripts/train_dreambc.py \
      --variant lora \
      --bundles-root rollouts/C16_g5_block \
      --exp-name smoke_lora \
      --num-train-steps 2 --batch-size 1 --save-interval 1 \
      --filter-mode all --no-wandb --overwrite

Example (full fine-tune, real run):

  python scripts/train_dreambc.py \
      --variant full \
      --bundles-root rollouts/C16_g5_block \
      --exp-name cube_full_v0 \
      --num-train-steps 6000 --batch-size 32 \
      --save-interval 1000 --keep-period 2000
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Optional

# Make repo-root scripts importable when run via `python scripts/train_dreambc.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# openpi imports happen after sys.path tweaks but before any data_loader calls.
import importlib.util  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as _opi_config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.weight_loaders as _weight_loaders  # noqa: E402

# ``openpi/scripts/train.py`` is shipped alongside the installed package but is
# *not* importable as ``openpi.scripts.train`` (it lives outside the ``src/``
# tree). Load it by file path so we can re-use its ``main(config)`` directly.
_OPENPI_TRAIN_PY = REPO_ROOT / "Ctrl-World" / "openpi" / "scripts" / "train.py"
if not _OPENPI_TRAIN_PY.is_file():
    raise FileNotFoundError(f"openpi train.py not found at {_OPENPI_TRAIN_PY}")
_spec = importlib.util.spec_from_file_location("_openpi_train_script", _OPENPI_TRAIN_PY)
train_module = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(train_module)

from bc_dataset import DreamBCEpisodeDataset  # noqa: E402
from dreambc_data_config import DREAMBC_REPO_ID, make_dreambc_data_config  # noqa: E402


# --------------------------------------------------------------------------- #
# DataLoader monkey-patch
# --------------------------------------------------------------------------- #


def _install_create_torch_dataset_patch(
    *,
    bundles_root: Path,
    filter_mode: str,
    include_list_path: Optional[Path],
    max_episodes: Optional[int],
) -> None:
    """Replace ``create_torch_dataset`` so DreamBC repo ids load locally.

    We delegate to the original implementation for any other repo id, so the
    patch is safe even if the openpi process needs to load some standard
    LeRobot dataset later (it does not in this launcher, but it's cheap to be
    defensive).
    """
    original_factory = _data_loader.create_torch_dataset

    include_list: list[str] | None = None
    if include_list_path is not None:
        include_list = [
            line.strip()
            for line in include_list_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def dreambc_factory(data_config, action_horizon, model_config):  # type: ignore[no-untyped-def]
        if data_config.repo_id != DREAMBC_REPO_ID:
            return original_factory(data_config, action_horizon, model_config)
        return DreamBCEpisodeDataset(
            bundle_dirs=[bundles_root],
            filter_mode=filter_mode,
            include_list=include_list,
            action_horizon=action_horizon,
            max_episodes=max_episodes,
        )

    _data_loader.create_torch_dataset = dreambc_factory


# --------------------------------------------------------------------------- #
# TrainConfig builders
# --------------------------------------------------------------------------- #


def _build_train_config(args: argparse.Namespace) -> _opi_config.TrainConfig:
    is_lora = args.variant == "lora"

    paligemma_variant = "gemma_2b_lora" if is_lora else "gemma_2b"
    action_expert_variant = "gemma_300m_lora" if is_lora else "gemma_300m"

    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    data = make_dreambc_data_config(
        asset_id="droid",
        assets_dir=args.assets_dir,
        prompt_from_task=False,
        enable_image_aug=args.enable_image_aug,
    )

    lr_schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=max(args.num_train_steps, args.warmup_steps + 1),
        decay_lr=args.decay_lr,
    )

    if is_lora:
        freeze_filter = pi0_config.Pi0Config(
            pi05=True,
            action_dim=args.action_dim,
            action_horizon=args.action_horizon,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
        ).get_freeze_filter()
        ema_decay: float | None = None  # LoRA fine-tuning conventionally disables EMA.
    else:
        import flax.nnx as nnx
        freeze_filter = nnx.Nothing
        ema_decay = None  # Disabled to halve TrainState memory; pi05 EMA is optional for short BC runs.

    return _opi_config.TrainConfig(
        name=args.config_name,
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=model,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.pi05_params),
        lr_schedule=lr_schedule,
        ema_decay=ema_decay,
        freeze_filter=freeze_filter,
        data=data,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period if args.keep_period > 0 else None,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=not args.no_wandb,
        fsdp_devices=args.fsdp_devices,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--variant", choices=("lora", "full"), required=True,
                   help="lora: gemma_2b_lora + gemma_300m_lora with frozen base weights; full: standard fine-tune of all params.")
    p.add_argument("--bundles-root", type=Path, required=True,
                   help="Directory containing BC bundles (recursively scanned for episode.json).")
    p.add_argument("--exp-name", required=True, help="Experiment name. Used in checkpoint dir and wandb run.")

    # Data filtering
    p.add_argument("--filter-mode", choices=("success_true", "include_list", "all"), default="success_true",
                   help="success_true: only episodes with success=true; include_list: only ids in --include-list; all: every episode.")
    p.add_argument("--include-list", type=Path, default=None,
                   help="Text file with one episode id/path per line. Required when --filter-mode=include_list.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="If set, cap the number of episodes loaded after filtering (debugging aid).")
    p.add_argument("--enable-image-aug", action="store_true",
                   help="Apply mild crop/brightness/contrast jitter to DROID camera images during training.")

    # Model / training scale
    p.add_argument("--action-dim", type=int, default=32, help="pi05 was trained with 32-dim padded actions; keep the default to reuse norm stats.")
    p.add_argument("--action-horizon", type=int, default=15, help="Must match the policy_action_horizon saved in the bundles.")
    p.add_argument("--num-train-steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Logging / checkpointing
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--keep-period", type=int, default=2000, help="Set to <=0 to disable.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--fsdp-devices", type=int, default=1)

    # Optimizer
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--peak-lr", type=float, default=2.5e-5)
    p.add_argument("--decay-lr", type=float, default=2.5e-6)

    # Paths
    p.add_argument("--config-name", default="pi05_dreambc",
                   help="TrainConfig.name. Used as the assets subdir & under checkpoint_base_dir.")
    p.add_argument("--project-name", default="DreamBC")
    p.add_argument("--pi05-params", default="gs://openpi-assets/checkpoints/pi05_droid/params",
                   help="Path or gs:// URI of the pretrained pi05_droid params to initialise from.")
    p.add_argument("--assets-dir", default="gs://openpi-assets/checkpoints/pi05_droid/assets",
                   help="Where the pi05_droid normalisation stats live. Local cache is auto-used.")
    p.add_argument("--assets-base-dir", default=str(REPO_ROOT / "training_runs" / "assets"),
                   help="Per-config asset base dir.")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "training_runs" / "checkpoints"),
                   help="Per-config checkpoint base dir. Final dir is <base>/<config-name>/<exp-name>/.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bundles_root = args.bundles_root.resolve()
    if not bundles_root.is_dir():
        print(f"error: --bundles-root does not exist: {bundles_root}", file=sys.stderr)
        return 2
    if args.filter_mode == "include_list" and args.include_list is None:
        print("error: --filter-mode=include_list requires --include-list", file=sys.stderr)
        return 2
    if args.include_list is not None and not args.include_list.is_file():
        print(f"error: --include-list file not found: {args.include_list}", file=sys.stderr)
        return 2

    Path(args.assets_base_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_base_dir).mkdir(parents=True, exist_ok=True)

    _install_create_torch_dataset_patch(
        bundles_root=bundles_root,
        filter_mode=args.filter_mode,
        include_list_path=args.include_list,
        max_episodes=args.max_episodes,
    )

    cfg = _build_train_config(args)
    print(
        f"[train_dreambc] variant={args.variant} bundles_root={bundles_root} "
        f"filter_mode={args.filter_mode} batch_size={cfg.batch_size} "
        f"num_train_steps={cfg.num_train_steps} ema_decay={cfg.ema_decay} "
        f"enable_image_aug={args.enable_image_aug}"
    )
    print(f"[train_dreambc] checkpoints -> {cfg.checkpoint_dir}")

    train_module.main(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
