"""openpi ``DataConfigFactory`` for DreamBC bundles.

The pi05_droid weight checkpoint expects the same per-key normalisation that
was used during the original DROID training, so we reuse those norm stats
from ``gs://openpi-assets/checkpoints/pi05_droid/assets`` (already cached
locally under ``~/.cache/openpi``).

This factory is deliberately minimal:
  * No ``repack_transforms``: our ``DreamBCEpisodeDataset`` already emits the
    canonical ``observation/...`` / ``actions`` / ``prompt`` keys.
  * ``data_transforms`` = ``DroidInputs`` + ``DroidOutputs`` (same as
    ``LeRobotDROIDDataConfig`` / ``RLDSDroidDataConfig``).
  * ``model_transforms`` = standard ``ModelTransformFactory`` (resize to
    224x224, tokenise prompt, pad states/actions to ``action_dim``).

We do not depend on LeRobot or HuggingFace at all; the dataset is fetched
from the local filesystem by ``DreamBCEpisodeDataset``.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
from typing_extensions import override

import openpi.models.model as _model
import openpi.policies.droid_policy as droid_policy
import openpi.training.config as _opi_config
import openpi.transforms as _transforms

# DROID image keys emitted by DreamBCEpisodeDataset (before DroidInputs repack).
_DROID_IMAGE_KEYS = (
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)


# The DataConfig.repo_id we use as a sentinel for the launcher's monkey-patch
# on ``create_torch_dataset``. Picked to be distinctive so we never collide
# with an actual LeRobot repo id.
DREAMBC_REPO_ID = "dreambc_local"


def _resize_uint8_hwc(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize for uint8 HWC images (PIL only, no JAX/torch)."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil.resize((width, height), Image.BILINEAR), dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class DroidImageAugmentation(_transforms.DataTransformFn):
    """Mild same-env augmentation on raw DROID observation images.

    Applied before :class:`droid_policy.DroidInputs`. 30% of samples pass through
    unchanged so the batch always contains clean examples.
    """

    pass_through_prob: float = 0.30
    crop_min_frac: float = 0.92
    crop_max_frac: float = 1.00
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15

    def __call__(self, data: dict) -> dict:
        rng = np.random.default_rng()
        if rng.random() < self.pass_through_prob:
            return data

        out = dict(data)
        for key in _DROID_IMAGE_KEYS:
            if key not in out:
                continue
            out[key] = self._augment_image(np.asarray(out[key], dtype=np.uint8), rng)
        return out

    def _augment_image(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = image.shape[:2]
        crop_frac = float(rng.uniform(self.crop_min_frac, self.crop_max_frac))
        crop_h = max(1, int(round(h * crop_frac)))
        crop_w = max(1, int(round(w * crop_frac)))
        top = int(rng.integers(0, max(1, h - crop_h + 1)))
        left = int(rng.integers(0, max(1, w - crop_w + 1)))
        cropped = image[top : top + crop_h, left : left + crop_w]
        image = _resize_uint8_hwc(cropped, h, w)

        brightness = 1.0 + float(rng.uniform(-self.brightness_jitter, self.brightness_jitter))
        contrast = 1.0 + float(rng.uniform(-self.contrast_jitter, self.contrast_jitter))
        image = np.clip((image.astype(np.float32) - 128.0) * contrast + 128.0 * brightness, 0, 255)
        return image.astype(np.uint8)


@dataclasses.dataclass(frozen=True)
class DreamBCDataConfig(_opi_config.DataConfigFactory):
    """``DataConfigFactory`` that wires up DreamBC bundles for openpi.

    ``repo_id`` is fixed to :data:`DREAMBC_REPO_ID`. ``assets`` should point
    at the original pi05 droid asset directory so that norm stats match the
    pretrained checkpoint.
    """

    repo_id: str = DREAMBC_REPO_ID
    enable_image_aug: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _opi_config.DataConfig:
        input_transforms: list[_transforms.DataTransformFn] = []
        if self.enable_image_aug:
            input_transforms.append(DroidImageAugmentation())
        input_transforms.append(droid_policy.DroidInputs(model_type=model_config.model_type))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = _opi_config.ModelTransformFactory()(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def make_dreambc_data_config(
    *,
    asset_id: str = "droid",
    assets_dir: str = "gs://openpi-assets/checkpoints/pi05_droid/assets",
    prompt_from_task: bool = False,
    enable_image_aug: bool = False,
    extra_base_config: dict[str, Any] | None = None,
) -> DreamBCDataConfig:
    """Helper that builds a :class:`DreamBCDataConfig` with sensible defaults.

    The default ``assets`` directory reuses the pi05_droid norm stats — which
    is what we want for fine-tuning, since our actions/state come from the
    same DROID action space.
    """
    base_kwargs: dict[str, Any] = {"prompt_from_task": prompt_from_task}
    if extra_base_config:
        base_kwargs.update(extra_base_config)
    return DreamBCDataConfig(
        repo_id=DREAMBC_REPO_ID,
        enable_image_aug=enable_image_aug,
        assets=_opi_config.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
        base_config=_opi_config.DataConfig(**base_kwargs),
    )
