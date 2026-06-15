"""Image dataset reading crops from HDF5 with albumentations augmentation.

Classical augmentation only (flips/rotations/blur/distortion/color) -- NO
generative augmentation per the project's hard rule on medical images.

Also provides:
  - ``train_aug`` : the proven "transV2" albumentations recipe used by the top
    ISIC-2024 open-source teams (full | light variant, ``variant`` selectable).
  - ``ResampledLesionDataset`` : per-epoch negative subsampling -- keep ALL
    positives (optionally upsampled), draw a fresh ``n_pos * neg_ratio`` random
    negative subset every epoch. Makes each epoch a few-thousand images instead
    of 400k, and is what 2nd/3rd/4th place all used for the imbalance.
  - ``sample_weights`` : per-sample weights for a WeightedRandomSampler (the old
    default; kept as a config-selectable alternative to negative subsampling).
  - ``tta_transforms``/``apply_tta`` : test-time flips + rot90 used at inference
    to mean-pool sigmoid probs (off by default so the OOF stays deterministic).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import ID, TARGET, HDF5Images


def train_aug(img_size: int = 128, variant: str = "transV2"):
    """Training augmentation pipeline.

    ``variant``:
      - "transV2" (default): the full proven recipe -- Transpose, V/H flips,
        RandomBrightnessContrast, OneOf[blur/noise], OneOf[distortion], CLAHE,
        HueSaturationValue, ShiftScaleRotate, CoarseDropout (~0.375*img hole),
        Resize, ImageNet Normalize, ToTensorV2. Best for CNN backbones.
      - "light": flips + transpose + mild brightness/contrast + ShiftScaleRotate,
        NO heavy distortion / blur / CoarseDropout. Gentler recipe for ViT
        backbones, which are more sensitive to aggressive geometric/photometric
        corruption at small resolution.
      - "minimal": resize + flips + normalize only (debugging / fast baseline).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    variant = str(variant).lower()
    norm_tt = [A.Normalize(), ToTensorV2()]

    if variant in ("minimal", "min"):
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Transpose(p=0.5),
            *norm_tt,
        ])

    if variant in ("light", "vit"):
        return A.Compose([
            A.Transpose(p=0.5), A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                               border_mode=0, p=0.7),
            A.Resize(img_size, img_size),
            *norm_tt,
        ])

    # --- "transV2": the full proven recipe ---
    hole = max(1, int(round(0.375 * img_size)))   # CoarseDropout hole ~= 0.375*img
    return A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(std_range=(0.04, 0.2)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.0),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                           border_mode=0, p=0.85),
        A.Resize(img_size, img_size),
        A.CoarseDropout(num_holes_range=(1, 1),
                        hole_height_range=(hole, hole),
                        hole_width_range=(hole, hole),
                        fill=0, p=0.7),
        *norm_tt,
    ])


def eval_aug(img_size: int = 128):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])


def sample_weights(meta, pos_frac: float = 0.5) -> np.ndarray:
    """Per-sample weights for a WeightedRandomSampler targeting ``pos_frac``.

    With ``num_samples = len(meta)`` (one epoch's worth) and replacement, draws
    converge to a positive fraction of ``pos_frac`` regardless of the true
    prevalence. Positives and negatives each get uniform weight within class;
    the cross-class ratio encodes the target fraction.

    Degenerate folds (all one class) fall back to uniform weights.
    """
    y = np.asarray(meta[TARGET].values, dtype=np.float64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype=np.float64)
    pos_frac = float(min(max(pos_frac, 1e-6), 1.0 - 1e-6))
    w_pos = pos_frac / n_pos
    w_neg = (1.0 - pos_frac) / n_neg
    return np.where(y == 1, w_pos, w_neg).astype(np.float64)


def tta_transforms(n_tta: int = 4) -> list:
    """List of (flip_h, flip_w, k_rot90) ops for batched TTA, length ``n_tta``.

    Index 0 is always the identity so n_tta=1 == no TTA. The pool below is
    geometric only (flips + 90deg rotations), keeping TTA classical and
    label-preserving for dermoscopy crops.
    """
    pool = [
        (False, False, 0),   # identity
        (True, False, 0),    # horizontal flip
        (False, True, 0),    # vertical flip
        (False, False, 1),   # rot90
        (True, False, 1),    # hflip + rot90
        (False, True, 1),    # vflip + rot90
        (False, False, 2),   # rot180
        (False, False, 3),   # rot270
    ]
    n_tta = max(1, int(n_tta))
    if n_tta <= len(pool):
        return pool[:n_tta]
    # repeat the pool if more views than presets are requested
    return [pool[i % len(pool)] for i in range(n_tta)]


def apply_tta(x: torch.Tensor, op) -> torch.Tensor:
    """Apply one (flip_h, flip_w, k_rot90) op to an NCHW batch."""
    flip_h, flip_w, k = op
    if flip_h:
        x = torch.flip(x, dims=[3])   # flip width -> horizontal flip
    if flip_w:
        x = torch.flip(x, dims=[2])   # flip height -> vertical flip
    if k:
        x = torch.rot90(x, k=int(k), dims=[2, 3])
    return x


class LesionDataset(Dataset):
    def __init__(self, meta, hdf5_path: str, transform, has_target: bool = True):
        self.ids = meta[ID].tolist()
        self.y = meta[TARGET].tolist() if has_target else [0] * len(self.ids)
        self.hdf5_path, self.transform, self.has_target = hdf5_path, transform, has_target
        self._imgs = None  # opened lazily per worker

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        if self._imgs is None:
            self._imgs = HDF5Images(self.hdf5_path)
        img = self._imgs[self.ids[i]]
        img = self.transform(image=img)["image"]
        return img, torch.tensor(self.y[i], dtype=torch.float32), self.ids[i]


class ResampledLesionDataset(Dataset):
    """Per-epoch negative-subsampling dataset (the proven imbalance handling).

    Each epoch's view = ALL positives (optionally upsampled ``pos_mult`` times)
    + a FRESH random subset of ``n_pos * neg_ratio`` negatives, reshuffled. Call
    :meth:`resample` at the top of every epoch to redraw the negatives (and
    reshuffle) -- the training loop owns when that happens. This shrinks an epoch
    from ~400k images to a few thousand, so epochs are tiny and fast, while still
    showing the model fresh negatives across epochs.

    ``neg_ratio`` is negatives-per-positive measured against the RAW positive
    count (not the upsampled one), matching how 2nd/3rd/4th place specified it:
      - 3rd place: pos_mult=2, neg_ratio~=1 (so ~2*n_pos pos vs n_pos neg -> ~2:1
        positives, i.e. roughly balanced after upsampling).
      - 4th place swept neg_ratio in 3..50.
    Default here: pos_mult=2, neg_ratio=7 -> per epoch ~2*393 pos + 7*393 neg
    ~= 786 pos + 2751 neg ~= 3.5k images.

    Validation must NOT use this -- the held-out fold stays full so OOF pAUC is
    honest. This dataset is for the TRAIN side only.
    """

    def __init__(self, meta, hdf5_path: str, transform, *,
                 neg_ratio: float = 7.0, pos_mult: int = 2, seed: int = 42):
        y = np.asarray(meta[TARGET].values).astype(int)
        ids = np.asarray(meta[ID].values)
        self.pos_ids = ids[y == 1]
        self.neg_ids = ids[y == 0]
        self.pos_mult = max(1, int(pos_mult))
        self.neg_ratio = float(neg_ratio)
        self.n_pos = int(len(self.pos_ids))
        # negatives to draw per epoch, capped at available negatives
        self.n_neg_draw = int(min(len(self.neg_ids),
                                  round(self.n_pos * self.neg_ratio)))
        self.hdf5_path, self.transform = hdf5_path, transform
        self._imgs = None
        self._rng = np.random.default_rng(int(seed))
        self._epoch_ids: np.ndarray = np.empty(0, dtype=object)
        self._epoch_y: np.ndarray = np.empty(0, dtype=np.float32)
        self.resample()

    def resample(self) -> None:
        """Redraw the negative subset and reshuffle the epoch order. Call once
        per epoch BEFORE iterating the DataLoader."""
        pos = np.tile(self.pos_ids, self.pos_mult)
        if self.n_neg_draw > 0:
            neg = self._rng.choice(self.neg_ids, size=self.n_neg_draw, replace=False)
        else:
            neg = np.empty(0, dtype=self.neg_ids.dtype)
        ids = np.concatenate([pos, neg])
        ys = np.concatenate([
            np.ones(len(pos), dtype=np.float32),
            np.zeros(len(neg), dtype=np.float32),
        ])
        order = self._rng.permutation(len(ids))
        self._epoch_ids = ids[order]
        self._epoch_y = ys[order]

    def __len__(self) -> int:
        return int(len(self._epoch_ids))

    def __getitem__(self, i: int):
        if self._imgs is None:
            self._imgs = HDF5Images(self.hdf5_path)
        isic_id = str(self._epoch_ids[i])
        img = self._imgs[isic_id]
        img = self.transform(image=img)["image"]
        return img, torch.tensor(self._epoch_y[i], dtype=torch.float32), isic_id
