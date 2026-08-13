"""Deterministic image-level train/val split shared by EEG code paths.

Both :class:`~src.data.eeg_datamodule.EegDataModule` and the raw preprocessing
pipeline (:mod:`src.preprocessing.build_variant`) must agree **exactly** on which
training images are validation, because MVNN may only be fitted on *train*
images (spec §3.1, §3.2, §5.9). Keeping the rule in one function removes any
chance of the two drifting apart.

Properties guaranteed here:

* the split is at **image** level, so all repetitions of an image stay together
  (never split across train/val);
* it is deterministic given ``(n_images, val_ratio, seed, subject)``;
* the official test set is always test — it is never touched by this function.
"""
from __future__ import annotations

import re
from typing import Dict

import numpy as np


def subject_num(subject: str) -> int:
    """Numeric part of a subject id (``sub-01`` → 1); 0 when absent."""
    m = re.search(r"(\d+)", str(subject))
    return int(m.group(1)) if m else 0


def image_level_split(n_images: int, val_ratio: float, seed: int,
                      subject: str = "") -> Dict[int, str]:
    """Assign each training image index to ``'train'`` or ``'val'``.

    Args:
        n_images: number of images in the training source.
        val_ratio: fraction carved out for validation.
        seed: base split seed (``dataset.split_seed``).
        subject: subject id — offsets the RNG so subjects differ.

    Returns:
        ``{image_index: 'train' | 'val'}`` for every index in ``range(n_images)``.
    """
    idx = np.arange(int(n_images))
    rng = np.random.default_rng(int(seed) + subject_num(subject))
    rng.shuffle(idx)
    n_val = int(round(float(val_ratio) * int(n_images)))
    return {int(i): ("val" if rank < n_val else "train")
            for rank, i in enumerate(idx)}


def train_image_indices(n_images: int, val_ratio: float, seed: int,
                        subject: str = "") -> np.ndarray:
    """Sorted indices of the images assigned to **train** (MVNN fit set)."""
    assign = image_level_split(n_images, val_ratio, seed, subject)
    return np.array(sorted(i for i, s in assign.items() if s == "train"),
                    dtype=np.int64)
