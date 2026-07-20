"""Per-voxel fMRI normalization fitted on the TRAIN split only (spec §3.5).

    mu_v    = mean(x_train[:, v])
    sigma_v = std(x_train[:, v])
    x_norm  = (x - mu_v) / (sigma_v + eps)

Fitting uses exclusively train rows; the same parameters are applied to
train/val/test. Parameters are persisted to ``.npz`` so runs are reproducible
and leakage-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FmriNormalizer:
    def __init__(self, mean=None, std=None, eps: float = 1e-6):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.eps = float(eps)

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, x_train: np.ndarray) -> "FmriNormalizer":
        x = np.asarray(x_train, dtype=np.float64)
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FmriNormalizer must be fitted (or loaded) first")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + self.eps)

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        return self.fit(x_train).transform(x_train)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x_norm, dtype=np.float32)
        return x_norm * (self.std + self.eps) + self.mean

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, eps=np.float32(self.eps))

    @classmethod
    def load(cls, path) -> "FmriNormalizer":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"], eps=float(data["eps"]))


def fit_normalizer_from_indices(fmri: np.ndarray, train_indices,
                                eps: float = 1e-6) -> FmriNormalizer:
    """Fit a normalizer using only ``train_indices`` rows of ``fmri``."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    return FmriNormalizer(eps=eps).fit(fmri[train_indices])
