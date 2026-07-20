"""Per-channel EEG normalization fitted on the TRAIN split only.

The temporal analogue of :class:`~src.data.fmri_normalization.FmriNormalizer`.
EEG trials are ``[C, T]`` (channels x time); statistics are computed **per
channel** (pooled over train trials and time), then applied to train/val/test:

    mu_c    = mean over train of x[:, c, :]
    sigma_c = std  over train of x[:, c, :]
    x_norm[c, t] = (x[c, t] - mu_c) / (sigma_c + eps)

Fitting uses exclusively train trials; the same parameters are applied to
val/test. Parameters are persisted to ``.npz`` so runs are reproducible and
leakage-free (spec §3.5, §7.4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class EegNormalizer:
    """Per-channel standardizer for EEG signals of shape ``[C, T]``.

    ``mean``/``std`` are length-``C`` vectors, broadcast over the time axis.
    """

    def __init__(self, mean=None, std=None, eps: float = 1e-6):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.eps = float(eps)

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, x_train: np.ndarray) -> "EegNormalizer":
        """Fit per-channel stats. ``x_train`` is ``[N, C, T]`` (train trials)."""
        x = np.asarray(x_train, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(f"EEG train array must be [N, C, T], got {x.shape}")
        # Pool over trials (axis 0) and time (axis 2) -> per-channel stats.
        self.mean = x.mean(axis=(0, 2)).astype(np.float32)
        self.std = x.std(axis=(0, 2)).astype(np.float32)
        return self

    def _reshape(self, arr: np.ndarray) -> np.ndarray:
        # mean/std are [C]; reshape so they broadcast against [..., C, T].
        return arr.reshape(-1, 1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("EegNormalizer must be fitted (or loaded) first")
        x = np.asarray(x, dtype=np.float32)
        mean = self._reshape(self.mean)
        std = self._reshape(self.std)
        return (x - mean) / (std + self.eps)

    def fit_transform(self, x_train: np.ndarray) -> np.ndarray:
        self.fit(x_train)
        return np.stack([self.transform(t) for t in np.asarray(x_train)], axis=0)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        x_norm = np.asarray(x_norm, dtype=np.float32)
        mean = self._reshape(self.mean)
        std = self._reshape(self.std)
        return x_norm * (std + self.eps) + mean

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, eps=np.float32(self.eps))

    @classmethod
    def load(cls, path) -> "EegNormalizer":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"], eps=float(data["eps"]))
