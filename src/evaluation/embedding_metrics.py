"""Regression-style metrics for predicted embeddings / low-level vectors.

Used for the CLIP head (MSE/MAE/cosine) and, in Experiment 3, for the
low-level PCA branch (per-component Pearson correlation, spec §9.5).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def per_component_pearson(pred: np.ndarray, target: np.ndarray,
                          eps: float = 1e-8) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p = pred - pred.mean(axis=0, keepdims=True)
    t = target - target.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0)) + eps
    return (num / den).astype(np.float64)


def embedding_regression_metrics(pred: np.ndarray,
                                 target: np.ndarray) -> Tuple[dict, np.ndarray]:
    """Return (metrics dict, per-component Pearson array)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    tgt_n = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    mean_cosine = float(np.mean(np.sum(pred_n * tgt_n, axis=1)))

    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    pearson = per_component_pearson(pred, target)
    metrics = {
        "mse": mse, "mae": mae, "mean_cosine": mean_cosine,
        "r2": float(r2), "mean_pearson": float(np.nanmean(pearson)),
        "median_pearson": float(np.nanmedian(pearson)),
        "n": int(pred.shape[0]), "dim": int(pred.shape[1]),
    }
    return metrics, pearson
