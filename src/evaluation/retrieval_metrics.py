"""Retrieval metrics between predicted and target CLIP embeddings (spec §8.3).

Query ``i`` (fMRI-predicted embedding) is ranked against a candidate pool of
target image embeddings; the correct candidate is index ``i`` (arrays are
assumed aligned). Reports Top-1/5/10, mean/median rank and mean cosine, plus
the chance levels for reference.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def compute_retrieval_metrics(pred: np.ndarray, target: np.ndarray,
                              ks: Sequence[int] = (1, 5, 10),
                              normalize: bool = True,
                              chunk: int = 2048) -> Tuple[dict, np.ndarray]:
    """Return (metrics dict, per-query 1-indexed ranks)."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    n = pred.shape[0]
    m = target.shape[0]
    if n == 0:
        return {"n_queries": 0}, np.array([])
    if normalize:
        pred = _l2norm(pred)
        target = _l2norm(target)

    ranks = np.empty(n, dtype=np.int64)
    diag_cos = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = pred[start:end] @ target.T  # [b, m]
        rows = np.arange(end - start)
        correct = np.arange(start, end)
        correct_sim = sims[rows, correct]
        diag_cos[start:end] = correct_sim
        # rank = 1 + number of candidates strictly more similar than the correct
        ranks[start:end] = (sims > correct_sim[:, None]).sum(axis=1) + 1

    metrics = {f"top{k}": float(np.mean(ranks <= k)) for k in ks}
    metrics.update({
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mean_cosine": float(diag_cos.mean()),
        "n_queries": int(n),
        "n_candidates": int(m),
    })
    for k in ks:
        metrics[f"chance_top{k}"] = float(k) / m
    return metrics, ranks


def topk_candidates(pred: np.ndarray, target: np.ndarray, k: int = 5,
                    normalize: bool = True) -> np.ndarray:
    """Return the indices of the top-``k`` candidates per query (for qualitative
    retrieval display in notebooks)."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if normalize:
        pred = _l2norm(pred)
        target = _l2norm(target)
    sims = pred @ target.T
    k = min(k, target.shape[0])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)
