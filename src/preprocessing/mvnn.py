"""Multivariate noise normalization (MVNN) — spatial whitening (spec §5.10).

Estimates ``W ≈ Sigma_noise^(-1/2)`` and transforms every time sample:

    X_white(t) = W X(t)

The noise covariance is estimated **only from training trials** (spec §3.1):

1. group training repetitions by ``image_id``;
2. subtract, from each repetition, the mean of that image's repetitions
   (the residual is the trial-to-trial noise);
3. concatenate residuals across trials and time points;
4. estimate a regularized spatial covariance with Ledoit–Wolf shrinkage;
5. take ``W = Sigma^(-1/2)`` via eigendecomposition;
6. floor the eigenvalues to avoid unstable inversions (important after CAR,
   which makes the spatial matrix rank-deficient — spec §8.9).

Two hard ordering rules from the spec:

* MVNN must be **fitted before** the train/val split is violated — i.e. only on
  trials whose image belongs to the *train* split, never validation or test;
* MVNN must be **applied to individual repetitions before averaging** (§5.10),
  otherwise the averaging would destroy the trial-to-trial variability the
  covariance estimate needs.

Covariance/eigendecomposition run in float64; the transformed output is cast
back to float32 (spec §6.2).

The Ledoit–Wolf shrinkage estimator is implemented here in plain NumPy rather
than imported from scikit-learn: it keeps the estimate chunked (the residual
matrix of one training session is ~2 GB) and avoids ``sklearn.covariance``,
whose native extension is blocked by the Application Control policy on this
machine. The formula is the standard LW estimator with a scaled-identity target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from ..utils import get_logger

logger = get_logger("mvnn")


@dataclass
class MVNN:
    """A fitted whitening transform for one scope (e.g. subject × session)."""

    W: np.ndarray                 # float64 [n_channels, n_channels]
    n_channels: int
    n_trials_used: int
    n_images_used: int
    scope: str = "subject_session"
    shrinkage: Optional[float] = None
    eigenvalue_floor: float = 1e-8
    stats: Dict = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply whitening to ``[..., n_channels, n_times]`` data."""
        return apply_mvnn(x, self)


def _covariance_shrunk(res: np.ndarray, estimator: str = "ledoit_wolf",
                       chunk: int = 200_000) -> tuple:
    """Chunked spatial covariance of ``res`` ``[n_channels, n_observations]``.

    ``'ledoit_wolf'`` returns the shrunk estimate
    ``Sigma = a*mu*I + (1-a)*S`` with the analytically optimal ``a``; a
    scaled-identity target, i.e. the standard Ledoit–Wolf estimator.
    ``'empirical'`` returns ``S`` unshrunk.
    """
    p, n = res.shape
    S = np.zeros((p, p), dtype=np.float64)
    sum_norm4 = 0.0
    for i in range(0, n, chunk):
        b = np.asarray(res[:, i:i + chunk], dtype=np.float64)
        S += b @ b.T
        norm2 = np.einsum("ij,ij->j", b, b)      # squared norm of each observation
        sum_norm4 += float((norm2 ** 2).sum())
    S /= float(n)

    if estimator == "empirical":
        return S, 0.0
    if estimator != "ledoit_wolf":
        raise ValueError(f"Unknown covariance estimator: {estimator}")

    trace = float(np.trace(S))
    mu = trace / p
    s_fro2 = float((S ** 2).sum())
    # d2 = ||S - mu I||_F^2 / p ; b2 = E||x x^T - S||_F^2 / p (capped at d2)
    d2 = (s_fro2 - 2.0 * mu * trace + p * mu * mu) / p
    b2_bar = (sum_norm4 / (float(n) ** 2) - s_fro2 / float(n)) / p
    b2 = min(max(b2_bar, 0.0), max(d2, 0.0))
    shrinkage = 0.0 if d2 <= 0 else b2 / d2
    sigma = shrinkage * mu * np.eye(p) + (1.0 - shrinkage) * S
    return sigma, float(shrinkage)


def fit_mvnn(epochs: np.ndarray, image_ids: Sequence[int],
             covariance: str = "ledoit_wolf", eigenvalue_floor: float = 1e-8,
             scope: str = "subject_session") -> MVNN:
    """Fit MVNN from **training** epochs only.

    Args:
        epochs: ``[n_trials, n_channels, n_times]`` — training repetitions.
        image_ids: length ``n_trials``; repetitions of one image share an id.
        covariance: ``'ledoit_wolf'`` (default, shrinkage) or ``'empirical'``.
        eigenvalue_floor: lower bound applied to eigenvalues before inversion.

    Returns:
        A fitted :class:`MVNN`.
    """
    x = np.asarray(epochs, dtype=np.float32)     # float32 keeps the peak low
    if x.ndim != 3:
        raise ValueError(f"MVNN expects [n_trials, n_channels, n_times], got {x.shape}")
    ids = np.asarray(image_ids)
    if len(ids) != x.shape[0]:
        raise ValueError("image_ids length must match the number of trials")

    n_trials, n_ch, n_times = x.shape

    # 1-2) residuals: subtract the per-image mean from each repetition.
    residuals = np.empty_like(x)
    uniq = np.unique(ids)
    for img in uniq:
        sel = np.flatnonzero(ids == img)
        block = x[sel]
        residuals[sel] = block - block.mean(axis=0, keepdims=True)

    # 3) concatenate across trials and time -> [n_channels, n_trials*n_times]
    res = residuals.transpose(1, 0, 2).reshape(n_ch, -1)

    # 4-5) regularized covariance (chunked, float64 accumulation)
    n_residual_samples = int(res.shape[1])
    sigma, shrink = _covariance_shrunk(res, covariance)
    del residuals, res

    # 6) W = Sigma^(-1/2) with an eigenvalue floor
    evals, evecs = np.linalg.eigh(sigma)
    floor = float(eigenvalue_floor) * max(float(evals.max()), 1e-30)
    n_floored = int((evals < floor).sum())
    evals = np.clip(evals, floor, None)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T

    stats = {"cov_trace": float(np.trace(sigma)),
             "cov_cond": float(evals.max() / evals.min()),
             "eig_min": float(evals.min()), "eig_max": float(evals.max()),
             "n_eigenvalues_floored": n_floored,
             "residual_samples": n_residual_samples}
    logger.info("MVNN fitted (%s): %d ch | %d trials / %d images | shrinkage=%s "
                "| cond=%.3g | floored=%d", scope, n_ch, n_trials, len(uniq),
                "%.4f" % shrink if shrink is not None else "n/a",
                stats["cov_cond"], n_floored)
    return MVNN(W=W, n_channels=n_ch, n_trials_used=int(n_trials),
                n_images_used=int(len(uniq)), scope=scope, shrinkage=shrink,
                eigenvalue_floor=float(eigenvalue_floor), stats=stats)


def apply_mvnn(epochs: np.ndarray, mvnn: MVNN) -> np.ndarray:
    """Whiten ``[..., n_channels, n_times]`` data with a fitted :class:`MVNN`.

    Applied to **individual repetitions**, before any averaging (spec §5.10).
    """
    x = np.asarray(epochs, dtype=np.float64)
    if x.shape[-2] != mvnn.n_channels:
        raise ValueError(f"MVNN was fitted for {mvnn.n_channels} channels but "
                         f"got data with {x.shape[-2]}")
    out = np.einsum("ij,...jt->...it", mvnn.W, x)
    return np.ascontiguousarray(out, dtype=np.float32)


def whitened_covariance(epochs: np.ndarray, image_ids: Sequence[int]) -> np.ndarray:
    """Noise covariance of (already whitened) epochs — QC figure (spec §17.7).

    Should be close to the identity when MVNN worked as intended.
    """
    x = np.asarray(epochs, dtype=np.float64)
    ids = np.asarray(image_ids)
    residuals = np.empty_like(x)
    for img in np.unique(ids):
        sel = np.flatnonzero(ids == img)
        block = x[sel]
        residuals[sel] = block - block.mean(axis=0, keepdims=True)
    res = residuals.transpose(1, 0, 2).reshape(x.shape[1], -1)
    return np.cov(res, bias=True)
