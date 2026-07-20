"""Baselines for fMRI -> CLIP (spec §16.2).

* :class:`MeanBaseline` — predicts the train-mean CLIP embedding for everything.
  It is a sanity floor: identical predictions give chance-level retrieval.
* :class:`RidgeRegression` — a regularized linear map, solved in the *dual*
  (sample space) so it scales to tens of thousands of voxels without forming a
  V×V matrix. Ridge is often a strong, honest fMRI baseline.
"""
from __future__ import annotations

import numpy as np

from .embedding_metrics import embedding_regression_metrics
from .eval_data import load_subject_matrices
from .retrieval_metrics import compute_retrieval_metrics


class MeanBaseline:
    def __init__(self):
        self.mean_ = None

    def fit(self, x_train, y_train):
        self.mean_ = np.asarray(y_train, dtype=np.float32).mean(axis=0)
        return self

    def predict(self, x):
        n = np.asarray(x).shape[0]
        return np.tile(self.mean_[None, :], (n, 1))


class RidgeRegression:
    """Regularized linear map, solved in whichever space is cheaper.

    The brain matrix is flattened to 2-D ``[N, D]`` first (a no-op for fMRI, which
    is already ``[N, V]``; for EEG it flattens ``[N, C, T] -> [N, C*T]`` — the
    natural linear baseline over channels x time). Then:

    * **primal** (``D <= N``): ``w = (XᵀX + aI)^{-1} Xᵀ Y``; predict ``X_test w``.
      Cheaper for EEG (``C*T`` features ≪ samples).
    * **dual** (``D > N``): ``predict = (X_test Xᵀ)(X Xᵀ + aI)^{-1} Y``. Cheaper
      for fMRI (tens of thousands of voxels ≫ samples), avoids a V×V matrix.

    Both give the same ridge solution, so the fMRI numbers are unchanged.
    """

    def __init__(self, alpha: float = 1000.0):
        self.alpha = float(alpha)
        self.mode_ = None
        self.x_train_ = None
        self.dual_ = None
        self.w_ = None

    @staticmethod
    def _flatten(x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x.reshape(x.shape[0], -1)

    def fit(self, x_train, y_train):
        x = self._flatten(x_train)
        y = np.asarray(y_train, dtype=np.float64)
        n, d = x.shape
        if d <= n:
            self.mode_ = "primal"
            gram = x.T @ x                       # [D, D]
            gram[np.diag_indices_from(gram)] += self.alpha
            self.w_ = np.linalg.solve(gram, x.T @ y)      # [D, C]
        else:
            self.mode_ = "dual"
            gram = x @ x.T                       # [N, N]
            gram[np.diag_indices_from(gram)] += self.alpha
            self.dual_ = np.linalg.solve(gram, y)         # [N, C]
            self.x_train_ = x
        return self

    def predict(self, x_test):
        x = self._flatten(x_test)
        if self.mode_ == "primal":
            return (x @ self.w_).astype(np.float32)
        return ((x @ self.x_train_.T) @ self.dual_).astype(np.float32)


def evaluate_baselines(cfg, datamodule, split: str = "test",
                       ridge_alpha: float = 1000.0, ks=(1, 5, 10)) -> dict:
    """Fit baselines on train, evaluate retrieval + embedding metrics on ``split``."""
    out = {"mean": {}, "ridge": {}, "ridge_alpha": ridge_alpha}
    for subject in datamodule.subjects:
        train = load_subject_matrices(cfg, datamodule, subject, "train",
                                      want=("fmri", "clip"))
        test = load_subject_matrices(cfg, datamodule, subject, split,
                                     want=("fmri", "clip"))
        if train.clip is None or test.clip is None:
            raise FileNotFoundError(
                "CLIP features missing; run precompute_clip first.")

        mean_model = MeanBaseline().fit(train.fmri, train.clip)
        mean_pred = mean_model.predict(test.fmri)
        m_ret, _ = compute_retrieval_metrics(mean_pred, test.clip, ks=ks)
        m_emb, _ = embedding_regression_metrics(mean_pred, test.clip)
        out["mean"][subject] = {"retrieval": m_ret, "embedding": m_emb}

        ridge = RidgeRegression(alpha=ridge_alpha).fit(train.fmri, train.clip)
        r_pred = ridge.predict(test.fmri)
        r_ret, _ = compute_retrieval_metrics(r_pred, test.clip, ks=ks)
        r_emb, _ = embedding_regression_metrics(r_pred, test.clip)
        out["ridge"][subject] = {"retrieval": r_ret, "embedding": r_emb}
    return out
