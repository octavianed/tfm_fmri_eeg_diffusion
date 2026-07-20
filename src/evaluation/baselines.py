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
    """Dual ridge: predict = (X_test X_train^T)(X_train X_train^T + a I)^{-1} Y."""

    def __init__(self, alpha: float = 1000.0):
        self.alpha = float(alpha)
        self.x_train_ = None
        self.dual_ = None

    def fit(self, x_train, y_train):
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        n = x.shape[0]
        gram = x @ x.T
        gram[np.diag_indices_from(gram)] += self.alpha
        self.dual_ = np.linalg.solve(gram, y)  # [n, c]
        self.x_train_ = x
        return self

    def predict(self, x_test):
        x_test = np.asarray(x_test, dtype=np.float64)
        return ((x_test @ self.x_train_.T) @ self.dual_).astype(np.float32)


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
