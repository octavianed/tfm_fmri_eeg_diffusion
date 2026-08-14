"""Load precomputed features and invert the VAE-PCA back to a latent."""
from __future__ import annotations

import pickle

import numpy as np

from ..utils import (clip_feature_path, vae_pca_feature_path,
                     vae_pca_model_path)
from ..utils.paths import vae_latent_path


def load_split_features(cfg, subject: str, split: str, kind: str = "clip"):
    """Load a [N, D] feature array for a (subject, split); None if missing."""
    if kind == "clip":
        path = clip_feature_path(cfg, subject, split)
    elif kind in ("low", "pca", "vae_pca"):
        path = vae_pca_feature_path(cfg, subject, split)
    elif kind in ("vae", "latent"):
        path = vae_latent_path(cfg, subject, split)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")
    return np.load(path) if path.exists() else None


def load_pca_bundle(cfg, subject: str) -> dict:
    path = vae_pca_model_path(cfg, subject)
    if not path.exists():
        raise FileNotFoundError(f"PCA model not found: {path}")
    return pickle.loads(path.read_bytes())


def inverse_pca_to_latent(bundle: dict, pca_vectors: np.ndarray) -> np.ndarray:
    """Map low-level PCA vectors back to SD-scaled latents ``[N, C, H, W]``."""
    pca = bundle["pca"]
    flat = pca.inverse_transform(np.asarray(pca_vectors, dtype=np.float32))
    shape = tuple(bundle["latent_shape"])
    return flat.reshape(-1, *shape).astype(np.float32)


def clip_norm_reference(cfg, subjects, split: str = "train",
                        stat: str = "median") -> Optional[float]:
    """Typical L2 norm of the **real** CLIP image embeddings (train split only).

    Used to calibrate the scale of *predicted* embeddings before the TokenAdapter
    (``generation.rescale_clip_pred``). Neither ``clip_pred`` loss term (cosine
    nor InfoNCE) constrains the norm, so a decoder's predictions can land on a
    shell of a completely different radius than the one the adapter was trained
    on — and the radius drifts from run to run, which would confound a
    generation comparison across preprocessing variants.

    Fitted on ``split='train'`` only, so it introduces no test leakage.
    Returns ``None`` when no CLIP features are available.
    """
    norms = []
    for subj in subjects:
        arr = load_split_features(cfg, subj, split, "clip")
        if arr is None or len(arr) == 0:
            continue
        norms.append(np.linalg.norm(np.asarray(arr, dtype=np.float64), axis=1))
    if not norms:
        return None
    alln = np.concatenate(norms)
    return float(np.median(alln) if stat == "median" else np.mean(alln))


def explained_variance(bundle: dict) -> np.ndarray:
    return np.asarray(bundle["explained_variance_ratio"], dtype=float)
