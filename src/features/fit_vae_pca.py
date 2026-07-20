"""Fit PCA on VAE latents using the TRAIN split only, then transform all splits.

Prevents leakage (spec §3.5, §9.2, §20): the PCA basis is learned from train
latents, and val/test are only *projected* with that basis. The fitted model
(plus latent shape / scaling factor / explained variance) is pickled so the
low-level branch can be inverted back to a latent for generation Option C.
"""
from __future__ import annotations

import pickle

import numpy as np

from ..utils import (get_logger, save_json, vae_pca_feature_path,
                     vae_pca_model_path)
from .precompute_vae_latents import load_vae_meta
from ..utils.paths import vae_latent_path

logger = get_logger("fit_vae_pca")


def fit_vae_pca(cfg, datamodule, overwrite: bool = False) -> dict:
    from sklearn.decomposition import PCA

    pca_dim = int(cfg.get("features.pca_dim", 512))
    seed = int(cfg.get("project.seed", 42))
    summary = {}
    for subj in datamodule.subjects:
        model_path = vae_pca_model_path(cfg, subj)
        train_path = vae_latent_path(cfg, subj, "train")
        if not train_path.exists():
            raise FileNotFoundError(
                f"Missing VAE latents for {subj} train: {train_path}. "
                f"Run precompute_vae_latents first.")
        if model_path.exists() and not overwrite:
            logger.info("[skip] PCA exists: %s", model_path)
            bundle = pickle.loads(model_path.read_bytes())
        else:
            x_train = np.load(train_path)
            n_comp = int(min(pca_dim, x_train.shape[0], x_train.shape[1]))
            pca = PCA(n_components=n_comp, svd_solver="randomized",
                      random_state=seed)
            pca.fit(x_train)
            meta = load_vae_meta(cfg, subj)
            bundle = {
                "pca": pca,
                "latent_shape": meta["latent_shape"],
                "scaling_factor": meta["scaling_factor"],
                "pca_dim": int(pca.n_components_),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
            model_path.write_bytes(pickle.dumps(bundle))
            logger.info("[pca] %s: %d comps, EVR sum=%.4f -> %s", subj,
                        pca.n_components_,
                        float(pca.explained_variance_ratio_.sum()), model_path)

        pca = bundle["pca"]
        for split in ("train", "val", "test"):
            lat_path = vae_latent_path(cfg, subj, split)
            if not lat_path.exists():
                continue
            out_path = vae_pca_feature_path(cfg, subj, split)
            if out_path.exists() and not overwrite:
                continue
            z = pca.transform(np.load(lat_path)).astype(np.float32)
            np.save(out_path, z)
            logger.info("[pca] transformed %s/%s -> %s %s", subj, split,
                        out_path, z.shape)

        evr = np.asarray(bundle["explained_variance_ratio"], dtype=float)
        summary[subj] = {"n_components": int(bundle["pca_dim"]),
                         "evr_sum": float(evr.sum())}
        save_json({"explained_variance_ratio": evr.tolist(),
                   "cumulative": np.cumsum(evr).tolist(),
                   "pca_dim": int(bundle["pca_dim"])},
                  model_path.with_suffix(".evr.json"))
    return summary
