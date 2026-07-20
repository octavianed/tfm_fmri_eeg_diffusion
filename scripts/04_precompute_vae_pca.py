#!/usr/bin/env python
"""Precompute frozen VAE latents and fit PCA on TRAIN only (spec §13.5)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule  # noqa: E402
from src.features import fit_vae_pca, precompute_vae_latents  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = get_logger("precompute_vae_pca")

    cfg = load_config(args.config, args.set)
    dm = build_datamodule(cfg).prepare()
    precompute_vae_latents(cfg, dm, splits=tuple(args.splits),
                           overwrite=args.overwrite)
    summary = fit_vae_pca(cfg, dm, overwrite=args.overwrite)
    for subj, s in summary.items():
        log.info("%s: %d PCA comps, explained variance sum=%.4f",
                 subj, s["n_components"], s["evr_sum"])


if __name__ == "__main__":
    main()
