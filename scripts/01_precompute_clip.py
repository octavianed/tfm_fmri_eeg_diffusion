#!/usr/bin/env python
"""Precompute frozen-CLIP image embeddings for all (subject, split) (spec §13.2)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule  # noqa: E402
from src.features import precompute_clip  # noqa: E402
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    log = get_logger("precompute_clip")

    cfg = load_config(args.config, args.set)
    dm = build_datamodule(cfg).prepare()
    results = precompute_clip(cfg, dm, splits=tuple(args.splits),
                              overwrite=args.overwrite)
    for key, shape in results.items():
        log.info("%s -> %s", key, shape)


if __name__ == "__main__":
    main()
