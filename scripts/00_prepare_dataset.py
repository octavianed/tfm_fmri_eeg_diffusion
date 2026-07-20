#!/usr/bin/env python
"""Prepare NSD Algonauts 2023: resolve subjects, build reproducible splits and
fit train-only fMRI normalization (spec §13.1)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None,
                    help="config overrides, e.g. dataset.subject_selection=all")
    ap.add_argument("--force", action="store_true",
                    help="rebuild metadata + normalization even if cached")
    args = ap.parse_args()
    log = get_logger("prepare")

    cfg = load_config(args.config, args.set)
    dm = build_datamodule(cfg).prepare(force_rebuild=args.force)

    log.info("Subjects: %s", dm.subjects)
    log.info("Voxel counts: %s", dm.voxel_counts)
    for split in ("train", "val", "test"):
        log.info("  %-5s: %d samples", split, len(dm.get_frame(split)))
    log.info("Done. Metadata + normalization written under data/processed/.")


if __name__ == "__main__":
    main()
