#!/usr/bin/env python
"""Experiment 1: train fMRIEncoder + CLIPHead. Supports stop/resume (spec §13.3).

    python scripts/02_train_fmri_to_clip.py --config configs/exp01_fmri_to_clip.yaml
    # resume after interruption:
    python scripts/02_train_fmri_to_clip.py --config configs/exp01_fmri_to_clip.yaml --resume
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import train_clip  # noqa: E402
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="'--resume' -> auto (last.pt); '--resume PATH' -> explicit")
    args = ap.parse_args()
    log = get_logger("train_clip")

    cfg = load_config(args.config, args.set)
    result = train_clip(cfg, resume=args.resume)
    log.info("Finished. Experiment dir: %s", result["experiment_dir"])
    for split, m in result.get("metrics", {}).items():
        log.info("  %s retrieval: %s", split, m["retrieval"])


if __name__ == "__main__":
    main()
