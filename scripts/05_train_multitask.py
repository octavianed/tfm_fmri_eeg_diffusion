#!/usr/bin/env python
"""Experiment 3: train multitask decoder (CLIP + low-level). Resumable (spec §13.6).

    python scripts/05_train_multitask.py --config configs/exp03_lowlevel_multitask.yaml
    python scripts/05_train_multitask.py --config configs/exp03_lowlevel_multitask.yaml --resume
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import train_multitask  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    args = ap.parse_args()
    log = get_logger("train_multitask")

    cfg = load_config(args.config, args.set)
    result = train_multitask(cfg, resume=args.resume)
    log.info("Finished. Experiment dir: %s", result["experiment_dir"])
    for split, m in result.get("metrics", {}).items():
        log.info("  %s: retrieval=%s | low-level in test_lowlevel_metrics.json",
                 split, m["retrieval"])


if __name__ == "__main__":
    main()
