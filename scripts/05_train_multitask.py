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
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training and just re-run the final evaluation "
                         "from the existing best.pt (e.g. after a crash in the "
                         "evaluation step)")
    args = ap.parse_args()
    log = get_logger("train_multitask")

    cfg = load_config(args.config, args.set)
    result = train_multitask(cfg, resume=args.resume, eval_only=args.eval_only)
    log.info("Finished. Experiment dir: %s", result["experiment_dir"])
    for split, m in result.get("metrics", {}).items():
        log.info("  %s: retrieval=%s | low-level in test_lowlevel_metrics.json",
                 split, m["retrieval"])


if __name__ == "__main__":
    main()
