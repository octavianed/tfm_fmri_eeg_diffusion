#!/usr/bin/env python
"""Sweep TokenAdapter checkpoints (best/last/epoch_XXXX.pt) and score each one
by generation quality (CLIP similarity, correct vs permuted vs zero) instead
of trusting the training loss — see docs/03_lowlevel_multitarea_y_generacion.md.

    python scripts/08_sweep_adapter_checkpoints.py --config configs/exp04_generation_v3.yaml

By default it discovers every adapter checkpoint under the experiment's own
checkpoints/ folder. Pass --num-samples to control how many test images are
used per checkpoint (small = fast) and --steps to shorten diffusion sampling
for a quick, rough pass before committing to the full config's step count.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import discover_adapter_checkpoints, sweep_adapter_checkpoints  # noqa: E402
from src.utils import ExtendOverrides, get_experiment_paths, get_logger, load_config  # noqa: E402


def resolve_decoder(cfg, explicit):
    if explicit:
        return explicit
    ck = cfg.get("generation.decoder_checkpoint")
    if ck:
        return ck
    from pathlib import Path
    out = Path(cfg.get("paths.output_dir", "outputs"))
    for name in ("exp03_lowlevel_multitask", "exp01_fmri_to_clip"):
        cand = out / name / "checkpoints" / "best.pt"
        if cand.exists():
            return cand
    raise FileNotFoundError("No decoder checkpoint found; pass --decoder-checkpoint.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--decoder-checkpoint", default=None)
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit checkpoint paths to sweep (default: "
                         "auto-discover best/last/epoch_XXXX.pt)")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--steps", type=int, default=None,
                    help="override num_inference_steps for a faster sweep")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--no-save-images", action="store_true",
                    help="only compute metrics; skip writing PNGs/grids to disk")
    args = ap.parse_args()
    log = get_logger("sweep")

    cfg = load_config(args.config, args.set)
    conditions = args.conditions or ["correct", "permuted", "zero"]
    decoder_ckpt = resolve_decoder(cfg, args.decoder_checkpoint)

    checkpoints = None
    if args.checkpoints:
        checkpoints = [(f"ckpt{i}", p) for i, p in enumerate(args.checkpoints)]

    result = sweep_adapter_checkpoints(
        cfg, decoder_ckpt, checkpoints=checkpoints, conditions=conditions,
        num_samples=args.num_samples, num_inference_steps=args.steps,
        split=args.split, save_images=not args.no_save_images)

    log.info("Resultados en: %s", result["out_dir"])
    log.info("\n%s", result["margins"].to_string(index=False))


if __name__ == "__main__":
    main()
