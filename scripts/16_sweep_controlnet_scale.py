#!/usr/bin/env python
"""Sweep the ControlNet conditioning scale (§25) — no retraining needed.

    python scripts/16_sweep_controlnet_scale.py \
        --config configs/fMRI/exp04_generation_controlnet_weak.yaml \
        --scales 0.0 0.25 0.5 0.75 1.0

``controlnet_conditioning_scale`` multiplies the ControlNet residuals inside the
frozen UNet, so it acts only at inference: every point reuses the SAME adapter
checkpoint. ``0.0`` is the ControlNet-disabled reference.

The value is chosen by absolute quality **and** by the correct−control margin,
not by which images look nicest: a scale where every condition looks good but
the margin collapses means the structure is coming from the ControlNet prior
rather than from the brain.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.controlnet_scale_sweep import (DEFAULT_SCALES,  # noqa: E402
                                                   sweep_controlnet_scale)
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402


def resolve_decoder(cfg, explicit):
    """Same resolution order as scripts 06/08/12."""
    if explicit:
        return explicit
    ck = cfg.get("generation.decoder_checkpoint")
    if ck:
        return ck
    out = Path(cfg.get("paths.output_dir", "outputs"))
    # Includes the renamed fMRI experiment; EEG/multimodal configs set
    # generation.decoder_checkpoint explicitly, so this is only a fallback.
    for name in ("exp03_fmri_lowlevel_multitask", "exp03_lowlevel_multitask",
                 "exp01_fmri_to_clip"):
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
    ap.add_argument("--adapter-checkpoint", default=None)
    ap.add_argument("--scales", nargs="*", type=float, default=None,
                    help=f"values to sweep (default: {list(DEFAULT_SCALES)})")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None,
                    help="override num_inference_steps for a faster sweep")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    log = get_logger("sweep_controlnet_scale")

    cfg = load_config(args.config, args.set)
    decoder = resolve_decoder(cfg, args.decoder_checkpoint)
    log.info("Decoder checkpoint: %s", decoder)

    result = sweep_controlnet_scale(
        cfg, decoder, adapter_checkpoint=args.adapter_checkpoint,
        scales=args.scales or DEFAULT_SCALES, conditions=args.conditions,
        num_samples=args.num_samples, num_inference_steps=args.steps,
        split=args.split, out_dir=args.out_dir, save_images=not args.no_images)

    margins = result["margins"]
    if len(margins):
        best_q = result["summary"].query("condition == 'correct'") \
            .sort_values("mean_clip_similarity", ascending=False)
        best_m = margins.sort_values("margin", ascending=False)
        log.info("Best absolute quality: scale=%s (clip_sim=%.4f)",
                 best_q.iloc[0]["controlnet_scale"],
                 best_q.iloc[0]["mean_clip_similarity"])
        log.info("Best correct-vs-control margin: scale=%s (margin=%+.4f)",
                 best_m.iloc[0]["controlnet_scale"], best_m.iloc[0]["margin"])
        log.info("Pick considering BOTH (§25), not the prettiest images.")
    log.info("Results -> %s", result["out_dir"])


if __name__ == "__main__":
    main()
