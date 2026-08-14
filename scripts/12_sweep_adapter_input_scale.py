#!/usr/bin/env python
"""Sweep the conditioning strength of a scale-invariant TokenAdapter (Option B).

    python scripts/12_sweep_adapter_input_scale.py \
        --config configs/EEG/exp04_63_generation.yaml \
        --adapter-checkpoint outputs/exp04_63_eeg_generation_normadapter/checkpoints/adapter_best.pt \
        --scales 0.6 0.8 1.0 1.2 1.4 1.8

Needs an adapter trained with ``generation.adapter_normalize_input=true``; then
``adapter_input_scale`` is an inference-only knob, so every value here reuses the
SAME checkpoint — **no retraining**. Training runs at 1.0, but a
stronger-than-nominal conditioning measurably raised CLIP similarity in this
project (docs/06_tokenadapter_y_generacion.md §2.3 bis), so 1.0 is not
necessarily the best value.

Scores each scale exactly like Experiment 5 (CLIP similarity generated vs real,
for correct/permuted/zero) and reports the correct−best-control margin, so a
value is picked by what matters instead of by assumption.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import sweep_adapter_input_scale  # noqa: E402
from src.generation.input_scale_sweep import DEFAULT_SCALES  # noqa: E402
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402


def resolve_decoder(cfg, explicit):
    """Same resolution order as scripts 06/08."""
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
    ap.add_argument("--adapter-checkpoint", default=None,
                    help="adapter trained with normalize_input=true (default: "
                         "generation.adapter_checkpoint or the experiment's adapter_best.pt)")
    ap.add_argument("--scales", nargs="*", type=float, default=None,
                    help=f"values to sweep (default: {list(DEFAULT_SCALES)})")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None,
                    help="override num_inference_steps for a faster sweep")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--no-save-images", action="store_true",
                    help="only compute metrics; skip writing PNGs/grids to disk")
    args = ap.parse_args()
    log = get_logger("sweep_scale")

    cfg = load_config(args.config, args.set)
    conditions = args.conditions or ["correct", "permuted", "zero"]
    decoder_ckpt = resolve_decoder(cfg, args.decoder_checkpoint)

    result = sweep_adapter_input_scale(
        cfg, decoder_ckpt, adapter_checkpoint=args.adapter_checkpoint,
        scales=args.scales or DEFAULT_SCALES, conditions=conditions,
        num_samples=args.num_samples, num_inference_steps=args.steps,
        split=args.split, save_images=not args.no_save_images)

    log.info("Saved to %s", result["out_dir"])
    margins = result["margins"]
    if len(margins):
        log.info("Margin (correct - best control) per scale:\n%s",
                 margins.to_string(index=False))
        best = margins.sort_values("margin", ascending=False).iloc[0]
        log.info("Recommended: adapter_input_scale=%g (margin %.4f, correct %.4f)",
                 best["input_scale"], best["margin"], best["correct"])
        log.info("Note: pick by margin over the controls, not by 'correct' alone — "
                 "a higher correct with an equally higher control means nothing.")


if __name__ == "__main__":
    main()
