#!/usr/bin/env python
"""Experiment 4: generate images with a frozen Stable Diffusion pipeline (spec §13.7).

Optionally trains the (only) trainable module — the token adapter — then
generates images for every configured condition. The condition set follows
``generation.conditioning_architecture``:

* ``legacy_adapter`` — correct / permuted / zero (brain only);
* ``text_adapter_concat`` — the above with the caption fixed, plus
  ``permuted_text`` (correct brain, permuted caption);
* ``text_adapter_concat_controlnet`` — the above plus the semantic/low-level
  single-branch ablations of §23.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule  # noqa: E402
from src.generation import (conditioning_metadata, describe_conditions,  # noqa: E402
                            generate_images, resolve_conditions,
                            train_token_adapter)
from src.utils import (get_experiment_paths, ExtendOverrides, get_logger, load_config)  # noqa: E402


def resolve_decoder(cfg, explicit):
    if explicit:
        return Path(explicit)
    ck = cfg.get("generation.decoder_checkpoint")
    if ck:
        return Path(ck)
    out = Path(cfg.get("paths.output_dir", "outputs"))
    # Includes the renamed fMRI experiment; EEG/multimodal configs set
    # generation.decoder_checkpoint explicitly, so this is only a fallback.
    for name in ("exp03_fmri_lowlevel_multitask", "exp03_lowlevel_multitask",
                 "exp01_fmri_to_clip"):
        cand = out / name / "checkpoints" / "best.pt"
        if cand.exists():
            return cand
    raise FileNotFoundError("No decoder checkpoint found; train Experiment 1 or 3, "
                            "or pass --decoder-checkpoint.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--decoder-checkpoint", default=None)
    ap.add_argument("--adapter-checkpoint", default=None)
    ap.add_argument("--train-adapter", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    log = get_logger("generate")

    cfg = load_config(args.config, args.set)
    meta = conditioning_metadata(cfg)
    log.info("Architecture: %s | text: %s%s | controlnet: %s",
             meta["conditioning_architecture"], meta["text_mode"],
             f" ({meta['caption_field']})" if meta["caption_field"] else "",
             meta["controlnet_model"] or "off")
    conditions = args.conditions or None
    log.info("Conditions: %s", describe_conditions(resolve_conditions(cfg, conditions)))
    adapter_ckpt = args.adapter_checkpoint

    if args.train_adapter or bool(cfg.get("generation.train_adapter", False)):
        log.info("Training token adapter (frozen UNet diffusion loss) ...")
        dm = build_datamodule(cfg).prepare()
        info = train_token_adapter(cfg, dm,
                                   resume=cfg.get("generation.adapter_resume"))
        adapter_ckpt = info["adapter_checkpoint"]
        log.info("Token adapter ready: %s (loss=%.4f)", adapter_ckpt,
                 info["best_loss"])

    decoder_ckpt = resolve_decoder(cfg, args.decoder_checkpoint)
    log.info("Decoder checkpoint: %s", decoder_ckpt)
    outputs = generate_images(cfg, decoder_ckpt, adapter_checkpoint=adapter_ckpt,
                              conditions=conditions, split=args.split)
    paths = get_experiment_paths(cfg, ensure=True)
    log.info("Generated %d samples per condition -> %s",
             len(outputs["image_ids"]), paths.generated)


if __name__ == "__main__":
    main()
