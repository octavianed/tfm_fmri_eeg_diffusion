#!/usr/bin/env python
"""Precompute the frozen SD text-encoder embeddings of the caption prompts (§11).

Encodes ``"Image of <caption>"`` for every stimulus image, aligned by
``(subject, split, feat_idx)`` exactly like the CLIP/VAE features, plus the
permuted-caption control, the empty prompt (CFG negative branch) and the generic
control prompt. Prompts are deduplicated before encoding, so this is fast and the
cache stays small.

Depends only on the images and the split — NOT on the brain data. One cache is
therefore shared by every EEG preprocessing variant, exactly like the CLIP and
VAE-PCA features.

    python scripts/13_precompute_text_embeddings.py --config configs/fMRI/exp04_generation_text_weak.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule, validate_caption_alignment  # noqa: E402
from src.data.captions import resolve_caption_field, text_mode  # noqa: E402
from src.features import precompute_text_embeddings  # noqa: E402
from src.utils import (ExtendOverrides, get_logger, load_config,  # noqa: E402
                       set_seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate-only", action="store_true",
                    help="only check caption alignment; do not run the encoder")
    args = ap.parse_args()
    log = get_logger("precompute_text")

    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.get("project.seed", 42)))
    mode = text_mode(cfg)
    if mode == "none":
        log.warning("generation.text.mode=none in %s — nothing to do. Set "
                    "generation.text.mode=weak|oracle (or use one of the "
                    "exp04_*_text_* configs).", args.config)
        return

    field = resolve_caption_field(cfg)
    log.info("Text mode=%s | caption field=%s", mode, field)
    dm = build_datamodule(cfg).prepare()

    for subject in dm.subjects:
        for split in args.splits:
            if len(dm.subject_split_frame(subject, split)) == 0:
                continue
            report = validate_caption_alignment(cfg, dm, subject, split, field)
            if not report["aligned"]:
                raise SystemExit(f"Caption alignment failed: {report}")
            log.info("[align] %s/%s: %d rows, %d unique captions, e.g. %r",
                     subject, split, report["num_rows"],
                     report["num_unique_captions"],
                     report["examples"][0][field] if report["examples"] else "")
    if args.validate_only:
        log.info("Alignment OK (--validate-only, no embeddings written).")
        return

    info = precompute_text_embeddings(cfg, dm, splits=tuple(args.splits),
                                      overwrite=args.overwrite)
    log.info("Text embeddings ready: %s", info.get("dir"))


if __name__ == "__main__":
    main()
