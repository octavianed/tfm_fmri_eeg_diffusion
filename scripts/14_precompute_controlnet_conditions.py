#!/usr/bin/env python
"""Cache the ControlNet spatial conditions used to TRAIN the token adapter (§8).

For every image: ground-truth VAE-PCA vector -> inverse PCA -> VAE decode ->
coarse RGB -> Canny (or depth) -> PNG. Deliberately NOT ``GT image -> Canny``:
at inference the structure arrives from the brain through the same PCA
bottleneck, so training on a perfect outline would create a train/inference gap.

Requires ``scripts/04_precompute_vae_pca.py`` to have run. Depends only on the
images + PCA, so one cache serves every EEG preprocessing variant.

    python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_datamodule  # noqa: E402
from src.generation import precompute_controlnet_conditions  # noqa: E402
from src.generation.conditioning import (controlnet_settings,  # noqa: E402
                                         uses_controlnet)
from src.utils import (ExtendOverrides, get_experiment_paths,  # noqa: E402
                       get_logger, load_config, set_seed)


def _qc(cfg, dm, log, n: int = 6):
    """Save a coarse-vs-condition strip so the edge density can be eyeballed."""
    import numpy as np

    from src.features.load_features import load_split_features
    from src.generation.controlnet_condition import (build_control_images,
                                                     coarse_images_from_pca,
                                                     save_condition_qc)
    from src.features.precompute_vae_latents import load_vae
    from src.utils import get_device
    import torch
    from src.data.image_transforms import tensor_to_pil

    device = get_device(cfg.get("runtime.device", "auto"))
    # Same fp16 + slicing as the main pass, and freed before it starts: two
    # float32 VAEs resident at once was part of what filled the GPU.
    vae = load_vae(cfg, device,
                   dtype=torch.float16 if device.type == "cuda" else None,
                   slicing=True)

    def decode(latents):
        z = torch.as_tensor(latents, dtype=vae.dtype, device=device)
        with torch.no_grad():
            imgs = vae.decode(z / vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1).float().cpu()
        return [tensor_to_pil(i) for i in imgs]

    subject = dm.subjects[0]
    for split in ("test", "val", "train"):
        pca = load_split_features(cfg, subject, split, "low")
        if pca is not None and len(pca):
            break
    if pca is None:
        return
    coarse = coarse_images_from_pca(cfg, decode, subject,
                                    np.asarray(pca[:n], dtype=np.float32))
    conds = build_control_images(cfg, coarse, device)
    density = [float((np.asarray(c.convert("L")) > 127).mean()) for c in conds]
    out = get_experiment_paths(cfg, ensure=True).figures / "controlnet_condition_qc.png"
    save_condition_qc(conds, coarse, out)
    log.info("QC strip -> %s | edge density %.3f%% (mean of %d samples)",
             out, 100 * float(np.mean(density)), len(density))
    # The VAE (and the `decode` closure holding it) die when this frame returns;
    # the caller then empties the CUDA cache so the main pass starts clean.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--qc-only", action="store_true",
                    help="only render the QC strip (no full cache)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cache only the first N images per split (smoke tests; "
                         "pair it with generation.adapter_max_train_samples)")
    args = ap.parse_args()
    log = get_logger("precompute_controlnet")

    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.get("project.seed", 42)))
    if not uses_controlnet(cfg):
        raise SystemExit(
            "This config does not use ControlNet "
            "(generation.conditioning_architecture must be "
            "'text_adapter_concat_controlnet' with generation.controlnet.enabled=true).")
    settings = controlnet_settings(cfg)
    log.info("ControlNet %s | condition=%s | training source=%s",
             settings["model"], settings["condition_type"],
             settings["training_condition_source"])

    dm = build_datamodule(cfg).prepare()
    _qc(cfg, dm, log)
    if args.qc_only:
        return
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # release the QC VAE before the main pass
    results = precompute_controlnet_conditions(
        cfg, dm, splits=tuple(args.splits), overwrite=args.overwrite,
        batch_size=int(cfg.get("features.vae_batch_size", 8)), limit=args.limit)
    for (subject, split), n in results.items():
        log.info("%s/%s: %d condition images", subject, split, n)


if __name__ == "__main__":
    main()
