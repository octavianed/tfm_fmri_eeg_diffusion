"""Generate images from fMRI-predicted representations under each condition.

The SAME sample set and generation seeds are used for the correct / permuted /
zero conditions so the comparison is fair (spec §10.4, §11.2). Permutation is a
full-split Sattolo derangement (via the ablation utilities); zero feeds a null
fMRI vector. If a token adapter is available the semantic (CLIP) branch drives
generation (Option B); the low-level branch can additionally seed img2img
(Option C).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..data import build_datamodule, load_image
from ..evaluation.ablation_eval import make_condition_input
from ..evaluation.eval_data import load_subject_matrices
from ..features.load_features import (clip_norm_reference,
                                      inverse_pca_to_latent, load_pca_bundle)
from ..models import build_model_from_checkpoint
from ..utils import (get_device, get_experiment_paths, get_logger, save_config,
                     save_json)
from .sd_pipeline import FrozenSDGenerator

logger = get_logger("generate")


def load_decoder(cfg, datamodule, checkpoint_path, device):
    model, state = build_model_from_checkpoint(
        cfg, checkpoint_path, device, voxel_counts=datamodule.voxel_counts)
    return model, {"clip_dim": state.get("clip_dim"),
                   "low_dim": state.get("low_dim"),
                   "use_lowlevel": bool(state.get("use_lowlevel", False))}


def select_samples(datamodule, split: str, n: int, seed: int) -> List[dict]:
    frame = datamodule.get_frame(split)
    rng = np.random.default_rng(seed)
    if n >= len(frame):
        idx = np.arange(len(frame))
    else:
        idx = np.sort(rng.choice(len(frame), size=n, replace=False))
    rows = frame.iloc[idx]
    return [{"subject": r.subject_id, "feat_idx": int(r.feat_idx),
             "image_path": str(r.image_path), "image_id": str(r.image_id)}
            for r in rows.itertuples()]


def _forward_full(model, fmri_np, subject, device, batch_size: int = 256):
    use_subject = getattr(model, "adapters", None) is not None
    clip, low = [], []
    with torch.no_grad():
        for start in range(0, len(fmri_np), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(fmri_np[start:start + batch_size])
            ).float().to(device)
            out = model(batch, subject=subject if use_subject else None)
            clip.append(out["clip"].float().cpu().numpy())
            if out.get("low") is not None:
                low.append(out["low"].float().cpu().numpy())
    return (np.concatenate(clip, 0),
            np.concatenate(low, 0) if low else None)


def predict_condition_embeddings(model, cfg, datamodule, selection, conditions,
                                 split, device, seed):
    subjects = sorted({s["subject"] for s in selection})
    noise_std = float(cfg.get("evaluation.noise_std", 1.0))
    cache = {}
    for subj in subjects:
        mats = load_subject_matrices(cfg, datamodule, subj, split, want=("fmri",))
        for cond in conditions:
            rng = np.random.default_rng(seed + hash(cond) % 100000)
            fin = make_condition_input(mats.fmri, cond, rng, noise_std)
            cache[(subj, cond)] = _forward_full(model, fin, subj, device)

    clip_by = {c: [] for c in conditions}
    low_by = {c: [] for c in conditions}
    for s in selection:
        for cond in conditions:
            clip_full, low_full = cache[(s["subject"], cond)]
            clip_by[cond].append(clip_full[s["feat_idx"]])
            low_by[cond].append(None if low_full is None else low_full[s["feat_idx"]])
    clip_out = {c: np.stack(v).astype(np.float32) for c, v in clip_by.items()}
    low_out = {}
    for c, v in low_by.items():
        low_out[c] = np.stack(v).astype(np.float32) if all(
            x is not None for x in v) else None
    return clip_out, low_out


def lowlevel_init_images(cfg, generator, selection, low_vectors):
    bundles = {}
    latents = []
    for i, s in enumerate(selection):
        subj = s["subject"]
        if subj not in bundles:
            bundles[subj] = load_pca_bundle(cfg, subj)
        lat = inverse_pca_to_latent(bundles[subj], low_vectors[i:i + 1])
        latents.append(lat[0])
    return generator.decode_latents_to_pil(np.stack(latents))


def save_condition_images(images, out_dir, image_ids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for img, iid in zip(images, image_ids):
        p = out_dir / f"{iid}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def resolve_clip_rescale(cfg, datamodule, generator=None) -> Optional[float]:
    """Resolve ``generation.rescale_clip_pred`` into a target norm (or ``None``).

    ``none`` (default) keeps the historical behaviour. ``train_median`` projects
    every predicted CLIP embedding onto the sphere whose radius is the median
    norm of the **real** train-split embeddings — the regime the TokenAdapter was
    trained in. A float value is used verbatim.
    """
    mode = cfg.get("generation.rescale_clip_pred", "none")
    target = None
    if mode is None or str(mode).lower() in ("none", "off", "false", ""):
        target = None
    elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
        target = float(mode)
    elif str(mode).lower() in ("train_median", "train_mean"):
        stat = "mean" if str(mode).lower().endswith("mean") else "median"
        target = clip_norm_reference(cfg, datamodule.subjects, "train", stat)
        if target is None:
            logger.warning("rescale_clip_pred=%s but no train CLIP features "
                           "found; skipping the rescaling.", mode)
    else:
        raise ValueError(f"Unknown generation.rescale_clip_pred: {mode!r}")

    if generator is not None:
        generator.rescale_to_norm = target
    if target is not None:
        logger.info("Calibrating predicted CLIP norm to %.3f (%s) before the adapter",
                    target, mode)
    return target


def generate_images(cfg, decoder_checkpoint, adapter_checkpoint=None,
                    conditions: Sequence[str] = ("correct", "permuted", "zero"),
                    device=None, split: Optional[str] = None) -> dict:
    device = device or get_device(cfg.get("runtime.device", "auto"))
    split = split or str(cfg.get("generation.split", "test"))
    paths = get_experiment_paths(cfg, ensure=True)
    save_config(cfg, paths.root / "config.yaml")
    dm = build_datamodule(cfg).prepare()

    model, meta = load_decoder(cfg, dm, decoder_checkpoint, device)
    n = int(cfg.get("generation.num_samples", 16))
    sample_seed = int(cfg.get("generation.sample_seed",
                              cfg.get("project.seed", 42)))
    selection = select_samples(dm, split, n, sample_seed)
    image_ids = [s["image_id"] for s in selection]

    clip_by, low_by = predict_condition_embeddings(
        model, cfg, dm, selection, conditions, split, device, sample_seed)

    generator = FrozenSDGenerator(cfg, device=device)
    clip_rescale = resolve_clip_rescale(cfg, dm, generator)
    adapter_used, adapter_loaded = None, False
    if generator.mode in ("adapter", "adapter_lowlevel"):
        adapter_ck = adapter_checkpoint or cfg.get(
            "generation.adapter_checkpoint",
            str(paths.checkpoints / "adapter_best.pt"))
        adapter_used = adapter_ck
        if adapter_ck and Path(adapter_ck).exists():
            generator.load_adapter(int(meta["clip_dim"]), adapter_ck)
            adapter_loaded = True
            logger.info("Token adapter loaded from %s", adapter_ck)
        else:
            logger.warning("No token adapter at %s; falling back to "
                           "unconditional prompt (rely on img2img init).",
                           adapter_ck)

    gen_seed = int(cfg.get("generation.seed", 123))
    gs = float(cfg.get("generation.guidance_scale", 3.0))
    steps = int(cfg.get("generation.num_inference_steps", 50))
    strength = float(cfg.get("generation.strength", 0.8))

    outputs = {"image_ids": image_ids, "subjects": [s["subject"] for s in selection]}
    # real reference images resized to the SD resolution
    size = int(cfg.get("features.vae_image_size", 512))
    reals = [load_image(s["image_path"]).resize((size, size)) for s in selection]
    outputs["real"] = reals
    save_condition_images(reals, paths.generated / "real", image_ids)

    for cond in conditions:
        init_images = None
        if generator.use_img2img and low_by.get(cond) is not None:
            init_images = lowlevel_init_images(cfg, generator, selection, low_by[cond])
        images = generator.generate(
            clip_by[cond], seed=gen_seed, guidance_scale=gs,
            num_inference_steps=steps, init_images=init_images, strength=strength)
        save_condition_images(images, paths.generated / cond, image_ids)
        outputs[cond] = images
        logger.info("Generated %d images for condition '%s'", len(images), cond)

    save_json({
        "mode": generator.mode, "conditions": list(conditions), "split": split,
        "num_samples": len(selection), "generation_seed": gen_seed,
        "sample_seed": sample_seed, "guidance_scale": gs,
        "num_inference_steps": steps, "strength": strength,
        "rescale_clip_pred": cfg.get("generation.rescale_clip_pred", "none"),
        "clip_pred_target_norm": clip_rescale,
        "sd_model": str(cfg.get("generation.sd_model", "")),
        "prompt_mode": str(cfg.get("generation.prompt_mode", "empty")),
        "decoder_checkpoint": str(decoder_checkpoint),
        # The RESOLVED adapter (CLI arg -> config -> experiment default), plus
        # whether it was actually loaded: recording the raw argument would log an
        # empty string whenever the path came from the config (spec §10.4).
        "adapter_checkpoint": str(adapter_used or ""),
        "adapter_loaded": bool(adapter_loaded),
        "image_ids": image_ids,
    }, paths.metadata / "generation_params.json")
    return outputs
