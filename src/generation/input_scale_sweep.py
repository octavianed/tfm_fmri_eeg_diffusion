"""Sweep the conditioning strength of a scale-invariant TokenAdapter (Option B).

When the adapter normalizes its input (``generation.adapter_normalize_input``),
it is scale-invariant by construction and the *strength* of the conditioning
becomes an explicit knob: ``generation.adapter_input_scale`` (training always
runs at 1.0). That knob only acts at inference, so several values can be
compared **without retraining** — exactly like sweeping ``guidance_scale``.

This matters because the A/B measured in ``docs/06_...md`` §2.3 bis showed that a
*stronger-than-nominal* conditioning raised CLIP similarity: the natural value
(1.0, matching training) is not necessarily the best one.

Everything expensive — the frozen SD pipeline, the brain→CLIP predictions and
the adapter itself — is loaded ONCE; only the scalar changes per point, so the
sweep costs little more than the generations themselves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..data import build_datamodule, load_image
from ..evaluation.generation_metrics import compute_generation_metrics
from ..features.clip_model import load_clip
from ..utils import get_device, get_experiment_paths, get_logger, save_json
from .checkpoint_sweep import margin_table
from .generate_from_fmri import (load_decoder, lowlevel_init_images,
                                 predict_condition_embeddings,
                                 resolve_clip_rescale, save_condition_images,
                                 select_samples)
from .make_grids import save_comparison_grid
from .sd_pipeline import FrozenSDGenerator

logger = get_logger("input_scale_sweep")

DEFAULT_SCALES = (0.6, 0.8, 1.0, 1.2, 1.4, 1.8)


def sweep_adapter_input_scale(cfg, decoder_checkpoint, adapter_checkpoint=None,
                              scales: Sequence[float] = DEFAULT_SCALES,
                              conditions: Sequence[str] = ("correct", "permuted", "zero"),
                              num_samples: int = 8,
                              num_inference_steps: Optional[int] = None,
                              device=None, split: Optional[str] = None,
                              out_dir=None, save_images: bool = True) -> dict:
    """Generate at each ``adapter_input_scale`` and score by CLIP similarity.

    Requires an adapter trained with ``normalize_input=True`` (Option B) — with a
    plain adapter the scale is unused inside ``forward`` and every point would
    produce byte-identical images, so this raises instead of wasting GPU hours.

    Returns a dict with ``summary`` (one row per scale × condition), ``margins``
    (correct − best control per scale), ``images`` and ``out_dir``.
    """
    device = device or get_device(cfg.get("runtime.device", "auto"))
    split = split or str(cfg.get("generation.split", "test"))
    paths = get_experiment_paths(cfg, ensure=True)
    out_dir = Path(out_dir) if out_dir else paths.root / "input_scale_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.get("generation.mode", "adapter"))
    if mode not in ("adapter", "adapter_lowlevel"):
        raise ValueError(
            f"generation.mode={mode!r} never loads the token adapter, so "
            f"adapter_input_scale would have no effect. Use --set "
            f"generation.mode=adapter (or adapter_lowlevel).")

    scales = [float(s) for s in scales]
    if not scales:
        raise ValueError("No scales to sweep")

    dm = build_datamodule(cfg).prepare()
    model, meta = load_decoder(cfg, dm, decoder_checkpoint, device)
    sample_seed = int(cfg.get("generation.sample_seed", cfg.get("project.seed", 42)))
    selection = select_samples(dm, split, num_samples, sample_seed)
    image_ids = [s["image_id"] for s in selection]

    clip_by, low_by = predict_condition_embeddings(
        model, cfg, dm, selection, conditions, split, device, sample_seed)

    size = int(cfg.get("features.vae_image_size", 512))
    reals = [load_image(s["image_path"]).resize((size, size)) for s in selection]
    clip_bundle = load_clip(cfg, device)

    generator = FrozenSDGenerator(cfg, device=device)
    clip_rescale = resolve_clip_rescale(cfg, dm, generator)
    adapter_ck = adapter_checkpoint or cfg.get(
        "generation.adapter_checkpoint", str(paths.checkpoints / "adapter_best.pt"))
    generator.load_adapter(int(meta["clip_dim"]), adapter_ck)

    if not getattr(generator.adapter, "normalize_input", False):
        raise ValueError(
            "This adapter is NOT scale-invariant (normalize_input=False), so "
            "generation.adapter_input_scale is ignored inside forward() and every "
            "scale would give identical images. Retrain the adapter with "
            "--set generation.adapter_normalize_input=true (Option B), or use "
            "generation.rescale_clip_pred (Option A) to change the effective "
            "conditioning strength of a plain adapter.")

    gen_seed = int(cfg.get("generation.seed", 123))
    gs = float(cfg.get("generation.guidance_scale", 3.0))
    steps = int(num_inference_steps or cfg.get("generation.num_inference_steps", 50))
    strength = float(cfg.get("generation.strength", 0.8))
    logger.info("Sweeping adapter_input_scale over %s | adapter=%s | %d samples, "
                "%d steps", scales, adapter_ck, len(selection), steps)

    rows, images_by_scale = [], {}
    for scale in scales:
        label = f"scale_{scale:g}"
        generator.adapter.input_scale = float(scale)
        logger.info("  input_scale=%g", scale)
        sc_dir = out_dir / label
        outputs = {"real": reals, "image_ids": image_ids}
        for cond in conditions:
            init_images = None
            if generator.use_img2img and low_by.get(cond) is not None:
                init_images = lowlevel_init_images(cfg, generator, selection, low_by[cond])
            images = generator.generate(
                clip_by[cond], seed=gen_seed, guidance_scale=gs,
                num_inference_steps=steps, init_images=init_images, strength=strength)
            outputs[cond] = images
            if save_images:
                save_condition_images(images, sc_dir / cond, image_ids)

            res = compute_generation_metrics(reals, images, clip_bundle, device, ks=(1, 5))
            m = res["metrics"]
            rows.append({"input_scale": scale, "condition": cond,
                         "mean_clip_similarity": m["mean_clip_similarity"],
                         "median_clip_similarity": m["median_clip_similarity"],
                         "clip_top1": m["clip_retrieval"].get("top1"),
                         "clip_top5": m["clip_retrieval"].get("top5"),
                         "mean_pixel_mse": m.get("mean_pixel_mse")})
            logger.info("    %-9s clip_sim=%.4f top1=%.3f", cond,
                        m["mean_clip_similarity"], m["clip_retrieval"].get("top1"))
        images_by_scale[label] = outputs
        if save_images:
            save_comparison_grid(outputs, sc_dir / "comparison_grid.png",
                                 column_order=("real",) + tuple(conditions))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "input_scale_sweep_summary.csv", index=False)
    margins = margin_table(summary, key="input_scale")
    margins.to_csv(out_dir / "input_scale_sweep_margins.csv", index=False)
    save_json({"mode": mode, "scales": scales, "num_samples": len(selection),
               "num_inference_steps": steps, "guidance_scale": gs,
               "strength": strength, "image_ids": image_ids,
               "adapter_checkpoint": str(adapter_ck),
               "adapter_normalize_input": True,
               "rescale_clip_pred": cfg.get("generation.rescale_clip_pred", "none"),
               "clip_pred_target_norm": clip_rescale},
              out_dir / "sweep_params.json")
    save_input_scale_figure(summary, margins, out_dir / "input_scale_sweep.png")
    logger.info("Sweep done -> %s", out_dir)
    return {"summary": summary, "margins": margins, "images": images_by_scale,
            "out_dir": str(out_dir)}


def save_input_scale_figure(summary: pd.DataFrame, margins: pd.DataFrame,
                            out_path) -> Optional[str]:
    """CLIP similarity per condition and the correct−control margin vs scale."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("Figure skipped: %s", exc)
        return None

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for cond, g in summary.groupby("condition"):
        g = g.sort_values("input_scale")
        ax[0].plot(g["input_scale"], g["mean_clip_similarity"], "-o", label=cond)
    ax[0].axvline(1.0, color="k", ls="--", lw=0.8, label="1.0 (as trained)")
    ax[0].set_xlabel("adapter_input_scale"); ax[0].set_ylabel("CLIP similarity (gen vs real)")
    ax[0].set_title("Generation quality vs conditioning strength")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    if len(margins):
        m = margins.sort_values("input_scale")
        ax[1].plot(m["input_scale"], m["margin"], "-s", color="tab:green")
        ax[1].axhline(0.0, color="k", lw=0.8)
        ax[1].axvline(1.0, color="k", ls="--", lw=0.8)
        ax[1].set_xlabel("adapter_input_scale"); ax[1].set_ylabel("correct − best control")
        ax[1].set_title("Margin over the negative controls")
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)
