"""Sweep ``generation.controlnet.conditioning_scale`` at inference time (§25).

The strength of the structural branch is a free knob that does NOT require
retraining the adapter, exactly like ``adapter_input_scale``: diffusers scales
the ControlNet residuals by it, so ``0.0`` disables the branch entirely and
``1.0`` applies it at full strength.

The value is **not** chosen by "which image looks nicest". Following the
project's criterion, each point is scored by two numbers (§25):

* absolute quality — mean CLIP similarity of the ``correct`` condition;
* margin — ``correct`` minus the best negative control.

A scale that makes every condition look good but collapses the margin is worse
than a weaker one that preserves it: it would mean the structure comes from the
ControlNet prior, not from the brain.

Everything expensive (SD pipeline, ControlNet, brain predictions, control
images) is built ONCE; only the scalar changes per point.
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
from .conditioning import (controlnet_settings, required_brain_conditions,
                           resolve_conditions, uses_controlnet)
from .generate_from_fmri import (build_control_inputs, build_text_inputs,
                                 load_decoder, lowlevel_init_images,
                                 predict_condition_embeddings,
                                 resolve_adapter_checkpoint,
                                 resolve_clip_rescale, save_condition_images,
                                 select_samples)
from .make_grids import save_comparison_grid
from .sd_pipeline import FrozenSDGenerator

logger = get_logger("controlnet_sweep")

#: Recommended starting grid (§25); 0.0 = ControlNet disabled.
DEFAULT_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def sweep_controlnet_scale(cfg, decoder_checkpoint, adapter_checkpoint=None,
                           scales: Sequence[float] = DEFAULT_SCALES,
                           conditions: Optional[Sequence[str]] = None,
                           num_samples: int = 8,
                           num_inference_steps: Optional[int] = None,
                           device=None, split: Optional[str] = None,
                           out_dir=None, save_images: bool = True) -> dict:
    """Generate at each ``controlnet_conditioning_scale`` and score the result."""
    if not uses_controlnet(cfg):
        raise ValueError(
            "This config has no ControlNet, so conditioning_scale does nothing. "
            "Use generation.conditioning_architecture=text_adapter_concat_controlnet "
            "with generation.controlnet.enabled=true.")
    device = device or get_device(cfg.get("runtime.device", "auto"))
    split = split or str(cfg.get("generation.split", "test"))
    paths = get_experiment_paths(cfg, ensure=True)
    out_dir = Path(out_dir) if out_dir else paths.root / "controlnet_scale_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = [float(s) for s in scales]
    if not scales:
        raise ValueError("No scales to sweep")

    # Only the conditions whose structural branch is ON are informative here:
    # a condition with structural='zero' is scale-independent by construction.
    specs = [s for s in resolve_conditions(cfg, conditions)
             if s.structural not in ("none", "zero")]
    if not specs:
        raise ValueError("No condition uses the ControlNet branch; nothing to sweep.")
    brain_conditions = required_brain_conditions(specs)

    dm = build_datamodule(cfg).prepare()
    model, meta = load_decoder(cfg, dm, decoder_checkpoint, device)
    sample_seed = int(cfg.get("generation.sample_seed", cfg.get("project.seed", 42)))
    selection = select_samples(dm, split, num_samples, sample_seed)
    image_ids = [s["image_id"] for s in selection]

    clip_by, low_by = predict_condition_embeddings(
        model, cfg, dm, selection, brain_conditions, split, device, sample_seed)

    size = int(cfg.get("features.vae_image_size", 512))
    reals = [load_image(s["image_path"]).resize((size, size)) for s in selection]
    clip_bundle = load_clip(cfg, device)

    generator = FrozenSDGenerator(cfg, device=device)
    resolve_clip_rescale(cfg, dm, generator)
    adapter_ck = resolve_adapter_checkpoint(cfg, paths, adapter_checkpoint)
    if adapter_ck and Path(adapter_ck).exists():
        generator.load_adapter(int(meta["clip_dim"]), adapter_ck)
    else:
        logger.warning("No adapter at %s; the semantic branch will be null.", adapter_ck)

    text_inputs = build_text_inputs(cfg, selection, split)
    control_inputs = build_control_inputs(cfg, generator, selection, low_by, specs,
                                          device)

    gen_seed = int(cfg.get("generation.seed", 123))
    gs = float(cfg.get("generation.guidance_scale", 3.0))
    steps = int(num_inference_steps or cfg.get("generation.num_inference_steps", 50))
    strength = float(cfg.get("generation.strength", 0.8))
    logger.info("Sweeping controlnet_conditioning_scale over %s | %d samples, "
                "%d steps, conditions=%s", scales, len(selection), steps,
                [s.name for s in specs])

    rows, images_by_scale = [], {}
    for scale in scales:
        label = f"cn_{scale:g}"
        sc_dir = out_dir / label
        outputs = {"real": reals, "image_ids": image_ids}
        for spec in specs:
            init_images = None
            if generator.use_img2img and low_by.get(spec.structural) is not None:
                init_images = lowlevel_init_images(cfg, generator, selection,
                                                   low_by[spec.structural])
            text = None if not text_inputs or text_inputs.get(spec.text) is None \
                else text_inputs[spec.text]["embeds"]
            images = generator.generate(
                clip_by[spec.semantic], seed=gen_seed, guidance_scale=gs,
                num_inference_steps=steps, init_images=init_images,
                strength=strength, text_embeds=text,
                control_images=control_inputs[spec.structural],
                controlnet_scale=scale)
            outputs[spec.name] = images
            if save_images:
                save_condition_images(images, sc_dir / spec.name, image_ids)
            res = compute_generation_metrics(reals, images, clip_bundle, device,
                                             ks=(1, 5))
            m = res["metrics"]
            rows.append({"controlnet_scale": scale, "condition": spec.name,
                         "mean_clip_similarity": m["mean_clip_similarity"],
                         "median_clip_similarity": m["median_clip_similarity"],
                         "clip_top1": m["clip_retrieval"].get("top1"),
                         "clip_top5": m["clip_retrieval"].get("top5"),
                         "mean_pixel_mse": m.get("mean_pixel_mse")})
            logger.info("  scale=%.2f %-18s clip_sim=%.4f", scale, spec.name,
                        m["mean_clip_similarity"])
        images_by_scale[label] = outputs
        if save_images:
            save_comparison_grid(outputs, sc_dir / "comparison_grid.png",
                                 column_order=("real",) + tuple(s.name for s in specs))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "controlnet_scale_sweep_summary.csv", index=False)
    margins = margin_table(summary, key="controlnet_scale")
    margins.to_csv(out_dir / "controlnet_scale_sweep_margins.csv", index=False)
    save_json({"scales": scales, "num_samples": len(selection),
               "num_inference_steps": steps, "guidance_scale": gs,
               "conditions": [s.to_dict() for s in specs], "image_ids": image_ids,
               "adapter_checkpoint": str(adapter_ck),
               "controlnet": controlnet_settings(cfg)},
              out_dir / "sweep_params.json")
    save_controlnet_sweep_figure(summary, margins,
                                 out_dir / "controlnet_scale_sweep.png")
    logger.info("Sweep done -> %s", out_dir)
    return {"summary": summary, "margins": margins, "images": images_by_scale,
            "out_dir": str(out_dir)}


def save_controlnet_sweep_figure(summary: pd.DataFrame, margins: pd.DataFrame,
                                 out_path) -> Optional[str]:
    """Quality per condition and the correct−control margin vs ControlNet scale."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for cond, g in summary.groupby("condition"):
        g = g.sort_values("controlnet_scale")
        ax[0].plot(g["controlnet_scale"], g["mean_clip_similarity"], "-o", label=cond)
    ax[0].set_xlabel("controlnet_conditioning_scale")
    ax[0].set_ylabel("similitud CLIP (generada vs real)")
    ax[0].set_title("Calidad frente a la intensidad de ControlNet")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    if len(margins):
        m = margins.sort_values("controlnet_scale")
        ax[1].plot(m["controlnet_scale"], m["margin"], "-s", color="tab:green")
        ax[1].axhline(0.0, color="k", lw=0.8)
        ax[1].set_xlabel("controlnet_conditioning_scale")
        ax[1].set_ylabel("correcto − mejor control")
        ax[1].set_title("Margen sobre los controles negativos")
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)
