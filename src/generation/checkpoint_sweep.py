"""Sweep several TokenAdapter checkpoints and score them by generation quality.

Training loss (a single-random-timestep diffusion MSE) is a noisy, unreliable
proxy for how well a checkpoint actually generates images end-to-end: an
adapter can drift into weight configurations with a similar (or even lower)
training loss that generate markedly *worse* images once run through full
multi-step ancestral sampling. This module evaluates several checkpoints
(``best``/``last``/periodic ``epoch_XXXX.pt`` snapshots) the same way
Experiment 5 evaluates a single one — CLIP similarity between generated and
real images, for the correct/permuted/zero conditions — so the checkpoint that
actually generates best (and shows the clearest correct-vs-controls margin)
can be picked directly, instead of trusting the training loss.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from ..data import build_datamodule, load_image
from ..evaluation.generation_metrics import compute_generation_metrics
from ..features.clip_model import load_clip
from ..utils import get_device, get_experiment_paths, get_logger, load_checkpoint, save_json
from .conditioning import required_brain_conditions, resolve_conditions
from .generate_from_fmri import (_condition_scale, build_control_inputs,
                                 build_text_inputs, load_decoder,
                                 lowlevel_init_images, resolve_clip_rescale,
                                 predict_condition_embeddings,
                                 save_condition_images, select_samples)
from .make_grids import save_comparison_grid
from .sd_pipeline import FrozenSDGenerator

logger = get_logger("checkpoint_sweep")


def discover_adapter_checkpoints(checkpoints_dir) -> list:
    """Find ``adapter_best.pt`` / ``adapter_last.pt`` / periodic ``epoch_XXXX.pt``
    snapshots (saved by :func:`src.generation.sd_pipeline.train_token_adapter`)
    under a single experiment's checkpoints directory."""
    d = Path(checkpoints_dir)
    found = []
    for label, name in (("best", "adapter_best.pt"), ("last", "adapter_last.pt")):
        p = d / name
        if p.exists():
            found.append((label, p))
    for p in sorted(d.glob("epoch_*.pt")):
        found.append((p.stem, p))
    return found


def _checkpoint_epoch(path) -> Optional[int]:
    try:
        state = load_checkpoint(path, map_location="cpu")
        epoch = state.get("epoch")
        return int(epoch) if epoch is not None else None
    except Exception:  # pragma: no cover
        return None


def sweep_adapter_checkpoints(cfg, decoder_checkpoint,
                              checkpoints: Optional[Sequence[Tuple[str, "os.PathLike"]]] = None,
                              conditions: Sequence[str] = ("correct", "permuted", "zero"),
                              num_samples: int = 6,
                              num_inference_steps: Optional[int] = None,
                              device=None, split: Optional[str] = None,
                              out_dir: Optional["os.PathLike"] = None,
                              save_images: bool = True) -> dict:
    """Generate with each checkpoint on the SAME small sample set and score by
    CLIP similarity to the real images.

    Loads the frozen SD pipeline (UNet/VAE/scheduler) and predicts the fMRI ->
    CLIP/low-level embeddings for the sample set only ONCE (both are
    checkpoint-independent); only the small adapter is swapped in per
    checkpoint, so sweeping several checkpoints is much cheaper than calling
    :func:`~src.generation.generate_from_fmri.generate_images` repeatedly.

    Returns a dict with:
      - ``summary``: tidy DataFrame, one row per (checkpoint, condition), with
        ``mean_clip_similarity``, ``clip_top1``, ``clip_top5``.
      - ``margins``: one row per checkpoint with ``correct - best_control``.
      - ``images``: the generated PIL images per checkpoint/condition (for
        rendering grids interactively, e.g. in a notebook).
      - ``out_dir``: where the CSVs, figure and (optionally) PNGs were saved.
    """
    device = device or get_device(cfg.get("runtime.device", "auto"))
    split = split or str(cfg.get("generation.split", "test"))
    paths = get_experiment_paths(cfg, ensure=True)
    out_dir = Path(out_dir) if out_dir else paths.root / "checkpoint_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    if checkpoints is None:
        checkpoints = discover_adapter_checkpoints(paths.checkpoints)
    checkpoints = [(label, Path(p)) for label, p in checkpoints if Path(p).exists()]
    if not checkpoints:
        raise FileNotFoundError(f"No adapter checkpoints found under {paths.checkpoints}")

    mode = str(cfg.get("generation.mode", "adapter"))
    if mode not in ("adapter", "adapter_lowlevel"):
        logger.warning(
            "generation.mode=%s never loads the token adapter, so every "
            "checkpoint would generate the SAME images. Use --set "
            "generation.mode=adapter (or adapter_lowlevel) to sweep "
            "adapter checkpoints meaningfully.", mode)

    # Resolve conditions exactly like Exp4 so a text/ControlNet architecture is
    # swept with its real conditioning (the positive CFG branch would otherwise
    # be missing the text half and the run would abort on a length mismatch).
    specs = resolve_conditions(cfg, conditions)
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
    clip_rescale = resolve_clip_rescale(cfg, dm, generator)
    text_inputs = build_text_inputs(cfg, selection, split)
    control_inputs = build_control_inputs(cfg, generator, selection, low_by, specs,
                                          device)
    gen_seed = int(cfg.get("generation.seed", 123))
    gs = float(cfg.get("generation.guidance_scale", 3.0))
    steps = int(num_inference_steps or cfg.get("generation.num_inference_steps", 50))
    strength = float(cfg.get("generation.strength", 0.8))

    rows, images_by_checkpoint = [], {}
    for label, ckpt_path in checkpoints:
        logger.info("Sweeping checkpoint '%s': %s", label, ckpt_path)
        epoch = _checkpoint_epoch(ckpt_path)
        if mode in ("adapter", "adapter_lowlevel"):
            generator.load_adapter(int(meta["clip_dim"]), ckpt_path)
        ck_dir = out_dir / label
        outputs = {"real": reals, "image_ids": image_ids}
        for spec in specs:
            init_images = None
            src = spec.structural if spec.structural in low_by else spec.semantic
            if generator.use_img2img and low_by.get(src) is not None:
                init_images = lowlevel_init_images(cfg, generator, selection, low_by[src])
            text = None if not text_inputs or text_inputs.get(spec.text) is None                 else text_inputs[spec.text]["embeds"]
            controls = None if control_inputs is None else control_inputs[spec.structural]
            images = generator.generate(
                clip_by[spec.semantic], seed=gen_seed, guidance_scale=gs,
                num_inference_steps=steps, init_images=init_images, strength=strength,
                text_embeds=text, control_images=controls,
                controlnet_scale=_condition_scale(cfg, spec))
            outputs[spec.name] = images
            if save_images:
                save_condition_images(images, ck_dir / spec.name, image_ids)

            res = compute_generation_metrics(reals, images, clip_bundle, device, ks=(1, 5))
            m = res["metrics"]
            rows.append({"checkpoint": label, "epoch": epoch, "condition": spec.name,
                        "mean_clip_similarity": m["mean_clip_similarity"],
                        "median_clip_similarity": m["median_clip_similarity"],
                        "clip_top1": m["clip_retrieval"].get("top1"),
                        "clip_top5": m["clip_retrieval"].get("top5"),
                        "mean_pixel_mse": m.get("mean_pixel_mse")})
        images_by_checkpoint[label] = outputs
        if save_images:
            save_comparison_grid(outputs, ck_dir / "comparison_grid.png",
                                 column_order=("real",) + tuple(s.name for s in specs))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "checkpoint_sweep_summary.csv", index=False)
    margins = _margin_table(summary)
    margins.to_csv(out_dir / "checkpoint_sweep_margins.csv", index=False)
    save_json({"mode": mode, "num_samples": len(selection), "num_inference_steps": steps,
              "guidance_scale": gs, "strength": strength, "image_ids": image_ids,
              "rescale_clip_pred": cfg.get("generation.rescale_clip_pred", "none"),
              "clip_pred_target_norm": clip_rescale,
              "checkpoints": [{"label": lbl, "path": str(p)} for lbl, p in checkpoints]},
             out_dir / "sweep_params.json")
    save_sweep_figure(summary, out_dir / "checkpoint_sweep_quality.png")
    logger.info("Sweep done -> %s", out_dir)
    return {"summary": summary, "margins": margins, "images": images_by_checkpoint,
            "out_dir": str(out_dir)}


#: Every negative control the margin must beat (spec §2: correct >> permuted ≈
#: zero ≈ noise). Ignoring one would *overstate* the margin.
CONTROL_CONDITIONS = ("permuted", "zero", "noise")


def margin_table(summary: pd.DataFrame, key: str = "checkpoint",
                 carry: Sequence[str] = (), sort_by: Optional[str] = None
                 ) -> pd.DataFrame:
    """Per ``key``: correct's mean CLIP similarity minus the BEST control's.

    Encodes the project's falsifiable criterion, so every sweep (checkpoints,
    adapter input scale, …) applies exactly the same rule. ``carry`` copies extra
    columns through (e.g. ``epoch``).
    """
    rows = []
    for k in summary[key].unique():
        sub = summary[summary[key] == k]
        vals = dict(zip(sub["condition"], sub["mean_clip_similarity"]))
        correct = vals.get("correct")
        controls = [vals[c] for c in CONTROL_CONDITIONS if c in vals]
        if correct is None or not controls:
            continue
        best_control = max(controls)
        row = {key: k}
        for col in carry:
            row[col] = sub[col].iloc[0]
        row.update({"correct": correct, "best_control": best_control,
                    "margin": correct - best_control,
                    "uses_fmri_signal": bool(correct > best_control)})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    return df.sort_values(sort_by or key, na_position="last")


def _margin_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper used by the checkpoint sweep."""
    return margin_table(summary, key="checkpoint", carry=("epoch",),
                        sort_by="epoch")


def save_sweep_figure(summary: pd.DataFrame, out_path) -> Optional[str]:
    """Line plot of mean CLIP similarity per condition, ordered by checkpoint
    epoch — the direct visualization of "does quality change over training"."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return None
    if summary.empty:
        return None
    order = (summary[["checkpoint", "epoch"]].drop_duplicates()
             .sort_values("epoch", na_position="last")["checkpoint"].tolist())
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond in summary["condition"].unique():
        sub = summary[summary["condition"] == cond].set_index("checkpoint").reindex(order)
        ax.plot(range(len(order)), sub["mean_clip_similarity"], marker="o", label=cond)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_xlabel("checkpoint (ordenado por época)")
    ax.set_ylabel("similitud CLIP media (generada vs real)")
    ax.set_title("Calidad de generación por checkpoint del adapter")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
