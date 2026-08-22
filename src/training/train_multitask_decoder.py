"""Core training loop for the fMRI decoder (CLIP-only or CLIP + low-level).

``run_training`` powers Experiment 1 (``use_lowlevel=False``) and Experiment 3
(``use_lowlevel=True``). It supports stop/resume from ``last.pt`` restoring the
full training state, periodic + best + last checkpoints with retention, and
never loads Stable Diffusion (spec §4, §7.5, §17).
"""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..data import build_datamodule
from ..evaluation.embedding_metrics import embedding_regression_metrics
from ..evaluation.eval_data import load_subject_matrices
from ..losses import build_loss
from ..models import build_model
from ..utils import (CheckpointManager, CSVLogger, JsonlLogger, get_device,
                     get_experiment_paths, get_logger, load_checkpoint,
                     make_grad_scaler, save_config, save_json, set_seed)
from ..utils.paths import normalization_path
from . import trainer_utils as tu

logger = get_logger("train")

CSV_FIELDS = ["epoch", "global_step", "lr", "train_total", "train_clip_cosine",
              "train_clip_infonce", "train_lowlevel_mse", "val_loss",
              "val_top1", "val_top5", "val_top10", "val_mean_rank",
              "val_mean_cosine", "best_metric", "is_best", "seconds"]


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_training(cfg, use_lowlevel: bool = False, resume=None,
                 eval_only: bool = False) -> dict:
    """Train the decoder, then evaluate ``best.pt`` on val/test.

    ``eval_only=True`` skips training entirely and just runs the final
    evaluation from the existing ``best.pt`` — useful when a run finished but
    the final evaluation failed, so the (expensive) training is not repeated.
    """
    set_seed(int(cfg.get("project.seed", 42)),
             deterministic=bool(cfg.get("runtime.deterministic", False)))
    device = get_device(cfg.get("runtime.device", "auto"))
    paths = get_experiment_paths(cfg, ensure=True)
    save_config(cfg, paths.root / "config.yaml")
    logger.info("Experiment dir: %s | device: %s | lowlevel: %s%s",
                paths.root, device, use_lowlevel,
                " | EVAL ONLY" if eval_only else "")

    dm = build_datamodule(cfg).prepare()
    clip_dim = tu.peek_feature_dim(cfg, dm, "clip", "train")
    if clip_dim is None:
        raise FileNotFoundError(
            "No precomputed CLIP features found. Run scripts/01_precompute_clip.py.")
    low_dim = None
    if use_lowlevel:
        low_dim = tu.peek_feature_dim(cfg, dm, "low", "train")
        if low_dim is None:
            raise FileNotFoundError(
                "use_lowlevel=True but no VAE-PCA features found. "
                "Run scripts/04_precompute_vae_pca.py.")

    model = build_model(cfg, dm.voxel_counts, clip_dim=clip_dim, low_dim=low_dim,
                        use_lowlevel=use_lowlevel).to(device)
    loss_fn = build_loss(cfg, use_lowlevel=use_lowlevel)

    kinds = ("clip", "low") if use_lowlevel else ("clip",)
    if eval_only:
        ckpt = CheckpointManager(paths.checkpoints)
        if not ckpt.best_path.exists():
            raise FileNotFoundError(
                f"--eval-only needs a trained checkpoint at {ckpt.best_path}")
        result = _finalize(cfg, model, dm, loss_fn, device, paths, use_lowlevel,
                           ckpt)
        tu.save_training_figures(paths.train_log, paths.figures)
        return result

    train_loader = dm.build_dataloader("train", shuffle=True, kinds=kinds)
    val_loader = dm.build_dataloader("val", shuffle=False, kinds=kinds)
    steps_per_epoch = max(1, len(train_loader))
    epochs = int(cfg.get("training.epochs", 100))

    optimizer = tu.build_optimizer(model, cfg)
    scheduler = tu.build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    amp = bool(cfg.get("training.mixed_precision", True)) and device.type == "cuda"
    scaler = make_grad_scaler(enabled=amp)

    monitor = str(cfg.get("checkpointing.monitor", "val_top5"))
    mode = tu.MONITOR_MODES.get(monitor, "max")
    early = tu.EarlyStopping(
        patience=int(cfg.get("training.early_stopping_patience", 10)),
        mode=mode)
    ckpt = CheckpointManager(
        paths.checkpoints,
        save_last=bool(cfg.get("checkpointing.save_last", True)),
        save_best=bool(cfg.get("checkpointing.save_best", True)),
        save_every_n_epochs=int(cfg.get("checkpointing.save_every_n_epochs", 1)),
        keep_last_n=int(cfg.get("checkpointing.keep_last_n", 3)),
        keep_first_n=int(cfg.get("checkpointing.keep_first_n", 0)),
        monitor_mode=mode)

    start_epoch, global_step, best_metric, best_epoch = 0, 0, None, 0
    resume_arg = resume if resume is not None else cfg.get("checkpointing.resume", None)
    resume_path = ckpt.find_resume(resume_arg)
    resume_hist = JsonlLogger(paths.resume_history)
    if resume_path is not None:
        logger.info("Resuming from checkpoint: %s", resume_path)
        state = load_checkpoint(resume_path, map_location=device)
        if state.get("subject_selection") not in (None, dm.subjects) \
                and state.get("subject_selection") != dm.subjects:
            logger.warning("Checkpoint subjects %s != current %s",
                           state.get("subject_selection"), dm.subjects)
        resumed = tu.restore_state(state, model, optimizer, scheduler, scaler,
                                   early, strict=bool(cfg.get(
                                       "checkpointing.strict_load", True)))
        start_epoch = resumed["start_epoch"]
        global_step = resumed["global_step"]
        best_metric = resumed["best_metric"]
        best_epoch = resumed["best_epoch"]
        tu.append_resume_history(resume_hist, resume_path, resumed)
    else:
        logger.info("Starting training from scratch (no resume).")

    csv_logger = CSVLogger(paths.train_log, CSV_FIELDS)
    extra = {
        "subject_selection": dm.subjects,
        "normalization_paths": [str(normalization_path(cfg, s)) for s in dm.subjects],
        "feature_paths": tu.feature_paths_summary(cfg, dm),
        "monitor": monitor, "monitor_mode": mode, "use_lowlevel": use_lowlevel,
        "clip_dim": clip_dim, "low_dim": low_dim, "voxel_counts": dm.voxel_counts,
    }
    save_rng = bool(cfg.get("checkpointing.save_rng_state", True))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loader.batch_sampler.set_epoch(epoch)
        train_stats, global_step = tu.train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, scheduler, device,
            cfg, use_lowlevel, global_step, epoch,
            log_every=int(cfg.get("training.log_every", 50)))
        val_metrics, _ = tu.validate(model, val_loader, loss_fn, device, cfg,
                                     use_lowlevel)
        monitor_value, _ = tu.get_monitor(val_metrics, monitor)
        if monitor_value is None:
            monitor_value = -val_metrics["val_loss"] if mode == "max" \
                else val_metrics["val_loss"]

        improved = early.step(monitor_value)
        if improved:
            best_metric, best_epoch = monitor_value, epoch

        state = tu.assemble_state(cfg, model, optimizer, scheduler, scaler,
                                  epoch, global_step, best_metric, best_epoch,
                                  early, extra, save_rng=save_rng)
        ckpt.save(state, epoch, is_best=improved)

        csv_logger.log({
            "epoch": epoch, "global_step": global_step, "lr": _current_lr(optimizer),
            "train_total": train_stats.get("total"),
            "train_clip_cosine": train_stats.get("clip_cosine"),
            "train_clip_infonce": train_stats.get("clip_infonce"),
            "train_lowlevel_mse": train_stats.get("lowlevel_mse"),
            "val_loss": val_metrics["val_loss"], "val_top1": val_metrics["val_top1"],
            "val_top5": val_metrics["val_top5"], "val_top10": val_metrics["val_top10"],
            "val_mean_rank": val_metrics["val_mean_rank"],
            "val_mean_cosine": val_metrics["val_mean_cosine"],
            "best_metric": best_metric, "is_best": int(improved),
            "seconds": round(time.time() - t0, 1)})
        logger.info("epoch %d | train %.4f | val_loss %.4f | %s %.4f%s | %.0fs",
                    epoch, train_stats.get("total", 0.0), val_metrics["val_loss"],
                    monitor, monitor_value, " *" if improved else "",
                    time.time() - t0)
        if early.should_stop:
            logger.info("Early stopping at epoch %d (best %s=%.4f @ epoch %d)",
                        epoch, monitor, best_metric, best_epoch)
            break

    result = _finalize(cfg, model, dm, loss_fn, device, paths, use_lowlevel, ckpt)
    tu.save_training_figures(paths.train_log, paths.figures)
    return result


def _finalize(cfg, model, dm, loss_fn, device, paths, use_lowlevel, ckpt) -> dict:
    if ckpt.best_path.exists():
        logger.info("Loading best checkpoint for final evaluation: %s",
                    ckpt.best_path)
        best = load_checkpoint(ckpt.best_path, map_location=device)
        model.load_state_dict(best["model_state_dict"], strict=False)
        # Drop the checkpoint (~2.9 GB: weights + AdamW state) BEFORE building the
        # eval loaders. Keeping it alive while DataLoader workers are spawned was
        # enough extra RAM pressure to make the Windows spawn pipe fail with
        # "OSError: [Errno 22] Invalid argument" right after training finished.
        del best

    out = {"experiment_dir": str(paths.root), "metrics": {}}
    for split in ("val", "test"):
        frame = dm.get_frame(split)
        if len(frame) == 0:
            continue
        # num_workers=0 on purpose. The final evaluation is a single pass, while
        # each worker process re-pickles the whole dataset — and the fMRI reader
        # holds memory-mapped arrays that pickle BY VALUE (measured: 1.56 GB for
        # subj01). On Windows (spawn) that is several GB pushed down a pipe per
        # worker: slow at best, and a hard OSError at worst. Reading straight
        # from the memmap in the main process is both safer and faster here.
        loader = dm.build_dataloader(
            split, shuffle=False, num_workers=0,
            kinds=("clip", "low") if use_lowlevel else ("clip",))
        val_metrics, (preds, targets, _subs) = tu.validate(
            model, loader, loss_fn, device, cfg, use_lowlevel)
        emb, _ = embedding_regression_metrics(preds, targets)
        metrics = {"retrieval": {k: val_metrics.get(f"val_{k}") for k in
                                 ("top1", "top5", "top10", "mean_rank",
                                  "mean_cosine")},
                   "embedding": emb, "loss": val_metrics["val_loss"]}
        save_json(metrics, paths.metrics / f"{split}_metrics.json")
        np.save(paths.embeddings / f"{split}_clip_pred.npy", preds)
        np.save(paths.embeddings / f"{split}_clip_target.npy", targets)
        if use_lowlevel:
            low_metrics = _lowlevel_eval(cfg, model, dm, device, split, paths)
            if low_metrics:
                metrics["lowlevel"] = low_metrics
                save_json(low_metrics, paths.metrics / f"{split}_lowlevel_metrics.json")
        out["metrics"][split] = metrics
        logger.info("[%s] top1=%.3f top5=%.3f mean_cos=%.3f", split,
                    metrics["retrieval"]["top1"] or 0.0,
                    metrics["retrieval"]["top5"] or 0.0,
                    emb["mean_cosine"])
    _write_summary(cfg, paths, out, use_lowlevel)
    return out


def _lowlevel_eval(cfg, model, dm, device, split, paths) -> dict:
    """Evaluate the low-level branch on ``split`` and save preds/targets/figure."""
    uses_adapter = getattr(model, "adapters", None) is not None
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for subj in dm.subjects:
            m = load_subject_matrices(cfg, dm, subj, split, want=("fmri", "low"))
            if m.fmri is None or m.low is None:
                continue
            for start in range(0, len(m.fmri), 256):
                batch = torch.from_numpy(
                    np.ascontiguousarray(m.fmri[start:start + 256])).float().to(device)
                out = model(batch, subject=subj if uses_adapter else None)
                preds.append(out["low"].float().cpu().numpy())
            targets.append(m.low)
    if not preds:
        return {}
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    emb, pearson = embedding_regression_metrics(preds, targets)
    np.save(paths.lowlevel / f"{split}_low_pred.npy", preds)
    np.save(paths.lowlevel / f"{split}_low_target.npy", targets)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(pearson)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("PCA component"); ax.set_ylabel("Pearson r")
        ax.set_title(f"Low-level prediction ({split})")
        fig.tight_layout()
        fig.savefig(paths.figures / f"pca_component_correlation_{split}.png", dpi=120)
        plt.close(fig)
    except Exception:  # pragma: no cover
        pass
    return emb


def _write_summary(cfg, paths, out, use_lowlevel) -> None:
    enc = cfg.get("model.fmri_encoder", {})
    lines = [
        f"# {cfg.get('experiment.name', 'experiment')} — summary", "",
        f"- **Objetivo:** {'CLIP + low-level multitask' if use_lowlevel else 'fMRI -> CLIP embedding'}",
        f"- **Sujetos:** {cfg.get('dataset.subject_selection')}",
        f"- **Modelo:** fMRIEncoder(hidden={enc.get('hidden_dim')}, "
        f"out={enc.get('output_dim')}) + CLIPHead"
        + (" + LowLevelHead" if use_lowlevel else ""),
        f"- **Config:** `{paths.root / 'config.yaml'}`", "",
    ]
    for split, m in out["metrics"].items():
        r = m["retrieval"]
        lines.append(f"## {split}")
        lines.append("- retrieval Top-1/5/10: "
                     f"{(r['top1'] or 0):.3f} / {(r['top5'] or 0):.3f} / {(r['top10'] or 0):.3f}")
        lines.append(f"- mean cosine: {m['embedding']['mean_cosine']:.3f}")
        if m.get("lowlevel"):
            lines.append(f"- low-level mean Pearson r: {m['lowlevel'].get('mean_pearson', 0):.3f}")
        lines.append("")
    lines.append("**Conclusion preliminar:** confirmar en el Experimento 2 "
                 "(retrieval ablation) que fMRI correcto supera claramente a "
                 "permutado/cero antes de atribuir el resultado a la senal cerebral.")
    (paths.report / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def train_multitask(cfg, resume=None, eval_only: bool = False) -> dict:
    return run_training(cfg, use_lowlevel=True, resume=resume,
                        eval_only=eval_only)
