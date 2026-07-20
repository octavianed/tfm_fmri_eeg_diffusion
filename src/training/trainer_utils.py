"""Training building blocks: optimizer/scheduler, early stopping, the train/val
loops, and — crucially — full checkpoint assembly and resume (spec §17).

A checkpoint restores the *entire* training state so an interrupted run
continues exactly: model, optimizer, scheduler, GradScaler, epoch, global_step,
best metric, early-stopping counter and RNG state.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..evaluation.retrieval_metrics import compute_retrieval_metrics
from ..features.load_features import load_split_features
from ..utils import (autocast, collect_library_versions, get_rng_state,
                     get_logger, set_rng_state)

logger = get_logger("trainer")

# metric -> "min" means lower is better
MONITOR_MODES = {
    "val_loss": "min", "val_mean_rank": "min", "val_median_rank": "min",
    "val_top1": "max", "val_top5": "max", "val_top10": "max",
    "val_mean_cosine": "max",
}


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max",
                 min_delta: float = 0.0):
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        improved = self.is_improvement(value)
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.patience > 0 and self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return improved

    def state_dict(self) -> dict:
        return {"patience": self.patience, "mode": self.mode,
                "min_delta": self.min_delta, "best": self.best,
                "num_bad_epochs": self.num_bad_epochs,
                "should_stop": self.should_stop}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            setattr(self, k, v)


def is_better(new: float, best: float, mode: str) -> bool:
    if best is None:
        return True
    return new > best if mode == "max" else new < best


# --- optim / sched ---------------------------------------------------------
def build_optimizer(model, cfg):
    lr = float(cfg.get("training.lr", 1e-4))
    wd = float(cfg.get("training.weight_decay", 0.01))
    betas = tuple(cfg.get("training.betas", [0.9, 0.999]))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)


def build_scheduler(optimizer, cfg, steps_per_epoch: int, epochs: int):
    kind = str(cfg.get("training.scheduler", "cosine")).lower()
    if kind in ("none", "null", ""):
        return None
    total = max(1, steps_per_epoch * epochs)
    warmup = int(float(cfg.get("training.warmup_ratio", 0.0)) * total)

    if kind == "cosine":
        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if kind == "linear":
        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return max(0.0, (total - step) / max(1, total - warmup))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {kind}")


def peek_feature_dim(cfg, datamodule, kind: str = "clip",
                     split: str = "train") -> Optional[int]:
    for subj in datamodule.subjects:
        arr = load_split_features(cfg, subj, split, kind)
        if arr is not None:
            return int(arr.shape[1])
    return None


# --- loops -----------------------------------------------------------------
def _batch_subject(batch, uses_adapter: bool) -> Optional[str]:
    return batch["subject_id"][0] if uses_adapter else None


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, scheduler,
                    device, cfg, use_lowlevel: bool, global_step: int,
                    epoch: int, log_every: int = 50) -> Tuple[dict, int]:
    model.train()
    amp = bool(cfg.get("training.mixed_precision", True)) and device.type == "cuda"
    grad_clip = cfg.get("training.grad_clip", 1.0)
    uses_adapter = getattr(model, "adapters", None) is not None
    meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    for it, batch in enumerate(loader):
        fmri = batch["fmri"].to(device, non_blocking=True)
        subject = _batch_subject(batch, uses_adapter)
        targets = {"clip": batch["clip_target"].to(device, non_blocking=True)}
        if use_lowlevel and "low_target" in batch:
            targets["low"] = batch["low_target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, enabled=amp):
            out = model(fmri, subject=subject)
            loss, parts = loss_fn(out, targets)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        global_step += 1
        bs = fmri.size(0)
        for k, v in parts.items():
            meters[k].update(float(v), bs)
        if log_every and (it % log_every == 0):
            logger.info("  epoch %d it %d/%d | loss %.4f", epoch, it,
                        len(loader), meters["total"].avg)
    return {k: m.avg for k, m in meters.items()}, global_step


def validate(model, loader, loss_fn, device, cfg, use_lowlevel: bool,
             ks=(1, 5, 10)) -> Tuple[dict, tuple]:
    model.eval()
    amp = bool(cfg.get("training.mixed_precision", True)) and device.type == "cuda"
    uses_adapter = getattr(model, "adapters", None) is not None
    meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)
    preds, targets, subs = [], [], []

    with torch.no_grad():
        for batch in loader:
            fmri = batch["fmri"].to(device, non_blocking=True)
            subject = _batch_subject(batch, uses_adapter)
            tgt = {"clip": batch["clip_target"].to(device, non_blocking=True)}
            if use_lowlevel and "low_target" in batch:
                tgt["low"] = batch["low_target"].to(device, non_blocking=True)
            with autocast(device, enabled=amp):
                out = model(fmri, subject=subject)
                _, parts = loss_fn(out, tgt)
            bs = fmri.size(0)
            for k, v in parts.items():
                meters[k].update(float(v), bs)
            preds.append(out["clip"].float().cpu().numpy())
            targets.append(tgt["clip"].float().cpu().numpy())
            subs.extend(list(batch["subject_id"]))

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    subs = np.asarray(subs)

    per_subject = []
    for s in np.unique(subs):
        mask = subs == s
        m, _ = compute_retrieval_metrics(preds[mask], targets[mask], ks=ks)
        per_subject.append(m)
    macro = {}
    if per_subject:
        for key in per_subject[0]:
            macro[key] = float(np.mean([m[key] for m in per_subject if key in m]))

    val = {"val_loss": meters["total"].avg,
           "parts": {k: m.avg for k, m in meters.items()}}
    for key in ("top1", "top5", "top10", "mean_rank", "median_rank", "mean_cosine"):
        val[f"val_{key}"] = macro.get(key)
    return val, (preds, targets, subs)


def get_monitor(val_metrics: dict, monitor: str) -> Tuple[Optional[float], str]:
    mode = MONITOR_MODES.get(monitor, "max")
    return val_metrics.get(monitor), mode


# --- checkpoint state ------------------------------------------------------
def assemble_state(cfg, model, optimizer, scheduler, scaler, epoch, global_step,
                   best_metric, best_epoch, early_stopping, extra: dict,
                   save_rng: bool = True) -> dict:
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_epoch": int(best_epoch),
        "early_stopping_state": early_stopping.state_dict(),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "library_versions": collect_library_versions(),
        "rng_state": get_rng_state() if save_rng else None,
    }
    state.update(extra)
    return state


def restore_state(state: dict, model, optimizer, scheduler, scaler,
                  early_stopping, strict: bool = True) -> dict:
    model.load_state_dict(state["model_state_dict"], strict=strict)
    if optimizer is not None and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    if early_stopping is not None and state.get("early_stopping_state"):
        early_stopping.load_state_dict(state["early_stopping_state"])
    if state.get("rng_state"):
        set_rng_state(state["rng_state"])
    return {
        "start_epoch": int(state.get("epoch", -1)) + 1,
        "global_step": int(state.get("global_step", 0)),
        "best_metric": state.get("best_metric"),
        "best_epoch": int(state.get("best_epoch", 0)),
    }


def feature_paths_summary(cfg, datamodule, splits=("train", "val", "test")) -> dict:
    from ..utils.paths import clip_feature_path, vae_pca_feature_path
    out = {}
    for subj in datamodule.subjects:
        out[subj] = {}
        for split in splits:
            out[subj][split] = {
                "clip": str(clip_feature_path(cfg, subj, split)),
                "low": str(vae_pca_feature_path(cfg, subj, split)),
            }
    return out


def append_resume_history(jsonl_logger, checkpoint_path, resumed):
    jsonl_logger.append({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "resumed_epoch": resumed["start_epoch"],
        "resumed_global_step": resumed["global_step"],
        "best_metric": resumed["best_metric"],
    })


def save_training_figures(train_log_csv, figures_dir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:  # pragma: no cover
        return
    from pathlib import Path
    p = Path(train_log_csv)
    if not p.exists():
        return
    df = pd.read_csv(p)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if {"epoch", "train_total"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["epoch"], df["train_total"], label="train")
        if "val_loss" in df.columns:
            ax.plot(df["epoch"], df["val_loss"], label="val")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
        ax.set_title("Training / validation loss")
        fig.tight_layout(); fig.savefig(figures_dir / "loss_curve.png", dpi=120)
        plt.close(fig)

    topk = [c for c in ("val_top1", "val_top5", "val_top10") if c in df.columns]
    if topk and "epoch" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in topk:
            ax.plot(df["epoch"], df[c], label=c)
        ax.set_xlabel("epoch"); ax.set_ylabel("retrieval accuracy"); ax.legend()
        ax.set_title("Validation retrieval Top-k")
        fig.tight_layout(); fig.savefig(figures_dir / "retrieval_topk.png", dpi=120)
        plt.close(fig)
