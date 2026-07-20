"""Control-condition evaluation: fMRI correct vs permuted vs zero (spec §8, §16).

This is the load-bearing experiment of the whole project: the fMRI signal is
only credited as *used* if the correct condition clearly beats the permuted and
zero controls. Permutation uses a Sattolo cyclic shuffle (a guaranteed
derangement, so no sample ever receives its own fMRI), applied within each
subject so voxel dimensions stay consistent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..utils import get_device, get_logger
from .embedding_metrics import embedding_regression_metrics
from .eval_data import load_subject_matrices
from .retrieval_metrics import compute_retrieval_metrics

logger = get_logger("ablation")

RETRIEVAL_KEYS = ("top1", "top5", "top10", "mean_rank", "median_rank",
                  "mean_cosine")


def sattolo_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random cyclic permutation of ``range(n)`` (no fixed points)."""
    idx = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def make_condition_input(fmri: np.ndarray, condition: str,
                         rng: np.random.Generator, noise_std: float = 1.0):
    if condition == "correct":
        return fmri
    if condition == "zero":
        return np.zeros_like(fmri)
    if condition == "permuted":
        if fmri.shape[0] < 2:
            return fmri
        return fmri[sattolo_derangement(fmri.shape[0], rng)]
    if condition in ("noise", "gaussian"):
        return rng.normal(0.0, noise_std, size=fmri.shape).astype(fmri.dtype)
    raise ValueError(f"Unknown condition: {condition}")


def _forward(model, fmri_np, subject, device, batch_size: int = 256) -> dict:
    use_subject = getattr(model, "adapters", None) is not None
    preds_clip, preds_low = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, fmri_np.shape[0], batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(fmri_np[start:start + batch_size])
            ).float().to(device)
            out = model(batch, subject=subject if use_subject else None)
            preds_clip.append(out["clip"].float().cpu().numpy())
            if out.get("low") is not None:
                preds_low.append(out["low"].float().cpu().numpy())
    result = {"clip": np.concatenate(preds_clip, axis=0)}
    result["low"] = np.concatenate(preds_low, axis=0) if preds_low else None
    return result


def _macro_average(per_subject: Dict[str, dict], block: str) -> dict:
    keys = set()
    for s in per_subject.values():
        keys.update(s[block].keys())
    return {k: float(np.mean([s[block][k] for s in per_subject.values()
                              if k in s[block]])) for k in keys}


def evaluate_ablation(model, cfg, datamodule, split: str = "test",
                      conditions: Sequence[str] = ("correct", "permuted", "zero"),
                      device=None, checkpoint_name: str = "",
                      ks=(1, 5, 10), seed: Optional[int] = None) -> dict:
    device = device or get_device(cfg.get("runtime.device", "auto"))
    model = model.to(device)
    seed = int(cfg.get("project.seed", 42) if seed is None else seed)
    noise_std = float(cfg.get("evaluation.noise_std", 1.0))
    batch_size = int(cfg.get("evaluation.infer_batch_size", 256))

    results: Dict[str, dict] = {c: {"subjects": {}} for c in conditions}
    predictions: Dict[str, dict] = {c: {} for c in conditions}

    for subject in datamodule.subjects:
        mats = load_subject_matrices(cfg, datamodule, subject, split,
                                     want=("fmri", "clip", "low"))
        if mats.fmri is None or mats.clip is None:
            raise FileNotFoundError(
                f"Missing fMRI/CLIP for {subject}/{split}; precompute first.")
        for cond in conditions:
            rng = np.random.default_rng(seed + hash(cond) % 100000)
            fmri_in = make_condition_input(mats.fmri, cond, rng, noise_std)
            pred = _forward(model, fmri_in, subject, device, batch_size)

            ret, ranks = compute_retrieval_metrics(pred["clip"], mats.clip, ks=ks)
            emb, _ = embedding_regression_metrics(pred["clip"], mats.clip)
            block = {"retrieval": ret, "embedding": emb}
            if pred["low"] is not None and mats.low is not None:
                low_emb, _ = embedding_regression_metrics(pred["low"], mats.low)
                block["lowlevel"] = low_emb
            results[cond]["subjects"][subject] = block
            predictions[cond][subject] = {
                "clip_pred": pred["clip"], "clip_target": mats.clip,
                "low_pred": pred["low"], "low_target": mats.low,
                "ranks": ranks, "image_ids": mats.image_ids,
                "image_paths": mats.image_paths}

    for cond in conditions:
        subs = results[cond]["subjects"]
        results[cond]["macro"] = {
            "retrieval": _macro_average(subs, "retrieval"),
            "embedding": _macro_average(subs, "embedding"),
        }

    summary = _build_summary(results, split, seed, checkpoint_name)
    return {"conditions": results, "summary": summary, "predictions": predictions}


def _build_summary(results: dict, split: str, seed: int,
                   checkpoint: str) -> pd.DataFrame:
    rows: List[dict] = []

    def add(metric, cond, subject, value):
        rows.append({"metric_name": metric, "condition": cond,
                     "subject_id": subject, "split": split,
                     "value": float(value), "seed": seed,
                     "checkpoint": checkpoint})

    for cond, data in results.items():
        for subject, block in data["subjects"].items():
            for key in RETRIEVAL_KEYS:
                if key in block["retrieval"]:
                    add(f"retrieval/{key}", cond, subject, block["retrieval"][key])
            for key in ("mse", "mae", "mean_cosine", "mean_pearson"):
                if key in block["embedding"]:
                    add(f"embedding/{key}", cond, subject, block["embedding"][key])
            if "lowlevel" in block:
                for key in ("mse", "mae", "mean_pearson"):
                    add(f"lowlevel/{key}", cond, subject, block["lowlevel"][key])
        for key in RETRIEVAL_KEYS:
            if key in data["macro"]["retrieval"]:
                add(f"retrieval/{key}", cond, "all", data["macro"]["retrieval"][key])
    return pd.DataFrame(rows)


def save_ablation_figures(summary: pd.DataFrame, figures_dir,
                          subject: str = "all") -> None:
    """Bar charts of Top-k and cosine by condition (spec §8.4)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    from pathlib import Path
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sub = summary[summary.subject_id == subject]
    conditions = list(dict.fromkeys(sub.condition.tolist()))

    topk = ["retrieval/top1", "retrieval/top5", "retrieval/top10"]
    have = [m for m in topk if m in set(sub.metric_name)]
    if have:
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / max(1, len(have))
        x = np.arange(len(conditions))
        for i, metric in enumerate(have):
            vals = [float(sub[(sub.metric_name == metric) &
                             (sub.condition == c)].value.mean())
                    for c in conditions]
            ax.bar(x + i * width, vals, width, label=metric.split("/")[-1])
        ax.set_xticks(x + width * (len(have) - 1) / 2)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("retrieval accuracy")
        ax.set_title(f"Top-k by condition ({subject})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "topk_by_condition.png", dpi=120)
        plt.close(fig)

    if "retrieval/mean_cosine" in set(sub.metric_name):
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [float(sub[(sub.metric_name == "retrieval/mean_cosine") &
                         (sub.condition == c)].value.mean()) for c in conditions]
        ax.bar(conditions, vals, color=["#2c7fb8", "#7fcdbb", "#c7c7c7"][:len(conditions)])
        ax.set_ylabel("mean cosine similarity")
        ax.set_title(f"Cosine by condition ({subject})")
        fig.tight_layout()
        fig.savefig(figures_dir / "cosine_by_condition.png", dpi=120)
        plt.close(fig)


def conclusion_from_summary(summary: pd.DataFrame,
                            metric: str = "retrieval/top5",
                            subject: str = "all",
                            margin: float = 1.0) -> dict:
    """Decide whether 'correct' clearly beats permuted/zero (spec §2, §8.5).

    ``margin`` is the minimum multiplicative advantage of correct over the best
    control required to declare the fMRI signal genuinely used.
    """
    sub = summary[(summary.metric_name == metric) & (summary.subject_id == subject)]
    vals = {r.condition: r.value for r in sub.itertuples()}
    correct = vals.get("correct")
    controls = [vals[c] for c in ("permuted", "zero") if c in vals]
    if correct is None or not controls:
        return {"decision": "undetermined", "reason": "missing conditions",
                "values": vals}
    best_control = max(controls)
    uses_brain = correct > best_control * margin
    return {
        "decision": "fmri_used" if uses_brain else "fmri_not_clearly_used",
        "metric": metric, "subject": subject, "correct": correct,
        "best_control": best_control, "margin": margin, "values": vals,
        "message": ("fMRI correcto supera claramente a los controles"
                    if uses_brain else
                    "fMRI correcto NO supera claramente a permutado/cero: "
                    "no se puede afirmar uso de senal cerebral real"),
    }
