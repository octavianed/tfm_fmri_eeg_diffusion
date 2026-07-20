#!/usr/bin/env python
"""Experiment 2: retrieval ablation (fMRI correct / permuted / zero) + baselines.

Loads the Experiment 1 checkpoint and evaluates whether the correct fMRI clearly
beats the negative controls — the project's core criterion (spec §8, §13.4)."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.data import FmriDataModule  # noqa: E402
from src.evaluation import (conclusion_from_summary, evaluate_ablation,  # noqa: E402
                            evaluate_baselines, save_ablation_figures)
from src.models import build_model_from_checkpoint  # noqa: E402
from src.utils import (get_device, get_experiment_paths, get_logger,  # noqa: E402
                       load_config, save_json)


def resolve_checkpoint(cfg, explicit):
    if explicit:
        return Path(explicit)
    ckpt = cfg.get("evaluation.checkpoint")
    if ckpt:
        return Path(ckpt)
    source = cfg.get("evaluation.source_experiment", "exp01_fmri_to_clip")
    return Path(cfg.get("paths.output_dir", "outputs")) / source / "checkpoints" / "best.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()
    log = get_logger("eval_ablation")

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("runtime.device", "auto"))
    paths = get_experiment_paths(cfg, ensure=True)
    split = args.split or cfg.get("evaluation.split", "test")
    conditions = args.conditions or list(cfg.get("evaluation.conditions",
                                                 ["correct", "permuted", "zero"]))

    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt} (train Experiment 1 first)")
    log.info("Evaluating checkpoint: %s | split=%s | conditions=%s",
             ckpt, split, conditions)

    dm = FmriDataModule(cfg).prepare()
    model, _ = build_model_from_checkpoint(cfg, ckpt, device, dm.voxel_counts)

    result = evaluate_ablation(model, cfg, dm, split=split, conditions=conditions,
                               device=device, checkpoint_name=str(ckpt))
    summary = result["summary"]
    summary.to_csv(paths.metrics / "summary_table.csv", index=False)

    for cond, data in result["conditions"].items():
        save_json({"macro": data["macro"], "subjects": data["subjects"]},
                  paths.metrics / f"retrieval_{cond}.json")
    # save 'correct' predicted embeddings for reference
    for subj, pred in result["predictions"].get("correct", {}).items():
        np.save(paths.embeddings / f"{subj}_{split}_clip_pred.npy", pred["clip_pred"])
        np.save(paths.embeddings / f"{subj}_{split}_clip_target.npy", pred["clip_target"])

    save_ablation_figures(summary, paths.figures, subject="all")

    if bool(cfg.get("evaluation.run_baselines", True)):
        log.info("Computing mean + ridge baselines ...")
        baselines = evaluate_baselines(cfg, dm, split=split,
                                       ridge_alpha=float(cfg.get("evaluation.ridge_alpha", 1000.0)))
        save_json(baselines, paths.metrics / "baselines.json")

    conclusion = conclusion_from_summary(summary, metric="retrieval/top5",
                                         subject="all")
    save_json(conclusion, paths.metrics / "conclusion.json")
    log.info("CONCLUSION: %s", conclusion["message"])
    log.info("  correct=%.4f vs best_control=%.4f (metric=%s)",
             conclusion.get("correct", float("nan")),
             conclusion.get("best_control", float("nan")), conclusion.get("metric"))


if __name__ == "__main__":
    main()
