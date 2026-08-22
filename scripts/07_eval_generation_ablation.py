#!/usr/bin/env python
"""Experiment 5: final generative comparison across conditions (spec §13.8).

Loads the images produced by Experiment 4, computes CLIP-based similarity /
retrieval (plus optional SSIM/LPIPS) per condition, derives the deltas the
project's criterion is stated in (correct−permuted, correct−zero, and — for the
text/ControlNet architectures — delta_text, delta_semantic, delta_lowlevel),
runs paired significance tests and writes grids plus a summary report.

The logic lives in ``src/evaluation/generation_ablation.py``; this file only
orchestrates.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.evaluation.generation_ablation import (build_report,  # noqa: E402
                                                compute_deltas, conclusion,
                                                paired_test, score_conditions)
from src.features import load_clip  # noqa: E402
from src.generation import case_grids, save_comparison_grid  # noqa: E402
from src.utils import (ExtendOverrides, get_device,  # noqa: E402
                       get_experiment_paths, get_logger, load_config, load_json,
                       save_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of conditions to score (default: all generated)")
    args = ap.parse_args()
    log = get_logger("eval_generation")

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("runtime.device", "auto"))
    paths = get_experiment_paths(cfg, ensure=True)

    source = cfg.get("generation.source_experiment", "exp04_generation")
    src_dir = Path(cfg.get("paths.output_dir", "outputs")) / source
    params = load_json(src_dir / "metadata" / "generation_params.json")
    requested = args.conditions or cfg.get("generation.conditions", None)
    log.info("Source: %s | %d samples | architecture=%s | text=%s", src_dir,
             len(params["image_ids"]),
             params.get("conditioning_architecture", "legacy_adapter"),
             params.get("text_mode", "none"))

    clip_bundle = load_clip(cfg, device)
    scored = score_conditions(cfg, src_dir, params, clip_bundle, device, requested)
    summary, per_sample, images = scored["summary"], scored["per_sample"], scored["images"]
    conditions = scored["conditions"]
    if summary.empty:
        raise SystemExit(f"No generated conditions found under {src_dir}/generated")

    summary.to_csv(paths.metrics / "summary_generation_metrics.csv", index=False)

    deltas = compute_deltas(per_sample)
    if len(deltas):
        deltas.to_csv(paths.metrics / "generation_deltas.csv", index=False)
        for row in deltas.itertuples():
            log.info("  %-24s %+0.4f (p=%s)  %s", row.delta, row.value,
                     "n/a" if row.t_pvalue is None else f"{row.t_pvalue:.4f}",
                     row.question)

    # Every condition against 'correct', so nothing is silently left untested.
    tests = {}
    if "correct" in per_sample:
        base = np.asarray(per_sample["correct"]["clip_similarity"], float)
        for cond in conditions:
            if cond == "correct":
                continue
            tests[f"correct_vs_{cond}"] = paired_test(
                base, np.asarray(per_sample[cond]["clip_similarity"], float))
    save_json(tests, paths.metrics / "statistical_tests.json")

    order = ("real",) + tuple(conditions)
    save_comparison_grid(images, paths.grids / "comparison_grid_ablation.png",
                         column_order=order, max_rows=min(8, len(params["image_ids"])))
    if "correct" in per_sample:
        case_grids(images, per_sample["correct"]["clip_similarity"], paths.grids,
                   column_order=order,
                   k=int(cfg.get("generation.num_case_examples", 5)))

    verdict = conclusion(summary)
    save_json(verdict, paths.metrics / "generation_conclusion.json")
    report = build_report(params, summary, deltas, verdict, tests)
    (paths.report / "exp05_summary.md").write_text(report, encoding="utf-8")
    log.info("Report written: %s", paths.report / "exp05_summary.md")
    log.info("correct=%.4f best_control=%.4f -> %s",
             verdict.get("correct", float("nan")),
             verdict.get("best_control", float("nan")), verdict["decision"])


if __name__ == "__main__":
    main()
