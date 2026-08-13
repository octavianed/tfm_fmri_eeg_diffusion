#!/usr/bin/env python
"""Experiment 5: final generative comparison correct / permuted / zero (spec §13.8).

Loads the images produced by Experiment 4, computes CLIP-based similarity /
retrieval (plus optional SSIM/LPIPS), runs paired significance tests and writes
comparison + best/median/worst grids and a summary report."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.evaluation import compute_generation_metrics  # noqa: E402
from src.features import load_clip  # noqa: E402
from src.generation import case_grids, save_comparison_grid  # noqa: E402
from src.utils import (ExtendOverrides, get_device,  # noqa: E402
                       get_experiment_paths, get_logger, load_config, load_json,
                       save_json)


def _load_images(folder, image_ids):
    folder = Path(folder)
    return [Image.open(folder / f"{iid}.png").convert("RGB") for iid in image_ids]


def _paired_tests(per_sample_sim):
    try:
        from scipy import stats
    except Exception:
        return {"note": "scipy not available; skipped significance tests"}
    out = {}
    correct = per_sample_sim.get("correct")
    for cond in ("permuted", "zero"):
        other = per_sample_sim.get(cond)
        if correct is None or other is None:
            continue
        t = stats.ttest_rel(correct, other)
        try:
            w = stats.wilcoxon(correct, other)
            w_p = float(w.pvalue)
        except Exception:
            w_p = None
        out[f"correct_vs_{cond}"] = {
            "mean_diff": float(np.mean(correct - other)),
            "t_stat": float(t.statistic), "t_pvalue": float(t.pvalue),
            "wilcoxon_pvalue": w_p}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    args = ap.parse_args()
    log = get_logger("eval_generation")

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("runtime.device", "auto"))
    paths = get_experiment_paths(cfg, ensure=True)

    source = cfg.get("generation.source_experiment", "exp04_generation")
    src_dir = Path(cfg.get("paths.output_dir", "outputs")) / source
    params = load_json(src_dir / "metadata" / "generation_params.json")
    image_ids = params["image_ids"]
    conditions = [c for c in cfg.get("generation.conditions",
                                     params.get("conditions", ["correct"]))
                  if (src_dir / "generated" / c).exists()]
    log.info("Source: %s | %d samples | conditions=%s", src_dir, len(image_ids),
             conditions)

    reals = _load_images(src_dir / "generated" / "real", image_ids)
    clip_bundle = load_clip(cfg, device)

    rows, per_sample_sim, outputs = [], {}, {"real": reals, "image_ids": image_ids}
    for cond in conditions:
        gen = _load_images(src_dir / "generated" / cond, image_ids)
        outputs[cond] = gen
        res = compute_generation_metrics(
            reals, gen, clip_bundle, device, ks=(1, 5),
            use_ssim=bool(cfg.get("generation.compute_ssim", False)),
            use_lpips=bool(cfg.get("generation.compute_lpips", False)))
        per_sample_sim[cond] = res["per_sample"]["clip_similarity"]
        m = res["metrics"]
        rows.append({"condition": cond,
                     "mean_clip_similarity": m["mean_clip_similarity"],
                     "clip_top1": m["clip_retrieval"].get("top1"),
                     "clip_top5": m["clip_retrieval"].get("top5"),
                     "mean_pixel_mse": m.get("mean_pixel_mse"),
                     "mean_ssim": m.get("mean_ssim"),
                     "mean_lpips": m.get("mean_lpips")})
        log.info("  %-9s clip_sim=%.4f top1=%.3f", cond,
                 m["mean_clip_similarity"], m["clip_retrieval"].get("top1", 0.0))

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.metrics / "summary_generation_metrics.csv", index=False)
    tests = _paired_tests(per_sample_sim)
    save_json(tests, paths.metrics / "statistical_tests.json")

    # grids
    save_comparison_grid(outputs, paths.grids / "comparison_grid_ablation.png",
                         column_order=("real",) + tuple(conditions),
                         max_rows=min(8, len(image_ids)))
    if "correct" in per_sample_sim:
        case_grids(outputs, per_sample_sim["correct"], paths.grids,
                   column_order=("real",) + tuple(conditions),
                   k=int(cfg.get("generation.num_case_examples", 5)))

    # conclusion
    correct_mean = summary.loc[summary.condition == "correct",
                               "mean_clip_similarity"].mean() \
        if "correct" in conditions else float("nan")
    controls = summary[summary.condition.isin(["permuted", "zero"])]["mean_clip_similarity"]
    best_control = float(controls.max()) if len(controls) else float("nan")
    uses_brain = bool(correct_mean > best_control) if not np.isnan(best_control) else False
    report = (
        f"# Experiment 5 — generative comparison\n\n"
        f"Samples: {len(image_ids)} | conditions: {conditions}\n\n"
        f"- correct mean CLIP similarity: {correct_mean:.4f}\n"
        f"- best control (permuted/zero): {best_control:.4f}\n\n"
        f"**Conclusion:** "
        + ("La generacion con fMRI correcto supera a los controles: hay evidencia "
           "de que la senal cerebral influye en la reconstruccion.\n"
           if uses_brain else
           "La generacion con fMRI correcto NO supera claramente a permutado/cero: "
           "no se puede atribuir la reconstruccion a la senal cerebral real.\n"))
    (paths.report / "exp05_summary.md").write_text(report, encoding="utf-8")
    log.info("Report written: %s", paths.report / "exp05_summary.md")
    log.info("correct=%.4f best_control=%.4f -> uses_brain=%s", correct_mean,
             best_control, uses_brain)


if __name__ == "__main__":
    main()
