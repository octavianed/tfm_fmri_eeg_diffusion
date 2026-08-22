"""Experiment 5: score every generated condition and derive the required deltas (§36).

Experiment 4 writes one folder of images per condition plus
``metadata/generation_params.json`` (which records where each condition landed
and what it means). This module reads them back, computes the generation metrics
per condition and turns them into the contrasts the project's falsifiable
criterion is stated in:

============================  ==========================================
delta                          contrast
============================  ==========================================
``delta_correct_permuted``     correct − permuted (the load-bearing one)
``delta_correct_zero``         correct − zero
``delta_brain``                text fixed: correct brain − permuted brain
``delta_text``                 brain fixed: correct text − permuted text
``delta_semantic``             ControlNet fixed: correct − permuted tokens
``delta_lowlevel``             tokens fixed: correct − permuted low-level
``delta_joint_brain``          both branches correct − both permuted
============================  ==========================================

Deltas are computed **per sample and then averaged**, so the paired t/Wilcoxon
tests measure exactly the quantity reported (every condition was generated from
the same images with the same diffusion seed — §24).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import get_logger
from .generation_metrics import compute_generation_metrics

logger = get_logger("generation_ablation")

#: Metric used for the conclusion; higher is better.
PRIMARY_METRIC = "mean_clip_similarity"

#: (delta name, positive condition, negative condition, what it answers).
DELTA_DEFINITIONS = (
    ("delta_correct_permuted", "correct", "permuted",
     "¿la señal cerebral correcta supera a la permutada?"),
    ("delta_correct_zero", "correct", "zero",
     "¿la señal cerebral correcta supera al control nulo?"),
    ("delta_text", "correct", "permuted_text",
     "con cerebro correcto, ¿aporta el caption correcto frente al permutado?"),
    ("delta_semantic", "correct", "semantic_permuted",
     "con ControlNet correcta, ¿aporta el CLIP cerebral?"),
    ("delta_semantic_zero", "correct", "semantic_zero",
     "con ControlNet correcta, ¿aporta el CLIP cerebral frente al nulo?"),
    ("delta_lowlevel", "correct", "lowlevel_permuted",
     "con pseudo-tokens correctos, ¿aporta la predicción VAE-PCA cerebral?"),
    ("delta_lowlevel_zero", "correct", "lowlevel_zero",
     "con pseudo-tokens correctos, ¿aporta la ControlNet frente a desactivarla?"),
)


def load_condition_images(source_dir, params: dict, condition: str,
                          image_ids: Sequence[str]) -> Optional[List[Image.Image]]:
    """Images of one condition, honouring the recorded layout (flat or nested)."""
    rel = (params.get("condition_dirs") or {}).get(condition, condition)
    folder = Path(source_dir) / "generated" / rel
    if not folder.exists():
        return None
    try:
        return [Image.open(folder / f"{iid}.png").convert("RGB") for iid in image_ids]
    except FileNotFoundError as exc:
        logger.warning("Condition '%s' is incomplete (%s); skipping.", condition, exc)
        return None


def available_conditions(source_dir, params: dict,
                         requested: Optional[Sequence[str]] = None) -> List[str]:
    names = list(requested or params.get("conditions", ["correct"]))
    dirs = params.get("condition_dirs") or {}
    out = []
    for name in names:
        rel = dirs.get(name, name)
        if (Path(source_dir) / "generated" / rel).exists():
            out.append(name)
        else:
            logger.warning("No generated images for condition '%s'", name)
    return out


def score_conditions(cfg, source_dir, params: dict, clip_bundle, device,
                     conditions: Optional[Sequence[str]] = None) -> dict:
    """Per-condition metrics + per-sample CLIP similarity."""
    image_ids = params["image_ids"]
    reals = load_condition_images(source_dir, {"condition_dirs": {"real": "real"}},
                                  "real", image_ids)
    if reals is None:
        raise FileNotFoundError(f"No 'real' reference images under {source_dir}")

    names = available_conditions(source_dir, params, conditions)
    rows, per_sample, images = [], {}, {"real": reals, "image_ids": image_ids}
    for name in names:
        gen = load_condition_images(source_dir, params, name, image_ids)
        if gen is None:
            continue
        res = compute_generation_metrics(
            reals, gen, clip_bundle, device, ks=(1, 5),
            use_ssim=bool(cfg.get("generation.compute_ssim", False)),
            use_lpips=bool(cfg.get("generation.compute_lpips", False)))
        m = res["metrics"]
        per_sample[name] = res["per_sample"]
        images[name] = gen
        rows.append({"condition": name,
                     "mean_clip_similarity": m["mean_clip_similarity"],
                     "median_clip_similarity": m["median_clip_similarity"],
                     "clip_top1": m["clip_retrieval"].get("top1"),
                     "clip_top5": m["clip_retrieval"].get("top5"),
                     "mean_pixel_mse": m.get("mean_pixel_mse"),
                     "mean_ssim": m.get("mean_ssim"),
                     "mean_lpips": m.get("mean_lpips")})
        logger.info("  %-20s clip_sim=%.4f top1=%.3f", name,
                    m["mean_clip_similarity"], m["clip_retrieval"].get("top1", 0.0))
    return {"summary": pd.DataFrame(rows), "per_sample": per_sample,
            "images": images, "conditions": names}


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test + Wilcoxon on the per-sample differences."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    out = {"n": int(diff.size), "mean_diff": float(diff.mean()),
           "std_diff": float(diff.std(ddof=1)) if diff.size > 1 else 0.0}
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        out["note"] = "scipy not available; no significance test"
        return out
    if diff.size > 1:
        t = stats.ttest_rel(a, b)
        out.update(t_stat=float(t.statistic), t_pvalue=float(t.pvalue))
        try:
            out["wilcoxon_pvalue"] = float(stats.wilcoxon(a, b).pvalue)
        except Exception:
            out["wilcoxon_pvalue"] = None
    return out


def compute_deltas(per_sample: Dict[str, dict],
                   metric: str = "clip_similarity") -> pd.DataFrame:
    """Every defined delta that both of its conditions are available for."""
    rows = []
    for name, pos, neg, question in DELTA_DEFINITIONS:
        if pos not in per_sample or neg not in per_sample:
            continue
        a = np.asarray(per_sample[pos][metric], float)
        b = np.asarray(per_sample[neg][metric], float)
        stats_ = paired_test(a, b)
        rows.append({"delta": name, "positive": pos, "negative": neg,
                     "metric": metric, "question": question,
                     "value": stats_["mean_diff"],
                     "t_pvalue": stats_.get("t_pvalue"),
                     "wilcoxon_pvalue": stats_.get("wilcoxon_pvalue"),
                     "n": stats_["n"]})
    # delta_brain / delta_joint_brain are aliases whose meaning depends on the
    # architecture; name them explicitly so the tables read the same way in
    # every experiment (§36).
    for alias, base in (("delta_brain", "delta_correct_permuted"),
                        ("delta_joint_brain", "delta_correct_permuted")):
        src = [r for r in rows if r["delta"] == base]
        if src:
            row = dict(src[0])
            row["delta"] = alias
            rows.append(row)
    return pd.DataFrame(rows)


def conclusion(summary: pd.DataFrame, metric: str = PRIMARY_METRIC,
               controls: Sequence[str] = ("permuted", "zero")) -> dict:
    """``correcto >> permutado ~ cero`` applied to the generated images."""
    vals = dict(zip(summary["condition"], summary[metric]))
    correct = vals.get("correct")
    present = [vals[c] for c in controls if c in vals]
    if correct is None or not present:
        return {"decision": "undetermined", "reason": "missing conditions",
                "values": vals}
    best_control = max(present)
    uses_brain = bool(correct > best_control)
    return {"decision": "brain_used" if uses_brain else "brain_not_clearly_used",
            "metric": metric, "correct": float(correct),
            "best_control": float(best_control),
            "margin": float(correct - best_control), "values": vals,
            "message": ("La generación con señal cerebral correcta supera a los "
                        "controles: hay evidencia de que la señal cerebral influye "
                        "en la reconstrucción."
                        if uses_brain else
                        "La generación con señal cerebral correcta NO supera "
                        "claramente a permutado/cero: no se puede atribuir la "
                        "reconstrucción a la señal cerebral real.")}


def _as_table(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is around, a fenced plain table otherwise.

    ``tabulate`` is an optional pandas dependency and is NOT installed in this
    project's venv; falling back keeps the report readable instead of crashing
    Experiment 5 over formatting.
    """
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def build_report(params: dict, summary: pd.DataFrame, deltas: pd.DataFrame,
                 verdict: dict, tests: dict) -> str:
    arch = params.get("conditioning_architecture", "legacy_adapter")
    text_mode = params.get("text_mode", "none")
    lines = [
        "# Experiment 5 — comparación generativa",
        "",
        f"- Arquitectura: `{arch}`",
        f"- Texto: `{text_mode}`"
        + (f" (campo `{params.get('caption_field')}`, plantilla "
           f"`{params.get('template')}`)" if text_mode != "none" else ""),
        "- ControlNet: "
        + (f"`{params.get('controlnet_model')}` "
           f"({params.get('controlnet_condition_type')}, escala "
           f"{(params.get('controlnet') or {}).get('conditioning_scale')})"
           if params.get("controlnet_enabled") else "desactivada"),
        f"- Muestras: {params.get('num_samples')} | semilla de difusión: "
        f"{params.get('generation_seed')} | pasos: "
        f"{params.get('num_inference_steps')} | CFG: {params.get('guidance_scale')}",
        "",
        "## Métricas por condición",
        "",
        _as_table(summary),
        "",
    ]
    if len(deltas):
        lines += ["## Deltas", "", _as_table(deltas), ""]
    lines += ["## Conclusión", "",
              f"- correcto: {verdict.get('correct', float('nan')):.4f}",
              f"- mejor control: {verdict.get('best_control', float('nan')):.4f}",
              f"- margen: {verdict.get('margin', float('nan')):+.4f}",
              "",
              f"**{verdict.get('message', 'sin conclusión')}**", ""]
    if text_mode != "none":
        lines += [
            "> Recordatorio metodológico: con un caption informativo (oracle) es "
            "esperable que `correcto ≈ permutado`; eso no implica que el decoder "
            "falle, sino que el texto puede dominar el condicionamiento (§38). "
            "Compara siempre el mismo delta entre modos de texto.", ""]
    if tests:
        lines += ["## Tests pareados", "", "```", str(tests), "```", ""]
    return "\n".join(lines)
