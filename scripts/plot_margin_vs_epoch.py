#!/usr/bin/env python
"""Figura: margen (correct - mejor_control) vs. época del adapter, para los dos
modos de generación (adapter / adapter_lowlevel), con los dos puntos que tienen
test de significación (época 0 y el checkpoint 'best') resaltados con su p-valor.

Lee directamente de los ficheros ya generados — no hay números copiados a mano:
  - Trayectoria completa (todas las épocas del barrido):
      outputs/exp04_generation_v4_f2/checkpoint_sweep_adapter/checkpoint_sweep_margins.csv
      outputs/exp04_generation_v4_f2/checkpoint_sweep__adapter_lowlevel/checkpoint_sweep_margins.csv
  - Los dos puntos con test pareado (época 0 y 'best'):
      outputs/exp05_v4f2_ep0_adapter/{metrics/summary_generation_metrics.csv, metrics/statistical_tests.json}
      outputs/exp05_v4f2_best_adapter/{...}
      outputs/exp05_v4f2_ep0_adapter_lowlevel/{...}
      outputs/exp05_v4f2_best_adapter_lowlevel/{...}

Uso:
    python scripts/plot_margin_vs_epoch.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# (nombre del modo, carpeta del sweep, exp05 de epoca 0, exp05 del 'best')
MODES = [
    ("adapter", OUT / "exp04_generation_v4_f2" / "checkpoint_sweep_adapter",
     OUT / "exp05_v4f2_ep0_adapter", OUT / "exp05_v4f2_best_adapter"),
    ("adapter_lowlevel",
     OUT / "exp04_generation_v4_f2" / "checkpoint_sweep__adapter_lowlevel",
     OUT / "exp05_v4f2_ep0_adapter_lowlevel", OUT / "exp05_v4f2_best_adapter_lowlevel"),
]

COLORS = {"adapter": "#2c7fb8", "adapter_lowlevel": "#d95f02"}


def tested_point(exp05_dir):
    """Devuelve (margin, best_control_cond, p_valor_wilcoxon) para un run de Exp5
    con test de significación (correct vs permuted/zero)."""
    summary = pd.read_csv(exp05_dir / "metrics" / "summary_generation_metrics.csv")
    tests = json.loads((exp05_dir / "metrics" / "statistical_tests.json").read_text())
    sims = dict(zip(summary["condition"], summary["mean_clip_similarity"]))
    correct = sims["correct"]
    controls = {c: sims[c] for c in ("permuted", "zero") if c in sims}
    best_cond = max(controls, key=controls.get)
    margin = correct - controls[best_cond]
    p = tests[f"correct_vs_{best_cond}"]["wilcoxon_pvalue"]
    return margin, best_cond, p


fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="black", lw=0.8, ls="--", zorder=1)

for mode, sweep_dir, exp05_ep0, exp05_best in MODES:
    color = COLORS[mode]

    # --- trayectoria completa (sin test de significacion) ---
    margins = pd.read_csv(sweep_dir / "checkpoint_sweep_margins.csv")
    margins = margins.sort_values("epoch")
    ax.plot(margins["epoch"], margins["margin"], "-o", color=color, alpha=0.55,
            ms=5, label=f"{mode} (barrido, sin test)", zorder=2)

    # --- los dos puntos con test pareado (epoca 0 y 'best') ---
    for exp05_dir, epoch_label in ((exp05_ep0, 0), (exp05_best, None)):
        margin, best_cond, p = tested_point(exp05_dir)
        epoch = epoch_label if epoch_label is not None else \
            margins.loc[margins["checkpoint"] == "best", "epoch"].iloc[0]
        significant = p < 0.05
        ax.scatter([epoch], [margin], s=170,
                   marker="*" if significant else "X",
                   facecolor=color if significant else "none",
                   edgecolor=color, linewidths=1.8, zorder=3)
        ax.annotate(f"p={p:.1e}" + ("*" if significant else " (n.s.)"),
                   (epoch, margin), textcoords="offset points",
                   xytext=(8, 10 if margin >= 0 else -16), fontsize=8, color=color)

ax.set_xlabel("época del adapter")
ax.set_ylabel("margen: correct − mejor control (similitud CLIP)")
ax.set_title("Dependencia de la fMRI correcta vs. época de entrenamiento del adapter\n"
             "(★ = significativo p<0.05, test de Wilcoxon pareado; ✕ = no significativo)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()

out_path = OUT / "exp04_generation_v4_f2" / "figures" / "margin_vs_epoch_significance.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print("Figura guardada en:", out_path)
plt.show()
