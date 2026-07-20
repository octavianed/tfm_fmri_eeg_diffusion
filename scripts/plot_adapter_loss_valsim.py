#!/usr/bin/env python
"""Figura combinada del entrenamiento del adaptador: pérdida de difusión (plana) frente a
la similitud CLIP de validación de la feature 2 (creciente), en ejes gemelos.

Motivación: ilustra de un vistazo el desajuste que sostiene §4.2.4/§4.2.6 de la memoria —
la pérdida apenas se mueve (y su mínimo cae en una época distinta de la seleccionada),
mientras la métrica de calidad con la que se elige el checkpoint sube de forma sostenida.
Marca el mínimo de pérdida y el máximo de similitud, que NO coinciden.

Lee `outputs/exp04_generation_v4_f2/logs/adapter_train_log.csv`.

Uso:
    python scripts/plot_adapter_loss_valsim.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # backend sin ventana: guarda a fichero sin bloquear
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs" / "exp04_generation_v4_f2" / "logs" / "adapter_train_log.csv"
OUT = ROOT / "outputs" / "exp04_generation_v4_f2" / "figures" / "adapter_loss_vs_valsim.png"

LOSS_COLOR = "#2c7fb8"   # azul: pérdida (eje izquierdo)
SIM_COLOR = "#d95f02"    # naranja: similitud CLIP (eje derecho)

df = pd.read_csv(LOG)
sim = df.dropna(subset=["val_clip_sim"])          # la feature 2 evalúa cada N épocas
ep_min_loss = int(df.loc[df["loss"].idxmin(), "epoch"])
ep_max_sim = int(sim.loc[sim["val_clip_sim"].idxmax(), "epoch"])

fig, ax_loss = plt.subplots(figsize=(8.5, 5))

# --- eje izquierdo: pérdida de difusión ---
ax_loss.plot(df["epoch"], df["loss"], "-o", color=LOSS_COLOR, ms=4,
             label="pérdida por época")
ax_loss.set_xlabel("época del adaptador")
ax_loss.set_ylabel("pérdida de difusión (MSE de ruido)", color=LOSS_COLOR)
ax_loss.tick_params(axis="y", labelcolor=LOSS_COLOR)

# --- eje derecho: similitud CLIP de validación (feature 2) ---
ax_sim = ax_loss.twinx()
ax_sim.plot(sim["epoch"], sim["val_clip_sim"], "-s", color=SIM_COLOR, ms=5,
            label="val_clip_sim (feature 2)")
ax_sim.set_ylabel("similitud CLIP de validación (generada↔real)", color=SIM_COLOR)
ax_sim.tick_params(axis="y", labelcolor=SIM_COLOR)

# --- marcadores de los dos óptimos, que NO coinciden ---
ax_loss.axvline(ep_min_loss, color=LOSS_COLOR, ls=":", alpha=0.7, label="_nolegend_")
ax_loss.annotate(f"mín. pérdida (ép. {ep_min_loss})", xy=(ep_min_loss, df["loss"].min()),
                 xytext=(ep_min_loss + 0.3, df["loss"].min() + 0.0004), color=LOSS_COLOR,
                 fontsize=8)
ax_sim.axvline(ep_max_sim, color=SIM_COLOR, ls=":", alpha=0.7, label="_nolegend_")
ax_sim.annotate(f"«best» = máx. similitud (ép. {ep_max_sim})",
                xy=(ep_max_sim, sim["val_clip_sim"].max()),
                xytext=(ep_max_sim - 9.5, sim["val_clip_sim"].max() - 0.008),
                color=SIM_COLOR, fontsize=8)

# leyenda combinada de ambos ejes (excluye las líneas verticales sin etiqueta)
lines = [l for l in ax_loss.get_lines() + ax_sim.get_lines()
         if not l.get_label().startswith("_")]
ax_loss.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8,
               ncol=2)
ax_loss.set_title("Entrenamiento del adaptador: la pérdida no baja, la similitud CLIP sí sube\n"
                  "(sus óptimos caen en épocas distintas, y ninguno es el checkpoint útil)")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("Figura guardada en:", OUT)
print(f"mín. pérdida: época {ep_min_loss} | máx. val_clip_sim: época {ep_max_sim}")
