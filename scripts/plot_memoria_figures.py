#!/usr/bin/env python
"""Figuras y tablas del capitulo de Resultados de la memoria.

Lee unicamente artefactos ya existentes en ``outputs/`` (JSON y CSV de metricas)
y produce, en ``outputs/_memoria/``:

Tablas (CSV, listas para copiar a la memoria)
    tabla_decodificacion.csv          una fila por linea de senal: Exp1, Exp3,
                                      controles, baselines y veredicto
    tabla_generacion_condiciones.csv  una fila por (corrida, condicion)
    tabla_generacion_deltas.csv       una fila por (corrida, contraste)

Figuras (PNG a 200 ppp)
    fig_5_5_eeg_preproc_top5.png            Top-5 por variante de preprocesado (EEG)
    fig_5_6_comparativa_modalidades.png     razon Top-5 / azar por modalidad
    fig_5_8_deltas_generacion.png           deltas pareados con significacion
    fig_5_10_similitud_por_arquitectura.png similitud por configuracion
    fig_5_11_efecto_estructura.png          efecto de la rama estructural

Las similitudes y sus deltas se expresan en PUNTOS PORCENTUALES (x100), para que
dos decimales sean informativos.

Uso:  python scripts/plot_memoria_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs")
DEST = OUT / "_memoria"
PP = 100.0

# (etiqueta, exp1, exp3, exp2 sobre Exp3, n candidatos, grupo)
LINEAS = [
    ("fMRI (NSD)", "exp01_fmri_to_clip", "exp03_fmri_lowlevel_multitask",
     "exp02_fmri_retrieval_ablation_exp3", 159, "fMRI"),
    ("Oficial 17 ch", "exp01_17_eeg_to_clip", "exp03_17_eeg_lowlevel_multitask",
     "exp02_17_eeg_retrieval_ablation_exp3", 200, "EEG oficial"),
    ("Oficial 63 ch", "exp01_63_eeg_to_clip", "exp03_63_eeg_lowlevel_multitask",
     "exp02_63_eeg_retrieval_ablation_exp3", 200, "EEG oficial"),
    ("Propio: baseline", "exp01_raw_baseline_eeg_to_clip",
     "exp03_raw_baseline_eeg_lowlevel_multitask",
     "exp02_raw_baseline_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: temporal 100-600", "exp01_raw_temporal_100_600_eeg_to_clip",
     "exp03_raw_temporal_100_600_eeg_lowlevel_multitask",
     "exp02_raw_temporal_100_600_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: temporal 200-400", "exp01_raw_temporal_200_400_eeg_to_clip",
     "exp03_raw_temporal_200_400_eeg_lowlevel_multitask",
     "exp02_raw_temporal_200_400_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: sin blanqueo", "exp01_raw_ablate_mvnn_eeg_to_clip",
     "exp03_raw_ablate_mvnn_eeg_lowlevel_multitask",
     "exp02_raw_ablate_mvnn_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: 17 canales", "exp01_raw_channels_17_eeg_to_clip",
     "exp03_raw_channels_17_eeg_lowlevel_multitask",
     "exp02_raw_channels_17_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: 100 Hz", "exp01_raw_sampling_100hz_eeg_to_clip",
     "exp03_raw_sampling_100hz_eeg_lowlevel_multitask",
     "exp02_raw_sampling_100hz_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: 0,5-40 Hz", "exp01_raw_frequency_0_5_40_eeg_to_clip",
     "exp03_raw_frequency_0_5_40_eeg_lowlevel_multitask",
     "exp02_raw_frequency_0_5_40_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: referencia CAR", "exp01_raw_reference_car_eeg_to_clip",
     "exp03_raw_reference_car_eeg_lowlevel_multitask",
     "exp02_raw_reference_car_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: linea base -100", "exp01_raw_baseline_minus100_eeg_to_clip",
     "exp03_raw_baseline_minus100_eeg_lowlevel_multitask",
     "exp02_raw_baseline_minus100_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
    ("Propio: trials sueltos", "exp01_raw_train_independent_trials_eeg_to_clip",
     "exp03_raw_train_independent_trials_eeg_lowlevel_multitask",
     "exp02_raw_train_independent_trials_eeg_retrieval_ablation_exp3", 200, "EEG propio"),
]

# (id, modalidad, arquitectura, experimento Exp5)
CORRIDAS = [
    ("Af", "fMRI", "A - solo cerebro", "exp05_generation_legacy_ablation"),
    ("Bf", "fMRI", "B - texto + cerebro", "exp05_generation_text_weak_ablation"),
    ("Cf", "fMRI", "C - + ControlNet", "exp05_generation_controlnet_weak_ablation"),
    ("Ae", "EEG 63", "A - solo cerebro", "exp05_63_eeg_generation_legacy_ablation"),
    ("Be", "EEG 63", "B - texto + cerebro", "exp05_63_eeg_generation_text_weak_ablation"),
    ("Ce", "EEG 63", "C - + ControlNet",
     "exp05_63_eeg_generation_controlnet_weak_ablation"),
    ("Ct", "EEG 100-600", "C - + ControlNet",
     "exp05_raw_temporal_100_600_eeg_generation_controlnet_weak_ablation"),
]

COND_ES = {"correct": "Correcta", "permuted": "Permutada", "zero": "Nula",
           "permuted_text": "Texto permutado",
           "semantic_permuted": "Semantica permutada",
           "semantic_zero": "Semantica nula",
           "lowlevel_permuted": "Estructura permutada",
           "lowlevel_zero": "Estructura nula"}

DELTA_ES = {"delta_correct_permuted": "D cerebro (correcta - permutada)",
            "delta_correct_zero": "D cerebro (correcta - nula)",
            "delta_text": "D texto", "delta_semantic": "D semantica",
            "delta_semantic_zero": "D semantica (nula)",
            "delta_lowlevel": "D estructura",
            "delta_lowlevel_zero": "D estructura (nula)"}

# Deltas cuya atribucion es CONJUNTA en la arquitectura C (ver docs/12_..._new.md)
CONJUNTO = {"C - + ControlNet": {"delta_correct_permuted", "delta_correct_zero"}}


def jload(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def tidy(exp, metric, cond):
    p = OUT / exp / "metrics" / "summary_table.csv"
    if not p.exists():
        return np.nan
    d = pd.read_csv(p)
    s = d[(d.metric_name == metric) & (d.condition == cond) & (d.subject_id == "all")]
    return float(s.value.iloc[0]) if len(s) else np.nan


def tabla_decodificacion():
    rows = []
    for label, e1, e3, e2, ncand, grupo in LINEAS:
        j1 = jload(OUT / e1 / "metrics" / "test_metrics.json")
        j3 = jload(OUT / e3 / "metrics" / "test_metrics.json")
        m1, m3 = j1.get("retrieval", {}), j3.get("retrieval", {})
        low = jload(OUT / e3 / "metrics" / "test_lowlevel_metrics.json")
        base = jload(OUT / e2 / "metrics" / "baselines.json")
        conc = jload(OUT / e2 / "metrics" / "conclusion.json")
        subj = next(iter(base.get("mean", {})), None)
        ridge = base.get("ridge", {}).get(subj, {}).get("retrieval", {}) if subj else {}
        media = base.get("mean", {}).get(subj, {}).get("retrieval", {}) if subj else {}
        rows.append({
            "grupo": grupo, "linea": label, "n_candidatos": ncand,
            "azar_top5_%": 100.0 * 5 / ncand,
            "exp1_top1_%": 100.0 * m1.get("top1", np.nan),
            "exp1_top5_%": 100.0 * m1.get("top5", np.nan),
            "exp3_top1_%": 100.0 * m3.get("top1", np.nan),
            "exp3_top5_%": 100.0 * m3.get("top5", np.nan),
            "exp3_top10_%": 100.0 * m3.get("top10", np.nan),
            "exp3_rango_medio": m3.get("mean_rank", np.nan),
            "exp3_coseno": j3.get("embedding", {}).get("mean_cosine", np.nan),
            "lowlevel_r2": low.get("r2", np.nan),
            "lowlevel_pearson": low.get("mean_pearson", np.nan),
            "ctrl_permutada_top5_%": 100.0 * tidy(e2, "retrieval/top5", "permuted"),
            "ctrl_nula_top5_%": 100.0 * tidy(e2, "retrieval/top5", "zero"),
            "ctrl_ruido_top5_%": 100.0 * tidy(e2, "retrieval/top5", "noise"),
            "media_top5_%": 100.0 * media.get("top5", np.nan),
            "ridge_top5_%": 100.0 * ridge.get("top5", np.nan),
            "veredicto": conc.get("decision", ""),
            "razon_sobre_azar": m3.get("top5", np.nan) * ncand / 5.0,
        })
    return pd.DataFrame(rows)


def tablas_generacion():
    cond_rows, delta_rows = [], []
    for cid, mod, arq, exp in CORRIDAS:
        s = pd.read_csv(OUT / exp / "metrics" / "summary_generation_metrics.csv")
        for r in s.itertuples():
            cond_rows.append({
                "config": cid, "modalidad": mod, "arquitectura": arq,
                "condicion": COND_ES.get(r.condition, r.condition),
                "similitud_pp": PP * r.mean_clip_similarity,
                "similitud_mediana_pp": PP * r.median_clip_similarity,
                "top1_%": 100.0 * r.clip_top1, "top5_%": 100.0 * r.clip_top5,
                "mse_pixel": r.mean_pixel_mse,
                "ssim": getattr(r, "mean_ssim", np.nan),
                "lpips": getattr(r, "mean_lpips", np.nan)})
        d = pd.read_csv(OUT / exp / "metrics" / "generation_deltas.csv")
        d = d[~d.delta.isin(["delta_joint_brain", "delta_brain"])]
        for r in d.itertuples():
            conj = r.delta in CONJUNTO.get(arq, set())
            sig = bool(r.t_pvalue < 0.05 and r.wilcoxon_pvalue < 0.05)
            delta_rows.append({
                "config": cid, "modalidad": mod, "arquitectura": arq,
                "delta": DELTA_ES.get(r.delta, r.delta), "delta_id": r.delta,
                "valor_pp": PP * r.value, "p_t": r.t_pvalue,
                "p_wilcoxon": r.wilcoxon_pvalue, "n": r.n,
                "atribucion": "conjunto" if conj else "limpio",
                "significativo": sig})
    return pd.DataFrame(cond_rows), pd.DataFrame(delta_rows)


def fig_eeg_preproc(dec):
    d = dec[dec.grupo.str.startswith("EEG")].sort_values("exp3_top5_%")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y, d["exp3_top5_%"], color="#2c7fb8", label="Decodificador (Exp3)")
    ax.plot(d["ridge_top5_%"], y, "D", color="#d95f02", ms=5, label="Baseline ridge")
    ax.plot(d["ctrl_permutada_top5_%"], y, "x", color="#555555", ms=6,
            label="Control permutado")
    ax.axvline(2.5, ls="--", lw=1, color="black", label="Azar (2,50 %)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["linea"], fontsize=9)
    ax.set_xlabel("Top-5 en el conjunto de prueba (%)")
    ax.set_title("Decodificacion semantica en EEG por preprocesamiento")
    for i, v in enumerate(d["exp3_top5_%"]):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DEST / "fig_5_5_eeg_preproc_top5.png", dpi=200)
    plt.close(fig)


def fig_comparativa(dec):
    sel = ["fMRI (NSD)", "Oficial 17 ch", "Oficial 63 ch", "Propio: baseline",
           "Propio: temporal 100-600", "Propio: temporal 200-400"]
    d = dec[dec.linea.isin(sel)].set_index("linea").loc[sel].reset_index()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar(d["linea"], d["razon_sobre_azar"],
                  color=["#1b7837"] + ["#2c7fb8"] * 4)
    ax.axhline(1, ls="--", lw=1, color="black")
    ax.set_yscale("log")
    ax.set_ylabel("Top-5 / azar (escala logaritmica)")
    ax.set_title("Decodificacion normalizada frente al azar")
    for b, v in zip(bars, d["razon_sobre_azar"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"x{v:.2f}",
                ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DEST / "fig_5_6_comparativa_modalidades.png", dpi=200)
    plt.close(fig)


def fig_deltas(dl):
    keep = ["delta_correct_permuted", "delta_text", "delta_semantic", "delta_lowlevel"]
    d = dl[dl.delta_id.isin(keep)].copy()
    orden = {k: i for i, k in enumerate(keep)}
    d["o"] = d.delta_id.map(orden)
    d = d.sort_values(["config", "o"], ascending=[False, True]).reset_index(drop=True)
    colores = {0: "#2c7fb8", 1: "#7570b3", 2: "#1b7837", 3: "#d95f02"}
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(d) + 1.8))
    ax.axvline(0, color="black", lw=1)
    for i, r in d.iterrows():
        col = colores[r.o]
        ax.plot(r.valor_pp, i, "o", ms=7, color=col,
                mfc=col if r.significativo else "white", mec=col)
        off = 0.25 if r.valor_pp >= 0 else -0.25
        ax.text(r.valor_pp + off, i, f"{r.valor_pp:+.2f}", va="center", fontsize=8,
                ha="left" if r.valor_pp >= 0 else "right")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{r.config} - {r.delta}" for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel("Delta de similitud semantica (puntos porcentuales)")
    ax.set_title("Contrastes pareados de generacion\n"
                 "relleno = significativo (t y Wilcoxon); hueco = no significativo",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DEST / "fig_5_8_deltas_generacion.png", dpi=200)
    plt.close(fig)


def fig_similitud_arquitectura(dc):
    piv = dc[dc.condicion.isin(["Correcta", "Nula"])].pivot_table(
        index="config", columns="condicion", values="similitud_pp")
    orden = [c for c in ["Af", "Bf", "Cf", "Ae", "Be", "Ce", "Ct"]
             if c in piv.index]
    piv = piv.loc[orden]
    x = np.arange(len(piv))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(x - w / 2, piv["Correcta"], w, label="Condicion correcta", color="#2c7fb8")
    ax.bar(x + w / 2, piv["Nula"], w, label="Condicion nula (control)", color="#bdbdbd")
    for i, (c, z) in enumerate(zip(piv["Correcta"], piv["Nula"])):
        ax.text(i - w / 2, c + 0.5, f"{c:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, z + 0.5, f"{z:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("Similitud semantica (puntos porcentuales)")
    ax.set_title("Condicion correcta frente al control nulo, por configuracion")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, float(piv.values.max()) * 1.18)
    fig.tight_layout()
    fig.savefig(DEST / "fig_5_10_similitud_por_arquitectura.png", dpi=200)
    plt.close(fig)


def fig_estructura(dc):
    cs = ["Cf", "Ce", "Ct"]
    conds = ["Correcta", "Estructura permutada", "Estructura nula"]
    piv = dc[dc.config.isin(cs) & dc.condicion.isin(conds)].pivot_table(
        index="config", columns="condicion", values="similitud_pp").loc[cs][conds]
    x = np.arange(len(cs))
    w = 0.26
    cols = ["#2c7fb8", "#d95f02", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for j, c in enumerate(conds):
        vals = piv[c]
        ax.bar(x + (j - 1) * w, vals, w, label=c, color=cols[j])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.5, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cs)
    ax.set_ylabel("Similitud semantica (puntos porcentuales)")
    ax.set_title("Efecto de la rama estructural (red de control)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, float(piv.values.max()) * 1.18)
    fig.tight_layout()
    fig.savefig(DEST / "fig_5_11_efecto_estructura.png", dpi=200)
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    dec = tabla_decodificacion()
    dc, dl = tablas_generacion()
    dec.round(4).to_csv(DEST / "tabla_decodificacion.csv", index=False)
    dc.round(4).to_csv(DEST / "tabla_generacion_condiciones.csv", index=False)
    dl.round(6).to_csv(DEST / "tabla_generacion_deltas.csv", index=False)
    fig_eeg_preproc(dec)
    fig_comparativa(dec)
    fig_deltas(dl)
    fig_similitud_arquitectura(dc)
    fig_estructura(dc)
    print(f"[ok] 3 tablas y 5 figuras en {DEST}")
    cols = ["linea", "exp3_top5_%", "ridge_top5_%", "lowlevel_r2", "veredicto"]
    print(dec[cols].to_string(index=False))


if __name__ == "__main__":
    main()
