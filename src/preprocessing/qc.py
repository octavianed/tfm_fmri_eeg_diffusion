"""Quality-control figures for a built preprocessing variant (spec §17).

Produced from the ``metadata.json`` a variant writes, so figures can be
regenerated without re-running the pipeline. Everything here is optional and
non-destructive: a failure to draw must never invalidate a built variant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils import get_logger

logger = get_logger("preproc_qc")


def save_qc_figures(meta: dict, out_dir) -> List[str]:
    """Write the QC figures of one subject/variant. Returns the saved paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    qc: List[Dict] = meta.get("qc", []) or []

    # 1) PSD before the main preprocessing (diagnostic only — never gates a notch)
    psd = [r for r in qc if r.get("psd_mean")]
    if psd:
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in psd[:8]:
            ax.semilogy(r["psd_freqs"], r["psd_mean"], lw=0.8,
                        label=f"{r['session']}/{r['source']}")
        ax.axvline(50, color="r", ls="--", lw=0.8, label="50 Hz (line)")
        ax.set_xlabel("Hz"); ax.set_ylabel("PSD"); ax.set_title(
            f"PSD before preprocessing — {meta.get('subject','')}")
        ax.legend(fontsize=6); fig.tight_layout()
        p = out_dir / "psd_before_preprocessing.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(str(p))

    # 2) events / repetitions per session
    if qc:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        labels = [f"{r['session']}\n{r['source']}" for r in qc]
        ax[0].bar(range(len(qc)), [r["n_events"] for r in qc])
        ax[0].set_xticks(range(len(qc))); ax[0].set_xticklabels(labels, fontsize=6)
        ax[0].set_title("events kept per recording")
        reps: Dict[int, int] = {}
        for r in qc:
            for k, v in (r.get("reps_per_image") or {}).items():
                reps[int(k)] = reps.get(int(k), 0) + int(v)
        if reps:
            ks = sorted(reps)
            ax[1].bar([str(k) for k in ks], [reps[k] for k in ks])
            ax[1].set_title("repetitions per image (per session)")
            ax[1].set_xlabel("n repetitions")
        fig.tight_layout()
        p = out_dir / "events_and_repetitions.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(str(p))

    # 3) MVNN diagnostics
    mv = meta.get("mvnn") or {}
    if mv:
        sess = sorted(mv)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].bar(sess, [mv[s].get("shrinkage") or 0 for s in sess])
        ax[0].set_title("Ledoit-Wolf shrinkage"); ax[0].tick_params(labelsize=7)
        ax[1].bar(sess, [mv[s].get("cov_cond") or 0 for s in sess])
        ax[1].set_yscale("log"); ax[1].set_title("noise cov. condition number")
        ax[1].tick_params(labelsize=7)
        vals = [mv[s].get("whitened_cov_diag_mean") for s in sess
                if mv[s].get("whitened_cov_diag_mean") is not None]
        if vals:
            ax[2].bar(sess[:len(vals)], vals)
            ax[2].axhline(1.0, color="k", ls="--", lw=0.8)
            ax[2].set_title("mean diag(cov) after MVNN (≈1 is good)")
            ax[2].tick_params(labelsize=7)
        fig.tight_layout()
        p = out_dir / "mvnn_diagnostics.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(str(p))

    # 4) final tensor shapes / channels summary
    lines = [f"variant: {meta.get('variant')}  hash: {meta.get('config_hash')}",
             f"subject: {meta.get('subject')}",
             f"channels ({len(meta.get('channels', []) or [])}): "
             f"{', '.join(meta.get('channels', []) or [])}"]
    for src in ("training", "test"):
        if src in meta:
            lines.append(f"{src}: shape={meta[src].get('shape')} "
                         f"n_reps={meta[src].get('n_repetitions')}")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    saved.append(str(out_dir / "summary.txt"))
    logger.info("QC figures written to %s", out_dir)
    return saved
