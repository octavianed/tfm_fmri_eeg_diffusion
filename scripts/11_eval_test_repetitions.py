"""Test-repetitions curve: retrieval as a function of R (spec §8.8, §16).

    python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_eeg_raw_baseline.yaml \
        --checkpoint outputs/exp01_eeg_raw_baseline/checkpoints/best.pt

Answers "how much of the benchmark result depends on massive repetition
averaging?". **No retraining is needed** — the trained decoder is evaluated on
test inputs built by averaging R of the 80 available repetitions:

    R ∈ {1, 2, 4, 8, 20, 40, 80}, ``n_draws`` reproducible subsets for R < 80
    (R = 80 is deterministic — it uses every repetition).

MVNN was already applied per repetition during preprocessing, so averaging here
respects the required order (whiten → average). Writes a tidy CSV with
mean/std per R plus a figure with uncertainty bands.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_datamodule  # noqa: E402
from src.evaluation import compute_retrieval_metrics  # noqa: E402
from src.features import load_split_features  # noqa: E402
from src.models import build_model_from_checkpoint  # noqa: E402
from src.utils import (ExtendOverrides, get_device,  # noqa: E402
                       get_experiment_paths, get_logger, load_config, save_json)

logger = get_logger("test_repetitions")


def _predict(model, x: np.ndarray, subject, device, batch_size: int = 256):
    import torch
    uses_adapter = getattr(model, "adapters", None) is not None
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            b = torch.from_numpy(x[i:i + batch_size]).float().to(device)
            r = model(b, subject=subject if uses_adapter else None)
            out.append(r["clip"].cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="decoder checkpoint (default: <experiment>/checkpoints/best.pt)")
    ap.add_argument("--values", nargs="*", type=int, default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = get_device(cfg.get("runtime.device", "auto"))
    paths = get_experiment_paths(cfg, ensure=True)
    ckpt = args.checkpoint or str(paths.best_ckpt)
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {ckpt}")

    values = args.values or list(cfg.get("evaluation.test_repetitions.values",
                                         [1, 2, 4, 8, 20, 40, 80]))
    n_draws = int(args.n_draws if args.n_draws is not None
                  else cfg.get("evaluation.test_repetitions.n_draws", 20))
    seed = int(cfg.get("evaluation.test_repetitions.seed",
                       cfg.get("project.seed", 42)))

    dm = build_datamodule(cfg).prepare()
    model, _ = build_model_from_checkpoint(cfg, ckpt, device, dm.voxel_counts)

    rows = []
    for subject in dm.subjects:
        reader = dm.subject_reader(subject)
        frame = dm.subject_split_frame(subject, args.split)
        target = load_split_features(cfg, subject, args.split, "clip")
        if target is None:
            raise FileNotFoundError(
                f"No precomputed CLIP features for {subject}/{args.split}; run "
                f"scripts/01_precompute_clip.py first.")
        img_idx = frame["img_index"].to_numpy()
        sources = frame["source"].astype(str).to_numpy()
        n_avail = reader.n_reps(str(sources[0]))
        norm = dm.normalizer(subject)
        logger.info("%s: %d test images, %d repetitions available",
                    subject, len(frame), n_avail)

        for R in values:
            R_eff = min(int(R), n_avail)
            draws = 1 if R_eff >= n_avail else n_draws
            per_draw = []
            for d in range(draws):
                rng = np.random.default_rng(seed + 1000 * R_eff + d)
                x = np.empty((len(frame), *dm.signal_shape[subject]), np.float32)
                for i, (ii, src) in enumerate(zip(img_idx, sources)):
                    sel = (np.arange(n_avail) if R_eff >= n_avail
                           else rng.choice(n_avail, size=R_eff, replace=False))
                    trials = np.stack([reader.get_trial(str(src), int(ii), int(r))
                                       for r in sel])
                    sig = trials.mean(axis=0)
                    x[i] = norm.transform(sig) if norm is not None else sig
                pred = _predict(model, x, subject, device,
                                int(cfg.get("evaluation.infer_batch_size", 256)))
                m, _ = compute_retrieval_metrics(pred, target, ks=(1, 5, 10))
                per_draw.append(m)
            agg = {"subject_id": subject, "split": args.split, "R": R_eff,
                   "n_draws": draws, "n_candidates": per_draw[0].get("n_candidates")}
            for key in ("top1", "top5", "top10", "mean_rank", "mean_cosine"):
                vals = [p[key] for p in per_draw if key in p]
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
            rows.append(agg)
            logger.info("  R=%2d (%d draws): top1=%.4f±%.4f top5=%.4f±%.4f",
                        R_eff, draws, agg["top1_mean"], agg["top1_std"],
                        agg["top5_mean"], agg["top5_std"])

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = paths.metrics / "test_repetitions_curve.csv"
    df.to_csv(out_csv, index=False)
    save_json({"values": values, "n_draws": n_draws, "seed": seed,
               "checkpoint": ckpt, "split": args.split},
              paths.metrics / "test_repetitions_params.json")
    logger.info("Saved %s", out_csv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for subject, g in df.groupby("subject_id"):
            g = g.sort_values("R")
            for key, style in (("top1", "-o"), ("top5", "--s")):
                ax.errorbar(g["R"], g[f"{key}_mean"], yerr=g[f"{key}_std"],
                            fmt=style, capsize=3, label=f"{subject} {key}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("test repetitions averaged (R)")
        ax.set_ylabel("retrieval accuracy")
        ax.set_title("Decoding vs number of averaged test repetitions")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        p = paths.figures / "test_repetitions_curve.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        logger.info("Saved %s", p)
    except Exception as exc:  # pragma: no cover
        logger.warning("Figure skipped: %s", exc)


if __name__ == "__main__":
    main()
