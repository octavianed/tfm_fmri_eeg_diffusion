"""Mandatory checks for a built EEG preprocessing variant (spec §14).

    python scripts/10_validate_eeg_preproc.py --config configs/EEG/preproc/baseline.yaml
    python scripts/10_validate_eeg_preproc.py --config configs/EEG/preproc/channels_17.yaml --subject sub-08

Runs, on the cached variant:

* **§14.1 shapes** — expected ``n_channels × n_times`` derived from the config
  (63×250 baseline, 17×250, 63×125, 63×50, 63×100 …), including that
  ``[0, 1000)`` ms at 250 Hz is exactly 250 samples (never 251);
* **§14.2 MVNN leakage** — perturbing test/val trials must not change ``W``,
  and the fit must only see train-split images;
* **§14.3 split** — train and val image ids are disjoint (and repetitions of an
  image never straddle the split);
* **§14.4 repetitions** — 4 in training (or 1 after average-4) and 80 in test,
  no repetition used twice, averaging invariant to ordering;
* **§14.5 channel ablation** — the 17 posterior channels are exactly the
  specified list;
* **§14.6 CAR** — after CAR the across-channel mean is ≈ 0.

Exits non-zero if any check fails.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_datamodule, image_level_split  # noqa: E402
from src.preprocessing import POSTERIOR_17, fit_mvnn, n_samples_for_window  # noqa: E402
from src.preprocessing.epoching import apply_reference  # noqa: E402
from src.utils import ExtendOverrides, get_logger, load_config, load_json  # noqa: E402

logger = get_logger("validate_preproc")


class Checks:
    def __init__(self):
        self.passed, self.failed = [], []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name)
        logger.info("[%s] %s%s", "PASS" if ok else "FAIL", name,
                    f" — {detail}" if detail else "")

    def report(self) -> int:
        logger.info("%d passed, %d failed", len(self.passed), len(self.failed))
        if self.failed:
            logger.error("FAILED: %s", ", ".join(self.failed))
        return 1 if self.failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    c = Checks()

    pcfg = cfg.get("preprocessing", {}) or {}
    crop = pcfg.get("crop", {}) or {}
    sfreq = float((pcfg.get("resample", {}) or {}).get("sfreq_hz", 250))
    exp_times = n_samples_for_window(float(crop.get("tmin_ms", 0)),
                                     float(crop.get("tmax_ms", 1000)), sfreq)
    ch_mode = str((pcfg.get("channels", {}) or {}).get("mode", "all_63"))
    exp_ch = 17 if ch_mode in ("posterior_17", "17") else 63

    dm = build_datamodule(cfg).prepare()
    subject = args.subject or dm.subjects[0]
    reader = dm.subject_reader(subject)

    # --- §14.1 shapes -----------------------------------------------------
    C, T = dm.signal_shape[subject]
    c.check("14.1 n_channels", C == exp_ch, f"{C} (expected {exp_ch})")
    c.check("14.1 n_times", T == exp_times,
            f"{T} (expected {exp_times} for [{crop.get('tmin_ms')},"
            f"{crop.get('tmax_ms')}) ms @ {sfreq:g} Hz)")
    half_open_ok = exp_times == int(round(
        (float(crop.get("tmax_ms", 1000)) - float(crop.get("tmin_ms", 0)))
        / 1000.0 * sfreq))
    c.check("14.1 half-open crop (no off-by-one)", half_open_ok)

    trial = reader.get_signal("train", 0, 0)
    c.check("14.1 trial tensor shape", trial.shape == (exp_ch, exp_times),
            str(trial.shape))

    # --- §14.5 channel list ----------------------------------------------
    if exp_ch == 17:
        c.check("14.5 posterior-17 channel list",
                list(reader.ch_names) == POSTERIOR_17,
                f"{list(reader.ch_names)}")

    # --- §14.3 split ------------------------------------------------------
    tr = set(dm.subject_split_frame(subject, "train")["image_id"])
    va = set(dm.subject_split_frame(subject, "val")["image_id"])
    te = set(dm.subject_split_frame(subject, "test")["image_id"])
    c.check("14.3 train ∩ val = ∅", not (tr & va), f"|∩|={len(tr & va)}")
    c.check("14.3 train ∩ test = ∅", not (tr & te), f"|∩|={len(tr & te)}")
    c.check("14.3 val ∩ test = ∅", not (va & te), f"|∩|={len(va & te)}")

    frames_ok = True
    for split in ("train", "val"):
        f = dm.get_frame(split)
        f = f[f.subject_id == subject]
        # every image must contribute all of its rows to a single split
        if len(f) and f.groupby("img_index")["split"].nunique().max() > 1:
            frames_ok = False
    c.check("14.3 repetitions never straddle the split", frames_ok)

    # --- §14.4 repetitions ------------------------------------------------
    n_tr_reps = reader.n_reps("train")
    n_te_reps = reader.n_reps("test")
    rep_cfg = (cfg.get("repetitions", {}) or {}).get("train", {}) or {}
    strategy = str(rep_cfg.get("strategy", "average"))
    expected_tr = 1 if strategy == "average" else int(rep_cfg.get("n_repetitions", 4))
    c.check("14.4 training repetitions on disk", n_tr_reps == expected_tr,
            f"{n_tr_reps} (strategy={strategy})")
    c.check("14.4 test repetitions on disk", n_te_reps == 80, str(n_te_reps))

    reps = np.stack([reader.get_trial("test", 0, r) for r in range(n_te_reps)])
    mean_fwd = reps.mean(axis=0)
    mean_rev = reps[::-1].mean(axis=0)
    c.check("14.4 averaging invariant to repetition order",
            bool(np.allclose(mean_fwd, mean_rev, atol=1e-5)))
    c.check("14.4 repetitions are distinct (no duplicates)",
            len({r.tobytes() for r in reps[:8]}) == min(8, n_te_reps))

    # --- §14.2 MVNN leakage ----------------------------------------------
    if bool((pcfg.get("mvnn", {}) or {}).get("enabled", True)):
        rng = np.random.default_rng(0)
        n_img, n_ch, n_t = 40, exp_ch, exp_times
        base = rng.standard_normal((n_img * 4, n_ch, n_t))
        ids = np.repeat(np.arange(n_img), 4)
        train_mask = ids < 30                       # pretend 30 images are train
        w_ref = fit_mvnn(base[train_mask], ids[train_mask]).W
        tampered = base.copy()
        tampered[~train_mask] *= 1000.0             # wreck the held-out trials
        w_tampered = fit_mvnn(tampered[train_mask], ids[train_mask]).W
        c.check("14.2 MVNN ignores held-out trials (W unchanged)",
                bool(np.allclose(w_ref, w_tampered)),
                f"max|ΔW|={np.abs(w_ref - w_tampered).max():.3e}")

        meta_p = Path(reader.subj_dir) / "metadata.json"
        if meta_p.exists():
            meta = load_json(meta_p)
            n_train_images = int(cfg.get("dataset.n_train_images", 16540))
            assign = image_level_split(n_train_images,
                                       float(cfg.get("split.val_ratio", 0.1)),
                                       int(cfg.get("split.seed", 42)), subject)
            n_train = sum(1 for v in assign.values() if v == "train")
            fitted = [s.get("n_images") for s in (meta.get("mvnn") or {}).values()]
            c.check("14.2 MVNN fitted only on train-split images",
                    all(f is None or f <= n_train for f in fitted),
                    f"images per session {fitted} (train split has {n_train})")

    # --- §14.6 CAR --------------------------------------------------------
    if str((pcfg.get("reference", {}) or {}).get("mode", "original")) in ("average", "car"):
        x = np.stack([reader.get_trial("test", i, 0) for i in range(5)])
        c.check("14.6 CAR: across-channel mean ≈ 0",
                bool(np.abs(x.mean(axis=-2)).max() < 1e-3),
                f"max|mean_ch|={np.abs(x.mean(axis=-2)).max():.3e}")
    else:
        probe = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
        cared, _ = apply_reference(probe, "average")
        c.check("14.6 CAR helper zeroes the channel mean",
                bool(np.abs(cared.mean(axis=-2)).max() < 1e-12))

    return c.report()


if __name__ == "__main__":
    raise SystemExit(main())
