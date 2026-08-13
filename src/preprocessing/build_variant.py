"""Build one preprocessing variant for one or more subjects (spec §9).

Order of operations, per subject and session (spec §9, branching from the raw
signal as early as possible so variants never chain onto each other):

    1. load raw EEG + events + metadata      6. epoch [-200, 1000) ms
    2. non-destructive QC                    7. baseline of the variant
    3. select the variant's channels         8. resample to the variant's rate
    4. filter the continuous signal          9. crop the variant's window
    5. apply the variant's reference        10. split (image level, shared rule)
    11. fit MVNN on TRAIN images only (per subject×session)
    12. apply MVNN to every individual repetition
    13/15. aggregate repetitions (train policy / test keeps all)
    16. save tensors + metadata + config + hash

The output mirrors the official derivatives — a dict with
``preprocessed_eeg_data`` ``[n_images, n_reps, n_channels, n_times]``,
``ch_names`` and ``times`` — so the existing EEG datamodule reads it unchanged.

**Merge rule (critical):** in THINGS-EEG2 each training image appears in exactly
2 of the 4 sessions (2 repetitions each). Repetitions are therefore gathered
**by image code**, never by concatenating sessions along the repetition axis.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.eeg_split import image_level_split
from ..utils import get_logger, save_json
from ..utils.checkpointing import collect_library_versions
from .epoching import apply_reference, epoch_continuous, n_samples_for_window
from .mvnn import apply_mvnn, fit_mvnn, whitened_covariance
from .things_raw_loader import (EXPECTED_TOTAL_REPS, discover_raw_sessions,
                                load_raw_session, quality_report,
                                resolve_raw_subjects, validate_channel_consistency)
from .filters import filter_continuous

logger = get_logger("build_variant")

#: The 17 posterior channels of the channel ablation (spec §8.2, §14.5).
POSTERIOR_17 = ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
                "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

_SOURCES = ("training", "test")


def preproc_config_hash(cfg) -> str:
    """Stable 8-char hash of the ``preprocessing``/``repetitions``/``split`` blocks.

    Used to detect a cache built with different parameters (spec §12, §13).
    """
    block = {"preprocessing": _plain(cfg.get("preprocessing", {})),
             "repetitions": _plain(cfg.get("repetitions", {})),
             "split": _plain(cfg.get("split", {})),
             "loading": _plain(cfg.get("loading", {}))}
    raw = json.dumps(block, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def _plain(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def resolve_channel_selection(ch_names: Sequence[str], mode: str) -> List[int]:
    """Indices of the channels kept by the variant (spec §8.2).

    ``'all_63'`` keeps everything; ``'posterior_17'`` selects exactly
    :data:`POSTERIOR_17` **from the raw signal**, never from the official
    17-channel derivative (which would confound channels with sampling rate).
    """
    names = [str(c) for c in ch_names]
    mode = str(mode or "all_63").lower()
    if mode in ("all_63", "all", "63"):
        return list(range(len(names)))
    if mode in ("posterior_17", "17"):
        missing = [c for c in POSTERIOR_17 if c not in names]
        if missing:
            raise ValueError(f"Channels {missing} not found in the raw montage")
        return [names.index(c) for c in POSTERIOR_17]
    raise ValueError(f"Unknown channels.mode: {mode}")


def _times_ms(crop_tmin: float, n_times: int, sfreq: float) -> np.ndarray:
    """Time vector (seconds) of the final window, like the official files."""
    return (crop_tmin / 1000.0 + np.arange(n_times) / float(sfreq)).astype(np.float64)


def _epoch_session(session, pcfg, backend: str) -> tuple:
    """Channels → filter → reference → epoch/baseline/resample/crop (steps 3-9)."""
    ch_idx = resolve_channel_selection(session.ch_names,
                                       pcfg.get("channels", {}).get("mode", "all_63"))
    ch_names = [session.ch_names[i] for i in ch_idx]
    data = np.ascontiguousarray(session.data[ch_idx])

    fcfg = pcfg.get("filter", {}) or {}
    if fcfg.get("enabled", True):
        data, filt_desc = filter_continuous(
            data, session.sfreq, fcfg.get("l_freq_hz"), fcfg.get("h_freq_hz"),
            notch=fcfg.get("notch_hz"), backend=backend)
    else:
        filt_desc = {"enabled": False}

    data, ref_desc = apply_reference(data, pcfg.get("reference", {}).get("mode", "original"))

    ecfg = pcfg.get("epoch", {}) or {}
    bcfg = pcfg.get("baseline", {}) or {}
    rcfg = pcfg.get("resample", {}) or {}
    ccfg = pcfg.get("crop", {}) or {}
    baseline = ((float(bcfg.get("tmin_ms", -200)), float(bcfg.get("tmax_ms", 0)))
                if bcfg.get("enabled", True) else None)
    epochs, ep_desc = epoch_continuous(
        data, session.events.onsets, session.sfreq,
        tmin_ms=float(ecfg.get("tmin_ms", -200)),
        tmax_ms=float(ecfg.get("tmax_ms", 1000)),
        baseline=baseline,
        target_sfreq=float(rcfg.get("sfreq_hz", session.sfreq)),
        crop=(float(ccfg.get("tmin_ms", 0)), float(ccfg.get("tmax_ms", 1000))),
        backend=backend)
    # Subset the image codes with the SAME mask the epoching used: a dropped
    # epoch is not necessarily the last one, so slicing would misalign them.
    kept = ep_desc.pop("kept_mask")
    codes = np.asarray(session.events.codes)[kept]
    if len(codes) != len(epochs):
        raise AssertionError(f"epoch/code mismatch: {len(epochs)} vs {len(codes)}")
    desc = {"filter": filt_desc, "reference": ref_desc, "epoching": ep_desc,
            "channels": ch_names}
    return epochs, codes, ch_names, desc


def build_variant(cfg, subjects=None, force: bool = False,
                  out_root: Optional[Path] = None, qc: bool = True) -> dict:
    """Build the configured variant and write it to the variant cache.

    Returns a summary dict; writes, per subject,
    ``<out_root>/<variant>/<subject>/preprocessed_eeg_{training,test}.npy`` plus
    ``metadata.json`` (and QC figures under ``qc/``).
    """
    from ..utils.paths import eeg_preproc_dir

    pcfg = cfg.get("preprocessing", {}) or {}
    repcfg = cfg.get("repetitions", {}) or {}
    scfg = cfg.get("split", {}) or {}
    root_dir = cfg.get("dataset.root_dir")
    variant = str(cfg.get("dataset.preproc_variant", "baseline"))
    backend = str(pcfg.get("filter", {}).get("backend", "auto"))
    # Resolve the filter backend up front: a missing MNE must fail before we
    # spend ~10 s loading a 3.2 GB raw session (and it pins the backend that
    # every session of this variant will use).
    from .filters import filter_backend
    backend = filter_backend(backend)
    logger.info("Filter backend: %s", backend)
    cfg_hash = preproc_config_hash(cfg)

    # The split must match the one the datamodule will use, otherwise MVNN would
    # be fitted on a different 'train' set than the experiment trains on. The
    # preprocessing spec names the keys split.*, the datamodule reads dataset.*;
    # refuse to run if both are present and disagree.
    seed = int(scfg.get("seed", cfg.get("dataset.split_seed",
                                        cfg.get("project.seed", 42))))
    val_ratio = float(scfg.get("val_ratio", cfg.get("dataset.val_ratio", 0.1)))
    ds_seed = cfg.get("dataset.split_seed", None)
    ds_ratio = cfg.get("dataset.val_ratio", None)
    if ds_seed is not None and int(ds_seed) != seed:
        raise ValueError(f"split.seed={seed} disagrees with dataset.split_seed="
                         f"{ds_seed}; the datamodule would use a different split.")
    if ds_ratio is not None and abs(float(ds_ratio) - val_ratio) > 1e-12:
        raise ValueError(f"split.val_ratio={val_ratio} disagrees with "
                         f"dataset.val_ratio={ds_ratio}; the datamodule would use "
                         f"a different split.")
    dtype_out = str(cfg.get("runtime.dtype_output", "float32"))

    sel = subjects or cfg.get("dataset.subject_selection", "all")
    subs = resolve_raw_subjects(root_dir, sel)
    base_dir = Path(out_root) if out_root else eeg_preproc_dir(cfg, variant)

    summary = {"variant": variant, "config_hash": cfg_hash, "subjects": {},
               "out_dir": str(base_dir)}
    for subject in subs:
        t0 = time.time()
        sub_dir = base_dir / subject
        done = all((sub_dir / f"preprocessed_eeg_{s}.npy").exists() for s in _SOURCES)
        if done and not force:
            # Spec §13: never reuse a cache whose parameters have changed.
            cached_hash = None
            meta_path = sub_dir / "metadata.json"
            if meta_path.exists():
                try:
                    cached_hash = json.loads(
                        meta_path.read_text(encoding="utf-8")).get("config_hash")
                except Exception:
                    cached_hash = None
            if cached_hash is not None and cached_hash != cfg_hash:
                raise ValueError(
                    f"{subject}: the cached variant '{variant}' was built with a "
                    f"different preprocessing config (hash {cached_hash}, current "
                    f"{cfg_hash}). Reusing it would silently mix parameters. "
                    f"Rebuild it with --force, or give this configuration its own "
                    f"dataset.preproc_variant name.")
            logger.info("%s: variant '%s' already built (hash %s) — use --force "
                        "to rebuild", subject, variant, cached_hash or "unknown")
            summary["subjects"][subject] = {"status": "cached", "dir": str(sub_dir),
                                            "config_hash": cached_hash}
            continue
        sub_dir.mkdir(parents=True, exist_ok=True)
        info = _build_subject(cfg, subject, root_dir, pcfg, repcfg, backend, seed,
                              val_ratio, dtype_out, sub_dir, cfg_hash, variant, qc)
        info["seconds"] = round(time.time() - t0, 1)
        summary["subjects"][subject] = info
        logger.info("%s: variant '%s' built in %.1f s -> %s", subject, variant,
                    info["seconds"], sub_dir)
    return summary


def _build_subject(cfg, subject, root_dir, pcfg, repcfg, backend, seed, val_ratio,
                   dtype_out, sub_dir, cfg_hash, variant, qc) -> dict:
    sessions = discover_raw_sessions(root_dir, subject)
    logger.info("%s: %d session(s) %s", subject, len(sessions), sessions)

    per_source: Dict[str, Dict[int, List[np.ndarray]]] = {s: defaultdict(list)
                                                          for s in _SOURCES}
    ch_names: Optional[List[str]] = None
    sfreq_out = float((pcfg.get("resample", {}) or {}).get("sfreq_hz", 1000))
    descs: Dict[str, dict] = {}
    qc_reports: List[dict] = []
    mvnn_stats: Dict[str, dict] = {}
    loaded_for_check = []

    mv_cfg = pcfg.get("mvnn", {}) or {}
    mvnn_enabled = bool(mv_cfg.get("enabled", True))

    for ses in sessions:
        # --- training recording: needed first (it defines this session's MVNN)
        tr = load_raw_session(root_dir, subject, ses, "training", seed=seed)
        if qc:
            qc_reports.append(quality_report(tr))
        loaded_for_check.append(tr)
        tr_ep, tr_codes, names, d = _epoch_session(tr, pcfg, backend)
        ch_names = ch_names or names
        descs.setdefault("training", d)
        del tr

        mv = None
        if mvnn_enabled:
            # Fit ONLY on training-split images of this session (spec §3.1/§5.10).
            n_train_images = int(cfg.get("dataset.n_train_images", 16540))
            assign = image_level_split(n_train_images, val_ratio, seed, subject)
            # codes are 1-based image indices
            is_train = np.array([assign.get(int(c) - 1, "train") == "train"
                                 for c in tr_codes])
            mv = fit_mvnn(tr_ep[is_train], tr_codes[is_train],
                          covariance=str(mv_cfg.get("covariance", "ledoit_wolf")),
                          eigenvalue_floor=float(mv_cfg.get("eigenvalue_floor", 1e-8)),
                          scope=str(mv_cfg.get("scope", "subject_session")))
            mvnn_stats[ses] = {"shrinkage": mv.shrinkage, "n_trials": mv.n_trials_used,
                               "n_images": mv.n_images_used, **(mv.stats or {})}
            if qc:
                mvnn_stats[ses]["whitened_cov_diag_mean"] = float(np.mean(np.diag(
                    whitened_covariance(apply_mvnn(tr_ep[is_train][:200], mv),
                                        tr_codes[is_train][:200]))))
            tr_ep = apply_mvnn(tr_ep, mv)          # applied to EVERY repetition

        for code, ep in zip(tr_codes, tr_ep):
            per_source["training"][int(code)].append(ep)
        del tr_ep, tr_codes

        # --- test recording of the same session, whitened with the same W
        te = load_raw_session(root_dir, subject, ses, "test", seed=seed)
        if qc:
            qc_reports.append(quality_report(te))
        te_ep, te_codes, _, d = _epoch_session(te, pcfg, backend)
        descs.setdefault("test", d)
        del te
        if mv is not None:
            te_ep = apply_mvnn(te_ep, mv)
        for code, ep in zip(te_codes, te_ep):
            per_source["test"][int(code)].append(ep)
        del te_ep, te_codes

    validate_channel_consistency(loaded_for_check)
    del loaded_for_check

    out_info = {"status": "built", "dir": str(sub_dir), "sessions": sessions,
                "channels": ch_names, "sfreq": sfreq_out, "mvnn": mvnn_stats,
                "config_hash": cfg_hash}
    for source in _SOURCES:
        arr, n_reps = _assemble(per_source[source], source, repcfg, dtype_out)
        crop = pcfg.get("crop", {}) or {}
        times = _times_ms(float(crop.get("tmin_ms", 0)), arr.shape[-1], sfreq_out)
        np.save(sub_dir / f"preprocessed_eeg_{source}.npy",
                {"preprocessed_eeg_data": arr, "ch_names": np.array(ch_names),
                 "times": times}, allow_pickle=True)
        out_info[source] = {"shape": list(arr.shape), "n_repetitions": int(n_reps)}
        logger.info("%s %s: saved %s", subject, source, arr.shape)
        per_source[source] = None
        del arr

    # Reproducibility record (spec §12): channels/order, final sampling rate,
    # windows, filter, reference, MVNN stats, versions, QC.
    meta = {"variant": variant, "config_hash": cfg_hash, "subject": subject,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sessions": sessions, "channels": ch_names,
            "n_channels": len(ch_names or []), "sfreq": sfreq_out,
            "preprocessing": _plain(pcfg), "repetitions": _plain(repcfg),
            "split": {"val_ratio": val_ratio, "seed": seed, "level": "image"},
            "stage_descriptions": descs, "mvnn": mvnn_stats,
            "library_versions": collect_library_versions(),
            "qc": qc_reports, **{k: v for k, v in out_info.items()
                                 if k in _SOURCES}}
    save_json(meta, sub_dir / "metadata.json")
    if qc:
        try:
            from .qc import save_qc_figures
            save_qc_figures(meta, sub_dir / "qc")
        except Exception as exc:  # pragma: no cover - figures are optional
            logger.warning("QC figures skipped: %s", exc)
    return out_info


def _assemble(by_code: Dict[int, List[np.ndarray]], source: str, repcfg: dict,
              dtype_out: str) -> tuple:
    """Group repetitions **by image code** and apply the repetition policy.

    Training: ``average`` (baseline, 4 reps → 1 observation) or ``independent``
    (keeps the 4 reps). Test: keeps every repetition so the R-curve (spec §16)
    can subsample them later.
    """
    codes = sorted(by_code)
    if not codes:
        raise ValueError(f"No {source} epochs were collected")
    expected = EXPECTED_TOTAL_REPS.get(source)
    counts = {c: len(by_code[c]) for c in codes}
    n_reps = int(min(counts.values()))
    bad = {c: n for c, n in counts.items() if n != expected}
    policy = str((repcfg.get(_policy_key(source), {}) or {}).get("on_missing", "fail"))
    if bad and expected is not None:
        msg = (f"{len(bad)} {source} image(s) do not have exactly {expected} "
               f"repetitions (e.g. {list(bad.items())[:5]})")
        if policy == "fail":
            raise ValueError(msg + " — set repetitions.<split>.on_missing=truncate "
                             "to proceed by truncating to the common minimum.")
        logger.warning(msg + f" — truncating every image to {n_reps} repetitions.")

    strategy = str((repcfg.get(_policy_key(source), {}) or {}).get("strategy",
                                                                  "average"))
    keep = n_reps if expected is None else min(n_reps, expected)
    first = by_code[codes[0]][0]
    n_ch, n_times = first.shape

    if source == "training" and strategy == "average":
        out = np.empty((len(codes), 1, n_ch, n_times), dtype=np.float32)
        for i, c in enumerate(codes):
            out[i, 0] = np.mean(np.stack(by_code[c][:keep], 0), axis=0)
        used = keep
    else:
        out = np.empty((len(codes), keep, n_ch, n_times), dtype=np.float32)
        for i, c in enumerate(codes):
            out[i] = np.stack(by_code[c][:keep], 0)
        used = keep
    return out.astype(np.dtype(dtype_out), copy=False), used


def _policy_key(source: str) -> str:
    return "train" if source == "training" else "test"
