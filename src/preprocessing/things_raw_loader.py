"""Loading layer for the THINGS-EEG2 **original release** raw EEG.

Verified on-disk structure (discovered, not assumed — spec §2.2, §5.1)::

    <root>/raw-eeg/<sub>/ses-0X/raw_eeg_training.npy
    <root>/raw-eeg/<sub>/ses-0X/raw_eeg_test.npy

Each file is a pickled dict with:

===================  =========================================================
``raw_eeg_data``     ``(64, n_samples)`` float64 — 63 EEG channels + ``stim``
``ch_names``         64 names, the last one being ``stim``
``ch_types``         63 ``eeg`` + 1 ``stim``
``sfreq``            1000 (Hz)
``highpass``         0.01 — hardware filter already applied at acquisition
``lowpass``          100  — idem
===================  =========================================================

**Events** live in the ``stim`` channel as single non-zero samples whose value is
the 1-based *image index*; ``99999`` marks target/catch trials, which are dropped
(spec §2.2, §5.1.7).

Observed trial structure (sub-08), which the merge logic must respect:

* **test**: 200 image codes × 20 repetitions per session (a few codes carry an
  extra one) × 4 sessions → 80 repetitions per image.
* **training**: each session covers **8270 images × 2 repetitions**, and every
  image appears in **exactly 2 of the 4 sessions** with an interleaved layout
  (``ses1∩ses2 = ∅``, ``ses3∩ses4 = ∅``, ``ses1∩ses3 = 4110`` …). The union of
  the four sessions is exactly ``1..16540`` → 4 repetitions per image. Therefore
  repetitions must be gathered **by image code**, never by blindly concatenating
  sessions along the repetition axis.

Extra occurrences beyond ``max_rep`` are dropped with a seeded random choice,
mirroring the official preprocessing code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import get_logger

logger = get_logger("raw_loader")

STIM_CHANNEL = "stim"
TARGET_CODE = 99999
#: repetitions kept per image *per session* (mirrors the official pipeline)
MAX_REP_PER_SESSION = {"training": 2, "test": 20}
#: total repetitions expected per image across all sessions
EXPECTED_TOTAL_REPS = {"training": 4, "test": 80}


@dataclass
class SessionEvents:
    """Events of one recording: onsets (samples) and 1-based image codes."""

    onsets: np.ndarray          # int64 [n_events]
    codes: np.ndarray           # int64 [n_events]
    n_targets_dropped: int = 0
    n_extra_dropped: int = 0

    def __len__(self) -> int:
        return len(self.onsets)

    @property
    def unique_codes(self) -> np.ndarray:
        return np.unique(self.codes)


@dataclass
class RawSession:
    """One loaded recording (subject × session × split source)."""

    subject: str
    session: str
    source: str                 # 'training' | 'test'
    data: np.ndarray            # float32 [n_eeg_channels, n_samples]
    ch_names: List[str]         # EEG channel names (no 'stim')
    sfreq: float
    events: SessionEvents
    info: Dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def raw_root(root_dir) -> Path:
    return Path(root_dir) / "raw-eeg"


def discover_raw_subjects(root_dir) -> List[str]:
    """List ``sub-NN`` folders available under ``<root>/raw-eeg``."""
    base = raw_root(root_dir)
    if not base.exists():
        raise FileNotFoundError(f"THINGS-EEG2 raw-eeg folder not found: {base}")
    subs = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    return sorted(subs)


def discover_raw_sessions(root_dir, subject: str) -> List[str]:
    """List the ``ses-0X`` folders of a subject (sorted)."""
    d = raw_root(root_dir) / subject
    if not d.exists():
        raise FileNotFoundError(f"Raw subject folder not found: {d}")
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith("ses-"))


def raw_session_path(root_dir, subject: str, session: str, source: str) -> Path:
    """Path of a raw recording; ``source`` is 'training' or 'test'."""
    fname = f"raw_eeg_{source}.npy"
    return raw_root(root_dir) / subject / session / fname


def resolve_raw_subjects(root_dir, selection) -> List[str]:
    available = discover_raw_subjects(root_dir)
    if selection == "all":
        return available
    requested = [selection] if isinstance(selection, str) else list(selection)
    missing = [s for s in requested if s not in available]
    if missing:
        raise ValueError(f"Raw EEG not available for {missing} under "
                         f"{raw_root(root_dir)} (available: {available}).")
    return sorted(set(requested))


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
def extract_events(stim: np.ndarray, source: str, seed: int = 42,
                   max_rep: Optional[int] = None,
                   selection: str = "seeded_random") -> SessionEvents:
    """Build :class:`SessionEvents` from a ``stim`` channel.

    Drops ``99999`` (target/catch) events and caps each image code to
    ``max_rep`` occurrences within this session. ``selection`` is
    ``'seeded_random'`` (mirrors the official pipeline) or ``'first'``.
    """
    onsets = np.flatnonzero(np.asarray(stim) != 0).astype(np.int64)
    codes = np.asarray(stim)[onsets].astype(np.int64)

    keep = codes != TARGET_CODE
    n_targets = int((~keep).sum())
    onsets, codes = onsets[keep], codes[keep]

    if max_rep is None:
        max_rep = MAX_REP_PER_SESSION.get(source)

    n_extra = 0
    if max_rep is not None:
        rng = np.random.default_rng(seed)
        sel = []
        for code in np.unique(codes):
            idx = np.flatnonzero(codes == code)
            if len(idx) > max_rep:
                n_extra += len(idx) - max_rep
                if selection == "first":
                    idx = idx[:max_rep]
                else:  # seeded_random
                    idx = np.sort(rng.choice(idx, size=max_rep, replace=False))
            sel.append(idx)
        order = np.sort(np.concatenate(sel)) if sel else np.array([], dtype=np.int64)
        onsets, codes = onsets[order], codes[order]

    return SessionEvents(onsets=onsets, codes=codes,
                         n_targets_dropped=n_targets, n_extra_dropped=n_extra)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw_session(root_dir, subject: str, session: str, source: str,
                     seed: int = 42, selection: str = "seeded_random",
                     max_rep: Optional[int] = None) -> RawSession:
    """Load one raw recording, dropping the ``stim`` channel from the data.

    Returns EEG data as ``float32 [63, n_samples]`` plus the parsed events.
    """
    path = raw_session_path(root_dir, subject, session, source)
    if not path.exists():
        raise FileNotFoundError(f"Raw recording not found: {path}")
    d = np.load(path, allow_pickle=True).item()
    for key in ("raw_eeg_data", "ch_names", "sfreq"):
        if key not in d:
            raise KeyError(f"{path} is missing the '{key}' key "
                           f"(found: {sorted(d.keys())})")

    raw = np.asarray(d["raw_eeg_data"])
    ch_names = [str(c) for c in np.asarray(d["ch_names"]).tolist()]
    sfreq = float(np.asarray(d["sfreq"]))

    if STIM_CHANNEL not in ch_names:
        raise ValueError(f"No '{STIM_CHANNEL}' channel in {path}: {ch_names}")
    stim_idx = ch_names.index(STIM_CHANNEL)
    events = extract_events(raw[stim_idx], source, seed=seed, max_rep=max_rep,
                            selection=selection)

    eeg_idx = [i for i, c in enumerate(ch_names) if c != STIM_CHANNEL]
    data = np.ascontiguousarray(raw[eeg_idx], dtype=np.float32)
    eeg_names = [ch_names[i] for i in eeg_idx]

    info = {"highpass": float(d.get("highpass", np.nan)),
            "lowpass": float(d.get("lowpass", np.nan)),
            "path": str(path), "n_events": len(events),
            "n_targets_dropped": events.n_targets_dropped,
            "n_extra_dropped": events.n_extra_dropped}
    logger.info("%s %s %s: data %s @%.0f Hz | %d events (dropped %d targets, "
                "%d extra reps)", subject, session, source, data.shape, sfreq,
                len(events), events.n_targets_dropped, events.n_extra_dropped)
    return RawSession(subject=subject, session=session, source=source, data=data,
                      ch_names=eeg_names, sfreq=sfreq, events=events, info=info)


# --------------------------------------------------------------------------
# Quality control (non-destructive — spec §3.4, §5.2)
# --------------------------------------------------------------------------
def quality_report(session: RawSession, expected_sfreq: float = 1000.0,
                   expected_channels: int = 63, psd_fmax: float = 120.0) -> dict:
    """Non-destructive QC of one session (never modifies the signal).

    Checks sampling rate, channel count, flat channels, amplitude outliers,
    event counts / repetitions per image, and the PSD (including a 50 Hz line
    ratio). The PSD check is **diagnostic only** and must not switch on a notch
    filter in the baseline.
    """
    data, ev = session.data, session.events
    report: Dict[str, object] = {
        "subject": session.subject, "session": session.session,
        "source": session.source, "sfreq": session.sfreq,
        "n_channels": int(data.shape[0]), "n_samples": int(data.shape[1]),
        "duration_s": float(data.shape[1] / session.sfreq),
        "n_events": int(len(ev)),
        "n_unique_images": int(len(ev.unique_codes)),
        "n_targets_dropped": int(ev.n_targets_dropped),
        "n_extra_dropped": int(ev.n_extra_dropped),
    }
    report["sfreq_ok"] = bool(abs(session.sfreq - expected_sfreq) < 1e-6)
    report["n_channels_ok"] = bool(data.shape[0] == expected_channels)

    stds = data.std(axis=1)
    flat = [session.ch_names[i] for i in np.flatnonzero(stds < 1e-12)]
    report["flat_channels"] = flat
    report["channel_std_min"] = float(stds.min())
    report["channel_std_max"] = float(stds.max())
    report["abs_max"] = float(np.abs(data).max())

    counts = np.unique(np.unique(ev.codes, return_counts=True)[1],
                       return_counts=True)
    report["reps_per_image"] = {int(k): int(v) for k, v in zip(*counts)}

    try:  # PSD is diagnostic; never gate the pipeline on it
        from scipy.signal import welch
        step = max(1, data.shape[1] // 200_000)   # subsample for speed
        f, p = welch(data[:, ::step], fs=session.sfreq / step,
                     nperseg=min(1024, data.shape[1] // step))
        mean_psd = p.mean(axis=0)
        band = (f >= 1) & (f <= psd_fmax)
        report["psd_freqs"] = f[band].tolist()
        report["psd_mean"] = mean_psd[band].tolist()
        line = np.argmin(np.abs(f - 50.0))
        neigh = (np.abs(f - 50.0) > 2) & (np.abs(f - 50.0) < 10)
        if neigh.any():
            report["line_50hz_ratio"] = float(mean_psd[line] / mean_psd[neigh].mean())
    except Exception as exc:  # pragma: no cover
        report["psd_error"] = str(exc)
    return report


def validate_channel_consistency(sessions: Sequence[RawSession]) -> None:
    """Raise if channel names/order differ across sessions (spec §5.2)."""
    if not sessions:
        return
    ref = sessions[0].ch_names
    for s in sessions[1:]:
        if s.ch_names != ref:
            raise ValueError(
                f"Channel names/order differ between {sessions[0].session} and "
                f"{s.session} for {s.subject}; refusing to merge sessions.")
