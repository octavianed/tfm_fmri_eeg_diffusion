"""Continuous filtering and resampling with a dual backend (spec §5.3, §5.7).

``mne`` is the reference implementation used by THINGS-EEG2 / NICE / ATM, so it
is preferred when installed: a zero-phase Hamming FIR applied with overlap-add
(a 0.1 Hz transition needs ~33k taps, which MNE handles efficiently). When MNE
is absent the code falls back to ``scipy`` — a zero-phase IIR (``sosfiltfilt``,
Butterworth) for the band-pass and polyphase ``resample_poly`` for resampling
(1000→250 and 1000→100 are exact integer ratios).

Whichever backend runs, every parameter is recorded in the returned description
and persisted with the variant metadata (spec §12).
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

import numpy as np

from ..utils import get_logger

logger = get_logger("preproc_filter")


_FALLBACK_WARNED = False


def filter_backend(prefer: str = "auto") -> str:
    """Return the backend to use: ``'mne'`` or ``'scipy'``.

    ``prefer='mne'`` raises if MNE is missing — the baseline pins it so a missing
    install fails loudly instead of silently producing an IIR-filtered variant
    that would not be comparable with the FIR ones (spec §3.3).
    ``prefer='auto'`` picks MNE when importable, else warns once and uses scipy.
    """
    prefer = str(prefer or "auto").lower()
    if prefer == "scipy":
        return "scipy"
    try:
        import mne  # noqa: F401
        return "mne"
    except Exception as exc:
        if prefer == "mne":
            raise ImportError(
                "preprocessing.filter.backend='mne' but mne is not importable "
                f"with this interpreter ({sys.executable}). Run the pipeline "
                "with the project venv, which has mne installed:\n"
                "  .tfm_fmri_diffusion_3_11/Scripts/python.exe "
                "scripts/09_preprocess_eeg_raw.py --config ...\n"
                "To deviate on purpose, pass "
                "--set preprocessing.filter.backend=scipy (and report it: the "
                "variant will use a zero-phase IIR, not the reference FIR)."
            ) from exc
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            logger.warning(
                "mne not available with %s — falling back to the scipy filter "
                "backend (zero-phase IIR). This variant will NOT be directly "
                "comparable with FIR-built ones; document the deviation.",
                sys.executable)
        return "scipy"


def describe_filter(l_freq, h_freq, notch, backend: str,
                    extra: Optional[Dict] = None) -> dict:
    """Reproducibility record of the filter actually applied (spec §5.3)."""
    d = {"l_freq_hz": None if l_freq is None else float(l_freq),
         "h_freq_hz": None if h_freq is None else float(h_freq),
         "notch_hz": None if notch is None else float(notch),
         "backend": backend,
         "type": "fir" if backend == "mne" else "iir_butterworth",
         "phase": "zero"}
    if backend == "mne":
        try:
            import mne
            d["mne_version"] = mne.__version__
            d["design"] = "firwin (hamming), overlap-add"
        except Exception:
            pass
    else:
        import scipy
        d["scipy_version"] = scipy.__version__
        d["design"] = "butterworth sos, sosfiltfilt (forward-backward)"
    if extra:
        d.update(extra)
    return d


def filter_continuous(data: np.ndarray, sfreq: float, l_freq: Optional[float],
                      h_freq: Optional[float], notch: Optional[float] = None,
                      backend: str = "auto", n_jobs: int = 1,
                      iir_order: int = 4) -> tuple:
    """Band-pass the **continuous** signal before epoching (spec §5.3).

    Args:
        data: ``[n_channels, n_samples]`` (float32 in, float32 out).
        sfreq: sampling rate in Hz.
        l_freq / h_freq: band edges (``None`` disables that side).
        notch: notch frequency; ``None`` in the baseline (spec §5.3, §3.4).
        backend: ``'auto' | 'mne' | 'scipy'``.

    Returns:
        ``(filtered [n_channels, n_samples] float32, description dict)``.
    """
    backend = filter_backend(backend)
    x = np.ascontiguousarray(data, dtype=np.float64)
    extra: Dict[str, object] = {}

    if backend == "mne":
        import mne
        with mne.utils.use_log_level("error"):
            if l_freq is not None or h_freq is not None:
                x = mne.filter.filter_data(x, sfreq=sfreq, l_freq=l_freq,
                                           h_freq=h_freq, method="fir",
                                           phase="zero", fir_design="firwin",
                                           n_jobs=n_jobs, verbose=False)
            if notch:
                x = mne.filter.notch_filter(x, Fs=sfreq, freqs=notch,
                                            method="fir", phase="zero",
                                            n_jobs=n_jobs, verbose=False)
        # Record the transition bandwidths MNE derives by default.
        if l_freq is not None:
            extra["l_trans_bandwidth"] = float(min(max(l_freq * 0.25, 2.0), l_freq))
        if h_freq is not None:
            extra["h_trans_bandwidth"] = float(min(max(h_freq * 0.25, 2.0),
                                                   sfreq / 2.0 - h_freq))
    else:
        from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos
        nyq = sfreq / 2.0
        if l_freq is not None and h_freq is not None:
            sos = butter(iir_order, [l_freq / nyq, min(h_freq, nyq * 0.999) / nyq],
                         btype="bandpass", output="sos")
        elif l_freq is not None:
            sos = butter(iir_order, l_freq / nyq, btype="highpass", output="sos")
        elif h_freq is not None:
            sos = butter(iir_order, min(h_freq, nyq * 0.999) / nyq,
                         btype="lowpass", output="sos")
        else:
            sos = None
        if sos is not None:
            x = sosfiltfilt(sos, x, axis=-1)
        if notch:
            b, a = iirnotch(notch / nyq, Q=30.0)
            x = sosfiltfilt(tf2sos(b, a), x, axis=-1)
        extra["iir_order"] = int(iir_order)

    desc = describe_filter(l_freq, h_freq, notch, backend, extra)
    logger.info("Filtered continuous data with %s: %s", backend,
                {k: desc[k] for k in ("l_freq_hz", "h_freq_hz", "notch_hz", "type")})
    return np.ascontiguousarray(x, dtype=np.float32), desc


def resample_epochs(epochs: np.ndarray, sfreq: float, target_sfreq: float,
                    backend: str = "auto") -> tuple:
    """Resample epoched data with antialias protection (spec §5.7).

    Never decimate by plain slicing. MNE uses FFT-based resampling (the
    reference behaviour of ``Epochs.resample``); scipy uses polyphase
    ``resample_poly``, which also applies the antialias filter.

    Args:
        epochs: ``[n_epochs, n_channels, n_times]``.

    Returns:
        ``(resampled [n_epochs, n_channels, n_times'], description dict)``.
    """
    if abs(float(sfreq) - float(target_sfreq)) < 1e-9:
        return np.ascontiguousarray(epochs, dtype=np.float32), {
            "method": "none", "sfreq_in": float(sfreq),
            "sfreq_out": float(target_sfreq)}

    backend = filter_backend(backend)
    x = np.ascontiguousarray(epochs, dtype=np.float64)
    if backend == "mne":
        import mne
        with mne.utils.use_log_level("error"):
            out = mne.filter.resample(x, up=float(target_sfreq), down=float(sfreq),
                                      axis=-1, verbose=False)
        desc = {"method": "mne.filter.resample (FFT, antialiased)",
                "mne_version": mne.__version__}
    else:
        from math import gcd
        from scipy.signal import resample_poly
        import scipy
        up, down = int(round(target_sfreq)), int(round(sfreq))
        g = gcd(up, down)
        up, down = up // g, down // g
        out = resample_poly(x, up, down, axis=-1)
        desc = {"method": f"scipy.resample_poly (up={up}, down={down}, antialiased)",
                "scipy_version": scipy.__version__}

    desc.update({"sfreq_in": float(sfreq), "sfreq_out": float(target_sfreq),
                 "backend": backend, "n_times_in": int(epochs.shape[-1]),
                 "n_times_out": int(out.shape[-1])})
    return np.ascontiguousarray(out, dtype=np.float32), desc
