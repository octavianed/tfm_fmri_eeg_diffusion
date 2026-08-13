"""Reference, epoching, baseline correction and temporal cropping.

All time windows use the **half-open** contract ``[tmin, tmax)`` (spec §5.8), so
``[0, 1000)`` ms at 250 Hz yields exactly **250** samples — never 251 because of
an accidentally included ``t = 1000 ms`` sample.

Epoching is done blockwise with NumPy over the already-filtered continuous
signal instead of materialising every epoch at once: a training session holds
16 540 trials × 63 ch × 1200 samples, i.e. ~5 GB in float32 (~10 GB in float64).
Each block is baseline-corrected, resampled and cropped before being stored, so
peak memory stays bounded.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..utils import get_logger
from .filters import resample_epochs

logger = get_logger("preproc_epoch")


def n_samples_for_window(tmin_ms: float, tmax_ms: float, sfreq: float) -> int:
    """Number of samples in the half-open window ``[tmin_ms, tmax_ms)``."""
    return int(round((float(tmax_ms) - float(tmin_ms)) / 1000.0 * float(sfreq)))


def _ms_to_sample(t_ms: float, sfreq: float) -> int:
    return int(round(float(t_ms) / 1000.0 * float(sfreq)))


def apply_reference(data: np.ndarray, mode: str = "original") -> tuple:
    """Apply the referencing scheme of the variant (spec §5.4, §8.9).

    ``'original'`` keeps the reference stored in the release (the baseline: no
    CAR is applied, not even implicitly). ``'average'`` (CAR) subtracts the
    across-channel mean at each time point — it must run **before** MVNN.

    Args:
        data: ``[n_channels, n_samples]`` (continuous) or ``[..., n_ch, n_times]``.
    """
    mode = str(mode or "original").lower()
    if mode in ("original", "none", "keep"):
        return data, {"mode": "original"}
    if mode in ("average", "car"):
        out = data - data.mean(axis=-2, keepdims=True)
        return np.ascontiguousarray(out, dtype=data.dtype), {"mode": "average"}
    raise ValueError(f"Unknown reference mode: {mode}")


def baseline_correct(epochs: np.ndarray, tmin_ms: float, sfreq: float,
                     base_tmin_ms: float, base_tmax_ms: float) -> np.ndarray:
    """Subtract the per-trial, per-channel mean of ``[base_tmin, base_tmax)``.

    ``X_c(t) <- X_c(t) - mean(X_c(t), t in [base_tmin, base_tmax))`` (spec §5.6).

    Args:
        epochs: ``[n_epochs, n_channels, n_times]`` starting at ``tmin_ms``.
    """
    start = _ms_to_sample(base_tmin_ms - tmin_ms, sfreq)
    stop = _ms_to_sample(base_tmax_ms - tmin_ms, sfreq)
    if start < 0 or stop > epochs.shape[-1] or stop <= start:
        raise ValueError(
            f"Baseline window [{base_tmin_ms}, {base_tmax_ms}) ms is outside the "
            f"epoch starting at {tmin_ms} ms with {epochs.shape[-1]} samples.")
    mean = epochs[..., start:stop].mean(axis=-1, keepdims=True)
    return epochs - mean


def crop_window(epochs: np.ndarray, tmin_ms: float, sfreq: float,
                crop_tmin_ms: float, crop_tmax_ms: float) -> np.ndarray:
    """Crop epochs to the half-open window ``[crop_tmin_ms, crop_tmax_ms)``."""
    start = _ms_to_sample(crop_tmin_ms - tmin_ms, sfreq)
    n = n_samples_for_window(crop_tmin_ms, crop_tmax_ms, sfreq)
    stop = start + n
    if start < 0 or stop > epochs.shape[-1]:
        raise ValueError(
            f"Crop [{crop_tmin_ms}, {crop_tmax_ms}) ms does not fit in an epoch "
            f"starting at {tmin_ms} ms with {epochs.shape[-1]} samples "
            f"(needs samples {start}:{stop}).")
    return np.ascontiguousarray(epochs[..., start:stop])


def epoch_continuous(data: np.ndarray, onsets: Sequence[int], sfreq: float,
                     tmin_ms: float, tmax_ms: float,
                     block_size: int = 2000,
                     baseline: Optional[Tuple[float, float]] = None,
                     target_sfreq: Optional[float] = None,
                     crop: Optional[Tuple[float, float]] = None,
                     backend: str = "auto") -> tuple:
    """Epoch, baseline-correct, resample and crop — blockwise (memory bounded).

    The order follows spec §9: epoch ``[tmin, tmax)`` → baseline → resample →
    crop. The pre-stimulus window must stay in the epoch until the baseline has
    been applied, which is why cropping happens last.

    Args:
        data: filtered continuous signal ``[n_channels, n_samples]``.
        onsets: event onsets in samples.
        baseline: ``(tmin_ms, tmax_ms)`` or ``None`` to skip (spec §8.10 C).
        target_sfreq: resample target; ``None`` keeps ``sfreq``.
        crop: final ``(tmin_ms, tmax_ms)``; ``None`` keeps the whole epoch.

    Returns:
        ``(epochs [n_epochs, n_channels, n_times] float32, description dict)``.
        ``description['kept_mask']`` is a boolean array over the input ``onsets``
        — callers MUST use it to subset their per-event metadata (image codes),
        because a dropped epoch need not be the last one.
    """
    onsets = np.asarray(onsets, dtype=np.int64)
    n_ch, n_samp = data.shape
    start_off = _ms_to_sample(tmin_ms, sfreq)
    n_times = n_samples_for_window(tmin_ms, tmax_ms, sfreq)

    starts = onsets + start_off
    ok = (starts >= 0) & (starts + n_times <= n_samp)
    if not ok.all():
        logger.warning("Dropping %d epoch(s) that fall outside the recording.",
                       int((~ok).sum()))
    starts = starts[ok]

    out_sfreq = float(target_sfreq or sfreq)
    blocks = []
    resample_desc = {"method": "none"}
    for b0 in range(0, len(starts), block_size):
        blk = starts[b0:b0 + block_size]
        # Gather via fancy indexing: [n_block, n_ch, n_times]
        idx = blk[:, None] + np.arange(n_times)[None, :]
        ep = np.ascontiguousarray(data[:, idx].transpose(1, 0, 2), dtype=np.float32)
        if baseline is not None:
            ep = baseline_correct(ep, tmin_ms, sfreq, baseline[0], baseline[1])
        if target_sfreq is not None and abs(out_sfreq - sfreq) > 1e-9:
            ep, resample_desc = resample_epochs(ep, sfreq, out_sfreq, backend=backend)
        if crop is not None:
            ep = crop_window(ep, tmin_ms, out_sfreq, crop[0], crop[1])
        blocks.append(ep)

    epochs = (np.concatenate(blocks, axis=0) if blocks
              else np.zeros((0, n_ch, n_times), dtype=np.float32))
    desc = {"tmin_ms": float(tmin_ms), "tmax_ms": float(tmax_ms),
            "n_epochs": int(epochs.shape[0]), "n_channels": int(epochs.shape[1]),
            "n_times": int(epochs.shape[-1]), "sfreq_out": out_sfreq,
            "baseline_ms": None if baseline is None else [float(baseline[0]),
                                                          float(baseline[1])],
            "crop_ms": None if crop is None else [float(crop[0]), float(crop[1])],
            "resample": resample_desc, "kept_mask": ok,
            "kept_epochs_mask_sum": int(ok.sum()),
            "dropped_out_of_bounds": int((~ok).sum())}
    return epochs, desc
