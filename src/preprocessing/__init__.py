"""Own EEG preprocessing pipeline for THINGS-EEG2 raw data (original release).

Turns the raw 63-channel recordings under ``<root>/raw-eeg/`` into the SAME
on-disk contract as the official preprocessed derivatives — a dict with
``preprocessed_eeg_data`` ``[n_images, n_repetitions, n_channels, n_times]``,
``ch_names`` and ``times`` — so the existing EEG datamodule, encoder and the
whole Exp1–Exp5 stack consume it unchanged.

The reference configuration (baseline) is: 63 channels, 0.1–100 Hz, epoch
−200…1000 ms, baseline −200…0 ms, 250 Hz, crop ``[0, 1000)`` ms, no ICA/ASR/CAR,
MVNN fitted on training only and applied BEFORE averaging repetitions, average-4
in training and average-80 in test → a ``63 × 250`` tensor per image.

Heavy dependencies (``mne``, ``sklearn``, ``matplotlib``) are imported lazily
inside functions, following the project convention.
"""
from .things_raw_loader import (RawSession, SessionEvents, discover_raw_sessions,
                                discover_raw_subjects, load_raw_session,
                                extract_events, quality_report)
from .filters import (filter_continuous, resample_epochs, filter_backend,
                      describe_filter)
from .epoching import (apply_reference, epoch_continuous, baseline_correct,
                       crop_window, n_samples_for_window)
from .mvnn import MVNN, fit_mvnn, apply_mvnn
from .build_variant import (build_variant, preproc_config_hash,
                            resolve_channel_selection, POSTERIOR_17)

__all__ = [
    "RawSession", "SessionEvents", "discover_raw_subjects",
    "discover_raw_sessions", "load_raw_session", "extract_events",
    "quality_report", "filter_continuous", "resample_epochs", "filter_backend",
    "describe_filter", "apply_reference", "epoch_continuous",
    "baseline_correct", "crop_window", "n_samples_for_window", "MVNN",
    "fit_mvnn", "apply_mvnn", "build_variant", "preproc_config_hash",
    "resolve_channel_selection", "POSTERIOR_17",
]
