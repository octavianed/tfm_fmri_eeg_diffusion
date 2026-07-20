"""Modality-aware datamodule factory.

A single entry point so the training/eval/generation code stays modality-agnostic:
it calls ``build_datamodule(cfg)`` and gets an fMRI or EEG datamodule depending on
``dataset.modality`` (default ``"fmri"``). Both expose the same public interface
(``prepare``, ``subjects``, ``voxel_counts``, ``get_frame``,
``subject_split_frame``, ``subject_reader``, ``normalizer``, ``input_dim``,
``load_feature_arrays``, ``build_dataset``, ``build_dataloader``).
"""
from __future__ import annotations


def build_datamodule(cfg):
    """Return the datamodule for the configured modality.

    ``dataset.modality: eeg`` → :class:`~src.data.eeg_datamodule.EegDataModule`
    (THINGS-EEG2); anything else → :class:`~src.data.datamodule.FmriDataModule`
    (NSD Algonauts 2023).
    """
    modality = str(cfg.get("dataset.modality", "fmri")).lower()
    if modality == "eeg":
        from .eeg_datamodule import EegDataModule
        return EegDataModule(cfg)
    from .datamodule import FmriDataModule
    return FmriDataModule(cfg)
