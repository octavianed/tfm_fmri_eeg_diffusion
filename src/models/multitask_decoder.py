"""Multitask decoder: fMRI -> shared representation -> CLIP + low-level heads.

The same object powers Experiment 1 (CLIP head only) and Experiment 3 (CLIP +
low-level). A per-subject input adapter is enabled automatically when more than
one subject is selected (spec §5, §9.3).
"""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils.logging import get_logger
from .adapters import SubjectAdapters
from .eeg_encoder import EEGEncoderTemporalConv
from .fmri_encoder import FMRIEncoder
from .heads import CLIPHead, LowLevelHead

logger = get_logger("model")


class MultitaskDecoder(nn.Module):
    """Brain signal -> shared representation -> CLIP + low-level heads.

    ``encoder_type`` selects the front-end: ``"fmri"`` uses the residual MLP
    :class:`FMRIEncoder` over a flat ``[B, V]`` vector; ``"eeg"`` uses the
    temporal-conv :class:`EEGEncoderTemporalConv` over ``[B, C, T]``. In the EEG
    case ``voxel_counts`` maps each subject to its ``(C, T)`` signal shape and
    subject adapters are disabled (per-subject normalization handles subjects).
    Everything downstream of ``self.encoder`` (heads, losses, retrieval) is
    identical across modalities.
    """

    def __init__(self, voxel_counts: Union[int, Dict[str, int]],
                 encoder_kwargs: dict, clip_dim: int,
                 low_dim: Optional[int] = None,
                 use_lowlevel: bool = False,
                 use_subject_adapter: bool = False,
                 common_dim: int = 4096,
                 clip_head_kwargs: Optional[dict] = None,
                 low_head_kwargs: Optional[dict] = None,
                 encoder_type: str = "fmri"):
        super().__init__()
        self.encoder_type = str(encoder_type)
        if not isinstance(voxel_counts, dict):
            voxel_counts = {"_single": voxel_counts}

        if self.encoder_type == "eeg":
            # values are (C, T) signal shapes; no subject adapter for EEG.
            self.voxel_counts = {str(k): tuple(int(x) for x in v)
                                 for k, v in voxel_counts.items()}
            self.use_subject_adapter = False
            self.adapters = None
            c, t = next(iter(self.voxel_counts.values()))
            self.encoder = EEGEncoderTemporalConv(in_channels=c, in_times=t,
                                                  **encoder_kwargs)
        else:
            self.voxel_counts = {str(k): int(v) for k, v in voxel_counts.items()}
            self.use_subject_adapter = bool(use_subject_adapter)
            if self.use_subject_adapter:
                self.adapters = SubjectAdapters(self.voxel_counts, common_dim)
                enc_in = common_dim
            else:
                self.adapters = None
                enc_in = next(iter(self.voxel_counts.values()))
            self.encoder = FMRIEncoder(in_features=enc_in, **encoder_kwargs)

        self.clip_head = CLIPHead(self.encoder.output_dim, clip_dim,
                                  **(clip_head_kwargs or {}))
        self.use_lowlevel = bool(use_lowlevel)
        if self.use_lowlevel:
            if not low_dim:
                raise ValueError("use_lowlevel=True requires low_dim")
            self.low_head = LowLevelHead(self.encoder.output_dim, low_dim,
                                         **(low_head_kwargs or {}))
        else:
            self.low_head = None

    def encode(self, fmri, subject: Optional[str] = None):
        if self.adapters is not None:
            if subject is None:
                raise ValueError("subject id required when subject adapters are on")
            x = self.adapters(fmri, subject)
        else:
            x = fmri
        return self.encoder(x)

    def forward(self, fmri, subject: Optional[str] = None) -> dict:
        h = self.encode(fmri, subject)
        out = {"h": h, "clip": self.clip_head(h)}
        if self.low_head is not None:
            out["low"] = self.low_head(h)
        return out


def _normalize_counts(counts) -> dict:
    """Comparable form of ``voxel_counts``: fMRI ints vs EEG ``(C, T)`` tuples."""
    out = {}
    for k, v in dict(counts).items():
        out[str(k)] = (tuple(int(x) for x in v)
                       if isinstance(v, (list, tuple)) else int(v))
    return out


def _batch_subject(subjects) -> Optional[str]:
    """Return the single subject id of a homogeneous batch (or None)."""
    if subjects is None:
        return None
    if isinstance(subjects, str):
        return subjects
    uniq = set(str(s) for s in subjects)
    if len(uniq) != 1:
        raise ValueError(f"Batch is not subject-homogeneous: {uniq}")
    return next(iter(uniq))


def build_model(cfg, voxel_counts: Dict[str, int], clip_dim: Optional[int] = None,
                low_dim: Optional[int] = None,
                use_lowlevel: Optional[bool] = None) -> MultitaskDecoder:
    """Construct a :class:`MultitaskDecoder` from config + resolved input dims.

    ``dataset.modality`` (or an explicit ``model.encoder_type``) selects the
    encoder: fMRI (residual MLP over ``voxel_counts`` = per-subject vertex counts)
    or EEG (temporal conv; ``voxel_counts`` = per-subject ``(C, T)`` shapes).
    """
    modality = str(cfg.get("dataset.modality", "fmri")).lower()
    encoder_type = str(cfg.get("model.encoder_type",
                               "eeg" if modality == "eeg" else "fmri")).lower()
    encoder_type = "eeg" if encoder_type in ("eeg", "eeg_temporalconv") else "fmri"

    if encoder_type == "eeg":
        eeg = cfg.get("model.eeg_encoder", {})
        encoder_kwargs = dict(
            hidden_dim=int(eeg.get("hidden_dim", 256)),
            output_dim=int(eeg.get("output_dim", 2048)),
            num_res_blocks=int(eeg.get("num_res_blocks", 2)),
            dropout=float(eeg.get("dropout", 0.3)),
            channel_dropout=float(eeg.get("channel_dropout", 0.1)),
            pool=str(eeg.get("pool", "attention")),
            kernel_size=int(eeg.get("kernel_size", 7)),
        )
    else:
        enc = cfg.get("model.fmri_encoder", {})
        encoder_kwargs = dict(
            hidden_dim=int(enc.get("hidden_dim", 4096)),
            output_dim=int(enc.get("output_dim", 2048)),
            dropout=float(enc.get("dropout", 0.2)),
            voxel_dropout=float(enc.get("voxel_dropout", 0.1)),
            num_res_blocks=int(enc.get("num_res_blocks", 2)),
            input_layernorm=bool(enc.get("input_layernorm", True)),
        )
    clip_dim = int(clip_dim or cfg.get("model.clip_head.output_dim", 768))
    clip_head_kwargs = dict(
        hidden_dim=cfg.get("model.clip_head.hidden_dim", None),
        dropout=float(cfg.get("model.clip_head.dropout", 0.0)),
        final_layernorm=bool(cfg.get("model.clip_head.final_layernorm", False)),
    )

    if use_lowlevel is None:
        use_lowlevel = bool(cfg.get("model.use_lowlevel", False))
    if use_lowlevel and low_dim is None:
        low_dim = int(cfg.get("model.lowlevel_head.output_dim",
                              cfg.get("features.pca_dim", 512)))
    low_head_kwargs = dict(
        hidden_dim=cfg.get("model.lowlevel_head.hidden_dim", None),
        dropout=float(cfg.get("model.lowlevel_head.dropout", 0.0)),
    )

    n_subjects = len(voxel_counts)
    # EEG uses per-subject normalization instead of flat-Linear subject adapters.
    use_adapter = (False if encoder_type == "eeg"
                   else bool(cfg.get("model.use_subject_adapter", n_subjects > 1)))
    common_dim = int(cfg.get("model.subject_adapter.common_dim", 4096))

    return MultitaskDecoder(
        voxel_counts=voxel_counts, encoder_kwargs=encoder_kwargs,
        clip_dim=clip_dim, low_dim=low_dim, use_lowlevel=use_lowlevel,
        use_subject_adapter=use_adapter, common_dim=common_dim,
        clip_head_kwargs=clip_head_kwargs, low_head_kwargs=low_head_kwargs,
        encoder_type=encoder_type)


def build_model_from_checkpoint(cfg, checkpoint_path, device,
                                voxel_counts: Optional[Dict[str, int]] = None):
    """Rebuild a decoder from a training checkpoint and load its weights.

    Reads the architecture dims (clip/low/use_lowlevel/voxel_counts) that
    :mod:`src.training` stored in the checkpoint, so evaluation/generation don't
    need to re-derive them.
    """
    from ..utils import load_checkpoint
    state = load_checkpoint(checkpoint_path, map_location=device)
    saved_vc = state.get("voxel_counts")
    # Guard against evaluating/generating with a differently-shaped input than the
    # one the checkpoint was trained on. This is silent for EEG (the temporal-conv
    # encoder pools over time, so a [63, 100] tensor loads the same weights as the
    # [63, 250] it was trained on) and would quietly produce meaningless metrics —
    # e.g. pointing an ablation config at the official derivative instead of the
    # raw variant the decoder was trained on.
    if voxel_counts is not None and saved_vc is not None:
        if _normalize_counts(voxel_counts) != _normalize_counts(saved_vc):
            if not bool(cfg.get("checkpointing.allow_input_shape_mismatch", False)):
                raise ValueError(
                    f"Input shape mismatch: the checkpoint {checkpoint_path} was "
                    f"trained on {saved_vc} but the datamodule provides "
                    f"{dict(voxel_counts)}. Check dataset.source / "
                    f"dataset.preproc_variant / dataset.channels in this config — "
                    f"they must match the run that produced the checkpoint. To "
                    f"override on purpose set "
                    f"checkpointing.allow_input_shape_mismatch=true.")
            logger.warning("Input shape mismatch allowed by config: checkpoint %s "
                           "vs datamodule %s", saved_vc, dict(voxel_counts))
    vc = voxel_counts or saved_vc
    if vc is None:
        raise ValueError("voxel_counts not in checkpoint; pass it explicitly.")
    model = build_model(cfg, vc, clip_dim=state.get("clip_dim"),
                        low_dim=state.get("low_dim"),
                        use_lowlevel=bool(state.get("use_lowlevel", False)))
    model.load_state_dict(state["model_state_dict"],
                          strict=bool(cfg.get("checkpointing.strict_load", False)))
    return model.to(device).eval(), state
