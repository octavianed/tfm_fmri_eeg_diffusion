"""EEG encoder: a temporal 1-D convolutional network for THINGS-EEG2.

Unlike the fMRI response (a per-image *spatial* vector handled by
:class:`~src.models.fmri_encoder.FMRIEncoder`), EEG is a *temporal, multichannel*
signal ``[C, T]`` (channels x time). This encoder follows the temporal-conv
recipe from the multimodal spec (spec §7.2): a temporal stem, a
depthwise/separable temporal conv, residual temporal blocks, temporal pooling,
and a linear projection to the shared representation ``h``.

It keeps the SAME output contract as ``FMRIEncoder`` — a ``forward`` mapping the
brain tensor to ``[B, output_dim]`` and an ``output_dim`` attribute — so the CLIP
and low-level heads (``src/models/heads.py``) attach unchanged. Only the
front-end differs by modality; everything downstream is shared.

Temporal convolutions use 'same' padding and the final pooling is over the time
axis, so the module is robust to the exact number of time samples ``T`` (e.g. a
different ``time_window_ms`` crop or sampling rate).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelDropout(nn.Module):
    """Drop whole EEG channels (all time samples of a channel at once).

    The temporal analogue of :class:`~src.models.fmri_encoder.VoxelDropout`:
    encourages the encoder not to rely on any single electrode.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = float(p)

    def forward(self, x):  # x: [B, C, T]
        if not self.training or self.p <= 0.0:
            return x
        mask = x.new_empty((x.shape[0], x.shape[1], 1)).bernoulli_(1.0 - self.p)
        return x * mask / (1.0 - self.p)


class ResidualTemporalBlock(nn.Module):
    """Pre-norm residual block of two temporal convolutions (same padding)."""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm(1) = LayerNorm over (C, T)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B, dim, T]
        h = self.norm(x)
        h = F.gelu(self.conv1(h))
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


class TemporalAttentionPooling(nn.Module):
    """Attention-weighted pooling over the time axis: [B, dim, T] -> [B, dim]."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):  # [B, dim, T]
        h = x.transpose(1, 2)                       # [B, T, dim]
        w = torch.softmax(self.score(h), dim=1)     # [B, T, 1]
        return (h * w).sum(dim=1)                    # [B, dim]


class EEGEncoderTemporalConv(nn.Module):
    """Temporal-conv EEG encoder: ``[B, C, T] -> [B, output_dim]``.

    Args:
        in_channels: number of EEG channels ``C`` (e.g. 17 or 63).
        in_times: number of time samples ``T`` (informational; the network is
            robust to ``T`` thanks to 'same' padding + temporal pooling).
        hidden_dim: width of the temporal feature maps.
        output_dim: dimension of the shared representation ``h`` (kept at 2048 to
            match the fMRI encoder so the heads/adapter dims are identical).
        num_res_blocks: number of residual temporal blocks.
        dropout: dropout inside the stem/blocks/head.
        channel_dropout: probability of dropping whole channels at the input.
        pool: 'attention' (learned temporal attention) or 'mean' (global avg).
    """

    def __init__(self, in_channels: int, in_times: int = 100,
                 hidden_dim: int = 256, output_dim: int = 2048,
                 num_res_blocks: int = 2, dropout: float = 0.3,
                 channel_dropout: float = 0.1, pool: str = "attention",
                 kernel_size: int = 7):
        super().__init__()
        self.in_channels = int(in_channels)
        self.in_times = int(in_times)
        self.in_features = int(in_channels) * int(in_times)
        self.output_dim = int(output_dim)

        self.input_norm = nn.BatchNorm1d(in_channels)
        self.channel_dropout = ChannelDropout(channel_dropout)

        pad = kernel_size // 2
        # Temporal stem: mixes channels and captures temporal structure.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Depthwise/separable temporal conv (per-feature temporal filtering).
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad,
                      groups=hidden_dim, bias=False),
            nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False),  # pointwise
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, kernel_size=5, dropout=dropout)
             for _ in range(num_res_blocks)])

        self.pool_kind = str(pool)
        self.pool = (TemporalAttentionPooling(hidden_dim)
                     if self.pool_kind == "attention" else None)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):  # x: [B, C, T]
        if x.dim() != 3:
            raise ValueError(f"EEG encoder expects [B, C, T], got {tuple(x.shape)}")
        x = self.input_norm(x)
        x = self.channel_dropout(x)
        x = self.stem(x)
        x = self.depthwise(x)
        for block in self.blocks:
            x = block(x)
        if self.pool is not None:
            h = self.pool(x)                    # [B, hidden]
        else:
            h = x.mean(dim=-1)                  # global average over time
        return self.head(h)                     # [B, output_dim]
