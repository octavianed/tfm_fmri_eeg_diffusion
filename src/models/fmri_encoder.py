"""fMRI encoder: maps a spatial fMRI response ``[B, V]`` to a latent ``[B, H]``.

This is deliberately an MLP with residual blocks, NOT the temporal 1D-conv
architecture used for EEG — the Algonauts fMRI response is a per-image spatial
vector, not a ``[channels, time]`` signal (spec §5.1.1, §20).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelDropout(nn.Module):
    """Dropout applied on the input voxel features (regularizes noisy voxels)."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training)

    def extra_repr(self):
        return f"p={self.p}"


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class FMRIEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 4096,
                 output_dim: int = 2048, dropout: float = 0.2,
                 voxel_dropout: float = 0.1, num_res_blocks: int = 2,
                 input_layernorm: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.output_dim = int(output_dim)
        self.input_norm = nn.LayerNorm(in_features) if input_layernorm else nn.Identity()
        self.voxel_dropout = VoxelDropout(voxel_dropout)
        self.proj = nn.Linear(in_features, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_res_blocks)])
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.voxel_dropout(x)
        x = self.drop(F.gelu(self.proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.out(x)
