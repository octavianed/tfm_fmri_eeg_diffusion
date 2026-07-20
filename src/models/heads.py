"""Prediction heads on top of the shared fMRI representation ``h``.

* :class:`CLIPHead`     -> predicts the CLIP image embedding (semantic target).
* :class:`LowLevelHead` -> predicts the VAE-PCA vector (low-level target).

Heads output *unnormalized* vectors; L2-normalization is applied in the losses
and retrieval code so a single convention is used everywhere.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim=None,
                 dropout: float = 0.0, final_layernorm: bool = False):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim) if final_layernorm else nn.Identity()

    def forward(self, h):
        return self.final_norm(self.net(h))


class CLIPHead(ProjectionHead):
    """Predicts a CLIP image embedding from ``h``."""


class LowLevelHead(ProjectionHead):
    """Predicts a low-level VAE-PCA vector from ``h``."""
