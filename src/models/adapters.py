"""Small trainable adapters.

* :class:`SubjectAdapters` projects each subject's variable-length voxel vector
  to a common dimension so one shared encoder can serve several subjects with
  different voxel counts (batches are subject-homogeneous — see the datamodule).
* :class:`TokenAdapter` maps a predicted CLIP image embedding to pseudo text
  tokens for a frozen Stable Diffusion 1.5 UNet (generation Option B, spec §10.3).
"""
from __future__ import annotations

from typing import Dict

import torch.nn as nn


class SubjectAdapters(nn.Module):
    def __init__(self, voxel_counts: Dict[str, int], common_dim: int,
                 bias: bool = True):
        super().__init__()
        self.common_dim = int(common_dim)
        self.proj = nn.ModuleDict(
            {subj: nn.Linear(int(v), common_dim, bias=bias)
             for subj, v in voxel_counts.items()})

    def forward(self, x, subject: str):
        if subject not in self.proj:
            raise KeyError(f"No adapter for subject '{subject}'. "
                           f"Known: {list(self.proj.keys())}")
        return self.proj[subject](x)


class TokenAdapter(nn.Module):
    """CLIP image embedding ``[B, clip_dim]`` -> SD tokens ``[B, num_tokens, cross_dim]``."""

    def __init__(self, clip_dim: int, cross_dim: int = 768, num_tokens: int = 77,
                 hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.cross_dim = int(cross_dim)
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * cross_dim),
        )

    def forward(self, clip_emb):
        out = self.net(clip_emb)
        return out.view(-1, self.num_tokens, self.cross_dim)
