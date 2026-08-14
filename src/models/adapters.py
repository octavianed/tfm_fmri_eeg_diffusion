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
import torch.nn.functional as F


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
    """CLIP image embedding ``[B, clip_dim]`` -> SD tokens ``[B, num_tokens, cross_dim]``.

    ``normalize_input=True`` makes the adapter **scale-invariant by construction**
    (generation Option B): the input is L2-normalized inside ``forward``, so
    ``adapter(x) == adapter(k·x)`` for any ``k > 0``.

    Why it matters: neither CLIP loss term of the decoder (cosine, InfoNCE)
    constrains the *norm* of ``clip_pred``, so its radius is an unsupervised free
    parameter that drifts from run to run (measured 0.54x-1.39x the real CLIP
    norm in this project). Since the adapter is otherwise almost scale
    equivariant, that drift acts like an uncontrolled conditioning strength.
    Normalizing here removes it, and ``input_scale`` re-exposes the same knob as
    something explicit and tunable at inference time — no retraining needed.

    ``normalize_input`` must match between training and inference, so it is
    stored in the adapter checkpoint and restored by
    :meth:`~src.generation.sd_pipeline.FrozenSDGenerator.load_adapter`.
    """

    def __init__(self, clip_dim: int, cross_dim: int = 768, num_tokens: int = 77,
                 hidden_dim: int = 1024, dropout: float = 0.0,
                 normalize_input: bool = False, input_scale: float = 1.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.cross_dim = int(cross_dim)
        self.normalize_input = bool(normalize_input)
        #: inference-time conditioning strength (only meaningful when
        #: ``normalize_input``); training always runs at 1.0.
        self.input_scale = float(input_scale)
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * cross_dim),
        )

    def forward(self, clip_emb):
        if self.normalize_input:
            clip_emb = F.normalize(clip_emb, dim=-1, eps=1e-8) * self.input_scale
        out = self.net(clip_emb)
        return out.view(-1, self.num_tokens, self.cross_dim)
