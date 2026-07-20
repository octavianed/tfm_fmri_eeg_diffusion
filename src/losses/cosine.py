"""Cosine-based losses for CLIP embedding regression (spec §7.4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity_loss(pred, target, eps: float = 1e-8):
    """``1 - cos(pred, target)`` averaged over the batch."""
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    tgt_n = F.normalize(target, dim=-1, eps=eps)
    cos = (pred_n * tgt_n).sum(dim=-1)
    return (1.0 - cos).mean()


def normalized_mse_loss(pred, target, eps: float = 1e-8):
    """MSE between L2-normalized prediction and target."""
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    tgt_n = F.normalize(target, dim=-1, eps=eps)
    return F.mse_loss(pred_n, tgt_n)


def mean_cosine_similarity(pred, target, eps: float = 1e-8) -> float:
    with torch.no_grad():
        pred_n = F.normalize(pred, dim=-1, eps=eps)
        tgt_n = F.normalize(target, dim=-1, eps=eps)
        return float((pred_n * tgt_n).sum(dim=-1).mean().item())
