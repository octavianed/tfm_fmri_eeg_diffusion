"""Symmetric InfoNCE (CLIP-style) contrastive loss over a batch (spec §7.4).

Similarity matrix ``S = pred_norm @ target_norm.T / temperature`` is treated as
logits; the correct pairing is the diagonal, and cross-entropy is applied in
both directions (fMRI->image and image->fMRI).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(pred, target, temperature: float = 0.07, eps: float = 1e-8):
    batch = pred.shape[0]
    if batch < 2:  # contrastive term is undefined for a single sample
        return pred.new_zeros(())
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    tgt_n = F.normalize(target, dim=-1, eps=eps)
    logits = (pred_n @ tgt_n.t()) / float(temperature)
    labels = torch.arange(batch, device=pred.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_p2t + loss_t2p)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, pred, target):
        return info_nce_loss(pred, target, self.temperature)
