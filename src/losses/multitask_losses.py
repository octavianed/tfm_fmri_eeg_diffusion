"""Weighted multitask loss combining CLIP cosine, InfoNCE and low-level MSE.

    L = lambda_cosine * (1 - cos)
      + lambda_contrastive * InfoNCE
      + lambda_nmse * normalized_mse            (optional)
      + lambda_lowlevel * MSE(low_pred, low)    (Experiment 3 only)

All lambdas are configurable (spec §7.4, §9.4).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .contrastive import info_nce_loss
from .cosine import cosine_similarity_loss, normalized_mse_loss


class MultitaskLoss(nn.Module):
    def __init__(self, lambda_cosine: float = 1.0, lambda_contrastive: float = 1.0,
                 lambda_lowlevel: float = 0.25, lambda_nmse: float = 0.0,
                 temperature: float = 0.07, use_lowlevel: bool = False):
        super().__init__()
        self.lambda_cosine = float(lambda_cosine)
        self.lambda_contrastive = float(lambda_contrastive)
        self.lambda_lowlevel = float(lambda_lowlevel)
        self.lambda_nmse = float(lambda_nmse)
        self.temperature = float(temperature)
        self.use_lowlevel = bool(use_lowlevel)

    def forward(self, outputs: dict, targets: dict):
        clip_pred = outputs["clip"]
        clip_tgt = targets["clip"]
        l_cos = cosine_similarity_loss(clip_pred, clip_tgt)
        l_con = info_nce_loss(clip_pred, clip_tgt, self.temperature)
        total = self.lambda_cosine * l_cos + self.lambda_contrastive * l_con
        parts = {"clip_cosine": l_cos.detach(), "clip_infonce": l_con.detach()}

        if self.lambda_nmse > 0:
            l_nmse = normalized_mse_loss(clip_pred, clip_tgt)
            total = total + self.lambda_nmse * l_nmse
            parts["clip_nmse"] = l_nmse.detach()

        if self.use_lowlevel and outputs.get("low") is not None \
                and targets.get("low") is not None:
            l_low = F.mse_loss(outputs["low"], targets["low"])
            total = total + self.lambda_lowlevel * l_low
            parts["lowlevel_mse"] = l_low.detach()

        parts["total"] = total.detach()
        return total, parts


def build_loss(cfg, use_lowlevel: bool = False) -> MultitaskLoss:
    return MultitaskLoss(
        lambda_cosine=cfg.get("losses.lambda_cosine", 1.0),
        lambda_contrastive=cfg.get("losses.lambda_contrastive", 1.0),
        lambda_lowlevel=cfg.get("losses.lambda_lowlevel", 0.25),
        lambda_nmse=cfg.get("losses.lambda_nmse", 0.0),
        temperature=cfg.get("losses.temperature", 0.07),
        use_lowlevel=use_lowlevel,
    )
