"""Loss functions: cosine, InfoNCE contrastive, and the multitask combination."""
from .cosine import (cosine_similarity_loss, mean_cosine_similarity,
                     normalized_mse_loss)
from .contrastive import InfoNCELoss, info_nce_loss
from .multitask_losses import MultitaskLoss, build_loss

__all__ = [
    "cosine_similarity_loss", "normalized_mse_loss", "mean_cosine_similarity",
    "info_nce_loss", "InfoNCELoss", "MultitaskLoss", "build_loss",
]
