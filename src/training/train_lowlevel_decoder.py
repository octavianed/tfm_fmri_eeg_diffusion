"""Train with the low-level (VAE-PCA) branch active.

Same core loop as multitask; kept as a named entry point for the low-level
experiment. Set ``losses.lambda_cosine``/``lambda_contrastive`` low in the
config to emphasize the low-level objective if desired (spec §9).
"""
from __future__ import annotations

from .train_multitask_decoder import run_training


def train_lowlevel(cfg, resume=None) -> dict:
    return run_training(cfg, use_lowlevel=True, resume=resume)
