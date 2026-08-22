"""Experiment 1 entry point: train fMRIEncoder + CLIPHead (no low-level head).

Thin wrapper over :func:`run_training` so scripts and notebooks share exactly
one training implementation (spec §7, §17 — notebooks must not duplicate logic).
"""
from __future__ import annotations

from .train_multitask_decoder import run_training


def train_clip(cfg, resume=None, eval_only: bool = False) -> dict:
    return run_training(cfg, use_lowlevel=False, resume=resume,
                        eval_only=eval_only)
