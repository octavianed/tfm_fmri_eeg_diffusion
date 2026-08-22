"""Deterministic derangements — the project's single source of shuffling.

The falsifiable criterion of the whole project (``correcto >> permutado ~ cero``)
rests on the *permuted* control being a genuine derangement: if a sample could
keep its own brain signal (or its own caption) the control would be
contaminated. Sattolo's algorithm draws a uniformly random **cyclic** permutation
of ``range(n)``, which has no fixed points by construction.

Both the brain ablation (:mod:`src.evaluation.ablation_eval`) and the textual
permutation (:mod:`src.data.captions`) use these helpers, so "permuted" means
exactly the same thing on both sides.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def sattolo_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random cyclic permutation of ``range(n)`` (no fixed points)."""
    idx = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def stable_hash(text: str) -> int:
    """Process-independent hash of a string.

    Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so a
    seed derived from it changes between runs — two invocations of the same
    experiment would draw *different* permutations. Everything that seeds a
    control condition must therefore go through this instead.
    """
    import zlib
    return int(zlib.crc32(str(text).encode("utf-8")))


def condition_seed(seed: int, condition: str) -> int:
    """Reproducible per-condition seed (``correct``/``permuted``/``zero``/...)."""
    return (int(seed) + stable_hash(condition) % 100000) % (2 ** 32)


def derangement(n: int, seed: int, salt: Optional[str] = None) -> np.ndarray:
    """Reproducible derangement of ``range(n)`` from an integer seed.

    ``salt`` mixes a string (e.g. ``"subj01/test"``) into the seed so that two
    groups of the same size do not receive the *same* permutation while still
    being fully determined by ``seed`` — the requirement is "deterministic and
    within-split", not "identical across splits".

    ``n < 2`` has no derangement; the identity is returned so callers can stay
    branch-free (a 1-element split cannot be a meaningful control anyway).
    """
    if n < 2:
        return np.arange(max(0, n))
    mixed = int(seed)
    if salt:
        mixed = (mixed + stable_hash(salt)) % (2 ** 32)
    return sattolo_derangement(n, np.random.default_rng(mixed))
