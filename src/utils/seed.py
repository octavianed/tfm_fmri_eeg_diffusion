"""Reproducibility helpers: seeding and RNG state capture/restore."""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

try:  # torch is optional at import time (kept out of light-weight tooling)
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover
                pass


def get_rng_state() -> dict:
    """Snapshot RNG state for checkpointing."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            try:
                state["cuda"] = torch.cuda.get_rng_state_all()
            except Exception:  # pragma: no cover
                state["cuda"] = None
    return state


def set_rng_state(state: "dict | None") -> None:
    """Restore RNG state captured by :func:`get_rng_state`."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if torch is not None and state.get("torch") is not None:
        try:
            torch.set_rng_state(state["torch"])
        except Exception:  # pragma: no cover
            pass
        if torch.cuda.is_available() and state.get("cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(state["cuda"])
            except Exception:  # pragma: no cover
                pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible workers."""
    if torch is not None:
        worker_seed = torch.initial_seed() % (2 ** 32)
    else:  # pragma: no cover
        worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
