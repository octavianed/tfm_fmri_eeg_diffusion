"""Full training-state checkpointing with retention and resume support.

A checkpoint stores the *complete* training state (model, optimizer, scheduler,
GradScaler, epoch, global_step, best metric, early-stopping counter, RNG state,
config and library versions) so an interrupted run resumes exactly — not just
the weights. See spec §17.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .logging import get_logger

logger = get_logger("checkpoint")

_LIB_MODULES = ["torch", "torchvision", "numpy", "sklearn", "transformers",
                "diffusers", "open_clip", "PIL", "scipy", "pandas", "yaml", "mne"]


def collect_library_versions() -> dict:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in _LIB_MODULES:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not-installed"
    return versions


def save_checkpoint(state: dict, path) -> None:
    """Atomically save ``state`` to ``path`` (write tmp then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


class CheckpointManager:
    """Handles saving ``last.pt`` / ``best.pt`` / ``epoch_XXXX.pt`` and retention.

    ``best.pt`` and ``last.pt`` are never pruned; only periodic ``epoch_*.pt``
    checkpoints beyond ``keep_last_n`` are removed. The FIRST ``keep_first_n``
    epochs (0, 1, ...) are always saved and never pruned — useful when early
    checkpoints matter (e.g. the token adapter, whose earliest epochs often
    generate best; see docs/03_...md §6).
    """

    def __init__(self, ckpt_dir, save_last=True, save_best=True,
                 save_every_n_epochs=1, keep_last_n=3, keep_first_n=0,
                 monitor_mode="max"):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_last = bool(save_last)
        self.save_best = bool(save_best)
        self.save_every_n_epochs = int(save_every_n_epochs or 0)
        self.keep_last_n = int(keep_last_n or 0)
        self.keep_first_n = int(keep_first_n or 0)
        self.monitor_mode = monitor_mode

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def epoch_path(self, epoch: int) -> Path:
        return self.dir / f"epoch_{epoch + 1:04d}.pt"

    def save(self, state: dict, epoch: int, is_best: bool = False) -> None:
        if self.save_last:
            save_checkpoint(state, self.last_path)
        periodic = bool(self.save_every_n_epochs
                        and (epoch + 1) % self.save_every_n_epochs == 0)
        first = epoch < self.keep_first_n
        if periodic or first:
            save_checkpoint(state, self.epoch_path(epoch))
            self._prune_epochs()
        if is_best and self.save_best:
            save_checkpoint(state, self.best_path)

    def _prune_epochs(self) -> None:
        if self.keep_last_n <= 0:
            return
        files = sorted(self.dir.glob("epoch_*.pt"), key=self._epoch_of)
        # Never prune the first keep_first_n epochs (filename number == epoch+1,
        # so the first N epochs are those with _epoch_of <= keep_first_n).
        prunable = [p for p in files if self._epoch_of(p) > self.keep_first_n]
        excess = len(prunable) - self.keep_last_n
        for p in prunable[:max(0, excess)]:
            try:
                p.unlink()
            except OSError:  # pragma: no cover
                pass

    @staticmethod
    def _epoch_of(p: Path) -> int:
        m = re.search(r"epoch_(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    def find_resume(self, resume) -> Optional[Path]:
        """Resolve ``resume`` (None | 'auto' | path) to a concrete checkpoint.

        Returns ``None`` when no resume should happen. Raises if an explicit
        path is given but missing.
        """
        if resume in (None, False, "", "null", "none", "None"):
            return None
        if resume in ("auto", True, "True"):
            return self.last_path if self.last_path.exists() else None
        p = Path(resume)
        if not p.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {p}")
        return p
