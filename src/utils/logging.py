"""Logging utilities: console logger, append-only CSV metric log, JSONL history."""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "fmri", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class CSVLogger:
    """Append-only CSV logger that preserves history across resumes.

    If the file already exists with a header, new rows are appended without
    rewriting the header — previous metrics are never lost when resuming.
    """

    def __init__(self, path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if not exists:
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self.fieldnames).writeheader()

    def log(self, row: dict) -> None:
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writerow(clean)


class JsonlLogger:
    """Append one JSON object per line (used for resume history)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def save_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
