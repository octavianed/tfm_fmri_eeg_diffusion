"""Resolve the ``subject_selection`` config into a concrete list of subjects.

Selection may be:
  * a single subject id  -> ``"subj01"``
  * a list of subject ids -> ``["subj01", "subj02"]``
  * the string ``"all"``  -> every ``subjNN`` folder found under ``root_dir``

This is the single place that decides which subjects a run uses, so no other
module hard-codes single-subject assumptions (spec §3.2).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

Selection = Union[str, Sequence[str]]

_SUBJ_RE = re.compile(r"^subj\d+$", re.IGNORECASE)


def _normalize_id(name: str) -> str:
    """Normalize e.g. 'SUBJ1' / 'subj_01' / 'subj1' -> 'subj01'."""
    m = re.search(r"(\d+)", name)
    if not m:
        return name.lower()
    return f"subj{int(m.group(1)):02d}"


def discover_subjects(root_dir) -> List[str]:
    """List all ``subjNN`` subdirectories under ``root_dir`` (sorted)."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root_dir does not exist: {root}")
    found = []
    for child in root.iterdir():
        if child.is_dir() and _SUBJ_RE.match(child.name):
            found.append(_normalize_id(child.name))
    return sorted(set(found))


def resolve_subjects(root_dir, selection: Selection) -> List[str]:
    """Return the concrete, validated, sorted list of subject ids."""
    available = discover_subjects(root_dir)
    if isinstance(selection, str) and selection.lower() == "all":
        if not available:
            raise RuntimeError(f"No 'subjNN' folders found under {root_dir}")
        return available

    if isinstance(selection, str):
        requested = [selection]
    else:
        requested = list(selection)

    requested = [_normalize_id(s) for s in requested]
    if available:
        missing = [s for s in requested if s not in available]
        if missing:
            raise ValueError(
                f"Requested subjects {missing} not found under {root_dir}. "
                f"Available: {available}")
    return sorted(set(requested))


def selection_tag(selection: Selection) -> str:
    """Filesystem-safe tag describing the selection (for metadata filenames)."""
    if isinstance(selection, str):
        return "all" if selection.lower() == "all" else _normalize_id(selection)
    ids = sorted({_normalize_id(s) for s in selection})
    return "-".join(ids) if ids else "none"


def subject_dir(root_dir, subject: str) -> Path:
    """Path to a subject's folder, tolerating a couple of naming variants."""
    root = Path(root_dir)
    for candidate in (subject, subject.upper(), subject.replace("subj", "subj0")):
        p = root / candidate
        if p.exists():
            return p
    # fall back to a case-insensitive scan
    for child in root.iterdir():
        if child.is_dir() and _normalize_id(child.name) == _normalize_id(subject):
            return child
    return root / subject
