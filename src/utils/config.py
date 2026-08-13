"""Configuration loading, merging, and CLI override utilities.

Configs are plain YAML files. An experiment config may declare a list of base
configs via the reserved key ``_base_`` (relative paths, resolved against the
config file's directory). Bases are merged left-to-right, then the current file
is overlaid on top. CLI overrides (``--set a.b=c``) are applied last.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class DotDict(dict):
    """Dict with attribute access and recursive wrapping.

    Nested dicts are wrapped lazily on read, so ``cfg.a.b`` and ``cfg["a"]["b"]``
    are interchangeable. ``cfg.get("a.b.c", default)`` resolves dotted paths.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc

    def get(self, path: str, default: Any = None) -> Any:
        """Nested get by dotted path, e.g. ``cfg.get("training.lr", 1e-4)``."""
        node: Any = self
        for part in str(path).split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        if isinstance(node, dict) and not isinstance(node, DotDict):
            node = DotDict(node)
        return node

    def to_dict(self) -> dict:
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out: dict = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (key in out and isinstance(out[key], Mapping)
                and isinstance(value, Mapping)):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: "str | os.PathLike") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _resolve_bases(raw: Mapping, config_dir: Path) -> dict:
    bases = raw.get("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base_rel in bases:
        base_path = (config_dir / base_rel).resolve()
        base_raw = load_yaml(base_path)
        base_resolved = _resolve_bases(base_raw, base_path.parent)
        merged = deep_merge(merged, base_resolved)
    current = {k: v for k, v in raw.items() if k != "_base_"}
    return deep_merge(merged, current)


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict, overrides: "Iterable[str] | None") -> dict:
    """Apply ``key.path=value`` style overrides in place and return cfg."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of form key.path=value")
        key, raw_value = item.split("=", 1)
        value = _coerce(raw_value)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set '{key}': '{part}' is not a mapping")
        node[parts[-1]] = value
    return cfg


class ExtendOverrides(argparse.Action):
    """argparse action that ACCUMULATES ``--set`` values across repetitions.

    With a plain ``nargs="*"`` argument, ``--set a=1 --set b=2`` silently keeps
    only the LAST occurrence and drops every earlier override without warning —
    a nasty failure mode, because the run then quietly uses the wrong dataset or
    experiment name. This action collects both supported spellings:

        --set a=1 b=2          (one flag, several pairs)
        --set a=1 --set b=2    (repeated flag)
    """

    def __call__(self, parser, namespace, values, option_string=None):
        current = list(getattr(namespace, self.dest, None) or [])
        current.extend(values if isinstance(values, (list, tuple)) else [values])
        setattr(namespace, self.dest, current)


def add_override_arg(parser, flag: str = "--set", help_extra: str = "") -> None:
    """Register the standard ``--set key.path=value`` override flag on a parser."""
    parser.add_argument(
        flag, nargs="*", action=ExtendOverrides, default=None,
        help=("config overrides as key.path=value; repeatable and/or "
              "space-separated, e.g. --set training.lr=5e-5 dataset.channels=63"
              + (f". {help_extra}" if help_extra else "")))


def load_config(config_path, overrides: "Iterable[str] | None" = None) -> DotDict:
    """Load a YAML config, resolving ``_base_`` inheritance and CLI overrides."""
    config_path = Path(config_path).resolve()
    raw = load_yaml(config_path)
    resolved = _resolve_bases(raw, config_path.parent)
    resolved = apply_overrides(resolved, overrides)
    resolved.setdefault("_config_path", str(config_path))
    return DotDict(resolved)


def save_config(cfg, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, DotDict) else _to_plain(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
