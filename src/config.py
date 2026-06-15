"""Minimal YAML config loader with dotted access and deep-merge of overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access: cfg.seed as well as cfg['seed']."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(*paths: str | Path) -> Config:
    """Load and deep-merge one or more YAML files (later files win)."""
    merged: dict = {}
    for p in paths:
        with open(p) as f:
            merged = _merge(merged, yaml.safe_load(f) or {})
    return Config(merged)
