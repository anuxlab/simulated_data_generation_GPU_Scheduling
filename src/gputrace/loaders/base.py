"""
Loader base class and registry.

Adding a new data source is exactly one file: subclass ``BaseLoader``,
implement ``_load_raw()`` (read the file(s) into whatever shape is natural
for the source) and ``_normalize()`` (map that shape onto the unified
schema in ``gputrace.schema``), and decorate the class with
``@register("your_source_name")``. Nothing else in the framework needs to
change — ``analyze()``, the CLI, and every generator only ever see the
unified schema.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Type

import pandas as pd

from .. import schema

_REGISTRY: Dict[str, Type["BaseLoader"]] = {}


def register(name: str):
    """Class decorator: register a BaseLoader subclass under ``name``."""

    def _decorator(cls: Type["BaseLoader"]) -> Type["BaseLoader"]:
        if name in _REGISTRY:
            raise ValueError(f"loader {name!r} already registered by {_REGISTRY[name]!r}")
        cls.loader_name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_loader(name: str) -> "BaseLoader":
    """Instantiate the loader registered under ``name``."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown loader {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_loaders() -> list[str]:
    return sorted(_REGISTRY)


class BaseLoader(abc.ABC):
    """Subclass this to add a new trace data source.

    Contract
    --------
    ``load(path)`` is the only public entry point and is provided for you —
    it calls ``_load_raw()`` then ``_normalize()`` then validates the result
    against the unified schema before returning it. Subclasses only
    implement the two private hooks.
    """

    loader_name: str = "base"

    def load(self, path: str | Path, **kwargs) -> pd.DataFrame:
        raw = self._load_raw(Path(path), **kwargs)
        df = self._normalize(raw)
        df["source"] = self.loader_name
        schema.validate(df, strict=True)
        return df

    @abc.abstractmethod
    def _load_raw(self, path: Path, **kwargs):
        """Read the raw file(s) at ``path`` into whatever shape is natural
        for this source (a DataFrame, dict of DataFrames, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _normalize(self, raw) -> pd.DataFrame:
        """Map the raw shape onto the unified schema's column set. Must
        return a DataFrame with every column in
        ``gputrace.schema.REQUIRED_COLUMNS`` except ``source`` (added by
        ``load()``)."""
        raise NotImplementedError
