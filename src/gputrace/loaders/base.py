"""Abstract base class every dataset loader implements, plus a small registry so new
sources can be plugged in with `@register("name")` and discovered by name from the CLI."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..schema import validate

_REGISTRY: dict[str, type["BaseLoader"]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_loader(name: str) -> "BaseLoader":
    if name not in _REGISTRY:
        raise KeyError(f"unknown loader '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_loaders() -> list[str]:
    return sorted(_REGISTRY)


class BaseLoader(ABC):
    """Subclass and implement `_load_raw(path)` to return a DataFrame with your
    source's native columns; implement `_normalize(raw)` to map them onto the
    unified schema in gputrace.schema. `load()` glues the two together and
    validates the result -- that's the only method the rest of the framework calls."""

    name: str = "base"

    @abstractmethod
    def _load_raw(self, path: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        ...

    def load(self, path: str, strict: bool = False) -> pd.DataFrame:
        raw = self._load_raw(path)
        norm = self._normalize(raw)
        norm["source"] = self.name
        return validate(norm, strict=strict)
