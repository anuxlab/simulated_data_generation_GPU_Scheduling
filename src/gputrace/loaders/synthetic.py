"""Pass-through loader for CSVs already in gputrace's unified schema -- i.e. traces
produced by gputrace's own scenario generator, or any external file you've already
mapped by hand. Lets `gputrace analyze` / `gputrace fit` work uniformly on real and
synthetic traces alike."""
from __future__ import annotations

import pandas as pd

from ..schema import REQUIRED_COLUMNS
from .base import BaseLoader, register


@register("synthetic")
class SyntheticLoader(BaseLoader):
    name = "synthetic"

    def _load_raw(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"synthetic loader expects unified-schema columns, missing: {missing}")
        return raw
