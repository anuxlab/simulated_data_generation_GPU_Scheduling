"""
Passthrough loader for traces already in gputrace's unified schema —
including traces this framework generated itself. Lets you run
``gputrace analyze`` on a synthetic scenario's output to sanity-check that
its distributional properties match what the scenario claims to stress
(e.g. confirm ``heavy_tail_demand`` output actually has a fatter tail than
the fitted source).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import schema
from .base import BaseLoader, register


@register("synthetic")
class SyntheticLoader(BaseLoader):
    def _load_raw(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        missing = [c for c in schema.REQUIRED_COLUMNS if c != "source" and c not in df.columns]
        if missing:
            raise schema.SchemaError(
                f"synthetic loader expects a unified-schema CSV; missing columns: {missing}"
            )
        return df.drop(columns=["source"], errors="ignore")
