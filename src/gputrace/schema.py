"""
Unified trace schema.

Every loader (real trace source) and every generator (synthetic scenario)
produces / consumes a pandas DataFrame with exactly this column set, so
analysis code and downstream consumers (simulators, other tools) never need
to know which data source or scenario produced a given trace.

Columns
-------
job_id       : str    unique identifier for the job within the trace
submit_time  : float  seconds since the start of the trace, when the job
                       was submitted / arrived at the scheduler
duration     : float  seconds of actual runtime, once running (excludes
                       queueing delay)
num_cpu      : float  number of (v)CPU cores requested
num_gpu      : float  number of GPU *devices* requested (whole count; see
                       ``gpu_milli`` in the extended schema notes below for
                       fractional-GPU export)
user         : str    tenant / submitting-user identifier
gpu_type     : str    requested GPU SKU, e.g. "V100", "A100", "T4", or
                       "" / "none" for CPU-only jobs
mem          : float  memory requested, in GiB
wait_time    : float  seconds between submit_time and actual start (queueing
                       delay). 0.0 for jobs that ran immediately.
status       : str    terminal status: "completed", "failed", "killed"
source       : str    which loader/scenario produced this row, e.g.
                       "alibaba2020" or "bursty_arrivals"

Fractional-GPU extension (optional column)
-------------------------------------------
gpu_milli    : float  OPTIONAL. Fraction of a single GPU device requested,
                       in milli-units (0-1000 per device requested), for
                       scenarios/loaders that model GPU sharing. When absent,
                       consumers should assume whole-device requests
                       (gpu_milli = 1000 per device in num_gpu).

This column is intentionally optional rather than mandatory: most real
cluster traces (Alibaba 2020, Google 2011) do not report sub-device sharing,
so making it required would force every loader to fabricate data it doesn't
have. Generators that DO model sharing (see
``generators/stress_scenarios.py::gpu_fragmentation_sharing``) populate it;
everything else leaves it unset and downstream consumers fall back to the
whole-device assumption. See ``exporters/k8s_sim.py`` for how a consumer
should handle both cases.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "job_id",
    "submit_time",
    "duration",
    "num_cpu",
    "num_gpu",
    "user",
    "gpu_type",
    "mem",
    "wait_time",
    "status",
    "source",
]

OPTIONAL_COLUMNS = ["gpu_milli"]

NUMERIC_COLUMNS = ["submit_time", "duration", "num_cpu", "num_gpu", "mem", "wait_time"]

VALID_STATUSES = {"completed", "failed", "killed"}


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the unified trace schema."""


def validate(df: pd.DataFrame, *, strict: bool = True) -> None:
    """Validate a DataFrame against the unified schema.

    Parameters
    ----------
    df : the DataFrame to check
    strict : if True (default), raise on any violation. If False, only
        raise on missing required columns; coerce/ignore softer issues
        (unknown status values, negative-but-fixable numerics) instead of
        raising, which is useful when validating messy real-world trace
        input before it has been cleaned by a loader's ``_normalize()``.

    Raises
    ------
    SchemaError on any violation (missing columns always raise regardless
    of ``strict``).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaError(f"column {col!r} must be numeric, got dtype {df[col].dtype}")
        if strict and (df[col] < 0).any():
            bad = df.loc[df[col] < 0, col].index.tolist()[:5]
            raise SchemaError(f"column {col!r} has negative values at rows {bad}")

    if strict:
        bad_status = set(df["status"].unique()) - VALID_STATUSES
        if bad_status:
            raise SchemaError(f"unknown status values: {bad_status}, expected one of {VALID_STATUSES}")

    if df["job_id"].duplicated().any():
        dupes = df.loc[df["job_id"].duplicated(), "job_id"].tolist()[:5]
        raise SchemaError(f"duplicate job_id values: {dupes}")


def empty_frame() -> pd.DataFrame:
    """An empty, correctly-typed unified-schema DataFrame. Useful as a
    starting point for generators/loaders and in tests."""
    cols = {
        "job_id": pd.Series(dtype="object"),
        "submit_time": pd.Series(dtype="float64"),
        "duration": pd.Series(dtype="float64"),
        "num_cpu": pd.Series(dtype="float64"),
        "num_gpu": pd.Series(dtype="float64"),
        "user": pd.Series(dtype="object"),
        "gpu_type": pd.Series(dtype="object"),
        "mem": pd.Series(dtype="float64"),
        "wait_time": pd.Series(dtype="float64"),
        "status": pd.Series(dtype="object"),
        "source": pd.Series(dtype="object"),
        "gpu_milli": pd.Series(dtype="float64"),
    }
    return pd.DataFrame(cols)
