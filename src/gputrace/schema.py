"""
Unified job-trace schema.

Every loader in `gputrace.loaders` normalizes its source-specific columns into this
schema, so everything downstream (analysis, scenario generation, benchmarking) is
completely source-agnostic. This is the contract that lets you add a new datacenter
trace (Azure Public Dataset, Microsoft Philly, your own private trace, ...) by writing
one small loader, with zero changes anywhere else in the framework.
"""
from __future__ import annotations

import pandas as pd

# Columns every normalized trace DataFrame MUST have.
REQUIRED_COLUMNS = [
    "job_id",       # str/int, unique per job
    "submit_time",  # float seconds since trace start (monotonic non-negative)
    "duration",     # float seconds, > 0
    "num_cpu",      # float, CPU cores/units requested, >= 0
    "num_gpu",      # float, GPU units requested, >= 0 (0 for CPU-only jobs; may be
                     #   fractional for shared-GPU jobs, as in the Alibaba trace)
]

# Columns that are useful but not guaranteed by every source. Loaders should fill
# with a sensible default (empty string / "UNKNOWN" / 0) rather than omit them, so
# the DataFrame shape is always identical across sources.
OPTIONAL_COLUMNS = [
    "user",         # str, submitting user/tenant id
    "gpu_type",     # str, e.g. "V100", "T4", "UNKNOWN"
    "mem",          # float, memory requested (GB), 0 if unknown
    "wait_time",    # float seconds, queueing delay before start, NaN if unknown
    "status",       # str, terminal status e.g. "Terminated", "Failed", "UNKNOWN"
    "source",       # str, which loader/dataset this row came from
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class SchemaValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Validate (and lightly repair) a normalized trace DataFrame.

    - Raises SchemaValidationError if any REQUIRED_COLUMNS are missing.
    - Fills any missing OPTIONAL_COLUMNS with defaults so shape is always consistent.
    - If strict=True, also enforces value constraints (duration>0, submit_time>=0,
      num_cpu/num_gpu>=0) and raises on violation instead of silently dropping rows.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"missing required columns: {missing}")

    df = df.copy()
    defaults = {"user": "UNKNOWN", "gpu_type": "UNKNOWN", "mem": 0.0,
                "wait_time": float("nan"), "status": "UNKNOWN", "source": "UNKNOWN"}
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = defaults[c]

    if strict:
        bad = df[(df["duration"] <= 0) | (df["submit_time"] < 0) |
                  (df["num_cpu"] < 0) | (df["num_gpu"] < 0)]
        if len(bad) > 0:
            raise SchemaValidationError(
                f"{len(bad)} rows violate value constraints "
                f"(duration>0, submit_time>=0, num_cpu/num_gpu>=0)")

    return df[ALL_COLUMNS]
