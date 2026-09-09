"""
Builds a small built-in reference ``AnalysisResult`` by fitting to a
hand-specified bootstrap sample (order-of-magnitude realistic for a shared
GPU cluster, loosely calibrated to publicly reported Alibaba/Google trace
statistics) rather than any single real trace file.

This exists so ``gputrace generate ...`` and the test suite work with zero
external data files. It is deliberately NOT presented as a substitute for
fitting your own production trace — the whole point of the loader
architecture is to fit real, source-specific heterogeneity. Treat this as a
smoke-test / demo fixture, not a benchmark input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis import analyze
from ..schema import empty_frame


def _bootstrap_dataframe(seed: int = 12345, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    duration = rng.lognormal(mean=5.2, sigma=1.4, size=n)          # ~seconds-to-hours, heavy right tail
    num_cpu = np.round(rng.gamma(shape=2.0, scale=2.0, size=n)) + 1
    num_gpu = rng.choice([0, 1, 2, 4, 8], size=n, p=[0.35, 0.35, 0.15, 0.1, 0.05])
    mem = rng.lognormal(mean=2.3, sigma=0.7, size=n)
    wait_time = rng.exponential(scale=45.0, size=n)
    submit_time = np.sort(rng.exponential(scale=3.0, size=n).cumsum())
    gpu_type = np.where(num_gpu > 0, rng.choice(["A100", "V100", "T4"], size=n), "")
    status = rng.choice(["completed", "failed", "killed"], size=n, p=[0.9, 0.06, 0.04])
    user = rng.integers(0, 150, size=n)

    df = empty_frame()
    df = pd.DataFrame(
        {
            "job_id": [f"ref_{i:06d}" for i in range(n)],
            "submit_time": submit_time,
            "duration": duration,
            "num_cpu": num_cpu,
            "num_gpu": num_gpu,
            "user": [f"user_{u:04d}" for u in user],
            "gpu_type": gpu_type,
            "mem": mem,
            "wait_time": wait_time,
            "status": status,
            "source": "builtin_reference",
        }
    )
    return df


def build_reference_fit():
    return analyze(_bootstrap_dataframe())
