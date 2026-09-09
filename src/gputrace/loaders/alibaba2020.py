"""
Loader for the Alibaba cluster-trace-gpu-v2020 release
(https://github.com/alibaba/clusterdata).

Accepts either:
  * the official 100K-job simulator sample (``pai_task_table_sample.csv``),
    or
  * the full ``pai_task_table`` release

Both ship with the same column layout; we key off column presence rather
than filename so either works transparently.

Expected raw columns (subset actually used):
    job_name, task_name, inst_num, status, start_time, end_time,
    plan_cpu, plan_mem, plan_gpu, gpu_type, user
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, register

_STATUS_MAP = {
    "Terminated": "completed",
    "Failed": "failed",
    "Killed": "killed",
    "Running": "completed",  # snapshot traces may catch jobs mid-flight
    "Waiting": "killed",     # never actually ran within the trace window
}


@register("alibaba2020")
class Alibaba2020Loader(BaseLoader):
    def _load_raw(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()

        # The public release uses plan_gpu as a PERCENTAGE of one device
        # (0-100, sometimes >100 for multi-GPU tasks packed into inst_num).
        # Convert to device count.
        plan_gpu = pd.to_numeric(df.get("plan_gpu", 0), errors="coerce").fillna(0)
        num_gpu = (plan_gpu / 100.0).clip(lower=0)

        inst_num = pd.to_numeric(df.get("inst_num", 1), errors="coerce").fillna(1).clip(lower=1)

        start = pd.to_numeric(df["start_time"], errors="coerce")
        end = pd.to_numeric(df["end_time"], errors="coerce")
        duration = (end - start).clip(lower=0)

        t0 = start.min()
        submit_time = (start - t0).fillna(0)

        status = df.get("status", "Terminated").map(_STATUS_MAP).fillna("completed")

        out = pd.DataFrame(
            {
                "job_id": df["job_name"].astype(str) + "_" + df["task_name"].astype(str),
                "submit_time": submit_time,
                "duration": duration.fillna(0),
                "num_cpu": pd.to_numeric(df.get("plan_cpu", 0), errors="coerce").fillna(0) / 100.0 * inst_num,
                "num_gpu": num_gpu * inst_num,
                "user": df.get("user", "unknown").astype(str),
                "gpu_type": df.get("gpu_type", "").fillna("").astype(str),
                "mem": pd.to_numeric(df.get("plan_mem", 0), errors="coerce").fillna(0) / 100.0 * inst_num,
                "wait_time": np.zeros(len(df)),  # not directly reported; 0 baseline
                "status": status,
            }
        )
        out = out.dropna(subset=["submit_time", "duration"]).reset_index(drop=True)
        out["job_id"] = out["job_id"] + "_" + out.index.astype(str)  # guarantee uniqueness
        return out
