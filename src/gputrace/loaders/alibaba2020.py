"""
Loader for Alibaba's cluster-trace-gpu-v2020 (github.com/alibaba/clusterdata).

Handles two input shapes, auto-detected by column presence:
1. The official simulator sample (`pai_job_duration_estimate_100K.csv` or
   `pai_job_no_estimate_100K.csv`) -- job-level rows, already close to our schema.
2. The full multi-table release's `pai_task_table.csv` (job_name, task_name, inst_num,
   status, start_time, end_time, plan_cpu, plan_mem, plan_gpu, gpu_type) -- task-level
   rows that get aggregated to job level (sum over tasks belonging to the same job_name).

plan_cpu/plan_gpu in the full release are in percent-of-one-unit (e.g. 100 = 1 core);
we convert to units to match the sample format.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseLoader, register


@register("alibaba2020")
class Alibaba2020Loader(BaseLoader):
    name = "alibaba2020"

    def _load_raw(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        cols = set(raw.columns)

        if {"job_id", "submit_time", "duration", "num_cpu", "num_gpu"}.issubset(cols):
            # already job-level (the simulator sample format)
            df = raw.copy()
            df["job_id"] = df["job_id"].astype(str)
            if "wait_time" not in df.columns:
                df["wait_time"] = float("nan")
            if "gpu_type" not in df.columns:
                df["gpu_type"] = "UNKNOWN"
            return df

        if {"job_name", "task_name", "start_time", "end_time", "plan_cpu", "plan_gpu"}.issubset(cols):
            # full-release task table -> aggregate to job level
            raw = raw.dropna(subset=["start_time", "end_time"]).copy()
            raw["duration"] = raw["end_time"] - raw["start_time"]
            raw = raw[raw["duration"] > 0]
            agg = raw.groupby("job_name").agg(
                submit_time=("start_time", "min"),
                end_time=("end_time", "max"),
                num_cpu=("plan_cpu", "sum"),
                num_gpu=("plan_gpu", "sum"),
                gpu_type=("gpu_type", lambda s: s.dropna().mode().iat[0] if s.dropna().size else "UNKNOWN"),
                status=("status", lambda s: s.mode().iat[0] if s.size else "UNKNOWN"),
            ).reset_index()
            agg["duration"] = agg["end_time"] - agg["submit_time"]
            agg["num_cpu"] = agg["num_cpu"] / 100.0   # plan_cpu is in % of a core
            agg["num_gpu"] = agg["num_gpu"] / 100.0   # plan_gpu is in % of a GPU
            agg = agg.rename(columns={"job_name": "job_id"})
            agg["job_id"] = agg["job_id"].astype(str)
            agg["wait_time"] = float("nan")
            return agg[["job_id", "submit_time", "duration", "num_cpu", "num_gpu",
                        "gpu_type", "status", "wait_time"]]

        raise ValueError(
            f"unrecognized alibaba2020 file shape, columns={sorted(cols)}. "
            "expected either the simulator job-sample columns or pai_task_table columns."
        )
