"""
Loader for the Google cluster trace, 2011-2 schema
(https://github.com/google/cluster-data).

This trace predates public GPU accounting, so it's included as a
differently-shaped CPU-only reference workload, not a GPU data source.
``num_gpu`` and ``gpu_type`` are always zero/empty for rows from this
loader — analysis code that segments by ``gpu_type`` will correctly show
this source as 100% CPU-only.

Expected raw columns (task events table, subset used):
    time, job_id, task_index, event_type, user, cpu_request, mem_request
Timestamps are microseconds since trace start, per Google's own docs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, register

# task event types, per the public schema documentation
_SUBMIT = 0
_SCHEDULE = 1
_FINISH = 4
_FAIL = 3
_KILL = 5


@register("google2011")
class GoogleClusterLoader(BaseLoader):
    def _load_raw(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path)

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["job_id"] = df["job_id"].astype(str) + "_" + df["task_index"].astype(str)

        submits = df[df["event_type"] == _SUBMIT].groupby("job_id")["time"].min()
        schedules = df[df["event_type"] == _SCHEDULE].groupby("job_id")["time"].min()
        ends = df[df["event_type"].isin([_FINISH, _FAIL, _KILL])].groupby("job_id")["time"].max()
        terminal_type = (
            df[df["event_type"].isin([_FINISH, _FAIL, _KILL])]
            .groupby("job_id")["event_type"]
            .last()
        )

        meta = (
            df.sort_values("time")
            .drop_duplicates("job_id")
            .set_index("job_id")[["user", "cpu_request", "mem_request"]]
        )

        job_ids = submits.index.intersection(ends.index)
        t0 = submits.min()

        micros_per_sec = 1_000_000
        submit_time = (submits.loc[job_ids] - t0) / micros_per_sec
        end_time = (ends.loc[job_ids] - t0) / micros_per_sec
        sched_time = schedules.reindex(job_ids)
        wait_time = ((sched_time - submits.loc[job_ids]) / micros_per_sec).fillna(0).clip(lower=0)
        duration = (end_time - submit_time - wait_time).clip(lower=0)

        status_map = {_FINISH: "completed", _FAIL: "failed", _KILL: "killed"}
        status = terminal_type.reindex(job_ids).map(status_map).fillna("completed")

        out = pd.DataFrame(
            {
                "job_id": job_ids,
                "submit_time": submit_time.values,
                "duration": duration.values,
                "num_cpu": meta.reindex(job_ids)["cpu_request"].fillna(0).values,
                "num_gpu": np.zeros(len(job_ids)),
                "user": meta.reindex(job_ids)["user"].fillna("unknown").astype(str).values,
                "gpu_type": [""] * len(job_ids),
                "mem": meta.reindex(job_ids)["mem_request"].fillna(0).values,
                "wait_time": wait_time.values,
                "status": status.values,
            }
        )
        return out.reset_index(drop=True)
