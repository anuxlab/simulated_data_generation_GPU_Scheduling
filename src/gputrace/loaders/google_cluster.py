"""
Loader for Google's public cluster trace, `task_events` table
(github.com/google/cluster-data, clusterdata-2011-2 schema):

    timestamp, missing_info, job_ID, task_index, machine_ID, event_type, user,
    scheduling_class, priority, CPU_request, memory_request, disk_space_request,
    different_machine_constraint

Notes / honest limitations (documented rather than silently guessed):
- This trace predates GPU accounting in the public release, so `num_gpu` is always 0
  and `gpu_type` is "NONE" for every row -- it's included in this framework as a
  **CPU-only reference workload** to stress-test how a GPU-scheduling algorithm
  degrades/behaves on a very differently-shaped arrival & sizing distribution than
  the GPU-heavy Alibaba trace, not as a source of real GPU demand data.
- CPU_request/memory_request are pre-normalized by Google to [0, 1] as a fraction of
  the largest machine's capacity in the cluster; we rescale by `cpu_scale`/`mem_scale`
  (constructor args, default 64 cores / 256 GB) to get unit-comparable numbers across
  sources. Adjust these if you know the actual machine shapes for your trace cell.
- event_type codes: 0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, 5=KILL, 6=LOST,
  7=UPDATE_PENDING, 8=UPDATE_RUNNING. Duration = first SUBMIT -> last terminal event
  (EVICT/FAIL/FINISH/KILL/LOST) for the same (job_ID, task_index).
"""
from __future__ import annotations

import pandas as pd

from .base import BaseLoader, register

_TERMINAL = {2, 3, 4, 5, 6}
_COLS = ["timestamp", "missing_info", "job_ID", "task_index", "machine_ID",
         "event_type", "user", "scheduling_class", "priority", "CPU_request",
         "memory_request", "disk_space_request", "different_machine_constraint"]


@register("google2011")
class GoogleClusterLoader(BaseLoader):
    name = "google2011"

    def __init__(self, cpu_scale: float = 64.0, mem_scale: float = 256.0):
        self.cpu_scale = cpu_scale
        self.mem_scale = mem_scale

    def _load_raw(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=_COLS)
        return df

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.dropna(subset=["job_ID", "task_index", "timestamp"]).copy()
        raw["task_key"] = raw["job_ID"].astype(str) + "_" + raw["task_index"].astype(str)

        submits = (raw[raw["event_type"] == 0]
                   .sort_values("timestamp")
                   .drop_duplicates("task_key", keep="first")
                   .set_index("task_key"))
        terminals = (raw[raw["event_type"].isin(_TERMINAL)]
                     .sort_values("timestamp")
                     .drop_duplicates("task_key", keep="last")
                     .set_index("task_key"))

        joined = submits.join(terminals[["timestamp"]], rsuffix="_end", how="inner")
        joined["duration"] = joined["timestamp_end"] - joined["timestamp"]
        joined = joined[joined["duration"] > 0]

        out = pd.DataFrame({
            "job_id": joined.index.to_series().values,
            "submit_time": joined["timestamp"].values,
            "duration": joined["duration"].values,
            "num_cpu": joined["CPU_request"].fillna(0).values * self.cpu_scale,
            "num_gpu": 0.0,
            "gpu_type": "NONE",
            "mem": joined["memory_request"].fillna(0).values * self.mem_scale,
            "user": joined["user"].fillna("UNKNOWN").astype(str).values,
            "wait_time": float("nan"),
            "status": "UNKNOWN",
        })
        return out
