"""
Export a unified-schema trace into the input format consumed by
``py_sim``'s ``k8s_sim`` package (see ``k8s_sim/gputrace_bridge.py`` on that
side).

This is the concrete answer to "how does gputrace plug into py_sim": the
unified schema is a job-arrival trace with no cluster topology, and
py_sim's scheduler needs both jobs *and* a cluster to place them on. This
module produces both halves as two files:

    pods.csv   one row per job, in milli-unit resource fields plus the
               submit_time/duration columns py_sim's event-driven runner
               (``k8s_sim/event_runtime.py``) consumes to drive arrivals
               and releases over simulated time.
    nodes.csv  a synthesized cluster sized to make the scenario's total
               resource demand land at a target average utilization — see
               ``_synthesize_nodes`` for exactly how, and why this is a
               starting point to tune rather than a ground truth.

Fractional-GPU handling
------------------------
If the input trace has a populated ``gpu_milli`` column (e.g. output of the
``gpu_fragmentation_sharing`` or ``llm_inference_bursty`` scenarios), that
value is used directly. Otherwise every requested GPU device is exported as
a full 1000 milli-units — see ``schema.py``'s note on why gpu_milli is
optional rather than mandatory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pods_frame(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if "gpu_milli" in df.columns and df["gpu_milli"].notna().any():
        gpu_milli_per_device = df["gpu_milli"].fillna(1000.0)
    else:
        gpu_milli_per_device = pd.Series(np.full(n, 1000.0), index=df.index)

    return pd.DataFrame(
        {
            "pod_id": df["job_id"],
            "submit_time": df["submit_time"],
            "duration": df["duration"],
            "milli_cpu": (df["num_cpu"] * 1000).round().astype(int),
            "milli_gpu": (df["num_gpu"].clip(lower=0) * gpu_milli_per_device).round().astype(int),
            "gpu_number": df["num_gpu"].clip(lower=0).round().astype(int),
            "gpu_type": df["gpu_type"].fillna(""),
            "user": df["user"],
        }
    ).sort_values("submit_time").reset_index(drop=True)


def _synthesize_nodes(df: pd.DataFrame, n_nodes: int, gpus_per_node: int,
                       target_utilization: float = 0.6) -> pd.DataFrame:
    """Size a homogeneous-ish cluster from the trace's own demand.

    There is no "correct" topology implied by a job-arrival trace alone —
    the same demand can be served by many nodes with small GPUs or few
    nodes with big GPUs. This picks node *count* and *GPU type mix* to
    roughly match the trace's requested GPU-type distribution, and sizes
    per-node CPU capacity so that, if every job in the trace were resident
    simultaneously (a deliberately pessimistic upper bound, not the actual
    expected concurrency), the cluster would be at ``target_utilization``.
    Treat this as a reasonable starting point to run experiments from, not
    a fitted or validated cluster spec — tune ``n_nodes`` /
    ``gpus_per_node`` per-experiment to hit the contention level you
    actually want to test.
    """
    gpu_type_counts = df.loc[df["num_gpu"] > 0, "gpu_type"].value_counts()
    if gpu_type_counts.empty:
        gpu_type_counts = pd.Series({"A100": 1})
    types = gpu_type_counts.index.tolist()
    type_weights = (gpu_type_counts / gpu_type_counts.sum()).tolist()

    total_gpu_demand = df["num_gpu"].sum()
    total_cpu_demand = df["num_cpu"].sum()
    total_gpu_capacity = max(total_gpu_demand / max(target_utilization, 1e-6), gpus_per_node)
    n_nodes_sized = max(1, int(np.ceil(total_gpu_capacity / gpus_per_node)))
    n_nodes = n_nodes if n_nodes else n_nodes_sized

    cpu_per_node = max(4, int(np.ceil((total_cpu_demand / max(target_utilization, 1e-6)) / n_nodes)))

    rng = np.random.default_rng(0)
    assigned_types = rng.choice(types, size=n_nodes, p=type_weights)

    return pd.DataFrame(
        {
            "node_id": [f"node_{i:04d}" for i in range(n_nodes)],
            "milli_cpu_capacity": cpu_per_node * 1000,
            "gpu_count": gpus_per_node,
            "milli_gpu_capacity": gpus_per_node * 1000,
            "gpu_type": assigned_types,
        }
    )


def export_scenario(df: pd.DataFrame, out_dir: Path, n_nodes: int = 0, gpus_per_node: int = 8,
                     target_utilization: float = 0.6) -> None:
    """Write ``pods.csv`` and ``nodes.csv`` for ``py_sim/k8s_sim`` under
    ``out_dir``.

    Parameters
    ----------
    df : unified-schema DataFrame (a gputrace scenario or loader output)
    out_dir : directory to write pods.csv / nodes.csv into (created if needed)
    n_nodes : fixed node count, or 0 to size automatically from demand (see
        ``_synthesize_nodes``)
    gpus_per_node : GPUs per synthesized node
    target_utilization : utilization the auto-sized cluster targets under
        the (pessimistic) all-jobs-concurrent assumption
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _pods_frame(df).to_csv(out_dir / "pods.csv", index=False)
    _synthesize_nodes(df, n_nodes, gpus_per_node, target_utilization).to_csv(out_dir / "nodes.csv", index=False)
