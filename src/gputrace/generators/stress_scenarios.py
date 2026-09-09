"""
Built-in stress-test scenarios. Each takes a fitted ``AnalysisResult``
(wrapped in a ``WorkloadScenarioGenerator``) plus a job count and
scenario-specific knobs, and returns a unified-schema DataFrame.

Every scenario shares the same *marginal* distributions (duration, CPU/GPU
demand) fitted from whatever source trace you pointed ``gputrace analyze``
at — what differs between scenarios is the *arrival process* and, for a
few, the resource-request *correlation structure*. That's deliberate: it
isolates "does scheduler X handle bursty arrival dynamics" from "does
scheduler X handle a different job-size mix", which would otherwise be
confounded if scenarios also silently changed the size distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..analysis import AnalysisResult
from .engine import WorkloadScenarioGenerator, register_scenario

_GPU_TYPES = ["A100", "V100", "T4", "H100"]


def _base_frame(gen: WorkloadScenarioGenerator, submit_times: np.ndarray, source: str,
                 gpu_type_weights: Optional[dict] = None) -> pd.DataFrame:
    n = len(submit_times)
    duration = gen.sample_field("duration", n)
    num_cpu = np.maximum(1, np.round(gen.sample_field("num_cpu", n)))
    num_gpu_raw = gen.sample_field("num_gpu", n)
    num_gpu = np.round(num_gpu_raw)
    mem = gen.sample_field("wait_time", n) * 0 + gen.rng.lognormal(mean=2.0, sigma=0.6, size=n)  # GiB
    wait_time = np.zeros(n)  # ground-truth queueing delay is for the simulator to produce, not the generator

    gpu_types = np.array(_GPU_TYPES)
    if gpu_type_weights is None:
        weights = np.ones(len(gpu_types)) / len(gpu_types)
    else:
        weights = np.array([gpu_type_weights.get(g, 0) for g in gpu_types])
        weights = weights / weights.sum()
    chosen_type = gen.rng.choice(gpu_types, size=n, p=weights)
    chosen_type = np.where(num_gpu > 0, chosen_type, "")

    users = np.array([f"user_{i:04d}" for i in gen.rng.integers(0, 200, size=n)])
    status = gen.rng.choice(
        ["completed", "failed", "killed"], size=n, p=[0.9, 0.06, 0.04]
    )

    df = pd.DataFrame(
        {
            "job_id": [f"{source}_{i:08d}" for i in range(n)],
            "submit_time": submit_times,
            "duration": duration,
            "num_cpu": num_cpu,
            "num_gpu": num_gpu,
            "user": users,
            "gpu_type": chosen_type,
            "mem": mem,
            "wait_time": wait_time,
            "status": status,
            "source": source,
        }
    )
    return df.sort_values("submit_time").reset_index(drop=True)


@register_scenario("baseline", "reference: homogeneous Poisson arrivals, fitted marginals, no injected stress")
def baseline(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    return _base_frame(gen, submit, "baseline")


@register_scenario("bursty_arrivals", "self-exciting (Hawkes) arrivals: clustered bursts, CV(inter-arrival) > 1")
def bursty_arrivals(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                     branching_ratio: float = 0.6, decay: float = 0.5, **_) -> pd.DataFrame:
    submit = gen.hawkes_arrivals(n_jobs, base_rate=rate, branching_ratio=branching_ratio, decay=decay)
    return _base_frame(gen, submit, "bursty_arrivals")


@register_scenario("diurnal_pattern", "day/night sinusoidal load cycle")
def diurnal_pattern(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                     amplitude: float = 0.8, period: float = 86400.0, **_) -> pd.DataFrame:
    submit = gen.diurnal_arrivals(n_jobs, base_rate=rate, amplitude=amplitude, period=period)
    return _base_frame(gen, submit, "diurnal_pattern")


@register_scenario("flash_crowd", "one concentrated arrival spike against a low background rate")
def flash_crowd(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 0.3,
                 burst_frac: float = 0.6, **_) -> pd.DataFrame:
    submit = gen.flash_crowd_arrivals(n_jobs, background_rate=rate, burst_frac=burst_frac)
    return _base_frame(gen, submit, "flash_crowd")


@register_scenario("cold_start_storm", "long idle gaps punctuated by rapid-fire short-job storms")
def cold_start_storm(gen: WorkloadScenarioGenerator, n_jobs: int, **_) -> pd.DataFrame:
    submit = gen.cold_start_arrivals(n_jobs)
    df = _base_frame(gen, submit, "cold_start_storm")
    # storms are characteristically short jobs (cold-start/invocation
    # pattern, not long training runs) — bias duration down
    df["duration"] = np.minimum(df["duration"], gen.rng.exponential(scale=15.0, size=len(df)))
    return df


@register_scenario("heavy_tail_demand", "resource-request sizes drawn with the infinite-mean cap raised, exposing tail behavior")
def heavy_tail_demand(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                       tail_cap_multiple: float = 20.0, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    original_cap = gen.cap_multiple
    gen.cap_multiple = tail_cap_multiple
    try:
        df = _base_frame(gen, submit, "heavy_tail_demand")
    finally:
        gen.cap_multiple = original_cap
    return df


@register_scenario("high_contention", "many jobs concentrated on one popular GPU SKU")
def high_contention(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 2.0,
                     hot_type: str = "A100", hot_weight: float = 0.85, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    weights = {t: (hot_weight if t == hot_type else (1 - hot_weight) / (len(_GPU_TYPES) - 1)) for t in _GPU_TYPES}
    return _base_frame(gen, submit, "high_contention", gpu_type_weights=weights)


@register_scenario("resource_starvation", "a small set of users submit disproportionately large jobs")
def resource_starvation(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                         whale_frac: float = 0.05, whale_multiplier: float = 8.0, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "resource_starvation")
    n_whales = max(1, int(len(df) * whale_frac))
    whale_idx = gen.rng.choice(df.index, size=n_whales, replace=False)
    df.loc[whale_idx, "num_gpu"] = np.round(df.loc[whale_idx, "num_gpu"] * whale_multiplier).clip(lower=1)
    df.loc[whale_idx, "num_cpu"] = np.round(df.loc[whale_idx, "num_cpu"] * whale_multiplier).clip(lower=1)
    whale_users = [f"whale_{i}" for i in range(min(5, n_whales))]
    df.loc[whale_idx, "user"] = gen.rng.choice(whale_users, size=n_whales)
    return df


@register_scenario("spot_preemption_churn", "short-lived jobs with a high kill rate, modeling spot/preemptible instances")
def spot_preemption_churn(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.5,
                           kill_rate: float = 0.35, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "spot_preemption_churn")
    n = len(df)
    status = gen.rng.choice(["completed", "killed"], size=n, p=[1 - kill_rate, kill_rate])
    df["status"] = status
    # a killed job's *reported* duration is how long it ran before eviction,
    # not how long it would have taken to finish
    killed = df["status"] == "killed"
    df.loc[killed, "duration"] = df.loc[killed, "duration"] * gen.rng.uniform(0.05, 0.6, size=killed.sum())
    return df


@register_scenario("gpu_fragmentation_sharing", "fractional-GPU requests (gpu_milli populated) stressing bin-packing/fragmentation")
def gpu_fragmentation_sharing(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.5,
                               share_frac: float = 0.7, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "gpu_fragmentation_sharing")
    n = len(df)
    is_shared = gen.rng.uniform(size=n) < share_frac
    gpu_milli = np.where(df["num_gpu"] > 0, 1000.0, 0.0)
    # sharing jobs request a fraction of a single device in common
    # increments (250/500/750 milli), the way GPU-sharing allocators
    # (MPS, MIG, time-slicing) actually quantize requests
    quanta = gen.rng.choice([250, 500, 750], size=n)
    gpu_milli = np.where(is_shared & (df["num_gpu"] > 0), quanta, gpu_milli)
    df["num_gpu"] = np.where(is_shared & (df["num_gpu"] > 0), 1.0, df["num_gpu"])
    df["gpu_milli"] = gpu_milli
    return df


@register_scenario("moe_expert_load_skew", "Zipf-skewed per-tenant demand, modeling hot experts/tenants in MoE-style serving")
def moe_expert_load_skew(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 2.0,
                          n_experts: int = 32, skew: float = 1.3, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "moe_expert_load_skew")
    weights = gen.zipf_weights(n_experts, skew=skew)
    experts = gen.rng.choice([f"expert_{i:03d}" for i in range(n_experts)], size=len(df), p=weights)
    df["user"] = experts
    return df


@register_scenario("llm_inference_bursty", "short high-QPS inference-style jobs with Hawkes arrivals and small resource footprints")
def llm_inference_bursty(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 4.0,
                          branching_ratio: float = 0.7, decay: float = 2.0, **_) -> pd.DataFrame:
    submit = gen.hawkes_arrivals(n_jobs, base_rate=rate, branching_ratio=branching_ratio, decay=decay)
    df = _base_frame(gen, submit, "llm_inference_bursty")
    df["duration"] = gen.rng.exponential(scale=2.0, size=len(df))  # seconds-scale requests
    df["num_gpu"] = np.minimum(df["num_gpu"], 1)
    df["gpu_milli"] = gen.rng.choice([125, 250, 500], size=len(df))
    return df


@register_scenario("long_context_kv_pressure", "a subset of jobs with very long duration and elevated memory, modeling long-context KV-cache pressure")
def long_context_kv_pressure(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                              long_frac: float = 0.15, mem_multiplier: float = 6.0, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "long_context_kv_pressure")
    n_long = max(1, int(len(df) * long_frac))
    long_idx = gen.rng.choice(df.index, size=n_long, replace=False)
    df.loc[long_idx, "duration"] = df.loc[long_idx, "duration"] * gen.rng.uniform(3, 10, size=n_long)
    df.loc[long_idx, "mem"] = df.loc[long_idx, "mem"] * mem_multiplier
    return df


@register_scenario("checkpoint_io_burst", "periodic synchronized duration spikes, modeling checkpoint/IO stalls across concurrent jobs")
def checkpoint_io_burst(gen: WorkloadScenarioGenerator, n_jobs: int, rate: float = 1.0,
                         checkpoint_interval: float = 1800.0, stall_frac: float = 0.2, **_) -> pd.DataFrame:
    submit = gen.poisson_arrivals(n_jobs, rate=rate)
    df = _base_frame(gen, submit, "checkpoint_io_burst")
    # jobs whose submit_time falls in a periodic checkpoint window get a
    # duration stall added, correlated in time across otherwise-independent
    # jobs (the thing that makes checkpoint storms a scheduling problem)
    in_window = (df["submit_time"] % checkpoint_interval) < (checkpoint_interval * stall_frac)
    stall = gen.rng.exponential(scale=30.0, size=len(df))
    df.loc[in_window, "duration"] = df.loc[in_window, "duration"] + stall[in_window]
    return df
