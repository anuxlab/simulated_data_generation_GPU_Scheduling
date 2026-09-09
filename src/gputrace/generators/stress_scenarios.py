"""
Registry of benchmark/stress-test scenarios for GPU scheduling algorithms.

Each scenario is a small function `(gen: WorkloadScenarioGenerator, n_jobs: int,
**kwargs) -> pd.DataFrame` registered under a name. Add your own by writing a
function and decorating it with `@register_scenario("your_name")` -- no other file
needs to change, and it immediately becomes available via
`gputrace generate --scenario your_name` and in `list_scenarios()`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .scenario_generator import WorkloadScenarioGenerator

_SCENARIOS: dict[str, Callable[..., pd.DataFrame]] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register_scenario(name: str, description: str = ""):
    def deco(fn):
        _SCENARIOS[name] = fn
        _DESCRIPTIONS[name] = description or fn.__doc__ or ""
        return fn
    return deco


def list_scenarios() -> dict[str, str]:
    return dict(_DESCRIPTIONS)


def run_scenario(name: str, gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame:
    if name not in _SCENARIOS:
        raise KeyError(f"unknown scenario '{name}'. available: {sorted(_SCENARIOS)}")
    return _SCENARIOS[name](gen, n_jobs, **kwargs)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

@register_scenario("baseline", "Reproduces the fitted empirical distributions as-is. "
                                "Calibration / sanity-check scenario -- your scheduler's baseline score.")
def baseline(gen: WorkloadScenarioGenerator, n_jobs: int, **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    num_gpu = np.where(gpu_type == "CPU", 0.0, num_gpu)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "baseline")


@register_scenario("bursty_arrivals", "Hawkes-style self-exciting arrivals: renewal backbone plus "
                                       "probabilistic flash-crowds of near-simultaneous jobs. "
                                       "Stress-tests admission control and queue-depth spikes.")
def bursty_arrivals(gen: WorkloadScenarioGenerator, n_jobs: int,
                     burst_prob: float = 0.35, burst_size_range=(5, 40), **_) -> pd.DataFrame:
    submit_time = gen.arrivals_bursty_hawkes(n_jobs, burst_prob=burst_prob, burst_size_range=burst_size_range)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "bursty_arrivals")


@register_scenario("heavy_tail_demand", "CPU/GPU request sizes redrawn from a fatter generalized-Pareto "
                                         "tail than observed. Stress-tests fragmentation and large-job starvation.")
def heavy_tail_demand(gen: WorkloadScenarioGenerator, n_jobs: int, heavy_shape: float = 0.9, **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    cpu_scale = gen.fits["num_cpu"]["best_params"][-1] * 1.5
    gpu_scale = gen.fits["num_gpu"]["best_params"][-1] * 1.5
    num_cpu = stats.genpareto.rvs(heavy_shape, loc=0, scale=cpu_scale, size=n_jobs, random_state=gen.rng) + 1
    num_gpu = stats.genpareto.rvs(heavy_shape, loc=0, scale=gpu_scale, size=n_jobs, random_state=gen.rng)
    num_gpu = np.where(gpu_type == "CPU", 0.0, num_gpu)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "heavy_tail_demand")


@register_scenario("diurnal_pattern", "Arrival rate modulated by a day/night sinusoid via thinning. "
                                       "Stress-tests autoscaling and capacity planning across the daily cycle.")
def diurnal_pattern(gen: WorkloadScenarioGenerator, n_jobs: int,
                     period_hours: float = 24.0, amplitude: float = 0.6, **_) -> pd.DataFrame:
    submit_time = gen.arrivals_diurnal(n_jobs, period_hours=period_hours, amplitude=amplitude)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "diurnal_pattern")


@register_scenario("high_contention", "GPU/CPU demand scaled up with capacity held fixed downstream. "
                                       "Finds the point where a policy's queueing/backlog collapses.")
def high_contention(gen: WorkloadScenarioGenerator, n_jobs: int,
                     gpu_scale_range=(1.8, 2.5), cpu_scale_range=(1.3, 1.6), **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs) * gen.rng.uniform(*cpu_scale_range)
    num_gpu = gen.sample_metric("num_gpu", n_jobs) * gen.rng.uniform(*gpu_scale_range)
    gpu_type = gen.sample_gpu_type(n_jobs)
    num_gpu = np.where(gpu_type == "CPU", 0.0, num_gpu)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "high_contention")


@register_scenario("flash_crowd", "A single extreme burst: N jobs (much larger than a normal Hawkes "
                                   "burst) submitted within a few seconds midway through the trace, "
                                   "on top of otherwise-normal arrivals. Stress-tests worst-case queueing "
                                   "and whether a scheduler's admission control has a breaking point.")
def flash_crowd(gen: WorkloadScenarioGenerator, n_jobs: int,
                 crowd_fraction: float = 0.25, crowd_width_s: float = 5.0, **_) -> pd.DataFrame:
    n_crowd = int(n_jobs * crowd_fraction)
    n_base = n_jobs - n_crowd
    base_times = gen.arrivals_renewal(n_base)
    crowd_center = float(np.median(base_times)) if n_base > 0 else 0.0
    crowd_times = crowd_center + gen.rng.uniform(-crowd_width_s / 2, crowd_width_s / 2, size=n_crowd)
    submit_time = np.sort(np.concatenate([base_times, crowd_times]))
    submit_time = submit_time - submit_time.min()
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "flash_crowd")


@register_scenario("resource_starvation", "Most jobs request near-maximal CPU/GPU (a bimodal mix of "
                                           "small and huge jobs, weighted toward huge). Stress-tests "
                                           "whether small jobs get starved behind large resource holders.")
def resource_starvation(gen: WorkloadScenarioGenerator, n_jobs: int,
                         big_job_fraction: float = 0.6, big_job_multiplier: float = 4.0, **_) -> pd.DataFrame:
    submit_time = gen.arrivals_renewal(n_jobs)
    duration = gen.sample_metric("duration", n_jobs)
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    is_big = gen.rng.random(n_jobs) < big_job_fraction
    num_cpu = np.where(is_big, num_cpu * big_job_multiplier, num_cpu)
    num_gpu = np.where(is_big, num_gpu * big_job_multiplier, num_gpu)
    gpu_type = gen.sample_gpu_type(n_jobs)
    num_gpu = np.where(gpu_type == "CPU", 0.0, num_gpu)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "resource_starvation")


@register_scenario("cold_start_storm", "Many very short jobs submitted in rapid succession at the very "
                                        "start of the trace (e.g. autoscaler cold-start / CI pipeline burst). "
                                        "Stress-tests scheduling overhead and short-job throughput.")
def cold_start_storm(gen: WorkloadScenarioGenerator, n_jobs: int,
                      storm_fraction: float = 0.4, storm_duration_scale: float = 0.1, **_) -> pd.DataFrame:
    n_storm = int(n_jobs * storm_fraction)
    n_rest = n_jobs - n_storm
    storm_times = np.sort(gen.rng.exponential(0.5, size=n_storm).cumsum())
    rest_times = gen.arrivals_renewal(n_rest) + (storm_times[-1] if n_storm else 0)
    submit_time = np.concatenate([storm_times, rest_times])
    duration = gen.sample_metric("duration", n_jobs)
    duration[:n_storm] *= storm_duration_scale  # storm jobs are short-lived
    num_cpu = gen.sample_metric("num_cpu", n_jobs)
    num_gpu = gen.sample_metric("num_gpu", n_jobs)
    gpu_type = gen.sample_gpu_type(n_jobs)
    return gen.assemble(submit_time, duration, num_cpu, num_gpu, gpu_type, "cold_start_storm")
