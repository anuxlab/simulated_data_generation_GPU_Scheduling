"""
``WorkloadScenarioGenerator``: the sampling/arrival-process primitives every
stress-test scenario is built from, plus the registry scenarios attach to.

Adding a new scenario is one function:

    def my_scenario(gen: WorkloadScenarioGenerator, n_jobs: int, **kwargs) -> pd.DataFrame:
        ...

decorated with ``@register_scenario("my_scenario", "what it stresses")`` in
``stress_scenarios.py`` (or your own module that imports from it) — it's
then immediately available via ``gputrace generate --scenario my_scenario``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..analysis import AnalysisResult, FieldFit

_SCENARIOS: Dict[str, "ScenarioSpec"] = {}


@dataclass
class ScenarioSpec:
    name: str
    stresses: str
    fn: Callable[..., pd.DataFrame]


def register_scenario(name: str, stresses: str):
    def _decorator(fn):
        if name in _SCENARIOS:
            raise ValueError(f"scenario {name!r} already registered")
        _SCENARIOS[name] = ScenarioSpec(name=name, stresses=stresses, fn=fn)
        return fn

    return _decorator


def list_scenarios() -> List[ScenarioSpec]:
    return sorted(_SCENARIOS.values(), key=lambda s: s.name)


def get_scenario(name: str) -> ScenarioSpec:
    if name not in _SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(_SCENARIOS)}")
    return _SCENARIOS[name]


# Distributions whose MLE-fit shape parameter can imply an infinite
# theoretical mean; sampling from these is capped (see
# WorkloadScenarioGenerator.sample_field below).
_UNBOUNDED_RISK = {"pareto", "genpareto"}


class WorkloadScenarioGenerator:
    """Wraps a fitted ``AnalysisResult`` and an RNG, and exposes sampling
    primitives scenario functions compose to build a synthetic trace.

    All randomness flows through ``self.rng`` (a ``numpy.random.Generator``)
    so that passing the same ``seed`` always reproduces the same trace —
    reproducibility is load-bearing for benchmark suites that sweep
    scenario x scale x seed.
    """

    def __init__(self, fit: AnalysisResult, seed: int = 0, cap_multiple: float = 5.0):
        self.fit = fit
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # multiple of the empirical max a capped (infinite-mean) fit is
        # allowed to sample beyond, rather than the raw unbounded tail
        self.cap_multiple = cap_multiple
        # empirical maxima, used only for capping infinite-mean fits
        self._empirical_max = {name: None for name in fit.fields}

    def set_empirical_max(self, field_name: str, value: float) -> None:
        self._empirical_max[field_name] = value

    # -- core scalar-field sampling --------------------------------------

    def sample_field(self, field_name: str, n: int, *, from_segment: Optional[str] = None) -> np.ndarray:
        """Sample ``n`` values from the best-fit distribution for
        ``field_name`` (or a gpu_type segment of it). Applies the
        infinite-mean safety cap when the winning fit's shape parameter
        implies an unbounded mean.
        """
        ff = self._field_fit(field_name, from_segment)

        if ff.low_cardinality and ff.empirical_samples:
            return self.rng.choice(np.asarray(ff.empirical_samples), size=n, replace=True)

        if not ff.fits:
            # nothing fit (too little data, and not low-cardinality either)
            # — fall back to a small positive exponential so callers still
            # get plausible values rather than a crash
            return self.rng.exponential(scale=60.0, size=n)

        best = ff.fits[0]
        dist = _dist_by_name(best.name)
        samples = dist.rvs(*best.params, size=n, random_state=self.rng)
        samples = np.clip(samples, a_min=0, a_max=None)

        if best.infinite_mean and best.name in _UNBOUNDED_RISK:
            cap = self._empirical_max.get(field_name) or np.percentile(samples, 99)
            samples = np.minimum(samples, cap * self.cap_multiple)
        return samples

    def _field_fit(self, field_name: str, from_segment: Optional[str]) -> FieldFit:
        if from_segment and from_segment in self.fit.segments_by_gpu_type:
            seg = self.fit.segments_by_gpu_type[from_segment]
            if field_name in seg:
                return seg[field_name]
        return self.fit.fields[field_name]

    # -- arrival processes -------------------------------------------------

    def poisson_arrivals(self, n: int, rate: float) -> np.ndarray:
        """Homogeneous Poisson arrival process: i.i.d. Exp(1/rate)
        inter-arrival times, cumulatively summed into submit times."""
        inter = self.rng.exponential(scale=1.0 / rate, size=n)
        return np.cumsum(inter)

    def hawkes_arrivals(
        self, n: int, base_rate: float, branching_ratio: float = 0.5, decay: float = 1.0
    ) -> np.ndarray:
        """Self-exciting (Hawkes) arrival process — each event temporarily
        raises the instantaneous arrival rate, producing the clustered,
        bursty arrival pattern real cluster traces show (CV > 1) that a
        homogeneous Poisson process cannot. Uses Ogata's thinning
        algorithm.

        branching_ratio: expected number of "child" events triggered per
            parent event (must be < 1 for a stationary process). Higher =
            burstier.
        decay: rate at which a parent's excitation fades (higher = shorter
            bursts).
        """
        times: List[float] = []
        t = 0.0
        # upper bound on intensity for thinning; grows if we underestimate
        lam_max = base_rate * (1.0 + 10 * branching_ratio)
        guard = 0
        while len(times) < n and guard < n * 200:
            guard += 1
            t += self.rng.exponential(scale=1.0 / lam_max)
            excitation = sum(
                branching_ratio * decay * np.exp(-decay * (t - s)) for s in times[-50:]
            )
            lam_t = base_rate + excitation
            if self.rng.uniform(0, lam_max) <= lam_t:
                times.append(t)
        return np.array(times[:n])

    def diurnal_arrivals(self, n: int, base_rate: float, amplitude: float = 0.8, period: float = 86400.0) -> np.ndarray:
        """Non-homogeneous Poisson process with a sinusoidal rate over a
        day/night cycle: rate(t) = base_rate * (1 + amplitude * sin(2*pi*t/period)).
        Generated via thinning against the peak rate.
        """
        peak_rate = base_rate * (1 + amplitude)
        times: List[float] = []
        t = 0.0
        guard = 0
        while len(times) < n and guard < n * 200:
            guard += 1
            t += self.rng.exponential(scale=1.0 / peak_rate)
            rate_t = base_rate * (1 + amplitude * np.sin(2 * np.pi * t / period))
            if self.rng.uniform(0, peak_rate) <= max(rate_t, 0):
                times.append(t)
        return np.array(times[:n])

    def flash_crowd_arrivals(self, n: int, background_rate: float, burst_frac: float = 0.6,
                              burst_start_frac: float = 0.4, burst_width_frac: float = 0.05) -> np.ndarray:
        """Mostly-background Poisson arrivals with one concentrated burst
        window (a step-function-like spike in rate), modeling a single
        extreme event rather than an ongoing bursty regime.
        """
        n_burst = int(n * burst_frac)
        n_bg = n - n_burst
        span = n / background_rate
        bg_times = np.sort(self.rng.uniform(0, span, size=n_bg))
        burst_center = span * burst_start_frac
        burst_width = max(span * burst_width_frac, 1e-6)
        burst_times = np.sort(self.rng.normal(loc=burst_center, scale=burst_width, size=n_burst))
        burst_times = np.clip(burst_times, 0, None)
        return np.sort(np.concatenate([bg_times, burst_times]))

    def cold_start_arrivals(self, n: int, groups: int = 40, idle_scale: float = 3000.0,
                             storm_size_range=(5, 60)) -> np.ndarray:
        """Long idle gaps between jobs punctuated by rapid-fire short-job
        'storms' — models the invocation pattern behind serverless /
        rapid-fire short-job cold-start problems.
        """
        times: List[float] = []
        t = 0.0
        while len(times) < n:
            gap = self.rng.exponential(scale=idle_scale)
            t += gap
            storm_size = int(self.rng.integers(*storm_size_range))
            burst = t + np.sort(self.rng.exponential(scale=0.5, size=storm_size).cumsum())
            times.extend(burst.tolist())
            t = burst[-1] if len(burst) else t
        return np.array(sorted(times[:n]))

    # -- popularity / skew --------------------------------------------------

    def zipf_weights(self, n_items: int, skew: float = 1.2) -> np.ndarray:
        """Normalized Zipf-law popularity weights over ``n_items`` items,
        for modeling skewed per-tenant/expert demand (see
        ``moe_expert_load_skew``)."""
        ranks = np.arange(1, n_items + 1)
        w = 1.0 / np.power(ranks, skew)
        return w / w.sum()


def _dist_by_name(name: str):
    from scipy import stats as _stats

    return getattr(_stats, name)
