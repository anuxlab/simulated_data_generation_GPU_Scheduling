"""
Core synthetic-trace generation engine. Samples job attributes (duration, num_cpu,
num_gpu, gpu_type) from a `fit_results` dict (the output of `gputrace.analysis.analyze`)
and combines them with a chosen arrival process to produce a trace in gputrace's
unified schema -- ready to feed into a scheduler for benchmarking/stress testing.

Individual *scenarios* (what varies between stress tests) are defined in
`stress_scenarios.py` as small functions registered against this engine, so adding a
new stress scenario never requires touching this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.distributions import CANDIDATES


class WorkloadScenarioGenerator:
    def __init__(self, fit_results: dict, seed: int = 0):
        self.fits = fit_results
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    # ---- sampling primitives, reused by every scenario -----------------------
    def sample_metric(self, metric: str, n: int, dist_name: str | None = None,
                       params: list | None = None) -> np.ndarray:
        info = self.fits[metric]
        dname = dist_name or info["best_dist"]
        p = params or info["best_params"]
        dist = CANDIDATES[dname][0]
        return dist.rvs(*p, size=n, random_state=self.rng)

    def sample_gpu_type(self, n: int) -> np.ndarray:
        seg = self.fits.get("segmentation_by_gpu_type")
        if not seg:
            return np.array(["UNKNOWN"] * n)
        types = [r["gpu_type"] for r in seg]
        weights = np.array([r["n"] for r in seg], dtype=float)
        weights /= weights.sum()
        return self.rng.choice(types, size=n, p=weights)

    # ---- arrival-process primitives -------------------------------------------
    def arrivals_renewal(self, n: int, cap_multiple: float = 5.0) -> np.ndarray:
        """Renewal process from the fitted inter-arrival distribution, capped at
        `cap_multiple` x the empirically observed max gap. See README: distributions
        with shape params implying infinite mean (Pareto b<=1 etc.) are common
        best-AIC fits for bursty arrival data, but are unsafe to sample from
        unbounded for simulation -- capping preserves the fitted burst shape while
        keeping simulated gaps inside the range the data actually supports."""
        if "inter_arrival" not in self.fits or n <= 1:
            return np.arange(n, dtype=float)
        cap = self.fits["inter_arrival"]["stats"]["max"] * cap_multiple
        ia = self.sample_metric("inter_arrival", n - 1)
        ia = np.clip(ia, 0, cap)
        return np.concatenate([[0.0], np.cumsum(ia)])

    def arrivals_bursty_hawkes(self, n: int, burst_prob: float = 0.35,
                                burst_size_range: tuple[int, int] = (5, 40),
                                burst_width: float = 30.0) -> np.ndarray:
        """Self-exciting arrivals: a renewal backbone, but with probability
        `burst_prob` a flash-crowd of near-simultaneous jobs is injected -- models
        multi-worker distributed jobs / retry storms much better than a pure
        renewal process."""
        if "inter_arrival" not in self.fits:
            return np.arange(n, dtype=float)
        ia_info = self.fits["inter_arrival"]
        ia_dist = CANDIDATES[ia_info["best_dist"]][0]
        params = ia_info["best_params"]
        times: list[float] = []
        t = 0.0
        while len(times) < n:
            gap = float(ia_dist.rvs(*params, random_state=self.rng))
            t += max(gap, 0)
            times.append(t)
            if self.rng.random() < burst_prob:
                k = int(self.rng.integers(*burst_size_range))
                jitter = self.rng.exponential(burst_width, size=k)
                times.extend((t + jitter).tolist())
        times = np.sort(np.array(times[:n]))
        return times - times[0]

    def arrivals_diurnal(self, n: int, period_hours: float = 24.0,
                          amplitude: float = 0.6) -> np.ndarray:
        """Renewal-process arrivals thinned by a day/night sinusoid, for
        autoscaling / capacity-planning scenarios."""
        base = self.arrivals_renewal(n * 3)
        phase = (base / 3600.0 % period_hours) / period_hours * 2 * np.pi
        rate_mod = 1 + amplitude * np.sin(phase - np.pi / 2)
        keep = self.rng.random(len(base)) < (rate_mod / rate_mod.max())
        kept = np.sort(base[keep])
        if len(kept) < n:
            extra = np.sort(base[~keep])[: n - len(kept)]
            kept = np.sort(np.concatenate([kept, extra]))
        kept = kept[:n]
        return kept - kept[0]

    # ---- assemble a trace DataFrame in the unified schema ----------------------
    def assemble(self, submit_time, duration, num_cpu, num_gpu, gpu_type,
                 scenario: str, users: int = 500) -> pd.DataFrame:
        n = len(submit_time)
        df = pd.DataFrame({
            "job_id": [f"{scenario}_{i}" for i in range(n)],
            "submit_time": np.round(submit_time, 3),
            "duration": np.round(np.maximum(duration, 0.001), 3),
            "num_cpu": np.round(np.maximum(num_cpu, 0), 3),
            "num_gpu": np.round(np.maximum(num_gpu, 0), 3),
            "gpu_type": gpu_type,
            "user": [f"synthetic_user_{i % users}" for i in range(n)],
            "wait_time": float("nan"),
            "status": "UNKNOWN",
            "source": f"synthetic:{scenario}",
        })
        return df.sort_values("submit_time").reset_index(drop=True)
