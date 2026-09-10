"""
Statistical analysis: fit candidate distributions to a unified-schema trace,
rank them, and flag fits that are unsafe to sample from for simulation.

``analyze(df)`` fits candidate distributions via MLE to ``duration``,
``num_cpu``, ``num_gpu``, ``wait_time``, and the inter-arrival-time process
(derived from sorted ``submit_time``), ranks them by AIC/BIC, reports the KS
statistic (the statistic itself, not just its p-value — with large trace
sizes KS p-values are ~always significant even for a good fit; the
*statistic* is the informative part of the test here), and estimates the
Hill tail index. It also segments ``duration`` by ``gpu_type`` (or any
categorical column) since pooling job classes together washes out real
heterogeneity.

Safety check
------------
Whenever the best-AIC-fit distribution has a shape parameter implying an
infinite theoretical mean (Pareto with shape <= 1, generalized Pareto with
shape >= 1 — both common winners for bursty inter-arrival or wait-time
data), the result is flagged via ``infinite_mean_warning``. The scenario
generator (see ``generators/engine.py``) caps sampling from such a fit at a
bounded multiple of the empirical max, rather than silently producing
simulated traces with unrealistic multi-day gaps that never occurred in the
source data. Best likelihood fit is not the same thing as safe to
extrapolate from.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

CANDIDATE_DISTS = {
    "lognorm": stats.lognorm,
    "weibull_min": stats.weibull_min,
    "gamma": stats.gamma,
    "pareto": stats.pareto,
    "genpareto": stats.genpareto,
    "burr12": stats.burr12,
    "fisk": stats.fisk,       # log-logistic
    "expon": stats.expon,
}

ANALYZED_FIELDS = ["duration", "num_cpu", "num_gpu", "wait_time", "inter_arrival"]


@dataclass
class DistFit:
    name: str
    params: List[float]
    aic: float
    bic: float
    ks_statistic: float
    ks_pvalue: float
    infinite_mean: bool = False


@dataclass
class FieldFit:
    field: str
    n: int
    fits: List[DistFit] = field(default_factory=list)
    best: Optional[str] = None
    hill_tail_index: Optional[float] = None
    infinite_mean_warning: bool = False
    # For low-cardinality fields (e.g. num_gpu taking values like
    # {0,1,2,4,8}), continuous MLE fits are unstable/degenerate — see
    # generators/engine.py. We store a bounded empirical sample instead and
    # the generator bootstraps from it rather than sampling the parametric
    # fit, whenever `low_cardinality` is True.
    low_cardinality: bool = False
    empirical_samples: List[float] = field(default_factory=list)


@dataclass
class AnalysisResult:
    n_jobs: int
    fields: Dict[str, FieldFit]
    segments_by_gpu_type: Dict[str, Dict[str, FieldFit]]

    def to_json(self, path: str | Path) -> None:
        def _default(o):
            if hasattr(o, "__dict__"):
                return asdict(o) if hasattr(o, "__dataclass_fields__") else o.__dict__
            return str(o)

        with open(path, "w") as fh:
            json.dump(_to_plain(self), fh, indent=2, default=_default)

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisResult":
        with open(path) as fh:
            raw = json.load(fh)
        return _from_plain(raw)


def _to_plain(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _from_plain(raw: dict) -> AnalysisResult:
    def field_fit(d):
        fits = [DistFit(**f) for f in d.get("fits", [])]
        return FieldFit(
            field=d["field"], n=d["n"], fits=fits, best=d.get("best"),
            hill_tail_index=d.get("hill_tail_index"),
            infinite_mean_warning=d.get("infinite_mean_warning", False),
            low_cardinality=d.get("low_cardinality", False),
            empirical_samples=d.get("empirical_samples", []),
        )

    fields = {k: field_fit(v) for k, v in raw["fields"].items()}
    segments = {
        seg: {k: field_fit(v) for k, v in inner.items()}
        for seg, inner in raw.get("segments_by_gpu_type", {}).items()
    }
    return AnalysisResult(n_jobs=raw["n_jobs"], fields=fields, segments_by_gpu_type=segments)


def _has_infinite_mean(name: str, params: tuple) -> bool:
    # scipy shape-parameter conventions: for `pareto`, params = (b, loc, scale)
    # with b = shape; mean is infinite for b <= 1.
    # for `genpareto`, params = (c, loc, scale) with c = shape; mean is
    # infinite for c >= 1.
    if name == "pareto":
        b = params[0]
        return b <= 1.0
    if name == "genpareto":
        c = params[0]
        return c >= 1.0
    return False


def _fit_one(dist_name: str, dist, data: np.ndarray) -> Optional[DistFit]:
    if len(data) < 10 or np.all(data == data[0]):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dist.fit(data)
        loglik = np.sum(dist.logpdf(data, *params))
        if not np.isfinite(loglik):
            return None
        k = len(params)
        n = len(data)
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        ks_stat, ks_p = stats.kstest(data, dist_name if hasattr(stats, dist_name) else dist.cdf,
                                      args=params)
        return DistFit(
            name=dist_name,
            params=[float(p) for p in params],
            aic=float(aic),
            bic=float(bic),
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_p),
            infinite_mean=_has_infinite_mean(dist_name, params),
        )
    except Exception:
        return None


def _hill_tail_index(data: np.ndarray, k_frac: float = 0.1) -> Optional[float]:
    """Hill estimator of the tail index, using the top ``k_frac`` fraction
    of order statistics. Returns None if there isn't enough positive data."""
    x = np.sort(data[data > 0])[::-1]
    k = max(int(len(x) * k_frac), 5)
    if len(x) < k + 1:
        return None
    top = x[:k]
    xk1 = x[k] if len(x) > k else x[-1]
    if xk1 <= 0:
        return None
    logs = np.log(top / xk1)
    if np.sum(logs) <= 0:
        return None
    return float(k / np.sum(logs))


_LOW_CARDINALITY_MAX_UNIQUE = 25
_LOW_CARDINALITY_MAX_UNIQUE_FRAC = 0.02
_EMPIRICAL_SAMPLE_CAP = 5000


def _is_low_cardinality(data: np.ndarray) -> bool:
    n_unique = len(np.unique(data))
    if n_unique <= _LOW_CARDINALITY_MAX_UNIQUE:
        return True
    return (n_unique / max(len(data), 1)) < _LOW_CARDINALITY_MAX_UNIQUE_FRAC


def _fit_field(data: np.ndarray, field_name: str) -> FieldFit:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    ff = FieldFit(field=field_name, n=len(data))
    if len(data) < 10:
        return ff

    if _is_low_cardinality(data):
        # Continuous MLE fits on near-discrete data (small integer counts
        # like num_gpu in {0,1,2,4,8}) are numerically unstable — the
        # optimizer can drive a shape parameter to an extreme value that
        # technically maximizes the (degenerate) likelihood on repeated
        # values, but samples from that "best fit" then explode to
        # physically nonsensical magnitudes. Bootstrap resampling from the
        # empirical distribution preserves the exact discrete support
        # instead, which is what a small-integer resource-count field
        # actually needs.
        ff.low_cardinality = True
        sample = data if len(data) <= _EMPIRICAL_SAMPLE_CAP else np.random.default_rng(0).choice(
            data, size=_EMPIRICAL_SAMPLE_CAP, replace=False
        )
        ff.empirical_samples = [float(x) for x in sample]
        ff.hill_tail_index = _hill_tail_index(data)
        return ff

    fits = []
    for name, dist in CANDIDATE_DISTS.items():
        # scale-family distributions (all of ours) need strictly positive
        # data for a meaningful MLE fit on this support
        usable = data[data > 0] if name != "expon" else data
        f = _fit_one(name, dist, usable)
        if f is not None:
            fits.append(f)

    fits.sort(key=lambda f: f.aic)
    ff.fits = fits
    if fits:
        ff.best = fits[0].name
        ff.infinite_mean_warning = fits[0].infinite_mean
    ff.hill_tail_index = _hill_tail_index(data)
    return ff


def analyze(df: pd.DataFrame, segment_by: str = "gpu_type") -> AnalysisResult:
    """Fit distributions to the standard fields of a unified-schema trace.

    Parameters
    ----------
    df : unified-schema DataFrame (see ``gputrace.schema``)
    segment_by : categorical column to segment ``duration`` fits by, in
        addition to the pooled fit (default: gpu_type). Pass None to skip
        segmentation.
    """
    df = df.sort_values("submit_time")
    inter_arrival = df["submit_time"].diff().dropna().to_numpy()

    fields: Dict[str, FieldFit] = {}
    for col in ["duration", "num_cpu", "num_gpu", "wait_time"]:
        fields[col] = _fit_field(df[col].to_numpy(), col)
    fields["inter_arrival"] = _fit_field(inter_arrival, "inter_arrival")

    segments: Dict[str, Dict[str, FieldFit]] = {}
    if segment_by and segment_by in df.columns:
        for value, group in df.groupby(segment_by):
            if len(group) < 10:
                continue
            key = str(value) if value else "(none)"
            segments[key] = {"duration": _fit_field(group["duration"].to_numpy(), "duration")}

    return AnalysisResult(n_jobs=len(df), fields=fields, segments_by_gpu_type=segments)
