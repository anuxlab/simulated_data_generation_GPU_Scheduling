"""MLE distribution-fitting engine: fits a bank of candidate distributions to a 1-D
positive-valued sample, ranks by AIC/BIC, reports KS statistic (+ subsampled p-value,
see the large-n caveat below), and a Hill tail-index estimate."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CANDIDATES = {
    "expon":       (stats.expon,       dict(floc=0)),
    "lognorm":     (stats.lognorm,     dict(floc=0)),
    "gamma":       (stats.gamma,       dict(floc=0)),
    "weibull_min": (stats.weibull_min, dict(floc=0)),
    "pareto":      (stats.pareto,      dict(floc=0)),
    "genpareto":   (stats.genpareto,   dict(floc=0)),
    "burr12":      (stats.burr12,      dict(floc=0)),
    "loglogistic": (stats.fisk,        dict(floc=0)),
}


def _loglik(dist, params, data):
    return float(np.sum(dist.logpdf(data, *params)))


def fit_all(data, candidates=None, sample_for_ks=5000, seed=0, min_n=20):
    """Fit every candidate distribution to `data` via MLE; returns (ranked_df, params_dict).
    Raises ValueError if fewer than `min_n` positive observations are available."""
    if candidates is None:
        candidates = CANDIDATES
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    n = len(data)
    if n < min_n:
        raise ValueError(f"need at least {min_n} positive observations, got {n}")

    rng = np.random.default_rng(seed)
    ks_sample = data if n <= sample_for_ks else rng.choice(data, sample_for_ks, replace=False)

    rows, fitted_params = [], {}
    for name, (dist, kwargs) in candidates.items():
        try:
            params = dist.fit(data, **kwargs)
            k = len(params) - sum(1 for v in kwargs.values() if isinstance(v, (int, float)))
            ll = _loglik(dist, params, data)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            ks_stat, ks_p = stats.kstest(ks_sample, dist.cdf, args=params)
            rows.append(dict(distribution=name, params=params, k=k, loglik=ll,
                              AIC=aic, BIC=bic, KS_stat=float(ks_stat),
                              KS_p_subsample=float(ks_p)))
            fitted_params[name] = params
        except Exception as e:
            rows.append(dict(distribution=name, params=None, k=None, loglik=np.nan,
                              AIC=np.nan, BIC=np.nan, KS_stat=np.nan,
                              KS_p_subsample=np.nan, error=str(e)))
    df = pd.DataFrame(rows).sort_values("AIC", na_position="last").reset_index(drop=True)
    df["delta_AIC"] = df["AIC"] - df["AIC"].min()
    return df, fitted_params


def summary_stats(data) -> dict:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("no finite data points")
    return dict(
        n=int(len(data)), mean=float(np.mean(data)), median=float(np.median(data)),
        std=float(np.std(data)), min=float(np.min(data)), max=float(np.max(data)),
        p25=float(np.percentile(data, 25)), p75=float(np.percentile(data, 75)),
        p95=float(np.percentile(data, 95)), p99=float(np.percentile(data, 99)),
        skew=float(stats.skew(data)), kurtosis=float(stats.kurtosis(data)),
        cv=float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else float("nan"),
    )


def tail_index_hill(data, k_frac: float = 0.05) -> dict:
    """Hill estimator for the tail index alpha of the top k_frac fraction of `data`.
    alpha <~ 2 signals a heavy (possibly infinite-variance) Pareto-like tail."""
    data = np.sort(np.asarray(data, dtype=float))
    data = data[data > 0]
    n = len(data)
    if n < 20:
        raise ValueError("need at least 20 positive observations for Hill estimator")
    k = max(int(n * k_frac), 10)
    tail = data[-k:]
    xmin = float(tail[0])
    alpha = 1 + k / np.sum(np.log(tail / xmin))
    return dict(alpha=float(alpha), k=int(k), xmin=xmin)


def infinite_mean_warning(dist_name: str, params: tuple) -> str | None:
    """Flags the classic 'best-AIC-fit-is-unsafe-to-sample-from' trap: Pareto/genpareto
    fits with shape<=1 have infinite theoretical mean and will produce unrealistic
    extreme values if sampled from directly for simulation (see README)."""
    if dist_name == "pareto" and params[0] <= 1:
        return f"pareto shape b={params[0]:.3f} <= 1 => infinite mean; cap draws before simulating."
    if dist_name == "genpareto" and params[0] >= 1:
        return f"genpareto shape c={params[0]:.3f} >= 1 => infinite mean; cap draws before simulating."
    return None
