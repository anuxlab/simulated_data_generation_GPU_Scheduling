"""Segment a normalized trace by a categorical column (gpu_type, user, source, ...)
and fit `duration` (or any chosen metric) separately per segment -- pooled fits wash
out real heterogeneity between job classes, which matters a lot for scheduling
simulation realism."""
from __future__ import annotations

import pandas as pd

from .distributions import fit_all, summary_stats


def segment_report(df: pd.DataFrame, by: str = "gpu_type", metric: str = "duration",
                    min_n: int = 50) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(by):
        series = sub[metric].dropna()
        series = series[series > 0]
        if len(series) < min_n:
            continue
        s = summary_stats(series)
        fit_df, _ = fit_all(series, min_n=min_n)
        rows.append(dict(
            **{by: key}, n=len(sub), **{f"{metric}_median": s["median"],
            f"{metric}_mean": s["mean"], f"{metric}_cv": s["cv"]},
            mean_num_cpu=sub["num_cpu"].mean(), mean_num_gpu=sub["num_gpu"].mean(),
            best_dist=fit_df.iloc[0]["distribution"],
        ))
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
