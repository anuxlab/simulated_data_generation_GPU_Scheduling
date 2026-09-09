"""Top-level orchestrator: runs the full statistical analysis pipeline on a normalized
trace DataFrame and returns a single JSON-serializable dict (used by both the CLI
report command and the scenario generator, which reads fitted params back out)."""
from __future__ import annotations

import pandas as pd

from .distributions import fit_all, infinite_mean_warning, summary_stats, tail_index_hill
from .segmentation import segment_report

CORE_METRICS = ["duration", "num_cpu", "wait_time"]


def analyze(df: pd.DataFrame, gpu_only_for_num_gpu: bool = True) -> dict:
    results: dict = {}
    metrics = {m: df[m].dropna() for m in CORE_METRICS if m in df.columns}
    metrics = {k: v[v > 0] for k, v in metrics.items()}
    if "num_gpu" in df.columns:
        gpu_series = df.loc[df["num_gpu"] > 0, "num_gpu"] if gpu_only_for_num_gpu else df["num_gpu"]
        metrics["num_gpu"] = gpu_series[gpu_series > 0]

    for name, series in metrics.items():
        if len(series) < 20:
            continue
        stats_ = summary_stats(series)
        fit_df, _ = fit_all(series)
        best = fit_df.iloc[0]
        tail = tail_index_hill(series) if len(series) >= 20 else None
        warn = infinite_mean_warning(best["distribution"], best["params"])
        results[name] = dict(
            stats=stats_,
            fit_ranking=fit_df.drop(columns=["params"]).to_dict("records"),
            best_dist=best["distribution"],
            best_params=[float(x) for x in best["params"]],
            hill_tail=tail,
            infinite_mean_warning=warn,
        )

    # arrival process
    if "submit_time" in df.columns and len(df) > 20:
        st = df["submit_time"].dropna().sort_values()
        inter_arrival = st.diff().dropna()
        inter_arrival = inter_arrival[inter_arrival >= 0]
        if len(inter_arrival) >= 20:
            ia_stats = summary_stats(inter_arrival)
            fit_df_ia, _ = fit_all(inter_arrival)
            best = fit_df_ia.iloc[0]
            results["inter_arrival"] = dict(
                stats=ia_stats,
                best_dist=best["distribution"],
                best_params=[float(x) for x in best["params"]],
                cv=ia_stats["cv"],
                poisson_like=bool(0.85 <= ia_stats["cv"] <= 1.15),
                infinite_mean_warning=infinite_mean_warning(best["distribution"], best["params"]),
            )

    if "gpu_type" in df.columns and df["gpu_type"].nunique() > 1:
        seg = segment_report(df, by="gpu_type", metric="duration")
        results["segmentation_by_gpu_type"] = seg.to_dict("records")

    results["n_jobs"] = int(len(df))
    results["sources"] = sorted(df["source"].unique().tolist()) if "source" in df.columns else []
    return results
