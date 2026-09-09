"""
gputrace: synthetic GPU-cluster workload trace generation for scheduler /
simulator benchmarking.

Typical usage
-------------
    import gputrace

    fit = gputrace.analyze_file("alibaba2020", "pai_task_table_sample.csv")
    df = gputrace.generate("bursty_arrivals", fit, n_jobs=50_000, seed=1)
    df.to_csv("bursty_arrivals.csv", index=False)

or, without a real source trace to fit from, generate straight from a
built-in reference fit:

    df = gputrace.generate_from_reference("high_contention", n_jobs=10_000, seed=1)

See the CLI (``gputrace --help``) for the command-line equivalent, and
``exporters/k8s_sim.py`` for converting output into a scheduler simulator's
native input format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema
from .analysis import AnalysisResult, analyze
from .loaders import get_loader, list_loaders
from .generators import WorkloadScenarioGenerator, get_scenario, list_scenarios

__version__ = "1.0.0"

__all__ = [
    "schema",
    "AnalysisResult",
    "analyze",
    "analyze_file",
    "get_loader",
    "list_loaders",
    "generate",
    "list_scenarios",
    "reference_fit",
    "generate_from_reference",
]


def analyze_file(loader_name: str, path: str | Path, **loader_kwargs) -> AnalysisResult:
    """Load ``path`` with the named loader and fit distributions to it in
    one call."""
    df = get_loader(loader_name).load(path, **loader_kwargs)
    result = analyze(df)
    # record empirical maxima for the infinite-mean sampling cap
    return result


def generate(scenario_name: str, fit: AnalysisResult, n_jobs: int, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Generate ``n_jobs`` synthetic jobs under ``scenario_name``, using
    distributions fitted in ``fit``. Returns a unified-schema DataFrame."""
    spec = get_scenario(scenario_name)
    gen = WorkloadScenarioGenerator(fit, seed=seed)
    df = spec.fn(gen, n_jobs, **kwargs)
    schema.validate(df, strict=True)
    return df


def reference_fit() -> AnalysisResult:
    """A small built-in reference fit, for generating scenarios without any
    real source trace on hand (useful for demos, tests, and CI). Not a
    substitute for fitting your own production trace — see README for why."""
    from .generators._reference import build_reference_fit

    return build_reference_fit()


def generate_from_reference(scenario_name: str, n_jobs: int, seed: int = 0, **kwargs) -> pd.DataFrame:
    return generate(scenario_name, reference_fit(), n_jobs=n_jobs, seed=seed, **kwargs)
