import numpy as np
import pandas as pd
import pytest

from gputrace.analysis import analyze
from gputrace.generators import WorkloadScenarioGenerator, list_scenarios, run_scenario
from gputrace.schema import ALL_COLUMNS, validate


@pytest.fixture
def fit_results(normalized_df):
    return analyze(normalized_df)


EXPECTED_SCENARIOS = {
    "baseline", "bursty_arrivals", "heavy_tail_demand", "diurnal_pattern",
    "high_contention", "flash_crowd", "resource_starvation", "cold_start_storm",
}


def test_all_expected_scenarios_registered():
    assert EXPECTED_SCENARIOS.issubset(set(list_scenarios()))


@pytest.mark.parametrize("scenario", sorted(EXPECTED_SCENARIOS))
def test_scenario_produces_valid_schema_and_row_count(fit_results, scenario):
    gen = WorkloadScenarioGenerator(fit_results, seed=7)
    df = run_scenario(scenario, gen, n_jobs=500)
    assert len(df) == 500
    validated = validate(df, strict=True)  # every row must satisfy value constraints
    assert list(validated.columns) == ALL_COLUMNS
    assert (df["submit_time"].diff().dropna() >= 0).all()  # sorted / monotonic
    assert (df["duration"] > 0).all()


@pytest.mark.parametrize("scenario", sorted(EXPECTED_SCENARIOS))
def test_scenario_is_reproducible_with_same_seed(fit_results, scenario):
    gen1 = WorkloadScenarioGenerator(fit_results, seed=123)
    df1 = run_scenario(scenario, gen1, n_jobs=300)
    gen2 = WorkloadScenarioGenerator(fit_results, seed=123)
    df2 = run_scenario(scenario, gen2, n_jobs=300)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_high_contention_increases_mean_demand_vs_baseline(fit_results):
    gen_a = WorkloadScenarioGenerator(fit_results, seed=1)
    baseline = run_scenario("baseline", gen_a, n_jobs=3000)
    gen_b = WorkloadScenarioGenerator(fit_results, seed=1)
    contention = run_scenario("high_contention", gen_b, n_jobs=3000)
    assert contention["num_cpu"].mean() > baseline["num_cpu"].mean()
    assert contention["num_gpu"].mean() > baseline["num_gpu"].mean()


def test_flash_crowd_has_a_visible_spike(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=5)
    df = run_scenario("flash_crowd", gen, n_jobs=2000, crowd_fraction=0.3, crowd_width_s=5.0)
    # bin arrivals into 1-second buckets; the busiest bucket should hold a large
    # share of all jobs -- that's the "crowd" landing in a few-second window.
    counts = df["submit_time"].round().value_counts()
    assert counts.max() > 0.05 * len(df)


def test_run_scenario_raises_on_unknown_name(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=1)
    with pytest.raises(KeyError):
        run_scenario("not_a_real_scenario", gen, n_jobs=100)


def test_arrivals_renewal_caps_infinite_mean_distributions():
    """Regression test for the Pareto/shape<=1 infinite-mean trap: even if the fitted
    inter-arrival distribution has infinite theoretical mean, sampled traces must stay
    within a bounded, sane multiple of the empirical max gap."""
    fake_fits = {"inter_arrival": {
        "best_dist": "pareto", "best_params": [0.5, 0.0, 1.0],
        "stats": {"max": 100.0, "mean": 10, "median": 3, "std": 1, "n": 1000,
                  "min": 0, "p25": 1, "p75": 5, "p95": 20, "p99": 50, "skew": 1, "kurtosis": 1, "cv": 1},
    }}
    gen = WorkloadScenarioGenerator(fake_fits, seed=0)
    times = gen.arrivals_renewal(2000, cap_multiple=5.0)
    gaps = np.diff(times)
    assert gaps.max() <= 100.0 * 5.0 + 1e-6
