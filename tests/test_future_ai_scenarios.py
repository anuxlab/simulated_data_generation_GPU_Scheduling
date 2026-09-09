import numpy as np
import pytest

from gputrace.analysis import analyze
from gputrace.generators import WorkloadScenarioGenerator, run_scenario


@pytest.fixture
def fit_results(normalized_df):
    return analyze(normalized_df)


def test_spot_preemption_churn_marks_status(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=3)
    df = run_scenario("spot_preemption_churn", gen, n_jobs=2000, preempt_rate=0.5)
    assert set(df["status"].unique()) <= {"PREEMPTED_RESTARTED", "COMPLETED"}
    frac = (df["status"] == "PREEMPTED_RESTARTED").mean()
    assert 0.3 < frac < 0.7  # roughly matches the requested 0.5 rate


def test_long_context_kv_pressure_grows_over_time(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=4)
    df = run_scenario("long_context_kv_pressure", gen, n_jobs=5000,
                       start_mult=1.0, end_mult=5.0, extreme_prob=0.0)
    df = df.sort_values("submit_time")
    first_half = df.iloc[: len(df) // 2]["num_cpu"].mean()
    second_half = df.iloc[len(df) // 2:]["num_cpu"].mean()
    assert second_half > first_half  # demand trends up across the trace window


def test_moe_expert_load_skew_is_actually_skewed(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=5)
    df = run_scenario("moe_expert_load_skew", gen, n_jobs=5000, n_experts=32, skew=1.5)
    counts = df["user"].value_counts()
    # top expert should get meaningfully more jobs than a uniform 1/32 share would give
    assert counts.iloc[0] > (len(df) / 32) * 2


def test_gpu_fragmentation_sharing_uses_only_fractional_values(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=6)
    df = run_scenario("gpu_fragmentation_sharing", gen, n_jobs=1000)
    nonzero = df.loc[df["num_gpu"] > 0, "num_gpu"]
    assert set(np.round(nonzero.unique(), 3)) <= {0.125, 0.25, 0.5, 1.0}


def test_checkpoint_io_burst_injects_cpu_only_short_jobs(fit_results):
    gen = WorkloadScenarioGenerator(fit_results, seed=7)
    df = run_scenario("checkpoint_io_burst", gen, n_jobs=2000, checkpoint_fraction=0.2)
    ckpt_like = df[(df["gpu_type"] == "CPU") & (df["num_gpu"] == 0)]
    assert len(ckpt_like) > 0


def test_llm_inference_bursty_durations_much_shorter_than_baseline(fit_results):
    gen_a = WorkloadScenarioGenerator(fit_results, seed=8)
    inf = run_scenario("llm_inference_bursty", gen_a, n_jobs=2000)
    gen_b = WorkloadScenarioGenerator(fit_results, seed=8)
    base = run_scenario("baseline", gen_b, n_jobs=2000)
    assert inf["duration"].median() < base["duration"].median()
