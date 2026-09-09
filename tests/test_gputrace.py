import numpy as np
import pandas as pd
import pytest

import gputrace as gt
from gputrace import schema
from gputrace.exporters.k8s_sim import export_scenario


@pytest.fixture(scope="module")
def fit():
    return gt.reference_fit()


def test_reference_fit_has_all_fields(fit):
    for name in ["duration", "num_cpu", "num_gpu", "wait_time", "inter_arrival"]:
        assert name in fit.fields
        assert fit.fields[name].n > 0


@pytest.mark.parametrize("scenario_name", [s.name for s in gt.list_scenarios()])
def test_every_scenario_produces_valid_schema(fit, scenario_name):
    df = gt.generate(scenario_name, fit, n_jobs=300, seed=7)
    schema.validate(df, strict=True)
    assert len(df) == 300
    assert (df["submit_time"].diff().dropna() >= 0).all(), "submit_time must be non-decreasing"
    assert df["source"].eq(scenario_name).all()


def test_reproducibility_same_seed(fit):
    a = gt.generate("bursty_arrivals", fit, n_jobs=200, seed=42)
    b = gt.generate("bursty_arrivals", fit, n_jobs=200, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ(fit):
    a = gt.generate("bursty_arrivals", fit, n_jobs=200, seed=1)
    b = gt.generate("bursty_arrivals", fit, n_jobs=200, seed=2)
    assert not a["submit_time"].equals(b["submit_time"])


def test_bursty_arrivals_more_clustered_than_baseline(fit):
    """Sanity check the Hawkes process actually produces higher
    inter-arrival-time coefficient of variation than the homogeneous
    Poisson baseline scenario, at matched mean rate."""
    baseline = gt.generate("baseline", fit, n_jobs=3000, seed=1, rate=2.0)
    bursty = gt.generate("bursty_arrivals", fit, n_jobs=3000, seed=1, rate=2.0)

    def cv(df):
        ia = df["submit_time"].diff().dropna()
        return ia.std() / ia.mean()

    assert cv(bursty) > cv(baseline)


def test_gpu_fragmentation_scenario_populates_gpu_milli(fit):
    df = gt.generate("gpu_fragmentation_sharing", fit, n_jobs=500, seed=1)
    assert "gpu_milli" in df.columns
    assert df["gpu_milli"].notna().any()
    # sharing quanta are only ever 250/500/750/1000
    assert set(df.loc[df["num_gpu"] > 0, "gpu_milli"].unique()).issubset({250, 500, 750, 1000})


def test_infinite_mean_fields_are_capped(fit):
    """Even if a field's best fit implied an infinite theoretical mean, no
    single sample should blow up to an unusable magnitude."""
    df = gt.generate("heavy_tail_demand", fit, n_jobs=2000, seed=1)
    assert df["duration"].max() < 1e6  # sane upper bound for a seconds-scale field


def test_schema_validate_rejects_missing_columns():
    bad = pd.DataFrame({"job_id": ["a"], "submit_time": [0.0]})
    with pytest.raises(schema.SchemaError):
        schema.validate(bad)


def test_schema_validate_rejects_negative_values():
    df = gt.generate("baseline", gt.reference_fit(), n_jobs=50, seed=1)
    df.loc[0, "duration"] = -5.0
    with pytest.raises(schema.SchemaError):
        schema.validate(df, strict=True)


def test_schema_validate_rejects_duplicate_job_ids():
    df = gt.generate("baseline", gt.reference_fit(), n_jobs=50, seed=1)
    df.loc[1, "job_id"] = df.loc[0, "job_id"]
    with pytest.raises(schema.SchemaError):
        schema.validate(df, strict=True)


def test_k8s_sim_exporter_writes_both_files(fit, tmp_path):
    df = gt.generate("high_contention", fit, n_jobs=800, seed=1)
    export_scenario(df, tmp_path, n_nodes=5, gpus_per_node=8)

    pods = pd.read_csv(tmp_path / "pods.csv")
    nodes = pd.read_csv(tmp_path / "nodes.csv")

    assert len(pods) == len(df)
    assert {"pod_id", "submit_time", "duration", "milli_cpu", "milli_gpu", "gpu_number", "gpu_type"}.issubset(pods.columns)
    assert len(nodes) == 5
    assert (nodes["milli_gpu_capacity"] == 8000).all()
    # pods should be sorted by submit_time for a downstream event-driven consumer
    assert (pods["submit_time"].diff().dropna() >= 0).all()


def test_k8s_sim_exporter_autosizes_nodes(fit, tmp_path):
    df = gt.generate("baseline", fit, n_jobs=1000, seed=1)
    export_scenario(df, tmp_path, n_nodes=0, gpus_per_node=8, target_utilization=0.5)
    nodes = pd.read_csv(tmp_path / "nodes.csv")
    assert len(nodes) >= 1


def test_list_loaders_includes_builtins():
    names = gt.list_loaders()
    assert {"alibaba2020", "google2011", "synthetic"}.issubset(set(names))


def test_analysis_json_roundtrip(fit, tmp_path):
    path = tmp_path / "fit.json"
    fit.to_json(path)
    reloaded = gt.AnalysisResult.from_json(path)
    assert reloaded.n_jobs == fit.n_jobs
    assert set(reloaded.fields) == set(fit.fields)


def test_synthetic_loader_roundtrip(fit, tmp_path):
    df = gt.generate("baseline", fit, n_jobs=200, seed=1)
    path = tmp_path / "trace.csv"
    df.to_csv(path, index=False)

    from gputrace.loaders import get_loader

    reloaded = get_loader("synthetic").load(path)
    assert len(reloaded) == len(df)
    assert reloaded["source"].eq("synthetic").all()
