from gputrace import loaders
from gputrace.schema import ALL_COLUMNS


def test_registry_lists_expected_loaders():
    names = loaders.available_loaders()
    assert {"alibaba2020", "google2011", "synthetic"}.issubset(set(names))


def test_alibaba_loader_normalizes_job_level_csv(alibaba_csv):
    df = loaders.get_loader("alibaba2020").load(alibaba_csv)
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 2000
    assert (df["duration"] > 0).all()
    assert (df["num_cpu"] >= 0).all()
    assert (df["num_gpu"] >= 0).all()
    assert df["source"].unique().tolist() == ["alibaba2020"]
    assert set(df["gpu_type"].unique()) <= {"MISC", "T4", "CPU", "P100", "V100", "UNKNOWN"}


def test_google_loader_normalizes_task_events(google_csv):
    df = loaders.get_loader("google2011").load(google_csv)
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) > 0
    assert (df["duration"] > 0).all()
    assert (df["num_gpu"] == 0).all()          # documented limitation of this source
    assert (df["gpu_type"] == "NONE").all()
    assert df["source"].unique().tolist() == ["google2011"]


def test_synthetic_loader_roundtrips_unified_schema(alibaba_csv, tmp_path):
    df1 = loaders.get_loader("alibaba2020").load(alibaba_csv)
    p = tmp_path / "roundtrip.csv"
    df1.to_csv(p, index=False)
    df2 = loaders.get_loader("synthetic").load(str(p))
    assert list(df2.columns) == ALL_COLUMNS
    assert len(df2) == len(df1)


def test_unknown_loader_raises_keyerror():
    import pytest
    with pytest.raises(KeyError):
        loaders.get_loader("does_not_exist")
