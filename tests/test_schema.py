import pandas as pd
import pytest

from gputrace.schema import ALL_COLUMNS, SchemaValidationError, validate


def test_validate_fills_optional_defaults():
    df = pd.DataFrame({
        "job_id": ["a", "b"], "submit_time": [0.0, 1.0], "duration": [10.0, 20.0],
        "num_cpu": [1.0, 2.0], "num_gpu": [0.0, 1.0],
    })
    out = validate(df)
    assert list(out.columns) == ALL_COLUMNS
    assert out["gpu_type"].tolist() == ["UNKNOWN", "UNKNOWN"]
    assert out["user"].tolist() == ["UNKNOWN", "UNKNOWN"]


def test_validate_raises_on_missing_required_column():
    df = pd.DataFrame({"job_id": ["a"], "submit_time": [0.0], "duration": [10.0]})
    with pytest.raises(SchemaValidationError):
        validate(df)


def test_validate_strict_rejects_bad_values():
    df = pd.DataFrame({
        "job_id": ["a"], "submit_time": [-5.0], "duration": [10.0],
        "num_cpu": [1.0], "num_gpu": [0.0],
    })
    with pytest.raises(SchemaValidationError):
        validate(df, strict=True)


def test_validate_strict_passes_good_values():
    df = pd.DataFrame({
        "job_id": ["a"], "submit_time": [0.0], "duration": [10.0],
        "num_cpu": [1.0], "num_gpu": [0.0],
    })
    out = validate(df, strict=True)
    assert len(out) == 1
