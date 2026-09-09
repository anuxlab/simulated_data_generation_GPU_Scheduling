from gputrace.analysis import analyze


def test_analyze_produces_expected_keys(normalized_df):
    results = analyze(normalized_df)
    assert "duration" in results
    assert "num_cpu" in results
    assert "inter_arrival" in results
    assert "segmentation_by_gpu_type" in results
    assert results["n_jobs"] == len(normalized_df)
    for metric in ("duration", "num_cpu"):
        assert "best_dist" in results[metric]
        assert "best_params" in results[metric]
        assert "stats" in results[metric]
        assert "hill_tail" in results[metric]


def test_analyze_is_json_serializable(normalized_df):
    import json
    results = analyze(normalized_df)
    json.dumps(results, default=str)  # must not raise


def test_analyze_segmentation_has_expected_gpu_types(normalized_df):
    results = analyze(normalized_df)
    seg = results["segmentation_by_gpu_type"]
    types = {r["gpu_type"] for r in seg}
    assert types.issubset({"MISC", "T4", "CPU", "P100", "V100"})
    assert len(seg) >= 2
