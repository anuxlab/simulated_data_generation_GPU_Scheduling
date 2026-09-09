import numpy as np
import pytest
from scipy import stats

from gputrace.analysis.distributions import (
    fit_all,
    infinite_mean_warning,
    summary_stats,
    tail_index_hill,
)


def test_fit_all_recovers_known_lognormal_params():
    rng = np.random.default_rng(0)
    true_sigma, true_scale = 0.8, 500.0
    data = stats.lognorm.rvs(true_sigma, loc=0, scale=true_scale, size=20000, random_state=rng)
    ranked, params = fit_all(data)
    assert ranked.iloc[0]["distribution"] == "lognorm"
    fitted_sigma, fitted_loc, fitted_scale = params["lognorm"]
    assert abs(fitted_sigma - true_sigma) < 0.05
    assert abs(fitted_scale - true_scale) / true_scale < 0.05


def test_fit_all_ranked_by_aic_ascending():
    rng = np.random.default_rng(1)
    data = stats.expon.rvs(scale=10, size=5000, random_state=rng)
    ranked, _ = fit_all(data)
    aics = ranked["AIC"].dropna().tolist()
    assert aics == sorted(aics)
    assert ranked.iloc[0]["delta_AIC"] == 0


def test_fit_all_raises_on_too_few_points():
    with pytest.raises(ValueError):
        fit_all([1, 2, 3])


def test_summary_stats_basic_properties():
    data = [1, 2, 3, 4, 5, 100]
    s = summary_stats(data)
    assert s["n"] == 6
    assert s["min"] == 1
    assert s["max"] == 100
    assert s["median"] == 3.5


def test_tail_index_hill_returns_reasonable_alpha_for_pareto():
    rng = np.random.default_rng(2)
    data = stats.pareto.rvs(2.5, size=20000, random_state=rng)
    result = tail_index_hill(data)
    assert 1.5 < result["alpha"] < 4.0


def test_infinite_mean_warning_flags_low_shape_pareto():
    assert infinite_mean_warning("pareto", (0.5, 0, 1)) is not None
    assert infinite_mean_warning("pareto", (3.0, 0, 1)) is None
    assert infinite_mean_warning("lognorm", (1.0, 0, 1)) is None
