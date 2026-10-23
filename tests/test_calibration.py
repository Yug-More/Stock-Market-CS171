"""Endpoint-date maturity and future-outcome adversaries for online intervals."""

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.calibration import (
    chronological_intervals,
    interval_metrics,
    interval_score,
    residual_quantile,
)


def sample(n=40):
    dates = pd.bdate_range("2020-01-01", periods=n + 5).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "date": dates[:n],
            "target_date": dates[5:],
            "origin_position": np.arange(n),
            "y_true": np.arange(n, dtype=float),
            "y_pred": np.zeros(n),
            "volatility": np.ones(n),
            "regime": ["low"] * n,
            "stress": ["outside_stress"] * n,
        }
    )


def test_quantile_finite_sample_correction_and_ties():
    assert residual_quantile([1.0, 2.0, 3.0, 4.0], 0.8) == 4
    assert residual_quantile([1.0, 1.0, 1.0, 1.0], 0.8) == 1
    assert np.isinf(residual_quantile([1.0, 2.0], 0.99))
    with pytest.raises(ValueError):
        residual_quantile([2.0, 1.0], 0.8)


def test_actual_endpoint_dates_control_maturity_and_warmup():
    result = chronological_intervals(
        sample(), method="trailing", levels=[0.8, 0.9], minimum=10, window=12
    )
    first = result[result.eligible].iloc[0]
    assert first.origin_position == 15
    assert first.calibration_n == 10
    assert first.residual_cutoff_date < first.date
    assert result.calibration_n.max() == 12
    assert (
        result.loc[result.eligible, "lower_90"] <= result.loc[result.eligible, "lower_80"]
    ).all()
    assert (
        result.loc[result.eligible, "upper_90"] >= result.loc[result.eligible, "upper_80"]
    ).all()


def test_future_outcomes_cannot_change_earlier_intervals():
    original = sample()
    altered = original.copy()
    altered.loc[30:, "y_true"] = 1e8
    kwargs = {"method": "expanding", "levels": [0.8, 0.9], "minimum": 10, "window": 12}
    before = chronological_intervals(original, **kwargs)
    after = chronological_intervals(altered, **kwargs)
    pd.testing.assert_frame_equal(
        before.loc[:34, ["lower_80", "upper_90"]], after.loc[:34, ["lower_80", "upper_90"]]
    )
    # Self-observed and nonmonotone endpoint dates are explicitly rejected.
    altered.loc[0, "target_date"] = altered.loc[0, "date"]
    with pytest.raises(ValueError, match="endpoints"):
        chronological_intervals(altered, **kwargs)


def test_coverage_width_and_interval_score_known_values():
    np.testing.assert_allclose(
        interval_score(np.array([0.0, 3.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.8),
        [2.0, 22.0],
    )
    records = pd.DataFrame(
        {
            "eligible": [True, True, False],
            "y_true": [0.0, 3.0, 100.0],
            "lower_80": [-1.0, -1.0, np.nan],
            "upper_80": [1.0, 1.0, np.nan],
        }
    )
    metrics = interval_metrics(records, 0.8)
    assert metrics["coverage"] == 0.5
    assert metrics["n"] == 2 and metrics["excluded"] == 1
    assert metrics["mean_width"] == 2 and metrics["interval_score"] == pytest.approx(12.0)


def test_volatility_scaling_and_missing_data():
    frame = sample()
    kwargs = {"levels": [0.8], "minimum": 10, "window": 12}
    raw = chronological_intervals(frame, method="trailing", **kwargs)
    scaled = chronological_intervals(frame, method="volatility_scaled", **kwargs)
    np.testing.assert_allclose(raw.upper_80, scaled.upper_80, equal_nan=True)
    frame.loc[2, "volatility"] = np.nan
    with pytest.raises(ValueError, match="volatility"):
        chronological_intervals(frame, method="volatility_scaled", **kwargs)


def test_independent_audit_rejects_changed_bounds_and_cutoffs():
    from marketpulse.experiments.research_audit import audit_intervals

    frame = sample()
    design = {"coverage_levels": [0.8, 0.9], "calibration_minimum": 10, "calibration_trailing": 12}
    records = chronological_intervals(
        frame, method="trailing", levels=[0.8, 0.9], minimum=10, window=12
    )
    assert audit_intervals(frame, records, "trailing", design) > 0
    records.loc[20, "upper_90"] += 1
    with pytest.raises(AssertionError):
        audit_intervals(frame, records, "trailing", design)


def test_count_tree_matches_known_quantiles_with_removal():
    from marketpulse.experiments.research_audit import ScoreTree

    tree = ScoreTree(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))
    for i in range(5):
        tree.update(i, 1)
    assert [tree.quantile(k) for k in range(1, 6)] == [1.0, 1.0, 3.0, 4.0, 5.0]
    tree.update(1, -1)
    assert [tree.quantile(k) for k in range(1, 5)] == [1.0, 3.0, 4.0, 5.0]
