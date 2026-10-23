import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.adaptive_calibration import adaptive_intervals
from marketpulse.evaluation.calibration import chronological_intervals


def source():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=500)
    origins = np.r_[np.arange(190), np.arange(200, 480)]
    return pd.DataFrame(
        {
            "date": dates[origins].strftime("%Y-%m-%d"),
            "target_date": dates[origins + 5].strftime("%Y-%m-%d"),
            "origin_position": origins,
            "y_true": rng.normal(size=len(origins)),
            "y_pred": 0.0,
            "regime": "low",
            "stress": "outside_stress",
            "volatility": 1.0,
        }
    )


def test_zero_feedback_matches_fixed_trailing_bounds():
    frame = source()
    actual = adaptive_intervals(frame, window=100, gamma=0, minimum=50)
    expected = chronological_intervals(
        frame, method="trailing", levels=[0.8, 0.9], minimum=50, window=100
    )
    for level in [80, 90]:
        np.testing.assert_allclose(
            actual[[f"lower_{level}", f"upper_{level}"]],
            expected[[f"lower_{level}", f"upper_{level}"]],
            equal_nan=True,
        )


def test_future_outcome_and_equal_endpoint_do_not_change_current_state():
    frame = source()
    first = adaptive_intervals(frame, window=100, gamma=0.05, minimum=50)
    changed = frame.copy()
    changed.loc[150:, "y_true"] += 100
    second = adaptive_intervals(changed, window=100, gamma=0.05, minimum=50)
    prior = frame.date <= frame.target_date.iloc[150]
    pd.testing.assert_frame_equal(first.loc[prior], second.loc[prior])
    current = frame.date == frame.target_date.iloc[150]
    assert current.sum() == 1
    assert (
        first.loc[first.eligible, "residual_cutoff_date"] < first.loc[first.eligible, "date"]
    ).all()
    assert (first.feedback_n <= first.calibration_n.cumsum()).all()


def test_own_errors_affect_state_only_after_maturity_and_clipping_is_recorded():
    frame = source()
    frame.loc[220:, "y_true"] = 1000.0
    result = adaptive_intervals(frame, window=100, gamma=0.05, minimum=50)
    assert result.clipped_90.max() > 0
    assert result.alpha_90.between(0.02, 0.5).all()
    assert result.feedback_n.iloc[0] == 0
    assert result.feedback_n.iloc[-1] > 0
    with pytest.raises(ValueError):
        adaptive_intervals(frame, window=20, gamma=0.01, minimum=50)
    frame.loc[10, "target_date"] = frame.date.iloc[10]
    with pytest.raises(ValueError, match="endpoints"):
        adaptive_intervals(frame, window=100, gamma=0.01, minimum=50)


def test_independent_feedback_auditor_detects_tampering():
    from marketpulse.experiments.adaptive_study import audit_adaptive

    frame = source()
    record = adaptive_intervals(frame, window=100, gamma=0.01, minimum=50)
    design = {"minimum": 50, "levels": [0.8, 0.9], "alpha_bounds": [0.02, 0.5]}
    variant = {"window": 100, "gamma": 0.01}
    assert audit_adaptive(frame, record, variant, design) == 2 * record.eligible.sum()
    changed = record.copy()
    changed.loc[200, "alpha_90"] += 0.01
    with pytest.raises(AssertionError):
        audit_adaptive(frame, changed, variant, design)


def test_independent_block_ci_matches_explicit_resampler_with_gaps_and_tail():
    from marketpulse.evaluation.reliability import moving_block_means
    from marketpulse.experiments.adaptive_study import explicit_bootstrap_bounds

    values = np.linspace(0, 1, 113)
    positions = np.r_[np.arange(57), np.arange(70, 126)]
    expected = np.quantile(
        moving_block_means(values[:, None], positions, 20, 500, 11171)[:, 0], [0.025, 0.975]
    )
    np.testing.assert_allclose(
        explicit_bootstrap_bounds(values, positions, 20, 500, 11171), expected
    )
