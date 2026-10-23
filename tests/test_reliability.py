"""Known arithmetic, dependent-series resampling, and pairing adversaries."""

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.reliability import (
    compare_losses,
    hac_variance,
    holm_adjust,
    moving_block_means,
    paired_losses,
    valid_block_starts,
)


def test_blocks_never_cross_structural_session_gaps():
    assert valid_block_starts(np.array([0, 1, 2, 10, 11, 12]), 3).tolist() == [0, 3]
    values = np.column_stack([np.arange(6), 2 * np.arange(6)])
    boot = moving_block_means(values, np.array([0, 1, 2, 10, 11, 12]), 3, 100, 171)
    np.testing.assert_allclose(boot[:, 1], 2 * boot[:, 0])
    np.testing.assert_array_equal(
        boot, moving_block_means(values, np.array([0, 1, 2, 10, 11, 12]), 3, 100, 171)
    )


def test_hac_matches_direct_bartlett_arithmetic():
    values = np.array([1.0, 2.0, 4.0, 8.0])
    centered = values - values.mean()
    expected = ((centered @ centered) / 4 + (centered[1:] @ centered[:-1]) / 4) / 4
    assert hac_variance(values, np.arange(4), 1) == pytest.approx(expected)
    # A gap removes cross-boundary covariance without changing gamma0.
    expected_gap = (
        (centered @ centered) / 4 + (centered[0] * centered[1] + centered[2] * centered[3]) / 4
    ) / 4
    assert hac_variance(values, np.array([0, 1, 5, 6]), 1) == pytest.approx(expected_gap)


def test_holm_known_family_and_missing_tests():
    np.testing.assert_allclose(holm_adjust(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm_adjust(np.array([0.01, np.nan])), [0.02, 1.0])
    with pytest.raises(ValueError):
        holm_adjust(np.array([1.1]))


def test_conditional_resampling_keeps_original_timeline():
    a = np.tile([1.0, 2.0, 3.0, 4.0], 100)
    b = np.full(400, 3.0)
    mask = np.arange(400) % 2 == 0
    result = compare_losses(a, b, np.arange(400), length=10, replicates=200, seed=171, mask=mask)
    assert result["n"] == 200
    assert result["timeline_n"] == 400
    assert result["mean_loss_difference"] == pytest.approx(-1.0)
    assert result["ci_lower"] <= result["mean_loss_difference"] <= result["ci_upper"]
    small = compare_losses(a, b, np.arange(400), length=40, replicates=200, seed=171, mask=mask)
    assert small["status"] == "descriptive_small_sample"
    assert "p_value" not in small


def test_missing_and_unpaired_observations_are_rejected():
    with pytest.raises(ValueError):
        compare_losses(np.array([np.nan]), np.ones(1), np.arange(1), length=1, replicates=2, seed=1)
    frame = pd.DataFrame(
        {
            "model": ["a", "b"],
            "date": ["2000-01-01", "2000-01-02"],
            "target_date": ["2000-01-03", "2000-01-03"],
            "origin_position": [1, 2],
            "y_true": [1.0, 1.0],
            "y_pred": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="unpaired"):
        paired_losses(frame, "a", "b")


def test_prefix_sum_bootstrap_matches_independent_row_resampling():
    positions = np.arange(50)
    values = np.column_stack([positions**2, positions, np.ones(50)])
    rng = np.random.default_rng(6171)
    starts = rng.integers(0, 44, size=(100, 7))
    short = rng.choice(np.arange(50), size=100)
    indices = np.concatenate(
        [(starts[:, :, None] + np.arange(7)).reshape(100, 49), short[:, None]], axis=1
    )
    expected = values[indices].mean(axis=1)
    np.testing.assert_allclose(moving_block_means(values, positions, 7, 100, 6171), expected)


def test_hac_detects_positive_serial_dependence_in_fixed_ar_series():
    rng = np.random.default_rng(171)
    values = np.zeros(2000)
    innovations = rng.normal(size=2000)
    for i in range(1, len(values)):
        values[i] = 0.8 * values[i - 1] + innovations[i]
    assert hac_variance(values, np.arange(len(values)), 19) > 3 * np.var(values) / len(values)
