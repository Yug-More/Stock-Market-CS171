"""Tests for the return target, metrics, split embargo, and protocol file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.metrics import (
    directional_accuracy,
    forecast_bias,
    mae,
    mse_skill_score,
    rmse,
)
from marketpulse.evaluation.splits import exclusive_training_end
from marketpulse.protocol import load_forecasting_protocol
from marketpulse.targets import forward_log_return


def test_forward_log_return_alignment() -> None:
    close = pd.Series([100.0, 110.0, 121.0])
    one_day = forward_log_return(close, 1)
    two_day = forward_log_return(close, 2)

    assert len(one_day) == 3
    assert one_day.iloc[0] == pytest.approx(np.log(1.1))
    assert one_day.iloc[1] == pytest.approx(np.log(121.0 / 110.0))
    assert np.isnan(one_day.iloc[2])
    assert two_day.iloc[0] == pytest.approx(np.log(1.21))
    assert np.isnan(two_day.iloc[1])
    assert np.isnan(two_day.iloc[2])


def test_forward_log_return_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="horizon"):
        forward_log_return(pd.Series([1.0, 2.0]), 0)
    with pytest.raises(ValueError, match="positive"):
        forward_log_return(pd.Series([1.0, 0.0]), 1)
    with pytest.raises(ValueError, match="non-numeric"):
        forward_log_return(pd.Series(["1.0", "nope"]), 1)


def test_metrics() -> None:
    actual = np.array([1.0, -2.0, 0.0])
    predicted = np.array([1.0, 2.0, 5.0])

    assert mae(actual, predicted) == pytest.approx(3.0)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((0.0 + 16.0 + 25.0) / 3.0))
    assert forecast_bias(actual, predicted) == pytest.approx(np.mean(predicted - actual))
    assert directional_accuracy(actual, predicted) == pytest.approx(0.5)
    assert mse_skill_score(3.0, 4.0) == pytest.approx(0.25)


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mae(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="same length"):
        rmse(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        forecast_bias(np.array([np.nan]), np.array([1.0]))
    with pytest.raises(ValueError, match="undefined"):
        directional_accuracy(np.array([0.0, 0.0]), np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive"):
        mse_skill_score(1.0, 0.0)


def test_exclusive_training_end() -> None:
    assert exclusive_training_end(10, 1) == 9
    assert exclusive_training_end(10, 5) == 5
    with pytest.raises(ValueError):
        exclusive_training_end(5, 5)
    with pytest.raises(ValueError):
        exclusive_training_end(10, True)  # type: ignore[arg-type]


def test_forecasting_protocol(repo_root: Path) -> None:
    protocol = load_forecasting_protocol(repo_root)
    assert protocol.primary_target == "forward_log_return"
    assert protocol.horizons == (1, 5, 20)
    assert protocol.seed == 171
    assert protocol.minimum_training_origins == 2520
    assert protocol.validation_window == 252
    assert protocol.test_window == 252
    assert protocol.step == 252
    assert protocol.nominal_coverage == pytest.approx(0.9)
