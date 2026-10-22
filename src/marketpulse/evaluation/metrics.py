"""Point-forecast metrics for return predictions.

MAPE is intentionally absent. Near-zero returns make it unstable, so it is
not a primary metric for this project.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""

    actual, predicted = _pair(y_true, y_pred)
    return float(np.mean(np.abs(predicted - actual)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    actual, predicted = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean prediction minus mean outcome.

    Positive bias means the forecasts are too high on average.
    """

    actual, predicted = _pair(y_true, y_pred)
    return float(np.mean(predicted - actual))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of nonzero outcomes whose sign matches the prediction.

    Zero outcomes are excluded. A zero prediction does not match a nonzero
    outcome. The result is undefined when every outcome is zero.
    """

    actual, predicted = _pair(y_true, y_pred)
    observed = actual != 0.0
    if not bool(np.any(observed)):
        raise ValueError("directional accuracy is undefined when every outcome is zero")
    matches = np.sign(predicted[observed]) == np.sign(actual[observed])
    return float(np.mean(matches))


def mse_skill_score(model_mse: float, baseline_mse: float) -> float:
    """Return 1 - model_mse / baseline_mse.

    Positive skill means the model has a smaller squared error than the
    baseline. The baseline MSE must be positive.
    """

    model = _finite_scalar(model_mse, "model_mse")
    baseline = _finite_scalar(baseline_mse, "baseline_mse")
    if baseline <= 0.0:
        raise ValueError("baseline_mse must be positive")
    return float(1.0 - model / baseline)


def _pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = _as_vector(y_true, "y_true")
    predicted = _as_vector(y_pred, "y_pred")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return actual, predicted


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_scalar(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number
