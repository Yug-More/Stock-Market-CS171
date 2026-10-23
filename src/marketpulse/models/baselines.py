"""Forecasts that do not use the model features."""

from __future__ import annotations

import numpy as np


def zero_return_forecast(n_origins: int) -> np.ndarray:
    """Predict a forward log return of zero at every origin."""

    if isinstance(n_origins, bool) or not isinstance(n_origins, int) or n_origins < 1:
        raise ValueError("n_origins must be a positive integer")
    return np.zeros(n_origins, dtype=float)


def historical_mean_forecast(training_targets: np.ndarray, n_origins: int) -> np.ndarray:
    """Predict the mean of the embargoed training targets.

    The mean is fixed from that training partition. It is not updated with
    validation or test outcomes, and it is not an expanding mean inside the
    evaluation window.
    """

    if isinstance(n_origins, bool) or not isinstance(n_origins, int) or n_origins < 1:
        raise ValueError("n_origins must be a positive integer")
    targets = np.asarray(training_targets, dtype=float)
    if targets.ndim != 1 or targets.size == 0 or not bool(np.isfinite(targets).all()):
        raise ValueError("training_targets must be a non-empty finite vector")
    return np.full(n_origins, float(targets.mean()), dtype=float)
