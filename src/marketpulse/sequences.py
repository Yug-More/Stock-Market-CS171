"""Sequence windows that end at the forecast origin."""

from __future__ import annotations

import numpy as np


def sequence_windows(features: np.ndarray, origins: np.ndarray, lookback: int) -> np.ndarray:
    """Return windows of shape ``(len(origins), lookback, n_features)``.

    The window for origin ``t`` is ``features[t - lookback + 1 : t + 1]``.
    It ends on row ``t`` and does not include row ``t + 1``.
    """

    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer of at least 2")
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    index = np.asarray(origins, dtype=int)
    if index.ndim != 1:
        raise ValueError("origins must be one-dimensional")
    if len(index) == 0:
        raise ValueError("origins must be non-empty")
    if np.any(index < lookback - 1) or np.any(index >= len(values)):
        raise ValueError("an origin does not have a complete lookback inside the feature array")
    return np.stack([values[origin - lookback + 1 : origin + 1] for origin in index])
