"""Transformations fit on training rows only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    """Column standardization using the training mean and population variance.

    A zero training variance is left unscaled so a constant column cannot
    divide the validation or test rows by zero.
    """

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        if data.ndim != 2 or data.shape[1] != len(self.mean):
            raise ValueError("values must be a two-dimensional array with one column per feature")
        return (data - self.mean) / self.scale


def fit_standardizer(training_values: np.ndarray) -> Standardizer:
    """Fit column means and scales on training rows and on no other rows."""

    data = np.asarray(training_values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        raise ValueError("training values must contain at least two rows and one column")
    if not bool(np.isfinite(data).all()):
        raise ValueError("training values must be finite")
    mean = data.mean(axis=0)
    scale = data.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return Standardizer(mean=mean, scale=scale)
