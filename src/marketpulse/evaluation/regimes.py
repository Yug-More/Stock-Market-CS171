"""Volatility terciles and the predeclared historical stress windows.

Thresholds come from the training origins of one walk-forward fold. Stress
dates are the ranges fixed in the research design. They are reporting groups,
not model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRESS_PERIODS = (
    ("dot_com", "Dot-com drawdown", "2000-03-01", "2002-12-31"),
    ("financial_crisis", "Financial crisis", "2008-09-01", "2009-03-31"),
    ("covid_crash", "COVID crash", "2020-02-15", "2020-04-30"),
)


@dataclass(frozen=True)
class StressPeriod:
    """An inclusive calendar window declared before any model was scored."""

    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


def stress_periods() -> tuple[StressPeriod, ...]:
    """Return the three research-design windows, with both endpoints included."""

    return tuple(
        StressPeriod(key, label, pd.Timestamp(start), pd.Timestamp(end))
        for key, label, start, end in STRESS_PERIODS
    )


def volatility_thresholds(training_volatility: np.ndarray) -> tuple[float, float]:
    """Return the training 1/3 and 2/3 quantiles of trailing volatility.

    The quantiles use NumPy's linear method. The test sample is not an argument.
    """

    values = np.asarray(training_volatility, dtype=float)
    if values.ndim != 1 or values.size < 3 or not bool(np.isfinite(values).all()):
        raise ValueError("training volatility must be a finite vector of at least three values")
    low_max, high_min = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    return float(low_max), float(high_min)


def assign_volatility_regime(volatility: np.ndarray, low_max: float, high_min: float) -> np.ndarray:
    """Assign low, medium, and high regimes from frozen training thresholds.

    A value on the lower threshold is low. A value on the upper threshold is
    medium. A value above the upper threshold is high. When the two training
    quantiles are equal, the medium bin is empty.
    """

    values = np.asarray(volatility, dtype=float)
    if values.ndim != 1 or values.size == 0 or not bool(np.isfinite(values).all()):
        raise ValueError("volatility must be a non-empty finite vector")
    if not np.isfinite(low_max) or not np.isfinite(high_min) or high_min < low_max:
        raise ValueError("thresholds must be finite and ordered")
    labels = np.full(values.shape, "medium", dtype=object)
    labels[values <= low_max] = "low"
    labels[values > high_min] = "high"
    return labels


def stress_membership(dates: pd.Series, period: StressPeriod) -> np.ndarray:
    """Return a boolean mask for origins whose decision date falls inside the window."""

    stamps = pd.to_datetime(dates)
    return ((stamps >= period.start) & (stamps <= period.end)).to_numpy()


def coverage_status(eligible_in_period: int, tested_in_period: int) -> str:
    """Describe whether the walk-forward test blocks cover a stress window.

    ``complete`` means every eligible origin in the window was scored.
    ``partial`` means some of those origins were scored and some were not.
    ``unavailable`` means no eligible origin in the window was scored.
    """

    if eligible_in_period < 0 or tested_in_period < 0:
        raise ValueError("counts must be non-negative")
    if tested_in_period > eligible_in_period:
        raise ValueError("tested origins cannot exceed eligible origins")
    if tested_in_period == 0:
        return "unavailable"
    if tested_in_period < eligible_in_period:
        return "partial"
    return "complete"
