"""One terminal chronological split with an embargo around each boundary.

This is the initial benchmark partition. It uses the window lengths from the
research protocol, placed at the end of the series. Repeated walk-forward
folds are a later experiment and are not produced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketpulse.evaluation.splits import exclusive_training_end


@dataclass(frozen=True)
class ChronologicalSplit:
    """Price-row indexes for one embargoed train, validation, and test cut."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    purged_before_validation: np.ndarray
    purged_before_test: np.ndarray
    horizon: int

    @property
    def test_size(self) -> int:
        return int(len(self.test))


def eligible_origins(
    feature_rows_finite: np.ndarray, target_finite: np.ndarray, lookback: int
) -> np.ndarray:
    """Origins whose target is finite and whose lookback rows are all finite."""

    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer of at least 2")
    finite_rows = np.asarray(feature_rows_finite, dtype=bool)
    finite_target = np.asarray(target_finite, dtype=bool)
    if finite_rows.shape != finite_target.shape or finite_rows.ndim != 1:
        raise ValueError("feature and target masks must be one-dimensional and aligned")
    n_rows = len(finite_rows)
    if n_rows < lookback:
        return np.array([], dtype=int)
    cumulative = np.cumsum(finite_rows)
    origins = np.arange(lookback - 1, n_rows)
    previous = origins - lookback
    left = np.zeros(len(origins), dtype=cumulative.dtype)
    positive = previous >= 0
    left[positive] = cumulative[previous[positive]]
    window_complete = (cumulative[origins] - left) == lookback
    selected = origins[window_complete & finite_target[origins]]
    return selected.astype(int)


def terminal_chronological_split(
    origins: np.ndarray,
    *,
    horizon: int,
    minimum_training_origins: int,
    validation_window: int,
    test_window: int,
) -> ChronologicalSplit:
    """Take the latest test window and the latest embargo-clean validation window.

    An origin ``t`` is eligible for the block that starts at ``start`` only
    when ``t + horizon < start``. The origins in between are purged because
    their labels read a price inside the later block. Training uses every
    remaining earlier origin that passes the same test against the validation
    block. The historical mean and every fitted model see only those training
    labels.
    """

    _require_positive("horizon", horizon)
    _require_positive("minimum_training_origins", minimum_training_origins)
    _require_positive("validation_window", validation_window)
    _require_positive("test_window", test_window)
    eligible = np.asarray(origins, dtype=int)
    if eligible.ndim != 1 or len(eligible) == 0:
        raise ValueError("origins must be a non-empty one-dimensional index")
    if np.any(np.diff(eligible) <= 0):
        raise ValueError("origins must be strictly increasing")

    test = eligible[-test_window:]
    if len(test) < test_window:
        raise ValueError("not enough origins for the test window")
    test_start = int(test[0])
    clean_for_test = eligible[eligible + horizon < test_start]
    if len(clean_for_test) < validation_window:
        raise ValueError("not enough embargo-clean origins for the validation window")
    validation = clean_for_test[-validation_window:]
    validation_start = int(validation[0])
    # Reuse the embargo rule: training stops before labels can see validation.
    training_stop = exclusive_training_end(validation_start, horizon)
    train = eligible[eligible < training_stop]
    if len(train) < minimum_training_origins:
        raise ValueError(
            "the embargoed training window has "
            f"{len(train)} origins, below the required {minimum_training_origins}"
        )

    assigned = np.concatenate([train, validation, test])
    purged = eligible[~np.isin(eligible, assigned)]
    return ChronologicalSplit(
        train=train.astype(int),
        validation=validation.astype(int),
        test=test.astype(int),
        purged_before_validation=purged[purged < validation_start].astype(int),
        purged_before_test=purged[purged >= validation_start].astype(int),
        horizon=horizon,
    )


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
