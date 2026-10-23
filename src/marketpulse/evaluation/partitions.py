"""Chronological splits with an embargo around each boundary.

The terminal split places one test window at the end of the series. The
walk-forward splits place that same window, then step backward by the
protocol step, and keep every earlier window that still has the minimum
training length. Test blocks do not share origins.
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
    split = ChronologicalSplit(
        train=train.astype(int),
        validation=validation.astype(int),
        test=test.astype(int),
        purged_before_validation=purged[purged < validation_start].astype(int),
        purged_before_test=purged[purged >= validation_start].astype(int),
        horizon=horizon,
    )
    assert_embargo(split)
    return split


def walk_forward_splits(
    origins: np.ndarray,
    *,
    horizon: int,
    minimum_training_origins: int,
    validation_window: int,
    test_window: int,
    step: int,
) -> tuple[ChronologicalSplit, ...]:
    """Expand the training history and step the test window forward by ``step``.

    Windows are anchored on the latest eligible origins, so the final fold is
    the terminal split. Each earlier fold uses the same validation and test
    lengths on the prefix that ends where the next test block begins. A fold
    whose embargoed training sample is shorter than the minimum is omitted,
    and every fold before it is omitted as well. ``step`` must be at least
    ``test_window`` so a test origin is scored only once.
    """

    _require_positive("step", step)
    if step < test_window:
        raise ValueError("step must be at least the test window so test origins are not repeated")
    eligible = np.asarray(origins, dtype=int)
    if eligible.ndim != 1 or len(eligible) == 0:
        raise ValueError("origins must be a non-empty one-dimensional index")
    if np.any(np.diff(eligible) <= 0):
        raise ValueError("origins must be strictly increasing")

    folds: list[ChronologicalSplit] = []
    end = len(eligible)
    while end >= test_window:
        try:
            split = terminal_chronological_split(
                eligible[:end],
                horizon=horizon,
                minimum_training_origins=minimum_training_origins,
                validation_window=validation_window,
                test_window=test_window,
            )
        except ValueError as exc:
            if "training window" not in str(exc) and "not enough" not in str(exc):
                raise
            break
        assert_embargo(split)
        folds.append(split)
        end -= step
    if not folds:
        raise ValueError("no walk-forward fold met the minimum training requirement")
    ordered = tuple(reversed(folds))
    _assert_walk_forward(ordered)
    return ordered


def assert_embargo(split: ChronologicalSplit) -> None:
    """Fail if a label in an earlier block can read a price in a later block."""

    if split.train.size == 0 or split.validation.size == 0 or split.test.size == 0:
        raise AssertionError("a walk-forward block is empty")
    if np.any(split.train + split.horizon >= int(split.validation.min())):
        raise AssertionError("a training label reaches the validation block")
    if np.any(split.validation + split.horizon >= int(split.test.min())):
        raise AssertionError("a validation label reaches the test block")
    train = set(split.train.tolist())
    validation = set(split.validation.tolist())
    test = set(split.test.tolist())
    if train & validation or train & test or validation & test:
        raise AssertionError("train, validation, and test origins overlap")


def _assert_walk_forward(folds: tuple[ChronologicalSplit, ...]) -> None:
    seen: set[int] = set()
    previous: ChronologicalSplit | None = None
    for fold in folds:
        test_ids = fold.test.tolist()
        if seen.intersection(test_ids):
            raise AssertionError("a test origin is scored in more than one fold")
        seen.update(test_ids)
        if previous is not None:
            if int(fold.test.min()) <= int(previous.test.max()):
                raise AssertionError("test blocks are not in chronological order")
            if int(fold.train[0]) != int(previous.train[0]):
                raise AssertionError("the expanding window changed its first training origin")
            if not set(previous.train.tolist()).issubset(set(fold.train.tolist())):
                raise AssertionError("a later fold dropped an earlier training origin")
            if int(fold.train[-1]) <= int(previous.train[-1]):
                raise AssertionError("the training window did not expand")
        previous = fold


def _require_positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
