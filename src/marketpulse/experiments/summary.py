"""Aggregate walk-forward predictions without treating fold means as pooled error."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketpulse.evaluation.regimes import (
    StressPeriod,
    coverage_status,
    stress_membership,
    stress_periods,
)
from marketpulse.experiments.benchmark import _metrics


def pooled_metrics(predictions: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    """Score each model on the concatenated test rows inside each group.

    Skill uses the zero-return and historical-mean predictions stored on those
    same rows. A historical-mean forecast keeps the training mean of its own fold.
    """

    required = {"dataset", "horizon", "model", "date", "fold", "y_true", "y_pred", *group_columns}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for key, frame in predictions.groupby(list(group_columns), sort=False):
        fields = _key_fields(group_columns, key)
        zero = _model_frame(frame, "zero_return")
        mean = _model_frame(frame, "historical_mean")
        for model in list(dict.fromkeys(frame["model"])):
            part = _model_frame(frame, str(model))
            if not part["date"].equals(zero["date"]) or not part["fold"].equals(zero["fold"]):
                raise AssertionError(f"{model} is not aligned with the zero-return rows")
            if not np.allclose(part["y_true"].to_numpy(), zero["y_true"].to_numpy()):
                raise AssertionError(f"{model} does not share the zero-return outcomes")
            metrics = _metrics(
                part["y_true"].to_numpy(dtype=float),
                part["y_pred"].to_numpy(dtype=float),
                zero["y_pred"].to_numpy(dtype=float),
                mean["y_pred"].to_numpy(dtype=float),
            )
            rows.append({**fields, "model": model, **metrics})
    result = pd.DataFrame(rows)
    result["rank"] = result.groupby(list(group_columns))["rmse"].rank(method="min")
    return result


def fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average fold RMSE separately from the pooled scores, and rank each fold.

    Rank 1 is the lowest test RMSE. Ties share that rank. A fold counts as a
    baseline win only when skill is strictly positive.
    """

    test = fold_metrics.loc[fold_metrics["partition"] == "test"].copy()
    if test.empty:
        raise ValueError("fold metrics have no test rows")
    test["rank"] = test.groupby(["dataset", "horizon", "fold"])["rmse"].rank(method="min")
    rows: list[dict[str, object]] = []
    for (dataset, horizon, model), frame in test.groupby(
        ["dataset", "horizon", "model"], sort=False
    ):
        skill_zero = frame["skill_vs_zero"].to_numpy(dtype=float)
        skill_mean = frame["skill_vs_historical_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "horizon": int(horizon),
                "model": model,
                "n_folds": int(frame["fold"].nunique()),
                "mean_fold_rmse": float(frame["rmse"].mean()),
                "median_fold_rmse": float(frame["rmse"].median()),
                "mean_rank": float(frame["rank"].mean()),
                "rank_std": float(frame["rank"].std(ddof=0)),
                "rank_changes": int(np.sum(np.diff(frame.sort_values("fold")["rank"]) != 0)),
                "times_best": int(np.sum(frame["rank"].to_numpy() == 1)),
                "folds_better_than_zero": int(np.sum(skill_zero > 0.0)),
                "folds_better_than_historical_mean": int(np.sum(skill_mean > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def stress_tables(predictions: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Score the predeclared windows and the origins outside all three windows.

    Coverage compares tested origins with every eligible origin whose decision
    date lies in the window. A partial window keeps its sample size and is not
    a complete episode.
    """

    periods = stress_periods()
    pieces = [_period_table(predictions, eligible, period) for period in periods]
    pieces.append(_outside_table(predictions, eligible, periods))
    return pd.concat(pieces, ignore_index=True)


def _period_table(
    predictions: pd.DataFrame, eligible: pd.DataFrame, period: StressPeriod
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (dataset, horizon), frame in predictions.groupby(["dataset", "horizon"], sort=False):
        eligible_dates = _eligible_dates(eligible, str(dataset), int(horizon))
        tested_dates = _tested_dates(frame)
        eligible_mask = stress_membership(eligible_dates, period)
        tested_mask = stress_membership(tested_dates, period)
        pieces.append(
            _scored_slice(
                frame,
                str(dataset),
                int(horizon),
                period.key,
                period.label,
                eligible_mask,
                tested_dates.loc[tested_mask],
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _outside_table(
    predictions: pd.DataFrame,
    eligible: pd.DataFrame,
    periods: tuple[StressPeriod, ...],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (dataset, horizon), frame in predictions.groupby(["dataset", "horizon"], sort=False):
        eligible_dates = _eligible_dates(eligible, str(dataset), int(horizon))
        tested_dates = _tested_dates(frame)
        eligible_mask = ~_any_period(eligible_dates, periods)
        tested_mask = ~_any_period(tested_dates, periods)
        pieces.append(
            _scored_slice(
                frame,
                str(dataset),
                int(horizon),
                "outside_stress",
                "Outside the stress windows",
                eligible_mask,
                tested_dates.loc[tested_mask],
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _scored_slice(
    frame: pd.DataFrame,
    dataset: str,
    horizon: int,
    key: str,
    label: str,
    eligible_mask: np.ndarray,
    tested_dates: pd.Series,
) -> pd.DataFrame:
    n_eligible = int(np.sum(eligible_mask))
    n_tested = int(len(tested_dates))
    status = coverage_status(n_eligible, n_tested)
    base = {
        "dataset": dataset,
        "horizon": horizon,
        "slice": key,
        "slice_label": label,
        "coverage": status,
        "n_eligible": n_eligible,
        "n_tested": n_tested,
    }
    selected = set(pd.to_datetime(tested_dates).dt.date.astype(str))
    part = frame.loc[frame["date"].isin(selected)].copy()
    if part.empty:
        return pd.DataFrame([{**base, "model": None, "n": 0}])
    scored = pooled_metrics(part, ("dataset", "horizon"))
    for column, value in base.items():
        if column not in ("dataset", "horizon"):
            scored[column] = value
    return scored


def _eligible_dates(eligible: pd.DataFrame, dataset: str, horizon: int) -> pd.Series:
    mask = (eligible["dataset"] == dataset) & (eligible["horizon"] == horizon)
    return pd.to_datetime(eligible.loc[mask, "date"])


def _tested_dates(frame: pd.DataFrame) -> pd.Series:
    zero = frame.loc[frame["model"] == "zero_return", "date"].drop_duplicates()
    return pd.to_datetime(zero)


def _any_period(dates: pd.Series, periods: tuple[StressPeriod, ...]) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    for period in periods:
        mask = mask | stress_membership(dates, period)
    return mask


def _model_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    part = frame.loc[frame["model"] == model].sort_values(["fold", "date"])
    if part.empty:
        raise ValueError(f"predictions do not include {model}")
    return part.reset_index(drop=True)


def _key_fields(columns: tuple[str, ...], key: object) -> dict[str, object]:
    if not isinstance(key, tuple):
        key = (key,)
    return dict(zip(columns, key, strict=True))
