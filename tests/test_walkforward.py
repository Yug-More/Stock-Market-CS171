"""Tests for walk-forward folds, pooled metrics, and regime assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.partitions import terminal_chronological_split, walk_forward_splits
from marketpulse.evaluation.regimes import (
    assign_volatility_regime,
    coverage_status,
    stress_membership,
    stress_periods,
    volatility_thresholds,
)
from marketpulse.experiments.benchmark import BenchmarkSettings, score_split
from marketpulse.experiments.summary import fold_summary, pooled_metrics, stress_tables
from marketpulse.features import build_features
from marketpulse.targets import forward_log_return


def test_walk_forward_expands_and_does_not_repeat_tests() -> None:
    origins = np.arange(100)
    folds = walk_forward_splits(
        origins,
        horizon=1,
        minimum_training_origins=10,
        validation_window=5,
        test_window=5,
        step=5,
    )
    assert len(folds) == 16
    np.testing.assert_array_equal(folds[-1].test, np.arange(95, 100))
    np.testing.assert_array_equal(folds[0].test, np.arange(20, 25))
    terminal = terminal_chronological_split(
        origins,
        horizon=1,
        minimum_training_origins=10,
        validation_window=5,
        test_window=5,
    )
    np.testing.assert_array_equal(folds[-1].test, terminal.test)
    np.testing.assert_array_equal(folds[-1].train, terminal.train)
    scored: list[int] = []
    for fold in folds:
        assert int(fold.train.max()) + fold.horizon < int(fold.validation.min())
        assert int(fold.validation.max()) + fold.horizon < int(fold.test.min())
        assert len(fold.train) >= 10
        scored.extend(fold.test.tolist())
    assert len(scored) == len(set(scored))
    assert list(folds[0].train) == list(folds[1].train[: len(folds[0].train)])


def test_overlapping_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="step"):
        walk_forward_splits(
            np.arange(50),
            horizon=1,
            minimum_training_origins=5,
            validation_window=5,
            test_window=10,
            step=5,
        )


def test_volatility_thresholds_ignore_later_observations() -> None:
    training = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    low_max, high_min = volatility_thresholds(training)
    expected_low, expected_high = np.quantile(training, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    assert low_max == pytest.approx(float(expected_low))
    assert high_min == pytest.approx(float(expected_high))
    labels = assign_volatility_regime(
        np.array([low_max, (low_max + high_min) / 2.0, high_min, high_min + 1.0]), low_max, high_min
    )
    assert list(labels) == ["low", "medium", "medium", "high"]
    shifted = volatility_thresholds(np.concatenate([training, [100.0]]))
    assert shifted != (low_max, high_min)


def test_stress_windows_match_the_research_design() -> None:
    periods = {period.key: period for period in stress_periods()}
    assert periods["dot_com"].start == pd.Timestamp("2000-03-01")
    assert periods["dot_com"].end == pd.Timestamp("2002-12-31")
    assert periods["financial_crisis"].start == pd.Timestamp("2008-09-01")
    assert periods["financial_crisis"].end == pd.Timestamp("2009-03-31")
    assert periods["covid_crash"].start == pd.Timestamp("2020-02-15")
    assert periods["covid_crash"].end == pd.Timestamp("2020-04-30")
    dates = pd.Series(pd.to_datetime(["2000-02-29", "2000-03-01", "2002-12-31", "2003-01-01"]))
    mask = stress_membership(dates, periods["dot_com"])
    assert list(mask) == [False, True, True, False]


def test_coverage_status_distinguishes_partial_windows() -> None:
    assert coverage_status(5, 5) == "complete"
    assert coverage_status(5, 2) == "partial"
    assert coverage_status(5, 0) == "unavailable"
    with pytest.raises(ValueError, match="tested"):
        coverage_status(2, 3)


def test_pooled_skill_uses_the_same_rows_as_the_baselines() -> None:
    predictions = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "model": "zero_return",
                "date": "2020-01-01",
                "y_true": 1.0,
                "y_pred": 0.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "model": "historical_mean",
                "date": "2020-01-01",
                "y_true": 1.0,
                "y_pred": 0.5,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "model": "linear_regression",
                "date": "2020-01-01",
                "y_true": 1.0,
                "y_pred": 1.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 1,
                "model": "zero_return",
                "date": "2020-01-02",
                "y_true": 1.0,
                "y_pred": 0.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 1,
                "model": "historical_mean",
                "date": "2020-01-02",
                "y_true": 1.0,
                "y_pred": 0.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 1,
                "model": "linear_regression",
                "date": "2020-01-02",
                "y_true": 1.0,
                "y_pred": 1.0,
            },
        ]
    )
    pooled = pooled_metrics(predictions, ("dataset", "horizon"))
    linear = pooled.loc[pooled["model"] == "linear_regression"].iloc[0]
    assert linear["n"] == 2
    assert linear["rmse"] == pytest.approx(0.0)
    assert linear["skill_vs_zero"] == pytest.approx(1.0)
    metrics = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "partition": "test",
                "model": "zero_return",
                "rmse": 2.0,
                "skill_vs_zero": 0.0,
                "skill_vs_historical_mean": -0.1,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "partition": "test",
                "model": "linear_regression",
                "rmse": 1.0,
                "skill_vs_zero": 0.2,
                "skill_vs_historical_mean": 0.1,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 1,
                "partition": "test",
                "model": "zero_return",
                "rmse": 2.0,
                "skill_vs_zero": 0.0,
                "skill_vs_historical_mean": 0.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 1,
                "partition": "test",
                "model": "linear_regression",
                "rmse": 3.0,
                "skill_vs_zero": -0.4,
                "skill_vs_historical_mean": -0.2,
            },
        ]
    )
    summary = fold_summary(metrics)
    linear_summary = summary.loc[summary["model"] == "linear_regression"].iloc[0]
    assert linear_summary["mean_fold_rmse"] == pytest.approx(2.0)
    assert linear_summary["times_best"] == 1
    assert linear_summary["folds_better_than_zero"] == 1


def test_stress_table_marks_an_untested_window_unavailable() -> None:
    predictions = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "model": "zero_return",
                "date": "2020-03-01",
                "y_true": 0.1,
                "y_pred": 0.0,
            },
            {
                "dataset": "synthetic",
                "horizon": 1,
                "fold": 0,
                "model": "historical_mean",
                "date": "2020-03-01",
                "y_true": 0.1,
                "y_pred": 0.0,
            },
        ]
    )
    eligible = pd.DataFrame(
        {
            "dataset": ["synthetic", "synthetic"],
            "horizon": [1, 1],
            "date": ["2001-06-01", "2020-03-01"],
        }
    )
    table = stress_tables(predictions, eligible)
    dot_com = table.loc[table["slice"] == "dot_com"].iloc[0]
    covid = table.loc[(table["slice"] == "covid_crash") & (table["model"] == "zero_return")].iloc[0]
    assert dot_com["coverage"] == "unavailable"
    assert int(dot_com["n_tested"]) == 0
    assert covid["coverage"] == "complete"
    assert int(covid["n"]) == 1


def test_two_folds_recompute_the_training_mean() -> None:
    market = _market(360)
    features = build_features(market).frame
    target = forward_log_return(market["close"], 1)
    origins = np.flatnonzero(features.notna().all(axis=1).to_numpy() & target.notna().to_numpy())
    folds = walk_forward_splits(
        origins,
        horizon=1,
        minimum_training_origins=40,
        validation_window=20,
        test_window=20,
        step=20,
    )
    assert len(folds) >= 2
    scored = [
        score_split(
            dataset_name="synthetic",
            index_id="TEST",
            horizon=1,
            feature_matrix=features.to_numpy(dtype=float),
            target_values=target.to_numpy(dtype=float),
            target=target,
            features=features,
            dates=pd.to_datetime(market["date"]),
            split=fold,
            settings=_settings(),
            seed=171,
            models=("zero_return", "historical_mean", "linear_regression"),
            fold_index=index,
            n_folds=len(folds),
        )
        for index, fold in enumerate(folds[:2])
    ]
    means = [
        model["details"]["training_mean"]
        for result in scored
        for model in result["models"]
        if model["model"] == "historical_mean"
    ]
    assert means[0] != pytest.approx(means[1])
    dates = [row["date"] for row in scored[0]["predictions"] if row["model"] == "zero_return"]
    later = [row["date"] for row in scored[1]["predictions"] if row["model"] == "zero_return"]
    assert set(dates).isdisjoint(later)


def test_real_series_final_fold_matches_the_terminal_benchmark() -> None:
    from marketpulse.data.catalog import find_project_root, load_catalog
    from marketpulse.data.loader import load_market_csv

    root = find_project_root()
    entry = next(item for item in load_catalog(root) if item.name == "NASDAQ-100")
    market = load_market_csv(root / entry.path, dataset_name=entry.name)
    features = build_features(market).frame
    target = forward_log_return(market["close"], 1)
    origins = np.flatnonzero(features.notna().all(axis=1).to_numpy() & target.notna().to_numpy())
    folds = walk_forward_splits(
        origins,
        horizon=1,
        minimum_training_origins=2520,
        validation_window=252,
        test_window=252,
        step=252,
    )
    dates = pd.to_datetime(market["date"])
    assert pd.Timestamp(dates.iloc[int(folds[-1].test[0])]).date().isoformat() == "2023-12-29"
    assert pd.Timestamp(dates.iloc[int(folds[-1].test[-1])]).date().isoformat() == "2024-12-30"
    assert len(folds[-1].test) == 252
    assert len(folds) > 2


def _market(n: int) -> pd.DataFrame:
    close = 100.0 * np.exp(np.linspace(0.0, 0.4, n))
    close[: n // 2] *= np.linspace(1.0, 1.2, n // 2)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2000-01-03", periods=n),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )


def _settings() -> BenchmarkSettings:
    return BenchmarkSettings(
        lookback=5,
        rf_n_estimators=4,
        rf_min_samples_leaf=1,
        rf_max_depth=(2,),
        lstm_hidden_size=(4,),
        lstm_learning_rate=0.01,
        lstm_batch_size=16,
        lstm_max_epochs=1,
        lstm_patience=1,
    )
