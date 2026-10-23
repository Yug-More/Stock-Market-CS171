"""Tests for causal features, embargoed splits, sequences, and model wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.partitions import eligible_origins, terminal_chronological_split
from marketpulse.evaluation.preprocessing import fit_standardizer
from marketpulse.experiments.benchmark import BenchmarkSettings, run_index_horizon
from marketpulse.features import FEATURE_COLUMNS, build_features
from marketpulse.models.baselines import historical_mean_forecast, zero_return_forecast
from marketpulse.models.classical import fit_linear_regression, select_random_forest
from marketpulse.protocol import ForecastingProtocol
from marketpulse.sequences import sequence_windows
from marketpulse.targets import forward_log_return


def _prices(n: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 100.0 + n - 1.0, n)
    dates = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )


def test_forward_targets_match_the_close_ratio() -> None:
    close = pd.Series([100.0 * (1.01**step) for step in range(21)])
    for horizon in (1, 5, 20):
        target = forward_log_return(close, horizon)
        assert len(target) == len(close)
        assert target.iloc[0] == pytest.approx(np.log(close.iloc[horizon] / close.iloc[0]))
        assert target.iloc[-horizon:].isna().all()
        assert int(target.isna().sum()) == horizon
        assert target.iloc[: len(close) - horizon].notna().all()


def test_features_use_only_prices_through_the_origin() -> None:
    market = _prices()
    original = build_features(market).frame.copy()
    changed = market.copy()
    changed.loc[changed.index[-1], "close"] = changed.loc[changed.index[-1], "close"] * 3.0
    updated = build_features(changed).frame
    pd.testing.assert_frame_equal(original.iloc[:-1], updated.iloc[:-1])
    assert not original.iloc[-1].equals(updated.iloc[-1])


def test_feature_build_is_reproducible() -> None:
    market = _prices()
    pd.testing.assert_frame_equal(build_features(market).frame, build_features(market).frame)


def test_eligible_origins_require_a_complete_lookback() -> None:
    finite_rows = np.array([False, True, True, True, True])
    finite_target = np.array([True, True, True, True, False])
    assert np.array_equal(eligible_origins(finite_rows, finite_target, 3), np.array([3]))


def test_rolling_return_matches_a_hand_calculation() -> None:
    market = _prices(40)
    features = build_features(market).frame
    close = market["close"]
    origin = 25
    manual_return = np.log(close.iloc[origin] / close.iloc[origin - 1])
    manual_mean = np.mean(
        [np.log(close.iloc[origin - lag] / close.iloc[origin - lag - 1]) for lag in range(5)]
    )
    manual_distance = np.log(close.iloc[origin] / close.iloc[origin - 19 : origin + 1].mean())
    assert features.loc[origin, "ret_0"] == pytest.approx(manual_return)
    assert features.loc[origin, "roll_mean_5"] == pytest.approx(manual_mean)
    assert features.loc[origin, "ma_distance_20"] == pytest.approx(manual_distance)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.iloc[:20].isna().any(axis=1).all()


def test_terminal_split_embargoes_overlapping_labels() -> None:
    split = terminal_chronological_split(
        np.arange(100),
        horizon=5,
        minimum_training_origins=20,
        validation_window=10,
        test_window=10,
    )
    assert np.array_equal(split.test, np.arange(90, 100))
    assert np.array_equal(split.validation, np.arange(75, 85))
    assert np.array_equal(split.train, np.arange(70))
    assert np.array_equal(split.purged_before_validation, np.arange(70, 75))
    assert np.array_equal(split.purged_before_test, np.arange(85, 90))
    assert int(split.train.max()) + split.horizon < int(split.validation.min())
    assert int(split.validation.max()) + split.horizon < int(split.test.min())


def test_short_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="training window"):
        terminal_chronological_split(
            np.arange(30),
            horizon=1,
            minimum_training_origins=100,
            validation_window=5,
            test_window=5,
        )


def test_sequence_window_ends_at_the_origin() -> None:
    features = np.arange(15, dtype=float).reshape(5, 3)
    windows = sequence_windows(features, np.array([2, 4]), lookback=3)
    assert windows.shape == (2, 3, 3)
    np.testing.assert_array_equal(windows[0], features[0:3])
    np.testing.assert_array_equal(windows[1], features[2:5])
    np.testing.assert_array_equal(windows[0, -1], features[2])
    with pytest.raises(ValueError, match="lookback"):
        sequence_windows(features, np.array([2]), lookback=1)


def test_standardizer_uses_training_rows_only() -> None:
    train = np.array([[0.0], [2.0]])
    later = np.array([[10.0]])
    scaler = fit_standardizer(train)
    assert scaler.transform(later)[0, 0] == pytest.approx(9.0)
    combined = fit_standardizer(np.vstack([train, later]))
    assert scaler.mean[0] != pytest.approx(combined.mean[0])


def test_baselines_do_not_read_evaluation_targets() -> None:
    assert np.all(zero_return_forecast(4) == 0.0)
    forecast = historical_mean_forecast(np.array([1.0, 3.0]), 3)
    assert np.all(forecast == pytest.approx(2.0))


def test_linear_scaler_and_forest_selection_stay_on_their_partitions() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 2))
    y_train = x_train[:, 0] * 0.1
    x_validation = rng.normal(loc=5.0, size=(15, 2))
    y_validation = x_validation[:, 0] * 0.1
    linear = fit_linear_regression(x_train, y_train)
    assert linear.named_steps["scale"].mean_[0] == pytest.approx(x_train[:, 0].mean())
    forest = select_random_forest(
        x_train,
        y_train,
        x_validation,
        y_validation,
        depths=(2, 4),
        n_estimators=5,
        min_samples_leaf=1,
        seed=7,
    )
    assert forest.selected_max_depth in (2, 4)
    assert set(forest.validation_rmse_by_depth) == {2, 4}
    repeated = select_random_forest(
        x_train,
        y_train,
        x_validation,
        y_validation,
        depths=(2, 4),
        n_estimators=5,
        min_samples_leaf=1,
        seed=7,
    )
    np.testing.assert_allclose(
        forest.model.predict(x_validation), repeated.model.predict(x_validation)
    )


def test_failed_validation_refuses_training() -> None:
    market = _prices(80)
    market.loc[0, "close"] = -1.0
    with pytest.raises(ValueError, match="failed data validation"):
        run_index_horizon(
            market,
            dataset_name="broken",
            index_id="TEST",
            horizon=1,
            protocol=_protocol(),
            settings=_settings(),
            models=("zero_return",),
        )


def test_synthetic_benchmark_uses_one_common_test_sample() -> None:
    result = run_index_horizon(
        _prices(220),
        dataset_name="synthetic",
        index_id="TEST",
        horizon=1,
        protocol=_protocol(),
        settings=_settings(),
        models=("zero_return", "historical_mean", "linear_regression", "random_forest"),
    )
    counts = {model["test"]["n"] for model in result["models"]}
    assert counts == {20}
    zero = next(model for model in result["models"] if model["model"] == "zero_return")
    assert zero["test"]["directional_accuracy"] == pytest.approx(0.0)
    mean_model = next(model for model in result["models"] if model["model"] == "historical_mean")
    assert mean_model["details"]["training_mean"] != pytest.approx(0.0)
    assert result["sample"]["purged_before_test"] == 1
    assert int(result["sample"]["train"]["n"]) >= 30
    dates = [row["date"] for row in result["predictions"] if row["model"] == "zero_return"]
    assert dates == sorted(dates)
    assert len(dates) == 20


def test_lstm_window_is_scored_on_the_origin_target() -> None:
    pytest.importorskip("torch")
    result = run_index_horizon(
        _prices(220),
        dataset_name="synthetic",
        index_id="TEST",
        horizon=5,
        protocol=_protocol(),
        settings=_settings(),
        models=("lstm",),
    )
    lstm = result["models"][0]
    assert lstm["details"]["selected_hidden_size"] == 4
    assert lstm["details"]["lookback"] == 5
    assert lstm["test"]["n"] == 20
    assert np.isfinite(lstm["test"]["rmse"])
    assert result["sample"]["purged_before_test"] == 5


def _protocol() -> ForecastingProtocol:
    return ForecastingProtocol(
        schema_version=1,
        primary_target="forward_log_return",
        horizons=(1,),
        seed=171,
        minimum_training_origins=30,
        validation_window=20,
        test_window=20,
        step=20,
        nominal_coverage=0.9,
    )


def _settings() -> BenchmarkSettings:
    return BenchmarkSettings(
        lookback=5,
        rf_n_estimators=8,
        rf_min_samples_leaf=1,
        rf_max_depth=(2, 4),
        lstm_hidden_size=(4,),
        lstm_learning_rate=0.01,
        lstm_batch_size=16,
        lstm_max_epochs=2,
        lstm_patience=1,
    )
