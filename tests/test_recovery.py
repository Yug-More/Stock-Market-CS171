"""Recovery must reject changed inputs and corrupt forecasts without overwriting them."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marketpulse.evaluation.partitions import eligible_origins, walk_forward_splits
from marketpulse.experiments.benchmark import BenchmarkSettings
from marketpulse.experiments.recovery import validate_fold
from marketpulse.experiments.walkforward import _run_task
from marketpulse.features import build_features
from marketpulse.targets import forward_log_return


def test_recovered_fold_is_checked_against_data_and_configuration(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(171)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 180)))
    market = pd.DataFrame(
        {"date": pd.bdate_range("2000-01-03", periods=len(close)), "close": close}
    )
    features = build_features(market).frame
    target = forward_log_return(market.close, 1)
    origins = eligible_origins(
        features.notna().all(axis=1).to_numpy(), target.notna().to_numpy(), 5
    )
    split = walk_forward_splits(
        origins,
        horizon=1,
        minimum_training_origins=40,
        validation_window=10,
        test_window=10,
        step=10,
    )[0]
    task = {
        "entry": SimpleNamespace(name="synthetic", index_id="TEST"),
        "horizon": 1,
        "fold_index": 0,
        "n_folds": 1,
        "market": market,
        "features": features,
        "target": target,
        "split": split,
    }
    settings = BenchmarkSettings(5, 4, 1, (2,), (4,), 0.01, 16, 1, 1)
    _run_task(task, settings, 171, tmp_path, tmp_path, "smoke")
    marker = tmp_path / "folds/synthetic/h1/fold_000.json"
    csv = marker.with_suffix(".csv")
    assert validate_fold(task, settings, 171, marker, csv)["validated"]
    original = csv.read_bytes()
    with pytest.raises(ValueError, match="incompatible forest"):
        validate_fold(task, replace(settings, rf_n_estimators=8), 171, marker, csv)
    assert csv.read_bytes() == original
    frame = pd.read_csv(csv)
    frame.loc[frame.model == "lstm", "y_pred"] = float("inf")
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_fold(task, settings, 171, marker, csv)


def test_default_protocol_steps_252_sessions_for_every_horizon():
    for horizon in (1, 5, 20):
        folds = walk_forward_splits(
            np.arange(5000),
            horizon=horizon,
            minimum_training_origins=2520,
            validation_window=252,
            test_window=252,
            step=252,
        )
        assert len(folds) > 1
        for previous, current in zip(folds, folds[1:], strict=False):
            assert current.test[0] - previous.test[0] == 252
            assert previous.train[-1] + horizon < previous.validation[0]
            assert previous.validation[-1] + horizon < previous.test[0]
