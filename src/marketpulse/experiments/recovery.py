"""Validate recovered fold artifacts against the current experiment plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.experiments.benchmark import ALL_MODELS, _metrics, _sample_record


def file_hash(path: Path) -> str:
    """SHA-256 identifies data, source, and persisted predictions."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_identity(root, mode, protocol, settings):
    """Capture the immutable inputs required to safely resume a run."""
    paths = sorted((root / "src/marketpulse").rglob("*.py"))
    paths += sorted((root / "configs").glob("*.toml"))
    paths += sorted((root / "data").glob("*.csv"))
    return {
        "schema_version": 1,
        "mode": mode,
        "protocol": asdict(protocol),
        "settings": asdict(settings),
        "inputs": {str(path.relative_to(root)): file_hash(path) for path in paths},
    }


def validate_fold(task, settings, seed, marker: Path, predictions_path: Path) -> dict:
    """Reject incomplete, misaligned, nonfinite, or incompatible recovered outputs.

    Legacy files have no execution revision or environment; validation can
    establish internal consistency, but cannot reconstruct absent provenance.
    """
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "dataset": task["entry"].name,
        "index_id": task["entry"].index_id,
        "horizon": task["horizon"],
        "lookback": settings.lookback,
        "seed": seed,
        "fold": task["fold_index"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{marker}: incompatible {key}")
    sample = _sample_record(
        pd.to_datetime(task["market"]["date"]), task["split"], task["target"], task["features"]
    )
    if payload["sample"] != sample:
        raise ValueError(f"{marker}: split boundaries differ from the plan")
    if set(payload.get("models_present", [])) != set(ALL_MODELS):
        raise ValueError(f"{marker}: incomplete model set")
    reports = {item["model"]: item for item in payload["models"]}
    if len(payload["models"]) != len(ALL_MODELS) or set(reports) != set(ALL_MODELS):
        raise ValueError(f"{marker}: duplicate or missing model reports")
    forest = reports["random_forest"]["details"]
    lstm = reports["lstm"]["details"]
    for key, value in {
        "n_estimators": settings.rf_n_estimators,
        "min_samples_leaf": settings.rf_min_samples_leaf,
        "seed": seed,
    }.items():
        if forest[key] != value:
            raise ValueError(f"{marker}: incompatible forest {key}")
    if set(map(int, forest["validation_rmse_by_depth"])) != set(settings.rf_max_depth):
        raise ValueError(f"{marker}: incompatible forest search")
    for key, value in {
        "lookback": settings.lookback,
        "learning_rate": settings.lstm_learning_rate,
        "batch_size": settings.lstm_batch_size,
        "max_epochs": settings.lstm_max_epochs,
        "patience": settings.lstm_patience,
        "seed": seed,
    }.items():
        if lstm[key] != value:
            raise ValueError(f"{marker}: incompatible LSTM {key}")
    if set(map(int, lstm["validation_rmse_by_hidden_size"])) != set(settings.lstm_hidden_size):
        raise ValueError(f"{marker}: incompatible LSTM search")
    if "predictions_sha256" in payload:
        if payload["predictions_sha256"] != file_hash(predictions_path):
            raise ValueError(f"{predictions_path}: checksum mismatch")
    frame = pd.read_csv(predictions_path)
    if set(frame["model"]) != set(ALL_MODELS):
        raise ValueError(f"{predictions_path}: incomplete predictions")
    split = task["split"]
    dates = pd.to_datetime(task["market"]["date"].iloc[split.test]).dt.strftime("%Y-%m-%d")
    actual = task["target"].iloc[split.test].to_numpy()
    training_mean = float(task["target"].iloc[split.train].mean())
    from marketpulse.evaluation.regimes import assign_volatility_regime, volatility_thresholds

    volatility = task["features"]["roll_vol_20"].to_numpy()
    lower, upper = volatility_thresholds(volatility[split.train])
    regimes = assign_volatility_regime(volatility[split.test], lower, upper)
    np.testing.assert_allclose(
        [payload["volatility_thresholds"]["low_max"], payload["volatility_thresholds"]["high_min"]],
        [lower, upper],
        rtol=1e-12,
        atol=1e-14,
    )
    for model in ALL_MODELS:
        part = frame.loc[frame["model"] == model]
        if part["date"].tolist() != dates.tolist():
            raise ValueError(f"{predictions_path}: {model} date alignment")
        for key in ("dataset", "index_id", "horizon"):
            if not bool((part[key] == expected[key]).all()):
                raise ValueError(f"{predictions_path}: {key} differs")
        if not bool((part["fold"] == task["fold_index"]).all()):
            raise ValueError(f"{predictions_path}: fold differs")
        np.testing.assert_allclose(part["y_true"], actual, rtol=1e-12, atol=1e-14)
        if not np.isfinite(part[["y_true", "y_pred", "volatility"]].to_numpy()).all():
            raise ValueError(f"{predictions_path}: nonfinite values")
        np.testing.assert_array_equal(part["regime"], regimes)
        np.testing.assert_allclose(part["volatility"], volatility[split.test], atol=1e-14)
        np.testing.assert_allclose(part["vol_low_max"], lower, atol=1e-14)
        np.testing.assert_allclose(part["vol_high_min"], upper, atol=1e-14)
        if model == "zero_return":
            np.testing.assert_array_equal(part["y_pred"], np.zeros(len(actual)))
        if model == "historical_mean":
            np.testing.assert_allclose(part["y_pred"], training_mean, atol=1e-14)
        metrics = _metrics(
            actual,
            part["y_pred"].to_numpy(),
            np.zeros(len(actual)),
            np.full(len(actual), training_mean),
        )
        for key, value in metrics.items():
            saved = reports[model]["test"][key]
            if value is None:
                if saved is not None:
                    raise ValueError(f"{marker}: undefined {key} differs")
            else:
                np.testing.assert_allclose(saved, value, rtol=1e-9, atol=1e-12)
    return {
        "marker": str(marker.relative_to(marker.parents[4])),
        "predictions_sha256": file_hash(predictions_path),
        "marker_sha256": file_hash(marker),
        "validated": True,
        "provenance": "recorded" if "execution" in payload else "legacy_unrecorded",
    }
