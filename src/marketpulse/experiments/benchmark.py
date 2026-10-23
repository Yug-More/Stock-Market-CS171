"""Terminal-split benchmark for one dataset and horizon.

The runner scores every requested model on the same test origins. Model
selection uses validation RMSE only. Repeated walk-forward folds are not run.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse import __version__
from marketpulse.data.validation import validate_market_frame
from marketpulse.evaluation.metrics import (
    directional_accuracy,
    forecast_bias,
    mae,
    mse_skill_score,
    rmse,
)
from marketpulse.evaluation.partitions import eligible_origins, terminal_chronological_split
from marketpulse.evaluation.preprocessing import fit_standardizer
from marketpulse.features import build_features
from marketpulse.models.baselines import historical_mean_forecast, zero_return_forecast
from marketpulse.models.classical import fit_linear_regression, select_random_forest
from marketpulse.protocol import ForecastingProtocol
from marketpulse.sequences import sequence_windows
from marketpulse.targets import forward_log_return

ALL_MODELS = (
    "zero_return",
    "historical_mean",
    "linear_regression",
    "random_forest",
    "lstm",
)


@dataclass(frozen=True)
class BenchmarkSettings:
    """Model settings that are fixed before the test partition is scored."""

    lookback: int
    rf_n_estimators: int
    rf_min_samples_leaf: int
    rf_max_depth: tuple[int, ...]
    lstm_hidden_size: tuple[int, ...]
    lstm_learning_rate: float
    lstm_batch_size: int
    lstm_max_epochs: int
    lstm_patience: int

    def for_smoke(self) -> BenchmarkSettings:
        """Return a short configuration for a wiring check, not a benchmark."""

        return BenchmarkSettings(
            lookback=self.lookback,
            rf_n_estimators=min(self.rf_n_estimators, 20),
            rf_min_samples_leaf=self.rf_min_samples_leaf,
            rf_max_depth=(self.rf_max_depth[0],),
            lstm_hidden_size=(min(self.lstm_hidden_size),),
            lstm_learning_rate=self.lstm_learning_rate,
            lstm_batch_size=self.lstm_batch_size,
            lstm_max_epochs=2,
            lstm_patience=1,
        )


def load_benchmark_settings(root: Path) -> BenchmarkSettings:
    """Read ``configs/benchmark.toml``."""

    import tomllib

    path = root / "configs" / "benchmark.toml"
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return BenchmarkSettings(
        lookback=int(payload["lookback"]),
        rf_n_estimators=int(payload["rf_n_estimators"]),
        rf_min_samples_leaf=int(payload["rf_min_samples_leaf"]),
        rf_max_depth=tuple(int(value) for value in payload["rf_max_depth"]),
        lstm_hidden_size=tuple(int(value) for value in payload["lstm_hidden_size"]),
        lstm_learning_rate=float(payload["lstm_learning_rate"]),
        lstm_batch_size=int(payload["lstm_batch_size"]),
        lstm_max_epochs=int(payload["lstm_max_epochs"]),
        lstm_patience=int(payload["lstm_patience"]),
    )


def run_index_horizon(
    market: pd.DataFrame,
    *,
    dataset_name: str,
    index_id: str,
    horizon: int,
    protocol: ForecastingProtocol,
    settings: BenchmarkSettings,
    models: tuple[str, ...] = ALL_MODELS,
) -> dict[str, object]:
    """Fit and score one dataset at one horizon on the terminal chronological split."""

    summary = validate_market_frame(market, dataset_name)
    if summary.validation_status == "failed":
        raise ValueError(f"{dataset_name} failed data validation and was not used for training")
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")

    built = build_features(market)
    target = forward_log_return(market["close"], horizon)
    finite_features = built.frame.notna().all(axis=1).to_numpy()
    origins = eligible_origins(finite_features, target.notna().to_numpy(), settings.lookback)
    split = terminal_chronological_split(
        origins,
        horizon=horizon,
        minimum_training_origins=protocol.minimum_training_origins,
        validation_window=protocol.validation_window,
        test_window=protocol.test_window,
    )
    feature_matrix = built.frame.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    dates = pd.to_datetime(market["date"])
    x_train = feature_matrix[split.train]
    y_train = y[split.train]
    x_validation = feature_matrix[split.validation]
    y_validation = y[split.validation]
    x_test = feature_matrix[split.test]
    y_test = y[split.test]
    zero_test = zero_return_forecast(len(y_test))
    mean_test = historical_mean_forecast(y_train, len(y_test))
    zero_validation = zero_return_forecast(len(y_validation))
    mean_validation = historical_mean_forecast(y_train, len(y_validation))

    predictions: list[dict[str, object]] = []
    model_reports: list[dict[str, object]] = []

    def add_model(
        name: str,
        validation_prediction: np.ndarray,
        test_prediction: np.ndarray,
        details: dict[str, object],
    ) -> None:
        model_reports.append(
            {
                "model": name,
                "details": details,
                "validation": _metrics(
                    y_validation, validation_prediction, zero_validation, mean_validation
                ),
                "test": _metrics(y_test, test_prediction, zero_test, mean_test),
            }
        )
        predictions.extend(
            _prediction_rows(
                dataset_name, index_id, horizon, name, dates, split.test, y_test, test_prediction
            )
        )

    if "zero_return" in models:
        _progress(dataset_name, horizon, "zero_return")
        add_model(
            "zero_return",
            zero_validation,
            zero_test,
            {"definition": "predict 0; fixed and not estimated"},
        )
    if "historical_mean" in models:
        _progress(dataset_name, horizon, "historical_mean")
        add_model(
            "historical_mean",
            mean_validation,
            mean_test,
            {
                "definition": "mean of embargoed training targets, held fixed",
                "training_mean": float(y_train.mean()),
            },
        )
    if "linear_regression" in models:
        _progress(dataset_name, horizon, "linear_regression")
        linear = fit_linear_regression(x_train, y_train)
        add_model(
            "linear_regression",
            linear.predict(x_validation),
            linear.predict(x_test),
            {"scaling": "StandardScaler fit on training origins only"},
        )
    if "random_forest" in models:
        _progress(dataset_name, horizon, "random_forest")
        forest = select_random_forest(
            x_train,
            y_train,
            x_validation,
            y_validation,
            depths=settings.rf_max_depth,
            n_estimators=settings.rf_n_estimators,
            min_samples_leaf=settings.rf_min_samples_leaf,
            seed=protocol.seed,
        )
        add_model(
            "random_forest",
            forest.model.predict(x_validation),
            forest.model.predict(x_test),
            {
                "selected_max_depth": forest.selected_max_depth,
                "validation_rmse_by_depth": {
                    str(depth): value for depth, value in forest.validation_rmse_by_depth.items()
                },
                "n_estimators": settings.rf_n_estimators,
                "min_samples_leaf": settings.rf_min_samples_leaf,
                "seed": protocol.seed,
                "scaling": "StandardScaler fit on training origins only",
            },
        )
    if "lstm" in models:
        _progress(dataset_name, horizon, "lstm")
        lstm_report, lstm_test = _fit_lstm(
            feature_matrix,
            y,
            split.train,
            split.validation,
            split.test,
            settings,
            protocol.seed,
        )
        add_model("lstm", lstm_report["validation_prediction"], lstm_test, lstm_report["details"])

    return {
        "dataset": dataset_name,
        "index_id": index_id,
        "horizon": horizon,
        "lookback": settings.lookback,
        "seed": protocol.seed,
        "sample": _sample_record(dates, split, target, built.frame),
        "models": model_reports,
        "predictions": predictions,
    }


def _progress(dataset_name: str, horizon: int, model: str) -> None:
    print(f"{dataset_name} horizon {horizon}: {model}", flush=True)


def _fit_lstm(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    settings: BenchmarkSettings,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    from marketpulse.models.lstm import predict_return_lstm, select_return_lstm

    finite = np.isfinite(feature_matrix).all(axis=1)
    standardizer = fit_standardizer(feature_matrix[train])
    scaled = np.full(feature_matrix.shape, np.nan, dtype=float)
    scaled[finite] = standardizer.transform(feature_matrix[finite])
    x_train = sequence_windows(scaled, train, settings.lookback)
    x_validation = sequence_windows(scaled, validation, settings.lookback)
    x_test = sequence_windows(scaled, test, settings.lookback)
    selected, scores = select_return_lstm(
        x_train,
        target[train],
        x_validation,
        target[validation],
        hidden_sizes=settings.lstm_hidden_size,
        learning_rate=settings.lstm_learning_rate,
        batch_size=settings.lstm_batch_size,
        max_epochs=settings.lstm_max_epochs,
        patience=settings.lstm_patience,
        seed=seed,
    )
    test_prediction = predict_return_lstm(selected.model, x_test)
    validation_prediction = predict_return_lstm(selected.model, x_validation)
    return {
        "validation_prediction": validation_prediction,
        "details": {
            "selected_hidden_size": selected.hidden_size,
            "validation_rmse_by_hidden_size": {str(size): value for size, value in scores.items()},
            "epochs_trained": selected.epochs_trained,
            "best_epoch": selected.best_epoch,
            "lookback": settings.lookback,
            "learning_rate": settings.lstm_learning_rate,
            "batch_size": settings.lstm_batch_size,
            "max_epochs": settings.lstm_max_epochs,
            "patience": settings.lstm_patience,
            "seed": seed,
            "scaling": "training-origin mean and scale applied before windowing",
            "history": [
                {"epoch": int(row["epoch"]), "validation_mse": row["validation_mse"]}
                for row in selected.history
            ],
            "reproducibility": (
                "Seeds are set for Python, NumPy, and PyTorch. CPU LSTM results "
                "are not guaranteed to be bitwise identical across library versions."
            ),
        },
    }, test_prediction


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_prediction: np.ndarray,
    mean_prediction: np.ndarray,
) -> dict[str, object]:
    model_mse = float(np.mean((y_pred - y_true) ** 2))
    zero_mse = float(np.mean((zero_prediction - y_true) ** 2))
    mean_mse = float(np.mean((mean_prediction - y_true) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
        "directional_accuracy": _directional(y_true, y_pred),
        "skill_vs_zero": _skill(model_mse, zero_mse),
        "skill_vs_historical_mean": _skill(model_mse, mean_mse),
    }


def _directional(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    try:
        return directional_accuracy(y_true, y_pred)
    except ValueError:
        return None


def _skill(model_mse: float, baseline_mse: float) -> float | None:
    if baseline_mse <= 0.0:
        return None
    return mse_skill_score(model_mse, baseline_mse)


def _prediction_rows(
    dataset_name: str,
    index_id: str,
    horizon: int,
    model: str,
    dates: pd.Series,
    origins: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for origin, actual, predicted in zip(origins, y_true, y_pred, strict=True):
        rows.append(
            {
                "dataset": dataset_name,
                "index_id": index_id,
                "horizon": horizon,
                "model": model,
                "date": pd.Timestamp(dates.iloc[int(origin)]).date().isoformat(),
                "y_true": float(actual),
                "y_pred": float(predicted),
            }
        )
    return rows


def _sample_record(
    dates: pd.Series,
    split: object,
    target: pd.Series,
    features: pd.DataFrame,
) -> dict[str, object]:
    from marketpulse.evaluation.partitions import ChronologicalSplit

    if not isinstance(split, ChronologicalSplit):
        raise TypeError("split must be a ChronologicalSplit")
    return {
        "rows": int(len(features)),
        "missing_feature_rows": int((~features.notna().all(axis=1)).sum()),
        "missing_target_rows": int(target.isna().sum()),
        "train": _block(dates, split.train),
        "validation": _block(dates, split.validation),
        "test": _block(dates, split.test),
        "purged_before_validation": int(len(split.purged_before_validation)),
        "purged_before_test": int(len(split.purged_before_test)),
        "horizon": split.horizon,
    }


def _block(dates: pd.Series, origins: np.ndarray) -> dict[str, object]:
    return {
        "n": int(len(origins)),
        "start_date": pd.Timestamp(dates.iloc[int(origins[0])]).date().isoformat(),
        "end_date": pd.Timestamp(dates.iloc[int(origins[-1])]).date().isoformat(),
    }


def environment_record(root: Path, mode: str) -> dict[str, object]:
    """Record the library versions used to produce a benchmark."""

    import sklearn

    versions: dict[str, str] = {
        "marketpulse": __version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not-installed"
    return {
        "mode": mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "git_sha": _git_sha(root),
    }


def write_benchmark_artifacts(output_dir: Path, payload: dict[str, object]) -> None:
    """Write metrics, predictions, and a test-RMSE figure when matplotlib is installed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = payload["predictions"]
    if not isinstance(predictions, list):
        raise TypeError("predictions must be a list")
    serializable = {key: value for key, value in payload.items() if key != "predictions"}
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    frame = pd.DataFrame(predictions)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    _write_metrics_csv(output_dir / "metrics.csv", serializable)
    _write_figure(output_dir / "test_rmse.png", serializable)


def _write_metrics_csv(path: Path, payload: dict[str, object]) -> None:
    rows: list[dict[str, object]] = []
    results = payload["results"]
    if not isinstance(results, list):
        raise TypeError("results must be a list")
    for result in results:
        for model in result["models"]:
            for partition in ("validation", "test"):
                metrics = model[partition]
                rows.append(
                    {
                        "dataset": result["dataset"],
                        "index_id": result["index_id"],
                        "horizon": result["horizon"],
                        "model": model["model"],
                        "partition": partition,
                        "n": metrics["n"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "bias": metrics["bias"],
                        "directional_accuracy": metrics["directional_accuracy"],
                        "skill_vs_zero": metrics["skill_vs_zero"],
                        "skill_vs_historical_mean": metrics["skill_vs_historical_mean"],
                        "test_start": result["sample"]["test"]["start_date"],
                        "test_end": result["sample"]["test"]["end_date"],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_figure(path: Path, payload: dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    results = payload["results"]
    if not isinstance(results, list) or not results:
        return
    datasets = list(dict.fromkeys(result["dataset"] for result in results))
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets, strict=True):
        subset = [result for result in results if result["dataset"] == dataset]
        horizons = [int(result["horizon"]) for result in subset]
        model_names = [model["model"] for model in subset[0]["models"]]
        x = np.arange(len(horizons))
        width = 0.15
        for offset, name in enumerate(model_names):
            heights = [
                next(model["test"]["rmse"] for model in result["models"] if model["model"] == name)
                for result in subset
            ]
            axis.bar(x + (offset - len(model_names) / 2) * width, heights, width, label=name)
        axis.set_xticks(x, [str(horizon) for horizon in horizons])
        axis.set_xlabel("Horizon (trading sessions)")
        axis.set_ylabel("Test RMSE")
        axis.set_title(dataset)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
