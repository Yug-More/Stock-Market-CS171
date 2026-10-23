"""Expanding-window walk-forward evaluation and the regime slices of its predictions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import validate_market_frame
from marketpulse.evaluation.partitions import eligible_origins, walk_forward_splits
from marketpulse.evaluation.regimes import assign_volatility_regime, volatility_thresholds
from marketpulse.experiments.benchmark import (
    ALL_MODELS,
    BenchmarkSettings,
    _sample_record,
    environment_record,
    score_split,
)
from marketpulse.experiments.figures import write_walkforward_figures
from marketpulse.experiments.recovery import experiment_identity, file_hash, validate_fold
from marketpulse.experiments.summary import fold_summary, pooled_metrics, stress_tables
from marketpulse.features import build_features
from marketpulse.protocol import ForecastingProtocol
from marketpulse.targets import forward_log_return


def run_walk_forward(
    root: Path,
    *,
    mode: str,
    output_dir: Path,
    protocol: ForecastingProtocol,
    settings: BenchmarkSettings,
    max_folds: int | None = None,
) -> None:
    """Refit every model on each expanding fold and write the regime tables.

    A fold file that already contains every requested model is left in place,
    so a rerun continues after the last finished fold. Features are built once
    per series. Models are not reused across folds.
    """

    if mode not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    catalogs = load_catalog(root)
    horizons = protocol.horizons
    if mode == "smoke":
        catalogs = catalogs[:1]
        horizons = horizons[:1]
        settings = settings.for_smoke()
        max_folds = 2 if max_folds is None else max_folds
    tasks = _plan(root, catalogs, horizons, protocol, settings, max_folds)
    identity = experiment_identity(root, mode, protocol, settings)
    identity["max_folds"] = max_folds
    manifest_path = output_dir / "manifest.json"
    # JSON normalizes tuple settings to lists before comparison.
    identity = json.loads(json.dumps(identity))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("run inputs changed; use a new output directory")
    else:
        recovered = []
        for task in tasks:
            directory = output_dir / "folds" / _slug(task["entry"].name) / f"h{task['horizon']}"
            marker = directory / f"fold_{task['fold_index']:03d}.json"
            csv = marker.with_suffix(".csv")
            if marker.exists() or csv.exists():
                if not marker.exists() or not csv.exists():
                    raise ValueError(f"partial fold preserved at {marker}; inspect before recovery")
                recovered.append(validate_fold(task, settings, protocol.seed, marker, csv))
        manifest_path.write_text(
            json.dumps(
                {
                    "identity": identity,
                    "recovery_environment": environment_record(root, mode),
                    "legacy_provenance": (
                        "Original execution environment and revision were not recorded; "
                        "recovery validation cannot establish those missing facts."
                    ),
                    "recovered": recovered,
                    "expected_folds": len(tasks),
                },
                indent=2,
            )
            + "\n"
        )
    planned = []
    for task in tasks:
        planned.append(
            {
                "dataset": task["entry"].name,
                "horizon": task["horizon"],
                "fold": task["fold_index"],
                "sample": _sample_record(
                    pd.to_datetime(task["market"]["date"]),
                    task["split"],
                    task["target"],
                    task["features"],
                ),
            }
        )
    (output_dir / "fold_plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    print(f"{mode}: {len(tasks)} walk-forward folds", flush=True)
    for task in tasks:
        _run_task(task, settings, protocol.seed, output_dir, root, mode)
    _write_summaries(output_dir, mode, root, protocol, settings)


def _plan(root, catalogs, horizons, protocol, settings, max_folds):
    tasks = []
    for entry in catalogs:
        market = load_market_csv(root / entry.path, dataset_name=entry.name)
        summary = validate_market_frame(market, entry.name)
        if summary.validation_status == "failed":
            raise ValueError(f"{entry.name} failed data validation and was not used")
        features = build_features(market).frame
        finite = features.notna().all(axis=1).to_numpy()
        for horizon in horizons:
            target = forward_log_return(market["close"], horizon)
            origins = eligible_origins(finite, target.notna().to_numpy(), settings.lookback)
            folds = walk_forward_splits(
                origins,
                horizon=horizon,
                minimum_training_origins=protocol.minimum_training_origins,
                validation_window=protocol.validation_window,
                test_window=protocol.test_window,
                step=protocol.step,
            )
            selected = folds if max_folds is None else folds[-max_folds:]
            offset = len(folds) - len(selected)
            for position, split in enumerate(selected):
                tasks.append(
                    {
                        "entry": entry,
                        "market": market,
                        "features": features,
                        "target": target,
                        "origins": origins,
                        "split": split,
                        "horizon": horizon,
                        "fold_index": offset + position,
                        "n_folds": len(folds),
                    }
                )
    return tasks


def _run_task(
    task: dict[str, object],
    settings: BenchmarkSettings,
    seed: int,
    output_dir: Path,
    root: Path,
    mode: str,
) -> None:
    entry = task["entry"]
    horizon = int(task["horizon"])
    fold_index = int(task["fold_index"])
    directory = output_dir / "folds" / _slug(entry.name) / f"h{horizon}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"fold_{fold_index:03d}.json"
    predictions_path = directory / f"fold_{fold_index:03d}.csv"
    if marker.is_file() or predictions_path.is_file():
        if not marker.is_file() or not predictions_path.is_file():
            raise ValueError(f"partial fold preserved at {marker}; inspect before recovery")
        validate_fold(task, settings, seed, marker, predictions_path)
        print(f"reuse {entry.name} horizon {horizon} fold {fold_index + 1}", flush=True)
        return
    started = time.monotonic()
    features = task["features"]
    result = score_split(
        dataset_name=entry.name,
        index_id=entry.index_id,
        horizon=horizon,
        feature_matrix=features.to_numpy(dtype=float),
        target_values=task["target"].to_numpy(dtype=float),
        target=task["target"],
        features=features,
        dates=pd.to_datetime(task["market"]["date"]),
        split=task["split"],
        settings=settings,
        seed=seed,
        models=ALL_MODELS,
        fold_index=fold_index,
        n_folds=int(task["n_folds"]),
    )
    frame = _with_regimes(pd.DataFrame(result["predictions"]), features, task["split"])
    _assert_common_sample(frame)
    thresholds = _thresholds(features, task["split"])
    payload = {key: value for key, value in result.items() if key != "predictions"}
    payload["volatility_thresholds"] = thresholds
    payload["models_present"] = [model["model"] for model in result["models"]]
    payload["execution"] = {
        **environment_record(root, mode),
        "settings": asdict(settings),
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_csv = predictions_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(predictions_path)
    payload["predictions_sha256"] = file_hash(predictions_path)
    temporary_marker = marker.with_suffix(".json.tmp")
    temporary_marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_marker.replace(marker)
    validate_fold(task, settings, seed, marker, predictions_path)


def _with_regimes(frame: pd.DataFrame, features: pd.DataFrame, split: object) -> pd.DataFrame:
    volatility = features["roll_vol_20"].to_numpy(dtype=float)
    low_max, high_min = volatility_thresholds(volatility[split.train])
    regimes = assign_volatility_regime(volatility[split.test], low_max, high_min)
    pieces: list[pd.DataFrame] = []
    for _, part in frame.groupby("model", sort=False):
        ordered = part.reset_index(drop=True)
        if len(ordered) != len(split.test):
            raise AssertionError("a model does not have one prediction per test origin")
        ordered["volatility"] = volatility[split.test]
        ordered["regime"] = regimes
        ordered["vol_low_max"] = low_max
        ordered["vol_high_min"] = high_min
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def _thresholds(features: pd.DataFrame, split: object) -> dict[str, float]:
    volatility = features["roll_vol_20"].to_numpy(dtype=float)
    low_max, high_min = volatility_thresholds(volatility[split.train])
    return {
        "low_max": low_max,
        "high_min": high_min,
        "validation_mean": float(np.mean(volatility[split.validation])),
        "test_mean": float(np.mean(volatility[split.test])),
    }


def _assert_common_sample(frame: pd.DataFrame) -> None:
    grouped = [part.sort_values("date") for _, part in frame.groupby("model", sort=False)]
    dates = grouped[0]["date"].tolist()
    outcomes = grouped[0]["y_true"].to_numpy()
    for part in grouped[1:]:
        if part["date"].tolist() != dates:
            raise AssertionError("models were scored on different dates")
        if not np.allclose(part["y_true"].to_numpy(), outcomes):
            raise AssertionError("models were scored on different outcomes")
    if len(dates) != len(set(dates)):
        raise AssertionError("a fold repeats a test date")


def _write_summaries(
    output_dir: Path,
    mode: str,
    root: Path,
    protocol: ForecastingProtocol,
    settings: BenchmarkSettings,
) -> None:
    plan = json.loads((output_dir / "fold_plan.json").read_text())
    expected = {
        output_dir
        / "folds"
        / _slug(item["dataset"])
        / f"h{item['horizon']}"
        / f"fold_{item['fold']:03d}.csv"
        for item in plan
    }
    actual = set(output_dir.glob("folds/*/*/*.csv"))
    if actual != expected:
        raise ValueError("saved fold files do not match the complete run plan")
    frames = [pd.read_csv(path) for path in sorted(expected)]
    if not frames:
        raise ValueError("the walk-forward run produced no fold predictions")
    predictions = pd.concat(frames, ignore_index=True)
    _assert_unique_tests(predictions)
    fold_metrics = _fold_metrics_from_markers(output_dir)
    pooled = pooled_metrics(predictions, ("dataset", "horizon"))
    summary = fold_summary(fold_metrics)
    regime = pooled_metrics(predictions, ("dataset", "horizon", "regime"))
    low = regime.loc[regime["regime"] == "low", ["dataset", "horizon", "model", "rmse"]]
    regime = regime.merge(
        low.rename(columns={"rmse": "low_regime_rmse"}),
        on=["dataset", "horizon", "model"],
        how="left",
    )
    regime["rmse_ratio_to_low"] = regime["rmse"] / regime["low_regime_rmse"].replace(0, np.nan)
    eligible = _eligible_table(root, protocol, settings, predictions)
    stress = stress_tables(predictions, eligible)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    summary.to_csv(output_dir / "fold_summary.csv", index=False)
    regime.to_csv(output_dir / "regime_metrics.csv", index=False)
    stress.to_csv(output_dir / "stress_metrics.csv", index=False)
    _threshold_table(output_dir).to_csv(output_dir / "volatility_thresholds.csv", index=False)
    (output_dir / "environment.json").write_text(
        json.dumps(environment_record(root, mode), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "horizons": list(protocol.horizons),
                "seed": protocol.seed,
                "minimum_training_origins": protocol.minimum_training_origins,
                "validation_window": protocol.validation_window,
                "test_window": protocol.test_window,
                "step": protocol.step,
                "lookback": settings.lookback,
                "mode": mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_walkforward_figures(output_dir, fold_metrics, pooled, summary, regime, stress)
    print(f"wrote {output_dir}", flush=True)


def _fold_metrics_from_markers(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("folds/*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = payload["sample"]
        thresholds = payload["volatility_thresholds"]
        for model in payload["models"]:
            for partition in ("validation", "test"):
                metrics = model[partition]
                details = model["details"]
                rows.append(
                    {
                        "dataset": payload["dataset"],
                        "index_id": payload["index_id"],
                        "horizon": payload["horizon"],
                        "fold": payload["fold"],
                        "model": model["model"],
                        "partition": partition,
                        "n": metrics["n"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "bias": metrics["bias"],
                        "directional_accuracy": metrics["directional_accuracy"],
                        "skill_vs_zero": metrics["skill_vs_zero"],
                        "skill_vs_historical_mean": metrics["skill_vs_historical_mean"],
                        "train_n": sample["train"]["n"],
                        "train_start": sample["train"]["start_date"],
                        "train_end": sample["train"]["end_date"],
                        "validation_start": sample["validation"]["start_date"],
                        "validation_end": sample["validation"]["end_date"],
                        "test_start": sample["test"]["start_date"],
                        "test_end": sample["test"]["end_date"],
                        "purged_before_validation": sample["purged_before_validation"],
                        "purged_before_test": sample["purged_before_test"],
                        "selected_max_depth": details.get("selected_max_depth"),
                        "selected_hidden_size": details.get("selected_hidden_size"),
                        "training_mean": details.get("training_mean"),
                        "vol_low_max": thresholds["low_max"],
                        "vol_high_min": thresholds["high_min"],
                        "validation_volatility_mean": thresholds["validation_mean"],
                        "test_volatility_mean": thresholds["test_mean"],
                    }
                )
    return pd.DataFrame(rows)


def _eligible_table(
    root: Path,
    protocol: ForecastingProtocol,
    settings: BenchmarkSettings,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Eligible decision dates for the datasets and horizons present in this run."""

    present = predictions[["dataset", "horizon"]].drop_duplicates()
    catalogs = {entry.name: entry for entry in load_catalog(root)}
    rows: list[dict[str, object]] = []
    for dataset, horizon in present.itertuples(index=False):
        entry = catalogs[str(dataset)]
        market = load_market_csv(root / entry.path, dataset_name=entry.name)
        features = build_features(market).frame
        target = forward_log_return(market["close"], int(horizon))
        origins = eligible_origins(
            features.notna().all(axis=1).to_numpy(),
            target.notna().to_numpy(),
            settings.lookback,
        )
        dates = pd.to_datetime(market["date"])
        for origin in origins:
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "date": pd.Timestamp(dates.iloc[int(origin)]).date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def _threshold_table(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("folds/*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": payload["dataset"],
                "horizon": payload["horizon"],
                "fold": payload["fold"],
                **payload["volatility_thresholds"],
            }
        )
    return pd.DataFrame(rows)


def _assert_unique_tests(predictions: pd.DataFrame) -> None:
    zero = predictions.loc[predictions["model"] == "zero_return"]
    duplicate = zero.duplicated(["dataset", "horizon", "date"])
    if bool(duplicate.any()):
        raise AssertionError("a test origin was scored in more than one fold")


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("&", "and")
