"""Gated future intake; retain the first frozen test window even on later downloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import validate_market_frame
from marketpulse.evaluation.calibration import interval_metrics, residual_quantile
from marketpulse.evaluation.partitions import eligible_origins
from marketpulse.evaluation.reliability import compare_losses, holm_adjust, paired_losses
from marketpulse.experiments.benchmark import (
    _metrics,
    environment_record,
    load_benchmark_settings,
    score_split,
)
from marketpulse.experiments.independent_validation import four_blocks, prospective_gate
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json
from marketpulse.features import build_features
from marketpulse.targets import forward_log_return


def frozen_window(origins: np.ndarray, dates: pd.Series, horizon: int, start: str):
    future = origins[pd.to_datetime(dates.iloc[origins]).to_numpy() >= np.datetime64(start)]
    if len(future) < 252:
        raise ValueError("the frozen prospective test is not yet mature: need 252 eligible origins")
    if pd.Timestamp(dates.iloc[future[0]]) != pd.Timestamp(start):
        raise ValueError("frozen first origin unavailable; cannot shift test dates")
    fixed = future[:252]
    split, calibration, test = four_blocks(origins[origins <= fixed[-1]], horizon)
    np.testing.assert_array_equal(test, fixed)
    return split, calibration, test


def prepare(root: Path, manifest: dict) -> tuple[dict, list[dict]]:
    protocol_path = root / "configs/independent_validation.json"
    frozen = json.loads((root / "results/phase9/rehearsal/manifest.json").read_text())[
        "protocol_sha256"
    ]
    if manifest.get("protocol_sha256") != frozen or file_hash(protocol_path) != frozen:
        raise ValueError("prospective protocol differs from the original freeze")
    historical = json.loads((root / "results/walkforward/full/manifest.json").read_text())
    for path, sha in historical["identity"]["inputs"].items():
        if file_hash(root / path) != sha:
            raise ValueError(f"frozen training dependency changed: {path}")
    protocol = json.loads(protocol_path.read_text())
    entries = manifest.get("datasets", [])
    if sorted(x["name"] for x in entries) != sorted(x.name for x in load_catalog(root)):
        raise ValueError("both frozen index populations are required exactly once")
    expected_ids = {entry.name: entry.index_id for entry in load_catalog(root)}
    settings = load_benchmark_settings(root)
    tasks = []
    for entry in entries:
        if entry.get("index_id") != expected_ids[entry["name"]]:
            raise ValueError("acquired index identifier differs from frozen population")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path
        if file_hash(path) != entry.get("sha256"):
            raise ValueError("acquired source checksum mismatch")
        market = load_market_csv(path, dataset_name=entry["name"])
        if validate_market_frame(market, entry["name"]).validation_status == "failed":
            raise ValueError("acquired market data failed validation")
        features = build_features(market).frame
        dates = pd.to_datetime(market.date)
        for horizon in protocol["horizons"]:
            target = forward_log_return(market.close, horizon)
            origins = eligible_origins(
                features.notna().all(axis=1).to_numpy(),
                target.notna().to_numpy(),
                settings.lookback,
            )
            split, calibration, test = frozen_window(
                origins, dates, horizon, protocol["earliest_test_origin"]
            )
            prospective_gate(dates, test, entry, protocol)
            realized = dates.iloc[int(test[-1]) + horizon].tz_localize("UTC")
            if pd.Timestamp(entry["retrieved_utc"]) < realized:
                raise ValueError("last test outcome was not acquired")
            tasks.append(
                dict(
                    entry=entry,
                    market=market,
                    features=features,
                    target=target,
                    split=split,
                    calibration=calibration,
                    test=test,
                    horizon=horizon,
                )
            )
    return protocol, tasks


def run(root: Path, manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    protocol, tasks = prepare(root, manifest)  # All gates run before any fit/output.
    if output.exists():
        raise ValueError("preserve previous prospective evidence")
    output.mkdir(parents=True)
    predictions = []
    for i, task in enumerate(tasks):
        entry = task["entry"]
        dates = pd.to_datetime(task["market"].date)
        result = score_split(
            dataset_name=entry["name"],
            index_id=entry["index_id"],
            horizon=task["horizon"],
            feature_matrix=task["features"].to_numpy(float),
            target_values=task["target"].to_numpy(float),
            target=task["target"],
            features=task["features"],
            dates=dates,
            split=task["split"],
            settings=load_benchmark_settings(root),
            seed=protocol["seed"],
        )
        frame = pd.DataFrame(result.pop("predictions"))
        lookup = {str(date.date()): pos for pos, date in enumerate(dates)}
        frame["origin_position"] = frame.date.map(lookup)
        frame["target_date"] = [
            str(dates.iloc[p + task["horizon"]].date()) for p in frame.origin_position
        ]
        calibration_dates = set(dates.iloc[task["calibration"]].dt.strftime("%Y-%m-%d"))
        frame["partition"] = np.where(frame.date.isin(calibration_dates), "calibration", "test")
        for model, part in frame.groupby("model"):
            cal = part[part.partition == "calibration"]
            scores = sorted(np.abs(cal.y_true - cal.y_pred))
            for level in protocol["intervals"]["levels"]:
                suffix = round(100 * level)
                radius = residual_quantile(scores, level)
                mask = frame.model.eq(model) & frame.partition.eq("test")
                frame.loc[mask, f"lower_{suffix}"] = frame.loc[mask, "y_pred"] - radius
                frame.loc[mask, f"upper_{suffix}"] = frame.loc[mask, "y_pred"] + radius
        frame.to_csv(
            output / f"cell_{i}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
        write_json(output / f"cell_{i}.json", result)
        predictions.append(frame[frame.partition == "test"])
    forecasts = pd.concat(predictions, ignore_index=True)
    metrics, comparisons, intervals = [], [], []
    for (dataset, horizon), part in forecasts.groupby(["dataset", "horizon"]):
        zero = part[part.model == "zero_return"].sort_values("date")
        mean = part[part.model == "historical_mean"].sort_values("date")
        for model, rows in part.groupby("model"):
            rows = rows.sort_values("date").copy()
            rows["eligible"] = True
            for level in protocol["intervals"]["levels"]:
                intervals.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model,
                        **interval_metrics(rows, level),
                    }
                )
            metrics.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "model": model,
                    **_metrics(
                        rows.y_true.to_numpy(),
                        rows.y_pred.to_numpy(),
                        zero.y_pred.to_numpy(),
                        mean.y_pred.to_numpy(),
                    ),
                }
            )
        for first, second in protocol["primary_pairs"]:
            timeline, a, b = paired_losses(part, first, second)
            for length in protocol["inference"]["block_lengths"]:
                effect = compare_losses(
                    a,
                    b,
                    timeline.origin_position.to_numpy(),
                    length=length,
                    replicates=protocol["inference"]["replicates"],
                    seed=protocol["inference"]["seed"],
                )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "first": first,
                        "second": second,
                        **effect,
                    }
                )
    table = pd.DataFrame(comparisons)
    primary = table.block_length.eq(20)
    assert primary.sum() == 24
    table["holm_p"] = np.nan
    table.loc[primary, "holm_p"] = holm_adjust(table.loc[primary, "p_value"].to_numpy())
    table.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "point_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "interval_metrics.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "acquisition": manifest,
            "environment": environment_record(root, "prospective"),
            "classification": (
                "prospective execution; independence contingent on "
                "truthful provenance/exposure review"
            ),
            "protocol_sha256": file_hash(root / "configs/independent_validation.json"),
        },
    )
    write_json(output / "checksums.json", inventory(output))
