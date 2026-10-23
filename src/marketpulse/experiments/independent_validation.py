"""Four-block frozen evaluation, with an explicit prospective-data gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.evaluation.calibration import interval_score, residual_quantile
from marketpulse.evaluation.partitions import ChronologicalSplit, terminal_chronological_split
from marketpulse.experiments.benchmark import (
    environment_record,
    load_benchmark_settings,
    score_split,
)
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import write_json
from marketpulse.experiments.walkforward import _plan
from marketpulse.protocol import load_forecasting_protocol


def four_blocks(
    origins: np.ndarray, horizon: int
) -> tuple[ChronologicalSplit, np.ndarray, np.ndarray]:
    """Reserve test, then calibration, then validation, purging actual row endpoints."""
    outer = terminal_chronological_split(
        origins,
        horizon=horizon,
        minimum_training_origins=2520,
        validation_window=252,
        test_window=252,
    )
    test, calibration = outer.test, outer.validation
    clean = origins[origins + horizon < calibration[0]]
    validation = clean[-252:]
    train = origins[origins + horizon < validation[0]]
    if len(train) < 2520 or len(validation) != 252:
        raise ValueError("insufficient four-block history")
    split = replace(outer, train=train, validation=validation, test=np.r_[calibration, test])
    # The existing trainer treats calibration+test as a prediction-only block.
    # Neither block is used for fitting, scaling, early stopping or selection.
    return split, calibration, test


def prospective_gate(dates: pd.Series, test: np.ndarray, manifest: dict, protocol: dict) -> None:
    required = {"source_url", "version", "retrieved_utc", "license", "adjustments", "sha256"}
    if required - manifest.keys() or any(not manifest[k] for k in required):
        raise ValueError("incomplete acquisition provenance")
    if manifest.get("prior_outcome_exposure") != "none declared":
        raise ValueError("independence requires an explicit prior-exposure declaration")
    if pd.to_datetime(dates.iloc[test[0]]) < pd.Timestamp(protocol["earliest_test_origin"]):
        raise ValueError("historical dates cannot be labeled independent")
    if pd.Timestamp(manifest["retrieved_utc"]) < pd.Timestamp(dates.iloc[test[-1]]).tz_localize(
        "UTC"
    ):
        raise ValueError("data were acquired before the final test origin")


def run_rehearsal(root: Path, output: Path) -> None:
    """Execute full-budget historical wiring only; never call this confirmation."""
    if output.exists():
        raise ValueError("preserve the previous rehearsal; choose a new output directory")
    output.mkdir(parents=True)
    protocol = load_forecasting_protocol(root)
    settings = load_benchmark_settings(root)
    tasks = _plan(root, load_catalog(root), protocol.horizons, protocol, settings, None)
    tasks = [t for t in tasks if t["fold_index"] == t["n_folds"] - 1]
    rows, plans, provenance = [], [], []
    for task in tasks:
        split, calibration, test = four_blocks(task["origins"], task["horizon"])
        dates = pd.to_datetime(task["market"].date)
        blocks = [split.train, split.validation, calibration, test]
        assert all(a[-1] + task["horizon"] < b[0] for a, b in zip(blocks, blocks[1:], strict=False))
        plans.append(
            {
                "dataset": task["entry"].name,
                "horizon": task["horizon"],
                "blocks": [
                    {
                        "name": name,
                        "n": len(block),
                        "first": str(dates.iloc[block[0]].date()),
                        "last": str(dates.iloc[block[-1]].date()),
                        "last_endpoint": str(dates.iloc[block[-1] + task["horizon"]].date()),
                    }
                    for name, block in zip(
                        ["train", "validation", "calibration", "test"], blocks, strict=True
                    )
                ],
            }
        )
        result = score_split(
            dataset_name=task["entry"].name,
            index_id=task["entry"].index_id,
            horizon=task["horizon"],
            feature_matrix=task["features"].to_numpy(float),
            target_values=task["target"].to_numpy(float),
            target=task["target"],
            features=task["features"],
            dates=dates,
            split=split,
            settings=settings,
            seed=171,
        )
        frame = pd.DataFrame(result.pop("predictions"))
        calibration_dates = dates.iloc[calibration].dt.strftime("%Y-%m-%d")
        frame["partition"] = np.where(frame.date.isin(calibration_dates), "calibration", "test")
        records = []
        for model, part in frame.groupby("model"):
            cal = part[part.partition == "calibration"]
            evaluated = part[part.partition == "test"]
            scores = sorted(np.abs(cal.y_true - cal.y_pred))
            for level in (0.8, 0.9):
                radius = residual_quantile(scores, level)
                lo, hi = evaluated.y_pred - radius, evaluated.y_pred + radius
                records.append(
                    {
                        "dataset": task["entry"].name,
                        "horizon": task["horizon"],
                        "model": model,
                        "nominal": level,
                        "n": len(evaluated),
                        "rmse": float(np.sqrt(np.mean((evaluated.y_true - evaluated.y_pred) ** 2))),
                        "coverage": float(
                            ((evaluated.y_true >= lo) & (evaluated.y_true <= hi)).mean()
                        ),
                        "mean_width": 2 * radius,
                        "interval_score": float(
                            interval_score(evaluated.y_true, lo, hi, level).mean()
                        ),
                    }
                )
        rows.extend(records)
        name = f"{task['entry'].name.replace(' ', '_')}_h{task['horizon']}"
        frame.to_csv(
            output / f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
        )
        write_json(output / f"{name}.json", result)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    for entry in load_catalog(root):
        task = next(t for t in tasks if t["entry"].name == entry.name)
        provenance.append(
            {
                "dataset": entry.name,
                "path": entry.path,
                "sha256": file_hash(root / entry.path),
                "first": str(task["market"].date.iloc[0]),
                "last": str(task["market"].date.iloc[-1]),
                "source": entry.documented_source,
                "acquisition_utc": None,
                "license": "not recorded",
                "adjustments": "price-index closes; dividend total returns unavailable",
                "independent": False,
            }
        )
    write_json(
        output / "manifest.json",
        {
            "classification": "retrospective infrastructure rehearsal",
            "independent_validation_completed": False,
            "blocker": "No acquired, provenance-complete, unexposed prospective observations; fixed future test is not mature.",
            "protocol_sha256": file_hash(root / "configs/independent_validation.json"),
            "environment": environment_record(root, "full"),
            "provenance": provenance,
            "plans": plans,
        },
    )
