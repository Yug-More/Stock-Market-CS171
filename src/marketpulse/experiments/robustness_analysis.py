"""Historical audit, matched-calendar results, and stochastic sensitivity summaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import (
    load_predictions,
    matched_membership,
    select_matched,
    write_json,
)
from marketpulse.experiments.summary import pooled_metrics


def audit_historical(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(root / ".venv/bin/python") if (root / ".venv/bin/python").exists() else "python",
            str(root / "scripts/verify_walkforward.py"),
        ],
        check=True,
    )
    frame = load_predictions(root)
    assert (pd.to_datetime(frame.target_date) > pd.to_datetime(frame.date)).all()
    base = root / "results/walkforward/full"
    audit = json.loads((base / "verification.json").read_text())
    audit.update(
        actual_target_dates_verified=len(frame),
        input_predictions_sha256=file_hash(base / "predictions.csv.gz"),
        original_artifact_inventory_sha256=file_hash(base / "artifact_checksums.json"),
        environment=environment_record(root, "historical_audit"),
        provenance={
            "legacy_without_execution_metadata": 210,
            "folds_with_execution_metadata": 72,
            "new_targeted_reruns": "18 folds, seed 171, separately identified",
            "cannot_reconstruct": [
                "legacy execution revision",
                "legacy runtime",
                "legacy hardware/dependency context",
            ],
        },
    )
    write_json(output / "historical_audit.json", audit)
    (output / "HISTORICAL_AUDIT.md").write_text(
        "# Independent historical audit\n\n282 folds and 355,320 predictions passed the original "
        "audit and fresh actual target-date reconstruction. Pooled metrics use individual losses, "
        "not mean fold RMSE. All recovered-file hashes remain unchanged.\n\n"
        "Predictions, date/label alignment, configured selection records, thresholds, regimes, "
        "and errors can be verified. Actual historical execution of 210 legacy training runs "
        "and their absent environment/revision/runtime cannot be independently proven. The 72 "
        "newer folds have execution metadata, though weights were not saved. A new seed-171 "
        "rerun of 18 time-stratified folds supplies separate targeted provenance; it does not "
        "repair the old records or reproduce all 282 original fits.\n"
    )


def matched_analysis(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame = load_predictions(root)
    membership = matched_membership(frame)
    membership.to_csv(
        output / "membership.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    metrics, counts = [], []
    for population, selected in (
        ("native", frame),
        ("origin_matched", select_matched(frame, membership, endpoints=False)),
        ("endpoint_matched", select_matched(frame, membership, endpoints=True)),
    ):
        scored = pooled_metrics(selected, ("dataset", "horizon"))
        scored["population"] = population
        metrics.append(scored)
        for row in scored.itertuples():
            native_n = int(
                frame[
                    (frame.dataset == row.dataset)
                    & (frame.horizon == row.horizon)
                    & (frame.model == row.model)
                ].shape[0]
            )
            counts.append(
                {
                    "dataset": row.dataset,
                    "horizon": row.horizon,
                    "model": row.model,
                    "population": population,
                    "n": row.n,
                    "native_n": native_n,
                    "removed": native_n - row.n,
                    "retained_fraction": row.n / native_n,
                }
            )
    pd.concat(metrics, ignore_index=True).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "counts.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "environment": environment_record(root, "matched"),
            "input_sha256": file_hash(root / "results/walkforward/full/predictions.csv.gz"),
            "primary": "origin and actual endpoint dates",
            "sensitivity": "origin only",
            "endpoint_disagreements": int(
                (membership.origin_match & ~membership.endpoint_match).sum()
            ),
        },
    )


def seed_summary(root: Path, output: Path, directory: Path | None = None) -> None:
    directory = directory or root / "results/robustness/multiseed"
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = sorted((directory / "runs").glob("*.json"))
    if len(reports) != manifest["expected_tasks"]:
        raise ValueError("seed study is incomplete")
    native = load_predictions(root)
    rows, fold_rows, reruns = [], [], []
    for seed in manifest["identity"]["seeds"]:
        pieces = []
        for marker in reports:
            payload = json.loads(marker.read_text())
            if payload["seed"] != seed:
                continue
            csv = marker.with_suffix(".csv.gz")
            if file_hash(csv) != payload["predictions_sha256"]:
                raise ValueError(f"seed checksum mismatch: {csv}")
            part = pd.read_csv(csv)
            original = native[
                (native.dataset == payload["dataset"])
                & (native.horizon == payload["horizon"])
                & (native.fold == payload["fold"])
            ]
            baseline = original[
                original.model.isin(["zero_return", "historical_mean", "linear_regression"])
            ][part.columns.drop("seed")]
            combined = pd.concat([part.drop(columns="seed"), baseline], ignore_index=True)
            pieces.append(combined)
            scored = pooled_metrics(combined, ("dataset", "horizon", "fold"))
            scored["seed"] = seed
            fold_rows.append(scored)
            if seed == 171:
                for model in ("random_forest", "lstm"):
                    fresh = part[part.model == model].sort_values("date")
                    old = original[original.model == model].sort_values("date")
                    assert fresh.date.tolist() == old.date.tolist()
                    difference = fresh.y_pred.to_numpy() - old.y_pred.to_numpy()
                    reruns.append(
                        {
                            "dataset": payload["dataset"],
                            "horizon": payload["horizon"],
                            "fold": payload["fold"],
                            "model": model,
                            "max_prediction_difference": float(np.max(np.abs(difference))),
                            "numerically_equal_at_1e_10": bool(
                                np.allclose(fresh.y_pred, old.y_pred, rtol=1e-10, atol=1e-12)
                            ),
                        }
                    )
        pooled = pooled_metrics(pd.concat(pieces, ignore_index=True), ("dataset", "horizon"))
        pooled["seed"] = seed
        rows.append(pooled)
    output.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output / "seed_metrics.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(output / "seed_fold_metrics.csv", index=False)
    pd.DataFrame(reruns).to_csv(output / "seed171_replication.csv", index=False)
    result.groupby(["dataset", "horizon", "model"]).agg(
        seeds=("seed", "nunique"),
        rmse_min=("rmse", "min"),
        rmse_max=("rmse", "max"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        skill_min=("skill_vs_zero", "min"),
        skill_max=("skill_vs_zero", "max"),
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
    ).reset_index().to_csv(output / "seed_stability.csv", index=False)
