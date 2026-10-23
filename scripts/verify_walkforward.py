"""Independently verify all planned folds and recompute published metric tables.

Run from the repository root: python scripts/verify_walkforward.py.
For a fresh checkout, first extract fold_predictions.tar.gz into the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from marketpulse.data.catalog import load_catalog
from marketpulse.experiments.benchmark import load_benchmark_settings
from marketpulse.experiments.recovery import experiment_identity, file_hash, validate_fold
from marketpulse.experiments.summary import fold_summary, pooled_metrics, stress_tables
from marketpulse.experiments.walkforward import (
    _eligible_table,
    _fold_metrics_from_markers,
    _plan,
    _slug,
    _threshold_table,
)
from marketpulse.protocol import load_forecasting_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/walkforward/full"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    protocol = load_forecasting_protocol(root)
    settings = load_benchmark_settings(root)
    manifest = json.loads((output / "manifest.json").read_text())
    identity = experiment_identity(root, "full", protocol, settings)
    identity["max_folds"] = None
    if json.loads(json.dumps(identity)) != manifest["identity"]:
        raise AssertionError("source, configuration, or data differ from the saved manifest")
    tasks = _plan(root, load_catalog(root), protocol.horizons, protocol, settings, None)
    tasks = sorted(
        tasks,
        key=lambda task: (_slug(task["entry"].name), f"h{task['horizon']}", task["fold_index"]),
    )
    expected_markers = {
        output
        / "folds"
        / _slug(task["entry"].name)
        / f"h{task['horizon']}"
        / f"fold_{task['fold_index']:03d}.json"
        for task in tasks
    }
    assert set(output.glob("folds/*/*/*.json")) == expected_markers
    assert set(output.glob("folds/*/*/*.csv")) == {
        path.with_suffix(".csv") for path in expected_markers
    }
    frames = []
    legacy = 0
    runtimes = []
    for task in tasks:
        directory = output / "folds" / _slug(task["entry"].name) / f"h{task['horizon']}"
        marker = directory / f"fold_{task['fold_index']:03d}.json"
        csv = marker.with_suffix(".csv")
        if not bool((np.diff(task["origins"]) == 1).all()):
            raise AssertionError(
                "published session-step interpretation requires consecutive origins"
            )
        result = validate_fold(task, settings, protocol.seed, marker, csv)
        legacy += result["provenance"] == "legacy_unrecorded"
        payload = json.loads(marker.read_text())
        if "execution" in payload:
            runtimes.append(payload["execution"]["elapsed_seconds"])
        volatility = task["features"]["roll_vol_20"].to_numpy()
        for key, indices in (
            ("validation_mean", task["split"].validation),
            ("test_mean", task["split"].test),
        ):
            np.testing.assert_allclose(
                payload["volatility_thresholds"][key], np.mean(volatility[indices]), rtol=1e-12
            )
        for model in payload["models"]:
            details = model["details"]
            for field, selected in (
                ("validation_rmse_by_depth", "selected_max_depth"),
                ("validation_rmse_by_hidden_size", "selected_hidden_size"),
            ):
                if field in details:
                    scores = {int(k): v for k, v in details[field].items()}
                    assert np.isfinite(list(scores.values())).all()
                    assert details[selected] == min(scores, key=lambda k: (scores[k], k))
                    np.testing.assert_allclose(
                        model["validation"]["rmse"], scores[details[selected]], rtol=1e-12
                    )
            if model["model"] == "lstm":
                history = details["history"]
                assert details["epochs_trained"] == len(history)
                assert (
                    details["best_epoch"]
                    == min(history, key=lambda r: r["validation_mse"])["epoch"]
                )
        frames.append(pd.read_csv(csv))
    predictions = pd.concat(frames, ignore_index=True)
    assert not predictions.duplicated(["dataset", "horizon", "model", "date"]).any()
    source = output / "predictions.csv"
    if not source.exists():
        source = output / "predictions.csv.gz"
    published = pd.read_csv(source)
    assert_frame_equal(predictions, published, check_exact=False, rtol=1e-10, atol=1e-14)
    pooled = pooled_metrics(predictions, ("dataset", "horizon"))
    saved = pd.read_csv(output / "pooled_metrics.csv")
    assert_frame_equal(pooled, saved, check_exact=False, rtol=1e-9, atol=1e-12)
    regimes = pooled_metrics(predictions, ("dataset", "horizon", "regime"))
    saved = pd.read_csv(output / "regime_metrics.csv")
    assert_frame_equal(regimes, saved[regimes.columns], check_exact=False, rtol=1e-9, atol=1e-12)
    eligible = _eligible_table(root, protocol, settings, predictions)
    stress = stress_tables(predictions, eligible)
    saved = pd.read_csv(output / "stress_metrics.csv")
    # CSV roundtrips infer nullable strings for unavailable model rows.
    stress = stress.astype({"model": "string"})
    saved = saved.astype({"model": "string"})
    assert_frame_equal(stress, saved, check_exact=False, rtol=1e-9, atol=1e-12)
    fold_metrics = _fold_metrics_from_markers(output)
    saved = pd.read_csv(output / "fold_metrics.csv")
    assert_frame_equal(fold_metrics, saved, check_exact=False, rtol=1e-9, atol=1e-12)
    summary = fold_summary(fold_metrics)
    saved = pd.read_csv(output / "fold_summary.csv")
    assert_frame_equal(summary, saved, check_exact=False, rtol=1e-9, atol=1e-12)
    thresholds = _threshold_table(output)
    saved = pd.read_csv(output / "volatility_thresholds.csv")
    assert_frame_equal(thresholds, saved, check_exact=False, rtol=1e-9, atol=1e-12)
    # Recovery records are immutable checksums of the untouched original files.
    for recovered in manifest["recovered"]:
        marker = output.parent / recovered["marker"]
        assert file_hash(marker) == recovered["marker_sha256"]
        assert file_hash(marker.with_suffix(".csv")) == recovered["predictions_sha256"]
    inventory = output / "artifact_checksums.json"
    if inventory.exists():
        for relative, expected_hash in json.loads(inventory.read_text()).items():
            if file_hash(output / relative) != expected_hash:
                raise AssertionError(f"artifact checksum mismatch: {relative}")
    result = {
        "planned_folds": len(tasks),
        "validated_folds": len(frames),
        "legacy_folds": legacy,
        "new_folds": len(frames) - legacy,
        "prediction_rows": len(predictions),
        "origin_horizon_pairs": len(predictions) // 5,
        "new_fold_runtime_seconds": sum(runtimes),
        "published_test_and_slice_metrics_recomputed": True,
        "fold_summaries_and_thresholds_verified": True,
        "recovered_file_hashes_unchanged": True,
    }
    print(json.dumps(result, indent=2))
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
