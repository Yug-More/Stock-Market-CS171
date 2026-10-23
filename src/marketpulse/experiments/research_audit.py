"""Independent arithmetic and endpoint-based audit of the Phase 6–8 deliverables."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import load_design, load_predictions, write_json


class ScoreTree:
    """Count-tree order statistics, independent of calibration's sorted-list method."""

    def __init__(self, scores: np.ndarray):
        self.grid = np.unique(scores)
        self.ranks = np.searchsorted(self.grid, scores)
        self.tree = np.zeros(len(self.grid) + 1, dtype=int)

    def update(self, row: int, delta: int) -> None:
        position = int(self.ranks[row]) + 1
        while position < len(self.tree):
            self.tree[position] += delta
            position += position & -position

    def quantile(self, rank: int) -> float:
        index = 0
        bit = 1 << (len(self.tree).bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index < len(self.tree) and self.tree[next_index] < rank:
                rank -= int(self.tree[next_index])
                index = next_index
            bit >>= 1
        return float(self.grid[index])


def audit_intervals(source: pd.DataFrame, records: pd.DataFrame, method: str, design: dict) -> int:
    """Check every mature-score population, quantile, and bound through a second algorithm."""
    source = source.sort_values("date").reset_index(drop=True)
    assert source.date.tolist() == records.date.tolist()
    assert source.target_date.tolist() == records.target_date.tolist()
    for column in ("y_true", "y_pred", "volatility", "origin_position"):
        np.testing.assert_allclose(source[column], records[column], rtol=1e-12, atol=1e-14)
    assert source.regime.tolist() == records.regime.tolist()
    assert source.stress.tolist() == records.stress.tolist()
    dates = source.date.to_numpy()
    targets = source.target_date.to_numpy()
    stops = np.searchsorted(targets, dates, side="left")
    assert np.all(stops <= np.arange(len(source)))
    firsts = (
        np.zeros(len(source), int)
        if method == "expanding"
        else np.maximum(0, stops - design["calibration_trailing"])
    )
    counts = stops - firsts
    np.testing.assert_array_equal(records.calibration_n, counts)
    eligibility = counts >= design["calibration_minimum"]
    np.testing.assert_array_equal(records.eligible, eligibility)
    present = stops > 0
    assert np.all(targets[stops[present] - 1] < dates[present])
    assert (
        records.loc[present, "residual_cutoff_date"].tolist()
        == targets[stops[present] - 1].tolist()
    )
    assert (
        records.loc[present, "calibration_first_origin"].tolist() == dates[firsts[present]].tolist()
    )
    assert (
        records.loc[present, "calibration_last_origin"].tolist()
        == dates[stops[present] - 1].tolist()
    )
    scores = np.abs(source.y_true.to_numpy() - source.y_pred.to_numpy())
    volatility = source.volatility.to_numpy()
    if method == "volatility_scaled":
        scores /= volatility
    tree = ScoreTree(scores)
    added = removed = checked = 0
    expected_bounds = np.full((len(source), len(design["coverage_levels"]), 2), np.nan)
    predictions = source.y_pred.to_numpy()
    for i in range(len(source)):
        while added < stops[i]:
            tree.update(added, 1)
            added += 1
        while removed < firsts[i]:
            tree.update(removed, -1)
            removed += 1
        if not eligibility[i]:
            continue
        for level_index, level in enumerate(design["coverage_levels"]):
            rank = math.ceil((int(counts[i]) + 1) * level)
            radius = tree.quantile(rank)
            if method == "volatility_scaled":
                radius *= volatility[i]
            expected_bounds[i, level_index] = predictions[i] - radius, predictions[i] + radius
            checked += 1
    for level_index, level in enumerate(design["coverage_levels"]):
        suffix = round(100 * level)
        np.testing.assert_allclose(
            records[[f"lower_{suffix}", f"upper_{suffix}"]].to_numpy(),
            expected_bounds[:, level_index],
            rtol=1e-10,
            atol=1e-13,
            equal_nan=True,
        )
    return checked


def audit_metrics(records: pd.DataFrame, row: pd.Series) -> None:
    part = (
        records
        if row["slice"] == "overall"
        else records[
            records.regime.eq(row["slice"].removeprefix("volatility_"))
            if row["slice"].startswith("volatility_")
            else records.stress.eq(row["slice"])
        ]
    )
    chosen = part[part.eligible]
    assert len(part) == row.n_origins and len(chosen) == row.n
    assert len(part) - len(chosen) == row.excluded
    if chosen.empty:
        assert pd.isna(row.get("coverage"))
        return
    suffix = round(100 * row.nominal)
    lo, hi = chosen[f"lower_{suffix}"].to_numpy(), chosen[f"upper_{suffix}"].to_numpy()
    actual = chosen.y_true.to_numpy()
    assert np.all(lo <= hi)
    covered = np.count_nonzero((actual >= lo) & (actual <= hi)) / len(chosen)
    alpha = 1 - row.nominal
    scores = (
        hi
        - lo
        + np.where(actual < lo, 2 * (lo - actual) / alpha, 0)
        + np.where(actual > hi, 2 * (actual - hi) / alpha, 0)
    )
    np.testing.assert_allclose(
        [covered, covered - row.nominal, np.mean(hi - lo), np.median(hi - lo), np.mean(scores)],
        [row.coverage, row.coverage_gap, row.mean_width, row.median_width, row.interval_score],
        rtol=1e-10,
        atol=1e-12,
    )


def audit_statistics(root: Path, source: pd.DataFrame) -> int:
    table = pd.read_csv(root / "results/statistical_reliability/comparisons.csv")
    membership = pd.read_csv(root / "results/robustness/matched/membership.csv.gz")
    for row in table.itertuples():
        group = source[(source.dataset == row.dataset) & (source.horizon == row.horizon)]
        if row.population == "endpoint_matched":
            keys = membership[membership.endpoint_match & (membership.horizon == row.horizon)].date
            group = group[group.date.isin(keys)]
        left = group[group.model == row.first].sort_values("date")
        right = group[group.model == row.second].sort_values("date")
        assert left.date.tolist() == right.date.tolist() and len(left) == row.n
        a, b = (
            left.y_pred.to_numpy() - left.y_true.to_numpy(),
            right.y_pred.to_numpy() - right.y_true.to_numpy(),
        )
        a, b = (a * a, b * b) if row.loss == "squared_error" else (np.abs(a), np.abs(b))
        np.testing.assert_allclose(row.mean_loss_difference, np.mean(a - b), rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(
            row.skill_vs_comparator, 1 - np.mean(a) / np.mean(b), rtol=1e-8, atol=1e-12
        )
        centered = (a - b) - np.mean(a - b)
        positions = left.origin_position.to_numpy()
        segments = np.r_[0, np.cumsum(np.diff(positions) != 1)]
        long_run = float(np.mean(centered**2))
        for lag in range(1, row.block_length):
            valid = segments[lag:] == segments[:-lag]
            covariance = np.sum((centered[lag:] * centered[:-lag])[valid]) / len(centered)
            long_run += 2 * (1 - lag / row.block_length) * covariance
        se = np.sqrt(max(long_run / len(centered), 0))
        np.testing.assert_allclose(row.hac_standard_error, se, rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(
            row.p_value, math.erfc(abs(np.mean(a - b) / se) / np.sqrt(2)), rtol=1e-7, atol=1e-14
        )
    primary = table[table.primary_family].copy()
    assert len(primary) == 18
    order = primary.p_value.sort_values().index
    for rank, index in enumerate(order):
        expected = min(
            1.0,
            max(
                (18 - j) * table.loc[other, "p_value"] for j, other in enumerate(order[: rank + 1])
            ),
        )
        np.testing.assert_allclose(
            table.loc[index, "holm_p_value"], expected, rtol=1e-10, atol=1e-14
        )
    return len(table)


def audit_seeds(root: Path, source: pd.DataFrame, design: dict) -> int:
    directory = root / "results/robustness/multiseed"
    manifest = json.loads((directory / "manifest.json").read_text())
    markers = sorted((directory / "runs").glob("*.json"))
    assert len(markers) == manifest["expected_tasks"] == 90
    for relative, expected in manifest["identity"]["inputs"].items():
        assert file_hash(root / relative) == expected
    from marketpulse.experiments.benchmark import load_benchmark_settings

    expected_settings = json.loads(json.dumps(asdict(load_benchmark_settings(root))))
    metrics = pd.read_csv(root / "results/robustness/seed_metrics.csv")
    pieces = []
    identifiers = set()
    for marker in markers:
        payload = json.loads(marker.read_text())
        key = (payload["dataset"], payload["horizon"], payload["fold"], payload["seed"])
        assert key not in identifiers
        identifiers.add(key)
        assert payload["seed"] in design["seeds"]
        assert payload["execution"]["settings"] == expected_settings
        csv = marker.with_suffix(".csv.gz")
        assert file_hash(csv) == payload["predictions_sha256"]
        fresh = pd.read_csv(csv)
        assert set(fresh.model) == {"random_forest", "lstm"}
        original = source[
            (source.dataset == payload["dataset"])
            & (source.horizon == payload["horizon"])
            & (source.fold == payload["fold"])
        ]
        old_marker = (
            root
            / "results/walkforward/full/folds"
            / marker.stem.split("_h")[0]
            / f"h{payload['horizon']}"
            / f"fold_{payload['fold']:03d}.json"
        )
        assert payload["sample"] == json.loads(old_marker.read_text())["sample"]
        for model in ("random_forest", "lstm"):
            new = fresh[fresh.model == model].sort_values("date")
            old = original[original.model == model].sort_values("date")
            assert new.date.tolist() == old.date.tolist()
            np.testing.assert_allclose(new.y_true, old.y_true, rtol=1e-12, atol=1e-14)
            assert np.isfinite(new.y_pred).all()
            assert new.regime.tolist() == old.regime.tolist()
            np.testing.assert_allclose(new.volatility, old.volatility, rtol=1e-12, atol=1e-14)
        for model in payload["models"]:
            details = model["details"]
            assert details["seed"] == payload["seed"]
            score_key = (
                "validation_rmse_by_depth"
                if model["model"] == "random_forest"
                else "validation_rmse_by_hidden_size"
            )
            selected_key = (
                "selected_max_depth"
                if model["model"] == "random_forest"
                else "selected_hidden_size"
            )
            scores = {int(k): v for k, v in details[score_key].items()}
            assert details[selected_key] == min(scores, key=lambda k: (scores[k], k))
        fresh["seed"] = payload["seed"]
        pieces.append(fresh)
    fresh = pd.concat(pieces, ignore_index=True)
    for row in metrics.itertuples():
        part = fresh[
            (fresh.dataset == row.dataset)
            & (fresh.horizon == row.horizon)
            & (fresh.seed == row.seed)
            & (fresh.model == row.model)
        ]
        if row.model not in ("random_forest", "lstm"):
            selected_folds = fresh[
                (fresh.dataset == row.dataset) & (fresh.horizon == row.horizon)
            ].fold.unique()
            part = source[
                (source.dataset == row.dataset)
                & (source.horizon == row.horizon)
                & source.fold.isin(selected_folds)
                & (source.model == row.model)
            ]
        assert len(part) == row.n == 756
        errors = part.y_pred.to_numpy() - part.y_true.to_numpy()
        np.testing.assert_allclose(
            [row.rmse, row.mae],
            [np.sqrt(np.mean(errors**2)), np.mean(np.abs(errors))],
            rtol=1e-10,
            atol=1e-13,
        )
    return len(markers)


def audit_cycle(root: Path) -> None:
    design = load_design(root)
    source = load_predictions(root)
    for area in (
        "robustness/audit",
        "robustness/matched",
        "statistical_reliability",
        "calibration",
    ):
        directory = root / "results" / area
        checks = json.loads((directory / "checksums.json").read_text())
        for relative, expected in checks.items():
            assert file_hash(directory / relative) == expected, (
                f"changed artifact: {area}/{relative}"
            )
        execution = json.loads((directory / "execution.json").read_text())
        revision = execution["environment"]["git_sha"]
        for relative, expected in execution["source_hashes"].items():
            if file_hash(root / relative) != expected:
                blob = subprocess.run(
                    ["git", "show", f"{revision}:{relative}"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                ).stdout
                assert hashlib.sha256(blob).hexdigest() == expected
    membership = pd.read_csv(root / "results/robustness/matched/membership.csv.gz")
    zeros = source[source.model == "zero_return"]
    for horizon in (1, 5, 20):
        a = zeros[(zeros.dataset == "NASDAQ-100") & (zeros.horizon == horizon)].set_index("date")
        b = zeros[(zeros.dataset == "S&P 500") & (zeros.horizon == horizon)].set_index("date")
        origins = set(a.index) & set(b.index)
        matched = {
            date for date in origins if a.loc[date, "target_date"] == b.loc[date, "target_date"]
        }
        saved = membership[membership.horizon == horizon]
        assert set(saved[saved.origin_match].date) == origins
        assert set(saved[saved.endpoint_match].date) == matched
    matched_metrics = pd.read_csv(root / "results/robustness/matched/metrics.csv")
    for row in matched_metrics.itertuples():
        group = source[(source.dataset == row.dataset) & (source.horizon == row.horizon)]
        if row.population != "native":
            field = "origin_match" if row.population == "origin_matched" else "endpoint_match"
            allowed = membership[(membership.horizon == row.horizon) & membership[field]].date
            group = group[group.date.isin(allowed)]
        part = group[group.model == row.model].sort_values("date")
        zero = group[group.model == "zero_return"].sort_values("date")
        mean = group[group.model == "historical_mean"].sort_values("date")
        assert part.date.tolist() == zero.date.tolist() == mean.date.tolist()
        assert len(part) == row.n
        error = part.y_pred.to_numpy() - part.y_true.to_numpy()
        mean_error = mean.y_pred.to_numpy() - mean.y_true.to_numpy()
        np.testing.assert_allclose(
            [row.rmse, row.mae, row.skill_vs_zero, row.skill_vs_historical_mean],
            [
                np.sqrt(np.mean(error**2)),
                np.mean(np.abs(error)),
                1 - np.mean(error**2) / np.mean(zero.y_true.to_numpy() ** 2),
                1 - np.mean(error**2) / np.mean(mean_error**2),
            ],
            rtol=1e-8,
            atol=1e-12,
        )
    statistical_rows = audit_statistics(root, source)
    seed_runs = audit_seeds(root, source, design)
    manifest = json.loads((root / "results/calibration/manifest.json").read_text())
    metrics = pd.read_csv(root / "results/calibration/metrics.csv")
    quantiles = interval_rows = 0
    for item in manifest["files"]:
        path = root / "results/calibration" / item["path"]
        assert file_hash(path) == item["sha256"]
        records = pd.read_csv(path)
        group = source[
            (source.dataset == item["dataset"])
            & (source.horizon == item["horizon"])
            & (source.model == item["model"])
        ]
        quantiles += audit_intervals(group, records, item["method"], design)
        interval_rows += len(records)
        chosen = metrics[
            (metrics.dataset == item["dataset"])
            & (metrics.horizon == item["horizon"])
            & (metrics.model == item["model"])
            & (metrics.method == item["method"])
        ]
        for _, row in chosen.iterrows():
            audit_metrics(records, row)
    result = {
        "historical_predictions": len(source),
        "seed_tasks": seed_runs,
        "paired_statistical_rows": statistical_rows,
        "interval_records": interval_rows,
        "independently_verified_quantiles_and_bounds": quantiles,
        "all_residual_populations_mature": True,
        "all_coverage_and_score_metrics_verified": True,
        "historical_provenance_repaired": False,
    }
    write_json(root / "results/robustness/final_audit.json", result)
    print(json.dumps(result, indent=2))
