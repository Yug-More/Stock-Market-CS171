"""Adversarial reconstruction using separate arithmetic, not production metric/inference helpers.

The shared raw CSV loader is reused for format decoding. This computational review
is not an independent human review or independent dataset validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.data.loader import load_market_csv


def holm(pvalues):
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    result = np.zeros(len(ordered))
    previous = 0.0
    for rank, (index, value) in enumerate(ordered):
        previous = min(1.0, max(previous, (len(ordered) - rank) * value))
        result[index] = previous
    return result


def inference(difference, positions, length=20, replicates=2000, seed=6171):
    d = np.asarray(difference)
    positions = np.asarray(positions)
    n = len(d)
    centered = d - d.mean()
    runs = np.cumsum(np.r_[False, np.diff(positions) != 1])
    variance_sum = float(centered @ centered)
    for lag in range(1, length):
        valid = runs[lag:] == runs[:-lag]
        products = centered[lag:] * centered[:-lag]
        variance_sum += 2 * (1 - lag / length) * products[valid].sum()
    se = math.sqrt(max(variance_sum, 0)) / n
    p = math.erfc(abs(d.mean() / se) / math.sqrt(2))

    # Convolution, rather than production prefix-sum resampling of two loss columns.
    def candidates(width):
        starts = np.arange(n - width + 1)
        starts = starts[positions[starts + width - 1] - positions[starts] == width - 1]
        return np.convolve(d, np.ones(width), mode="valid")[starts]

    sums = candidates(length)
    rng = np.random.default_rng(seed)
    count, remainder = divmod(n, length)
    estimates = sums[rng.integers(0, len(sums), size=(replicates, count))].sum(axis=1)
    if remainder:
        short = candidates(remainder)
        estimates += rng.choice(short, size=replicates)
    bounds = np.quantile(estimates / n, [0.025, 0.975])
    return d.mean(), se, p, bounds


def run(root: Path) -> dict:
    inputs = {}

    def track(path):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def csv(path):
        return pd.read_csv(track(path))

    def js(path):
        return json.loads(track(path).read_text())

    track("src/marketpulse/scientific_review.py")
    track("scripts/scientific_verification.py")
    track("configs/datasets.toml")
    track("src/marketpulse/data/loader.py")
    track("src/marketpulse/data/catalog.py")
    track("tests/test_scientific_review.py")
    original_design = tomllib.loads(track("configs/robustness.toml").read_text())
    markets, lookups = {}, {}
    for entry in load_catalog(root):
        market = load_market_csv(track(entry.path), dataset_name=entry.name)
        dates = market.date.dt.strftime("%Y-%m-%d").to_numpy()
        markets[entry.name] = (dates, market.close.to_numpy())
        lookups[entry.name] = {date: i for i, date in enumerate(dates)}

    def positions(frame):
        return np.array(
            [lookups[d][date] for d, date in zip(frame.dataset, frame.date, strict=True)]
        )

    def labels(frame):
        for (dataset, horizon), part in frame.groupby(["dataset", "horizon"]):
            pos = positions(part)
            dates, close = markets[dataset]
            np.testing.assert_allclose(
                part.y_true, np.log(close[pos + horizon] / close[pos]), rtol=1e-10, atol=1e-14
            )
            if "target_date" in part:
                assert np.array_equal(part.target_date, dates[pos + horizon])

    native = csv("results/walkforward/full/predictions.csv.gz")
    labels(native)
    assert not native.duplicated(["dataset", "horizon", "model", "date"]).any()
    native["position"] = positions(native)
    paired_groups = 0
    for _, frame in native.groupby(["dataset", "horizon"]):
        reference = None
        for _, model in frame.groupby("model"):
            keys = model.sort_values("position")[["position", "y_true", "fold"]].to_numpy()
            if reference is not None:
                np.testing.assert_array_equal(keys, reference)
            reference = keys
        paired_groups += 1

    folds = js("results/walkforward/full/fold_plan.json")
    for fold in folds:
        lookup = lookups[fold["dataset"]]
        h, sample = fold["horizon"], fold["sample"]
        train, validation, test = [sample[k] for k in ["train", "validation", "test"]]
        assert lookup[train["end_date"]] + h < lookup[validation["start_date"]]
        assert lookup[validation["end_date"]] + h < lookup[test["start_date"]]
        part = native[
            native.dataset.eq(fold["dataset"]) & native.horizon.eq(h) & native.fold.eq(fold["fold"])
        ]
        assert len(part) == 5 * test["n"] == 1260
        assert part.date.min() == test["start_date"] and part.date.max() == test["end_date"]
        train_pos = np.arange(lookup[train["start_date"]], lookup[train["end_date"]] + 1)
        assert len(train_pos) == train["n"]
        close = markets[fold["dataset"]][1]
        mean = np.log(close[train_pos + h] / close[train_pos]).mean()
        np.testing.assert_allclose(part[part.model.eq("historical_mean")].y_pred, mean, atol=1e-14)
        assert part[part.model.eq("zero_return")].y_pred.eq(0).all()

    def metrics(frame, expected):
        checked = 0
        for (dataset, horizon), cell in frame.groupby(["dataset", "horizon"]):
            models = {m: p.sort_values("date") for m, p in cell.groupby("model")}
            losses = {m: np.mean((p.y_pred - p.y_true) ** 2) for m, p in models.items()}
            for model, part in models.items():
                row = expected[
                    expected.dataset.eq(dataset)
                    & expected.horizon.eq(horizon)
                    & expected.model.eq(model)
                ].iloc[0]
                y, pred = part.y_true.to_numpy(), part.y_pred.to_numpy()
                nonzero = y != 0
                values = [
                    len(y),
                    np.mean(np.abs(pred - y)),
                    np.sqrt(losses[model]),
                    np.mean(pred - y),
                    np.mean(np.sign(pred[nonzero]) == np.sign(y[nonzero])),
                    1 - losses[model] / losses["zero_return"],
                    1 - losses[model] / losses["historical_mean"],
                    1 + sum(loss < losses[model] for loss in losses.values()),
                ]
                actual = [
                    row.n,
                    row.mae,
                    row.rmse,
                    row.bias,
                    row.directional_accuracy,
                    row.skill_vs_zero,
                    row.skill_vs_historical_mean,
                    row["rank"],
                ]
                np.testing.assert_allclose(actual, values, rtol=1e-10, atol=1e-13)
                checked += 1
        return checked

    metric_rows = metrics(native, csv("results/walkforward/full/pooled_metrics.csv"))
    selected = []
    for h, frame in native.groupby("horizon"):
        base = frame[frame.model.eq("zero_return")]
        a, b = [p for _, p in base.groupby("dataset")]
        dates_a, dates_b = markets[a.dataset.iloc[0]][0], markets[b.dataset.iloc[0]][0]
        a = a.assign(endpoint=dates_a[a.position.to_numpy() + h])
        b = b.assign(endpoint=dates_b[b.position.to_numpy() + h])
        common = a[["date", "endpoint"]].merge(b[["date", "endpoint"]], on=["date", "endpoint"])
        assert len(common) == 2232
        selected.append(frame[frame.date.isin(common.date)])
    matched = csv("results/robustness/matched/metrics.csv")
    metric_rows += metrics(pd.concat(selected), matched[matched.population.eq("endpoint_matched")])

    reports, seed_frames = [], []
    for directory in ["robustness", "phase10"]:
        for path in sorted((root / f"results/{directory}/multiseed/runs").glob("*.json")):
            report = js(path)
            stochastic = csv(path.with_suffix(".csv.gz"))
            labels(stochastic)
            old = native[
                native.dataset.eq(report["dataset"])
                & native.horizon.eq(report["horizon"])
                & native.fold.eq(report["fold"])
                & native.model.isin(["zero_return", "historical_mean", "linear_regression"])
            ]
            frame = pd.concat([stochastic, old], ignore_index=True)
            frame["seed"] = report["seed"]
            seed_frames.append(frame)
            reports.append(report)
    seeds = pd.concat(seed_frames, ignore_index=True)
    assert len(reports) == 150
    expected_seed = csv("results/phase10/seed_metrics.csv")
    for seed, frame in seeds.groupby("seed"):
        assert frame.groupby(["dataset", "horizon", "model"]).size().eq(1260).all()
        metric_rows += metrics(frame, expected_seed[expected_seed.seed.eq(seed)])
    seeds["loss"] = (seeds.y_pred - seeds.y_true) ** 2
    loss = seeds.groupby(["dataset", "horizon", "model", "date"], as_index=False).loss.mean()
    saved_loss = csv("results/phase10/matched_losses.csv.gz")
    joined = saved_loss.merge(
        loss, on=["dataset", "horizon", "model", "date"], validate="one_to_one"
    )
    np.testing.assert_allclose(joined.mean_seed_loss, joined.loss, atol=1e-14)

    stat_vectors = []
    for path, primary_column, adjusted_column, frame in [
        (
            "results/statistical_reliability/comparisons.csv",
            "primary_family",
            "holm_p_value",
            native,
        ),
        ("results/phase10/comparisons.csv", "primary", "holm_p", saved_loss),
    ]:
        table = csv(path)
        primary = table[table[primary_column]]
        vectors = []
        for row in primary.itertuples():
            cell = frame[frame.dataset.eq(row.dataset) & frame.horizon.eq(row.horizon)]
            a, b = [cell[cell.model.eq(m)].sort_values("date") for m in [row.first, row.second]]
            assert a.date.tolist() == b.date.tolist()
            if "mean_seed_loss" in a:
                d = a.mean_seed_loss.to_numpy() - b.mean_seed_loss.to_numpy()
                pos = a.origin_position.to_numpy()
            else:
                d = (a.y_pred.to_numpy() - a.y_true.to_numpy()) ** 2
                d -= (b.y_pred.to_numpy() - b.y_true.to_numpy()) ** 2
                pos = a.position.to_numpy()
            vectors.append((row, d, pos))
        seed = (
            original_design["bootstrap_seed"]
            if primary_column == "primary_family"
            else js("configs/phase10.json")["bootstrap_seed"]
        )
        stat_vectors.append((primary, adjusted_column, vectors, seed))

    interval_metrics = csv("results/phase11/all_methods_metrics.csv")
    adaptive_primary = csv("results/phase11/score_comparisons.csv")
    adaptive_primary = adaptive_primary[adaptive_primary.primary]
    diagnostics = csv("results/phase11/diagnostics.csv")
    vectors, nesting, bound_pairs, interval_rows = [], {}, 0, 0
    fixed = {}
    for path in sorted((root / "results/calibration/records").glob("*.csv.gz")):
        record = csv(path)
        fixed[path.name] = record
    for path in sorted((root / "results/phase11/records").glob("*.csv.gz")):
        record = csv(path)
        # The suffix uniquely identifies the variant; source forecasts match by date.
        stem, variant_suffix = path.name.split("_aci_")
        variant = "aci_" + variant_suffix.removesuffix(".csv.gz")
        reference = fixed[stem + "_trailing.csv.gz"]
        assert record.date.equals(reference.date) and record.eligible.equals(reference.eligible)
        assert record.target_date.equals(reference.target_date)
        part = record[record.eligible].copy()
        ref = reference[reference.eligible]
        assert (part.residual_cutoff_date < part.date).all()
        # Resolve identifiers from the source filename, without inventing population metadata.
        sample = diagnostics[diagnostics.variant.eq(variant)]
        match = [
            r
            for r in sample.itertuples()
            if stem
            == f"{r.dataset.lower().replace('&', 'and').replace(' ', '_')}_h{r.horizon}_{r.model}"
        ]
        assert len(match) == 1
        row = match[0]
        y = ref.y_true.to_numpy()
        for level in [80, 90]:
            lo, hi = part[f"lower_{level}"].to_numpy(), part[f"upper_{level}"].to_numpy()
            assert np.all(lo <= hi) and np.isfinite(np.c_[lo, hi]).all()
            coverage = ((lo <= y) & (y <= hi)).mean()
            score = (
                hi - lo + 2 / (1 - level / 100) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
            )
            expected = interval_metrics[
                interval_metrics.dataset.eq(row.dataset)
                & interval_metrics.horizon.eq(row.horizon)
                & interval_metrics.model.eq(row.model)
                & interval_metrics.method.eq(variant)
                & interval_metrics.slice.eq("overall")
                & interval_metrics.nominal.eq(level / 100)
            ].iloc[0]
            np.testing.assert_allclose(
                [coverage, np.mean(hi - lo), score.mean()],
                [expected.coverage, expected.mean_width, expected.interval_score],
                rtol=1e-10,
                atol=1e-13,
            )
            interval_rows += 1
            bound_pairs += len(part)
            if variant == "aci_w504_g001":
                lower, upper = ref[f"lower_{level}"].to_numpy(), ref[f"upper_{level}"].to_numpy()
                reference_score = (
                    upper
                    - lower
                    + 2 / (1 - level / 100) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
                )
                statrow = adaptive_primary[
                    adaptive_primary.dataset.eq(row.dataset)
                    & adaptive_primary.horizon.eq(row.horizon)
                    & adaptive_primary.model.eq(row.model)
                    & adaptive_primary.nominal.eq(level / 100)
                ].iloc[0]
                vectors.append((statrow, score - reference_score, part.origin_position.to_numpy()))
        crossings = int(((part.lower_80 < part.lower_90) | (part.upper_80 > part.upper_90)).sum())
        assert crossings == row.nesting_violations and len(part) == row.eligible
        for level in [80, 90]:
            counter = record[f"clipped_{level}"].to_numpy()
            assert np.all(np.diff(counter) >= 0)
            assert int(counter[-1]) == getattr(row, f"clipped_{level}")
        nesting[variant] = nesting.get(variant, 0) + crossings
    # Match vector order to the saved family order, not filesystem order.
    vectors.sort(key=lambda item: int(item[0].name))
    design = js("configs/phase11.json")
    stat_vectors.append((adaptive_primary, "holm_p", vectors, design["bootstrap_seed"]))
    tested = 0
    for table, adjusted_column, vectors, seed in stat_vectors:
        assert len(vectors) == len(table)
        pvalues = []
        for row, d, pos in vectors:
            mean, se, p, bounds = inference(d, pos, seed=seed)
            np.testing.assert_allclose(
                [mean, se, p, *bounds],
                [
                    row.mean_loss_difference,
                    row.hac_standard_error,
                    row.p_value,
                    row.ci_lower,
                    row.ci_upper,
                ],
                rtol=1e-8,
                atol=1e-12,
            )
            pvalues.append(p)
            tested += 1
        np.testing.assert_allclose(holm(pvalues), table[adjusted_column], rtol=1e-8, atol=1e-12)
    assert tested == 108 and bound_pairs == 2780000
    return {
        "status": "passed",
        "scope": "computational review of existing evidence",
        "independent_dataset_validation": False,
        "external_human_peer_review": False,
        "prediction_rows_raw_label_checked": len(native) + 150 * 504,
        "native_model_forecasts": len(native),
        "fold_boundaries_and_frozen_means": len(folds),
        "paired_native_cells": paired_groups,
        "metric_rows_recomputed": metric_rows,
        "multiseed_tasks": len(reports),
        "primary_inference_and_bootstrap_rows": tested,
        "separate_holm_family_sizes": [len(t[0]) for t in stat_vectors],
        "adaptive_overall_metric_rows": interval_rows,
        "adaptive_bound_pairs": bound_pairs,
        "nesting_by_variant": nesting,
        "inputs": inputs,
        "limits": [
            "Shared raw CSV format loader reused; arithmetic independently implemented.",
            "Fixed/regime interval metrics and quantile/state histories use existing audits.",
            "No new fits, weights, data acquisition or human independence established.",
            "Approximate inference is reproducible, not proven valid under arbitrary dependence.",
        ],
    }
