"""Matched, seed-averaged loss inference and independent Phase 10 evidence audits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.evaluation.reliability import compare_losses, holm_adjust
from marketpulse.experiments.benchmark import environment_record, load_benchmark_settings
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import (
    enrich_predictions,
    inventory,
    load_predictions,
    matched_membership,
    select_matched,
    write_json,
)
from marketpulse.experiments.summary import pooled_metrics


def matched_seed_losses(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average losses, never predictions; reject even a one-day endpoint mismatch."""
    if not frames or any(frame.empty for frame in frames):
        raise ValueError("nonempty seed populations required")
    reference = (
        frames[0].sort_values(["dataset", "horizon", "model", "date"]).reset_index(drop=True)
    )
    keys = ["dataset", "horizon", "model", "date", "target_date", "origin_position", "y_true"]
    if reference.duplicated(["dataset", "horizon", "model", "date"]).any():
        raise ValueError("duplicate seed forecast origins")
    losses = []
    for frame in frames:
        frame = frame.sort_values(["dataset", "horizon", "model", "date"]).reset_index(drop=True)
        if not frame[keys].equals(reference[keys]):
            raise ValueError("seed populations or target endpoints differ")
        losses.append((frame.y_true.to_numpy() - frame.y_pred.to_numpy()) ** 2)
    result = reference.drop(columns=["y_pred", "seed"], errors="ignore").copy()
    result["mean_seed_loss"] = np.mean(losses, axis=0)
    return result


def load_expanded(root: Path) -> tuple[list[pd.DataFrame], list[dict]]:
    native = load_predictions(root)
    reports = []
    for directory, expected in [("robustness", 90), ("phase10", 60)]:
        base = root / "results" / directory / "multiseed"
        manifest = json.loads((base / "manifest.json").read_text())
        paths = sorted((base / "runs").glob("*.json"))
        assert len(paths) == manifest["expected_tasks"] == expected
        for relative, sha in manifest["identity"]["inputs"].items():
            assert file_hash(root / relative) == sha, relative
        for path in paths:
            report = json.loads(path.read_text())
            assert file_hash(path.with_suffix(".csv.gz")) == report["predictions_sha256"]
            reports.append({"path": path, "report": report})
    frames = []
    for seed in [171, 271, 371, 471, 571]:
        pieces = []
        for item in reports:
            report = item["report"]
            if report["seed"] != seed:
                continue
            part = pd.read_csv(item["path"].with_suffix(".csv.gz"))
            old = native[
                (native.dataset == report["dataset"])
                & (native.horizon == report["horizon"])
                & (native.fold == report["fold"])
            ]
            for model in ["random_forest", "lstm"]:
                a = part[part.model == model].sort_values("date")
                b = old[old.model == model].sort_values("date")
                assert a.date.tolist() == b.date.tolist()
                np.testing.assert_allclose(a.y_true, b.y_true, rtol=1e-12)
                assert len(a) == 252
            baseline = old[
                old.model.isin(["zero_return", "historical_mean", "linear_regression"])
            ].copy()
            baseline["seed"] = seed
            columns = list(part.columns)
            pieces.extend([part, baseline[columns]])
        frame = enrich_predictions(root, pd.concat(pieces, ignore_index=True))
        assert len(frame) == 30 * 252 * 5
        frames.append(frame)
    return frames, reports


def comparisons(losses: pd.DataFrame, design: dict) -> pd.DataFrame:
    rows = []
    for (dataset, horizon), frame in losses.groupby(["dataset", "horizon"]):
        for first, second in design["primary_pairs"]:
            a = frame[frame.model == first].sort_values("origin_position").reset_index(drop=True)
            b = frame[frame.model == second].sort_values("origin_position").reset_index(drop=True)
            assert a[["date", "target_date", "y_true"]].equals(b[["date", "target_date", "y_true"]])
            slices = [("overall", np.ones(len(a), bool))]
            slices += [
                (str(regime), a.regime.eq(regime).to_numpy())
                for regime in ["low", "medium", "high"]
            ]
            slices += [
                (f"fold_{int(fold)}", a.fold.eq(fold).to_numpy())
                for fold in sorted(a.fold.unique())
            ]
            for slice_name, mask in slices:
                for length in design["block_lengths"] if slice_name == "overall" else [20]:
                    result = compare_losses(
                        a.mean_seed_loss.to_numpy(),
                        b.mean_seed_loss.to_numpy(),
                        a.origin_position.to_numpy(),
                        length=length,
                        replicates=design["bootstrap_replicates"],
                        seed=design["bootstrap_seed"],
                        mask=mask,
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "first": first,
                            "second": second,
                            "slice": slice_name,
                            **result,
                        }
                    )
    table = pd.DataFrame(rows)
    primary = table.slice.eq("overall") & table.block_length.eq(20)
    assert primary.sum() == 30
    table["primary"] = primary
    table["holm_p"] = np.nan
    table.loc[primary, "holm_p"] = holm_adjust(table.loc[primary, "p_value"].to_numpy())
    return table


def analyze(root: Path) -> None:
    output = root / "results/phase10"
    if (output / "checksums.json").exists():
        raise ValueError("preserve published analysis; choose a new version")
    design = json.loads((root / "configs/phase10.json").read_text())
    frames, reports = load_expanded(root)
    metrics, folds, regimes, calendars, calendar_counts = [], [], [], [], []
    for seed, frame in zip(design["seeds"], frames, strict=True):
        membership = matched_membership(frame)
        for horizon, dates in membership.groupby("horizon"):
            calendar_counts.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "shared_origins": int(dates.origin_match.sum()),
                    "shared_endpoints": int(dates.endpoint_match.sum()),
                }
            )
        matched = select_matched(frame, membership, endpoints=True)
        if not matched.empty:
            calendar_metrics = pooled_metrics(matched, ("dataset", "horizon"))
            calendar_metrics["seed"] = seed
            calendars.append(calendar_metrics)
        for group, target in [
            (("dataset", "horizon"), metrics),
            (("dataset", "horizon", "fold"), folds),
            (("dataset", "horizon", "regime"), regimes),
        ]:
            score = pooled_metrics(frame, group)
            score["seed"] = seed
            target.append(score)
    for name, rows in [
        ("seed_metrics", metrics),
        ("fold_metrics", folds),
        ("regime_metrics", regimes),
    ]:
        pd.concat(rows, ignore_index=True).to_csv(output / f"{name}.csv", index=False)
    common = (
        pd.concat(calendars, ignore_index=True)
        if calendars
        else pd.DataFrame(columns=["dataset", "horizon", "model", "seed", "n", "rmse"])
    )
    common.to_csv(output / "common_calendar_metrics.csv", index=False)
    pd.DataFrame(calendar_counts).to_csv(output / "common_calendar_counts.csv", index=False)
    losses = matched_seed_losses(frames)
    losses.to_csv(
        output / "matched_losses.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    comparisons(losses, design).to_csv(output / "comparisons.csv", index=False)
    replication = []
    native = load_predictions(root)
    for item in reports:
        report = item["report"]
        if report["seed"] != 171 or "phase10" not in str(item["path"]):
            continue
        fresh = pd.read_csv(item["path"].with_suffix(".csv.gz"))
        for model in ["random_forest", "lstm"]:
            a = fresh[fresh.model == model].sort_values("date")
            b = native[
                (native.dataset == report["dataset"])
                & (native.horizon == report["horizon"])
                & (native.fold == report["fold"])
                & (native.model == model)
            ].sort_values("date")
            replication.append(
                {
                    "dataset": report["dataset"],
                    "horizon": report["horizon"],
                    "fold": report["fold"],
                    "model": model,
                    "max_abs_difference": float(
                        np.max(np.abs(a.y_pred.to_numpy() - b.y_pred.to_numpy()))
                    ),
                }
            )
    pd.DataFrame(replication).to_csv(output / "seed171_replication.csv", index=False)
    settings = load_benchmark_settings(root)
    new_reports = [x["report"] for x in reports if "phase10" in str(x["path"])]
    write_json(
        output / "analysis_manifest.json",
        {
            "environment": environment_record(root, "full"),
            "config_sha256": file_hash(root / "configs/phase10.json"),
            "new_tasks": len(new_reports),
            "new_training_seconds": sum(r["execution"]["elapsed_seconds"] for r in new_reports),
            "settings": settings.__dict__,
            "combined_folds": 30,
            "temporal_origins_per_cell": 1260,
            "new_stochastic_predictions": 60 * 252 * 2,
            "source_sha256": file_hash(Path(__file__)),
            "historical_inputs": {
                str(p.relative_to(root)): file_hash(p)
                for p in [
                    root / "results/walkforward/full/predictions.csv.gz",
                    root / "results/robustness/multiseed/manifest.json",
                ]
            },
        },
    )
    figures(root)
    report(root)
    verify(root, seal=True)


def figures(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "results/phase10"
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    metrics = pd.read_csv(output / "seed_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, ((dataset, horizon), part) in zip(
        axes.flat, metrics.groupby(["dataset", "horizon"]), strict=True
    ):
        for i, (_model, values) in enumerate(part.groupby("model")):
            ax.scatter(np.arange(len(values)) * 0.05 + i, values.rmse, s=14)
        ax.set_xticks(range(5), sorted(part.model.unique()), rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{dataset}, h={horizon}; 1,260 origins")
        ax.set_ylabel("RMSE (log return)")
    fig.suptitle("Expanded retrospective sensitivity: five seeds, five separated folds/cell")
    fig.tight_layout()
    fig.savefig(directory / "seed_rmse.png", dpi=220)
    plt.close(fig)
    table = pd.read_csv(output / "comparisons.csv")
    part = table[table.primary & table.second.eq("zero_return")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in part.iterrows():
        ax.plot([row.ci_lower, row.ci_upper], [i, i], color="steelblue")
        ax.scatter(row.mean_loss_difference, i, color="navy", s=20)
    ax.set_yticks(
        range(len(part)), [f"{r.dataset} h{r.horizon}: {r.first}" for r in part.itertuples()]
    )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Mean seed MSE − zero MSE; marginal 95% moving-block CI (L=20)")
    ax.set_title("Negative favors ML; retrospective, not independent confirmation")
    fig.tight_layout()
    fig.savefig(directory / "paired_effects.png", dpi=220)
    plt.close(fig)


def report(root: Path) -> None:
    output = root / "results/phase10"
    metrics = pd.read_csv(output / "seed_metrics.csv")
    table = pd.read_csv(output / "comparisons.csv")
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    path = root / "docs/PHASE10_ROBUSTNESS.md"
    text = path.read_text().split("\n## Executed findings")[0]
    minutes = manifest["new_training_seconds"] / 60
    text += (
        f"\n## Executed findings\n\nAll 60 new full-budget tasks completed in {minutes:.2f} "
        "summed training minutes: 120 stochastic model-fold evaluations and 30,240 new "
        "predictions. Combined coverage is 150 tasks on 30 distinct folds, five seeds "
        "and 1,260 temporal origins per cell. Deterministic forecasts are reused.\n\n"
    )
    for (dataset, horizon), part in metrics.groupby(["dataset", "horizon"]):
        for model in ["random_forest", "lstm"]:
            cell = part[part.model == model]
            text += (
                f"- {dataset} h{horizon}, {model}: "
                f"RMSE {cell.rmse.min():.6f}–{cell.rmse.max():.6f}, "
                f"zero-baseline skill {cell.skill_vs_zero.min():.4f}–"
                f"{cell.skill_vs_zero.max():.4f}, "
                f"ranks {cell['rank'].min():.0f}–{cell['rank'].max():.0f}.\n"
            )
    primary = table[table.primary]
    text += (
        "\nPrimary effects use seed-averaged MSE and 1,260 observations, not 6,300 seed "
        "rows. All 30 results, marginal CIs, HAC p-values and Holm values are below. "
        "Regime and individual-fold results are exploratory.\n\n"
    )
    for row in primary.itertuples():
        text += (
            f"- {row.dataset} h{row.horizon}: {row.first} vs {row.second}: "
            f"ΔMSE {row.mean_loss_difference:.7g}, "
            f"CI [{row.ci_lower:.7g}, {row.ci_upper:.7g}], Holm p={row.holm_p:.5g}.\n"
        )
    text += """
## Interpretation and reproduction

The original 282-fold inference remains separate. Expanding this subset does not
revise its sample size or eliminate selection bias. Rank reversals across these
populations show period sensitivity, not a reason to retune. Loss averaging
conditions on five fixed seeds, not all possible optimizer realizations. Regime
estimates retain the full timeline; their selected days are not independent.
Training histories, index construction and source quality remain unequal.
`common_calendar_metrics.csv` additionally restricts the expanded sample to the
intersection of both indexes' origins AND actual target endpoints. There are zero
shared origins/endpoints in this five-fold-per-cell selection
(as recorded in `common_calendar_counts.csv`), so that metric file is header-only:
no cross-index matched seed comparison is available. The earlier full-history
matched analysis remains available and does not equalize training history.

Run `python scripts/phase10.py train` (checksum-checked resume),
`python scripts/phase10.py analyze` (unsealed output only), and
`python scripts/phase10.py verify`. Seed-171 reruns check 24 newly selected
model/fold pairs against original predictions. Earlier evidence is immutable.

![Seed sensitivity](../results/phase10/figures/seed_rmse.png)

![Paired effects](../results/phase10/figures/paired_effects.png)
"""
    path.write_text(text)


def verify(root: Path, *, seal: bool = False) -> None:
    """Independent arithmetic on saved rows, actual endpoints, settings and HAC."""
    from dataclasses import asdict

    output = root / "results/phase10"
    frames, reports = load_expanded(root)
    settings = json.loads(json.dumps(asdict(load_benchmark_settings(root))))
    for item in reports:
        payload = item["report"]
        assert payload["execution"]["settings"] == settings
        from marketpulse.experiments.walkforward import _slug

        slug = _slug(payload["dataset"])
        historical = root / "results/walkforward/full/folds" / slug
        historical = historical / f"h{payload['horizon']}" / f"fold_{payload['fold']:03d}.json"
        assert payload["sample"] == json.loads(historical.read_text())["sample"]
        predictions = pd.read_csv(item["path"].with_suffix(".csv.gz"))
        for model in payload["models"]:
            details = model["details"]
            is_rf = model["model"] == "random_forest"
            key = "depth" if is_rf else "hidden_size"
            grid = details[f"validation_rmse_by_{key}"]
            assert sorted(map(int, grid)) == ([4, 8] if is_rf else [16, 32])
            selected = min(grid, key=lambda value: (grid[value], int(value)))
            chosen = details["selected_max_depth" if is_rf else "selected_hidden_size"]
            assert chosen == int(selected)
            assert details["seed"] == payload["seed"]
            part = predictions[predictions.model == model["model"]]
            error = part.y_true.to_numpy() - part.y_pred.to_numpy()
            np.testing.assert_allclose(model["test"]["rmse"], np.sqrt(np.mean(error**2)))
            assert details.get("n_estimators", 200) == 200
            assert details.get("max_epochs", 25) == 25
    common_counts = pd.read_csv(output / "common_calendar_counts.csv")
    for seed, frame in zip([171, 271, 371, 471, 571], frames, strict=True):
        membership = matched_membership(frame)
        for horizon, dates in membership.groupby("horizon"):
            row = common_counts[
                (common_counts.seed == seed) & (common_counts.horizon == horizon)
            ].iloc[0]
            assert row.shared_origins == dates.origin_match.sum() == 0
            assert row.shared_endpoints == dates.endpoint_match.sum() == 0
    assert pd.read_csv(output / "common_calendar_metrics.csv").empty
    metrics = pd.read_csv(output / "seed_metrics.csv")
    for seed, frame in zip([171, 271, 371, 471, 571], frames, strict=True):
        for (dataset, horizon, model), part in frame.groupby(["dataset", "horizon", "model"]):
            row = metrics[
                (metrics.seed == seed)
                & (metrics.dataset == dataset)
                & (metrics.horizon == horizon)
                & (metrics.model == model)
            ].iloc[0]
            zero = frame[
                (frame.dataset == dataset)
                & (frame.horizon == horizon)
                & (frame.model == "zero_return")
            ].sort_values("date")
            part = part.sort_values("date")
            squared = (part.y_true.to_numpy() - part.y_pred.to_numpy()) ** 2
            expected = [
                np.sqrt(squared.mean()),
                np.abs(part.y_true - part.y_pred).mean(),
                1 - squared.mean() / np.mean(zero.y_true.to_numpy() ** 2),
            ]
            np.testing.assert_allclose(
                row[["rmse", "mae", "skill_vs_zero"]].to_numpy(float),
                expected,
                rtol=1e-10,
                atol=1e-13,
            )
    for filename, groups in [
        ("fold_metrics.csv", ["dataset", "horizon", "fold"]),
        ("regime_metrics.csv", ["dataset", "horizon", "regime"]),
    ]:
        derived = pd.read_csv(output / filename)
        checked = 0
        for seed, frame in zip([171, 271, 371, 471, 571], frames, strict=True):
            for key, block in frame.groupby(groups):
                cell = derived[derived.seed == seed]
                for column, value in zip(groups, key, strict=True):
                    cell = cell[cell[column] == value]
                errors = {}
                for model, part in block.groupby("model"):
                    error = part.y_true.to_numpy() - part.y_pred.to_numpy()
                    errors[model] = np.sqrt(np.mean(error**2))
                for model, error in errors.items():
                    row = cell[cell.model == model].iloc[0]
                    assert row.n == len(block[block.model == model])
                    np.testing.assert_allclose(row.rmse, error, rtol=1e-11)
                    assert row["rank"] == 1 + sum(v < error for v in errors.values())
                    checked += 1
        assert checked == len(derived)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["config_sha256"] == file_hash(root / "configs/phase10.json")
    assert manifest["source_sha256"] == file_hash(Path(__file__))
    loss = matched_seed_losses(frames)
    saved = pd.read_csv(output / "matched_losses.csv.gz")
    np.testing.assert_allclose(saved.mean_seed_loss, loss.mean_seed_loss, rtol=1e-11, atol=1e-15)
    table = pd.read_csv(output / "comparisons.csv")
    for row in table.itertuples():
        part = loss[(loss.dataset == row.dataset) & (loss.horizon == row.horizon)]
        a = part[part.model == row.first].sort_values("origin_position")
        b = part[part.model == row.second].sort_values("origin_position")
        if row.slice == "overall":
            mask = np.ones(len(a), bool)
        elif row.slice.startswith("fold_"):
            mask = a.fold.eq(int(row.slice.split("_")[1])).to_numpy()
        else:
            mask = a.regime.eq(row.slice).to_numpy()
        d = a.mean_seed_loss.to_numpy() - b.mean_seed_loss.to_numpy()
        effect = d[mask].mean()
        assert row.n == mask.sum()
        np.testing.assert_allclose(row.mean_loss_difference, effect, atol=1e-15)
        if row.primary:
            # Independent explicit block arithmetic, unlike production prefix sums.
            positions = a.origin_position.to_numpy()
            starts = [i for i in range(len(d) - 19) if np.all(np.diff(positions[i : i + 20]) == 1)]
            block_means = np.array([np.mean(d[i : i + 20]) for i in starts])
            rng = np.random.default_rng(10171)
            draws = rng.integers(0, len(starts), size=(2000, len(d) // 20))
            bounds = np.quantile(block_means[draws].mean(axis=1), [0.025, 0.975])
            np.testing.assert_allclose([row.ci_lower, row.ci_upper], bounds, rtol=1e-8, atol=1e-14)
        if row.status != "approximate_inference":
            continue
        influence = mask * (d - effect) * len(d) / mask.sum()
        centered = influence - influence.mean()
        positions = a.origin_position.to_numpy()
        n = len(d)
        segments = np.r_[0, np.cumsum(np.diff(positions) != 1)]
        variance = np.dot(centered, centered) / n
        for lag in range(1, int(row.block_length)):
            pairs = segments[lag:] == segments[:-lag]
            covariance = (
                sum(float(centered[i] * centered[i - lag]) for i in range(lag, n) if pairs[i - lag])
                / n
            )
            variance += 2 * (1 - lag / row.block_length) * covariance
        se = math.sqrt(max(variance / n, 0))
        np.testing.assert_allclose(row.hac_standard_error, se, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(
            row.p_value, math.erfc(abs(effect / se) / math.sqrt(2)), rtol=1e-8, atol=1e-14
        )
    primary = table[table.primary].sort_values("p_value")
    adjusted = np.minimum(
        1, np.maximum.accumulate(primary.p_value.to_numpy() * np.arange(30, 0, -1))
    )
    np.testing.assert_allclose(primary.holm_p, adjusted, rtol=1e-8, atol=1e-14)
    result = {
        "status": "passed",
        "full_budget_tasks": 150,
        "new_tasks": 60,
        "metric_rows": len(metrics),
        "paired_rows": len(table),
        "primary_family": 30,
        "temporal_n_per_cell": 1260,
        "independent_confirmation": False,
    }
    if seal:
        write_json(output / "audit.json", result)
        write_json(output / "checksums.json", inventory(output))
    else:
        assert json.loads((output / "checksums.json").read_text()) == inventory(output)
    print(json.dumps(result))
