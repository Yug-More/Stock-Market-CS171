"""Materialized adaptive intervals, paired proper scores, and independent feedback audits."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.evaluation.adaptive_calibration import adaptive_intervals
from marketpulse.evaluation.calibration import interval_metrics, interval_score
from marketpulse.evaluation.reliability import compare_losses, holm_adjust, moving_block_means
from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.research_audit import ScoreTree, audit_metrics
from marketpulse.experiments.robustness_data import inventory, load_predictions, write_json
from marketpulse.experiments.walkforward import _slug


def attach(source: pd.DataFrame, record: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "y_true", "y_pred", "regime", "stress", "volatility"]
    return record.merge(source[columns], on="date", validate="one_to_one")


def audit_adaptive(source: pd.DataFrame, record: pd.DataFrame, variant: dict, design: dict) -> int:
    """Second algorithm: endpoint searches, count-tree ranks and saved past interval feedback."""
    source = source.sort_values("date").reset_index(drop=True)
    assert source.date.tolist() == record.date.tolist()
    assert source.target_date.tolist() == record.target_date.tolist()
    np.testing.assert_array_equal(source.origin_position, record.origin_position)
    dates, ends = source.date.to_numpy(), source.target_date.to_numpy()
    stops = np.searchsorted(ends, dates, side="left")
    firsts = np.maximum(0, stops - variant["window"])
    counts = stops - firsts
    eligible = counts >= design["minimum"]
    np.testing.assert_array_equal(record.calibration_n, counts)
    np.testing.assert_array_equal(record.eligible, eligible)
    present = stops > 0
    assert record.loc[present, "residual_cutoff_date"].tolist() == ends[stops[present] - 1].tolist()
    actual, prediction = source.y_true.to_numpy(), source.y_pred.to_numpy()
    tree = ScoreTree(np.abs(actual - prediction))
    added = removed = feedback = checked = 0
    levels = design["levels"]
    states = {level: 1 - level for level in levels}
    clipped = dict.fromkeys(levels, 0)
    expected_feedback = 0
    expected_alphas = np.empty((len(source), len(levels)))
    expected_clips = np.empty((len(source), len(levels)), int)
    feedback_counts = np.empty(len(source), int)
    saved_lower = record[[f"lower_{round(100 * p)}" for p in levels]].to_numpy()
    saved_upper = record[[f"upper_{round(100 * p)}" for p in levels]].to_numpy()
    lower = np.full((len(source), len(levels)), np.nan)
    upper = lower.copy()
    for i in range(len(source)):
        while added < stops[i]:
            tree.update(added, 1)
            added += 1
        while removed < firsts[i]:
            tree.update(removed, -1)
            removed += 1
        while feedback < stops[i]:
            if eligible[feedback]:
                expected_feedback += 1
                for j, level in enumerate(levels):
                    suffix = round(100 * level)
                    miss = int(
                        actual[feedback] < saved_lower[feedback, j]
                        or actual[feedback] > saved_upper[feedback, j]
                    )
                    candidate = states[level] + variant["gamma"] * ((1 - level) - miss)
                    low, high = design["alpha_bounds"]
                    clipped[level] += int(candidate < low or candidate > high)
                    states[level] = max(low, min(high, candidate))
            feedback += 1
        feedback_counts[i] = expected_feedback
        for j, level in enumerate(levels):
            suffix = round(100 * level)
            expected_alphas[i, j] = states[level]
            expected_clips[i, j] = clipped[level]
            if eligible[i]:
                rank = math.ceil((int(counts[i]) + 1) * (1 - states[level]))
                q = tree.quantile(rank)
                lower[i, j], upper[i, j] = prediction[i] - q, prediction[i] + q
                checked += 1
    np.testing.assert_array_equal(record.feedback_n, feedback_counts)
    for j, level in enumerate(levels):
        suffix = round(100 * level)
        np.testing.assert_allclose(record[f"alpha_{suffix}"], expected_alphas[:, j], atol=1e-13)
        np.testing.assert_array_equal(record[f"clipped_{suffix}"], expected_clips[:, j])
        np.testing.assert_allclose(
            record[[f"lower_{suffix}", f"upper_{suffix}"]],
            np.c_[lower[:, j], upper[:, j]],
            rtol=1e-10,
            atol=1e-13,
            equal_nan=True,
        )
    return checked


def reference_records(root: Path, dataset: str, horizon: int, model: str) -> pd.DataFrame:
    name = f"{_slug(dataset)}_h{horizon}_{model}_trailing.csv.gz"
    return pd.read_csv(root / "results/calibration/records" / name)


def run(root: Path) -> None:
    output = root / "results/phase11"
    if output.exists():
        raise ValueError("preserve previous adaptive experiment; output must be new")
    output.mkdir(parents=True)
    (output / "records").mkdir()
    started = time.monotonic()
    design = json.loads((root / "configs/phase11.json").read_text())
    source = load_predictions(root)
    metrics, comparisons, coverage, files = [], [], [], []
    for (dataset, horizon, model), part in source.groupby(["dataset", "horizon", "model"]):
        part = part.sort_values("date").reset_index(drop=True)
        baseline = reference_records(root, dataset, horizon, model)
        assert part.date.tolist() == baseline.date.tolist()
        for variant in design["variants"]:
            name = variant["name"]
            print(f"adaptive {dataset} h{horizon} {model} {name}", flush=True)
            record = adaptive_intervals(
                part,
                window=variant["window"],
                gamma=variant["gamma"],
                minimum=design["minimum"],
                levels=tuple(design["levels"]),
                alpha_bounds=tuple(design["alpha_bounds"]),
            )
            np.testing.assert_array_equal(record.eligible, baseline.eligible)
            path = output / "records" / f"{_slug(dataset)}_h{horizon}_{model}_{name}.csv.gz"
            record.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            files.append(
                {
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "model": model,
                    "variant": name,
                    "path": str(path.relative_to(output)),
                    "sha256": file_hash(path),
                    "rows": len(record),
                }
            )
            complete = attach(part, record)
            slices = {"overall": complete}
            slices.update(
                {
                    f"volatility_{r}": complete[complete.regime == r]
                    for r in ["low", "medium", "high"]
                }
            )
            slices.update(
                {
                    p: complete[complete.stress == p]
                    for p in ["dot_com", "financial_crisis", "covid_crash"]
                }
            )
            for label, block in slices.items():
                for level in design["levels"]:
                    metrics.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model": model,
                            "method": name,
                            "slice": label,
                            **interval_metrics(block, level),
                        }
                    )
            selected = complete[complete.eligible]
            ref = baseline[baseline.eligible]
            positions = selected.origin_position.to_numpy()
            for level in design["levels"]:
                suffix = round(100 * level)
                a = interval_score(
                    selected.y_true, selected[f"lower_{suffix}"], selected[f"upper_{suffix}"], level
                )
                b = interval_score(
                    ref.y_true, ref[f"lower_{suffix}"], ref[f"upper_{suffix}"], level
                )
                for length in (
                    design["block_lengths"] if name == design["primary_variant"] else [20]
                ):
                    comparison = compare_losses(
                        a,
                        b,
                        positions,
                        length=length,
                        replicates=design["bootstrap_replicates"],
                        seed=design["bootstrap_seed"],
                    )
                    comparisons.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model": model,
                            "variant": name,
                            "nominal": level,
                            "primary": name == design["primary_variant"] and length == 20,
                            **comparison,
                        }
                    )
                covered = (
                    (selected.y_true >= selected[f"lower_{suffix}"])
                    & (selected.y_true <= selected[f"upper_{suffix}"])
                ).to_numpy(float)
                boot = moving_block_means(
                    covered[:, None],
                    positions,
                    20,
                    design["bootstrap_replicates"],
                    design["bootstrap_seed"],
                )
                bounds = np.quantile(boot[:, 0], [0.025, 0.975])
                coverage.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model,
                        "variant": name,
                        "nominal": level,
                        "coverage": float(covered.mean()),
                        "ci_lower": bounds[0],
                        "ci_upper": bounds[1],
                        "n": len(covered),
                        "block_length": 20,
                    }
                )
    table = pd.DataFrame(comparisons)
    table["holm_p"] = np.nan
    primary = table.primary
    assert primary.sum() == 60
    table.loc[primary, "holm_p"] = holm_adjust(table.loc[primary, "p_value"].to_numpy())
    table.to_csv(output / "score_comparisons.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage_uncertainty.csv", index=False)
    adaptive = pd.DataFrame(metrics)
    adaptive.to_csv(output / "adaptive_metrics.csv", index=False)
    old = pd.read_csv(root / "results/calibration/metrics.csv")
    pd.concat([old, adaptive], ignore_index=True).to_csv(
        output / "all_methods_metrics.csv", index=False
    )
    write_json(
        output / "manifest.json",
        {
            "configuration": design,
            "files": files,
            "environment": environment_record(root, "adaptive"),
            "elapsed_seconds": time.monotonic() - started,
            "inputs": {
                p: file_hash(root / p)
                for p in [
                    "configs/phase11.json",
                    "src/marketpulse/evaluation/adaptive_calibration.py",
                    "src/marketpulse/experiments/adaptive_study.py",
                    "results/walkforward/full/predictions.csv.gz",
                    "results/calibration/manifest.json",
                ]
            },
            "coverage_guarantee": "none",
            "classification": "retrospective sensitivity",
        },
    )
    write_outputs(root)
    verify(root, seal=True)


def write_outputs(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "results/phase11"
    directory = output / "figures"
    directory.mkdir()
    metrics = pd.read_csv(output / "all_methods_metrics.csv")
    for label, slice_name in [("overall", "overall"), ("high_volatility", "volatility_high")]:
        rows = metrics[
            (metrics.model == "zero_return")
            & metrics.nominal.eq(0.9)
            & metrics.slice.eq(slice_name)
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, ((dataset, horizon), part) in zip(
            axes.flat, rows.groupby(["dataset", "horizon"]), strict=True
        ):
            for row in part.itertuples():
                ax.scatter(row.mean_width, row.coverage, label=row.method, s=25)
            ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"{dataset} h{horizon}")
            ax.set_xlabel("Mean interval width (log return)")
            ax.set_ylabel("Empirical coverage")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"Zero-return 90% intervals: {label}; retrospective, matched eligibility")
        fig.tight_layout(rect=(0, 0.10, 1, 0.95))
        fig.savefig(directory / f"{label}_tradeoffs.png", dpi=220)
        plt.close(fig)
    path = root / "docs/PHASE11_ADAPTIVE_UNCERTAINTY.md"
    text = path.read_text().split("\n## Executed findings")[0]
    manifest = json.loads((output / "manifest.json").read_text())
    total = sum(f["rows"] for f in manifest["files"])
    text += (
        "\n## Executed findings\n\n"
        f"Generation/comparisons took {manifest['elapsed_seconds']:.2f} seconds "
        f"before independent auditing; {total:,} saved records in "
        f"{len(manifest['files'])} files.\n\n"
    )
    chosen = metrics[
        (metrics.model == "zero_return") & metrics.nominal.eq(0.9) & metrics.slice.eq("overall")
    ]
    for row in chosen.itertuples():
        text += (
            f"- {row.dataset} h{row.horizon} {row.method}: "
            f"coverage {row.coverage:.4f}, width {row.mean_width:.5f}, "
            f"proper score {row.interval_score:.5f}, n={row.n:.0f}.\n"
        )
    text += """
All models, levels, regimes and stress slices are in `all_methods_metrics.csv`;
unavailable stress episodes retain n=0. Marginal bootstrap coverage CIs are in
`coverage_uncertainty.csv`. Proper-score effects, marginal CIs and the fixed
60-test Holm family are in `score_comparisons.csv`. No method is independently
validated.

![Overall trade-offs](../results/phase11/figures/overall_tradeoffs.png)

![High-volatility trade-offs](../results/phase11/figures/high_volatility_tradeoffs.png)
"""
    path.write_text(text)


def explicit_bootstrap_bounds(values, positions, length, replicates, seed):
    """Independent convolution block sums, including a shortened final block."""
    n = len(values)
    starts = np.array(
        [i for i in range(n - length + 1) if positions[i + length - 1] - positions[i] == length - 1]
    )
    sums = np.convolve(values, np.ones(length), mode="valid")[starts]
    rng = np.random.default_rng(seed)
    count, remainder = divmod(n, length)
    draws = rng.integers(0, len(starts), size=(replicates, count))
    means = sums[draws].sum(axis=1)
    if remainder:
        short = np.array(
            [
                i
                for i in range(n - remainder + 1)
                if positions[i + remainder - 1] - positions[i] == remainder - 1
            ]
        )
        picked = rng.choice(short, size=replicates)
        means += np.convolve(values, np.ones(remainder), mode="valid")[picked]
    return np.quantile(means / n, [0.025, 0.975])


def verify(root: Path, *, seal: bool = False) -> None:
    output = root / "results/phase11"
    manifest = json.loads((output / "manifest.json").read_text())
    design = manifest["configuration"]
    for path, sha in manifest["inputs"].items():
        assert file_hash(root / path) == sha, path
    source = load_predictions(root)
    metrics = pd.read_csv(output / "adaptive_metrics.csv")
    checked = records_count = 0
    coverage_table = pd.read_csv(output / "coverage_uncertainty.csv")
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert file_hash(path) == entry["sha256"]
        record = pd.read_csv(path)
        part = (
            source[
                (source.dataset == entry["dataset"])
                & (source.horizon == entry["horizon"])
                & (source.model == entry["model"])
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )
        variant = next(v for v in design["variants"] if v["name"] == entry["variant"])
        checked += audit_adaptive(part, record, variant, design)
        records_count += len(record)
        complete = attach(part, record)
        rows = metrics[
            (metrics.dataset == entry["dataset"])
            & (metrics.horizon == entry["horizon"])
            & (metrics.model == entry["model"])
            & (metrics.method == entry["variant"])
        ]
        for _, row in rows.iterrows():
            audit_metrics(complete, row)
    table = pd.read_csv(output / "score_comparisons.csv")
    primary = table[table.primary].sort_values("p_value")
    assert len(primary) == 60
    adjusted = np.minimum(
        1, np.maximum.accumulate(primary.p_value.to_numpy() * np.arange(60, 0, -1))
    )
    np.testing.assert_allclose(primary.holm_p, adjusted, atol=1e-13)
    # Reconstruct score vectors independently and audit every paired point/HAC result.
    for entry in manifest["files"]:
        record = pd.read_csv(output / entry["path"])
        part = (
            source[
                (source.dataset == entry["dataset"])
                & (source.horizon == entry["horizon"])
                & (source.model == entry["model"])
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )
        complete = attach(part, record)
        selected = complete[complete.eligible]
        baseline = reference_records(root, entry["dataset"], entry["horizon"], entry["model"])
        reference = baseline[baseline.eligible]
        assert selected.date.tolist() == reference.date.tolist()
        for level in design["levels"]:
            suffix = round(100 * level)
            covered = (
                (selected.y_true >= selected[f"lower_{suffix}"])
                & (selected.y_true <= selected[f"upper_{suffix}"])
            ).to_numpy(float)
            cell = coverage_table[
                (coverage_table.dataset == entry["dataset"])
                & (coverage_table.horizon == entry["horizon"])
                & (coverage_table.model == entry["model"])
                & (coverage_table.variant == entry["variant"])
                & (coverage_table.nominal == level)
            ].iloc[0]
            bounds = explicit_bootstrap_bounds(
                covered,
                selected.origin_position.to_numpy(),
                20,
                design["bootstrap_replicates"],
                design["bootstrap_seed"],
            )
            np.testing.assert_allclose([cell.ci_lower, cell.ci_upper], bounds, atol=1e-12)
            np.testing.assert_allclose(cell.coverage, covered.mean(), atol=1e-13)
            assert cell.n == len(covered)
        rows = table[
            (table.dataset == entry["dataset"])
            & (table.horizon == entry["horizon"])
            & (table.model == entry["model"])
            & (table.variant == entry["variant"])
        ]
        for row in rows.itertuples():
            suffix = round(100 * row.nominal)

            def scores(block, suffix=suffix, nominal=row.nominal):
                actual = block.y_true.to_numpy()
                lo, hi = block[f"lower_{suffix}"].to_numpy(), block[f"upper_{suffix}"].to_numpy()
                return (
                    hi
                    - lo
                    + 2 / (1 - nominal) * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))
                )

            d = scores(selected) - scores(reference)
            np.testing.assert_allclose(row.mean_loss_difference, d.mean(), atol=1e-13)
            if row.primary:
                bounds = explicit_bootstrap_bounds(
                    d,
                    selected.origin_position.to_numpy(),
                    20,
                    design["bootstrap_replicates"],
                    design["bootstrap_seed"],
                )
                np.testing.assert_allclose([row.ci_lower, row.ci_upper], bounds, atol=1e-12)
            n = len(d)
            centered = d - d.mean()
            variance = np.dot(centered, centered) / n
            positions = selected.origin_position.to_numpy()
            segments = np.r_[0, np.cumsum(np.diff(positions) != 1)]
            for lag in range(1, int(row.block_length)):
                mask = segments[lag:] == segments[:-lag]
                variance += (
                    2
                    * (1 - lag / row.block_length)
                    * np.sum(centered[lag:][mask] * centered[:-lag][mask])
                    / n
                )
            se = math.sqrt(max(variance / n, 0))
            np.testing.assert_allclose(row.hac_standard_error, se, rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(
                row.p_value, math.erfc(abs(d.mean() / se) / math.sqrt(2)), rtol=1e-8, atol=1e-13
            )
    result = {
        "status": "passed",
        "records": records_count,
        "independent_bounds": checked,
        "metric_rows": len(metrics),
        "paired_score_rows": len(table),
        "primary_family": 60,
        "strict_endpoint_feedback": True,
        "distribution_free_guarantee": False,
    }
    if seal:
        write_json(output / "audit.json", result)
        write_json(output / "checksums.json", inventory(output))
    else:
        assert json.loads((output / "checksums.json").read_text()) == inventory(output)
    print(json.dumps(result))
