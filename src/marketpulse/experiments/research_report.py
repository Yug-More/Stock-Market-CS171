"""Generate measured research tables and interpretation from audited extension artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("unavailable")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(root: Path) -> None:
    matched = pd.read_csv(root / "results/robustness/matched/metrics.csv")
    counts = pd.read_csv(root / "results/robustness/matched/counts.csv")
    seeds = pd.read_csv(root / "results/robustness/seed_stability.csv")
    replication = pd.read_csv(root / "results/robustness/seed171_replication.csv")
    statistics = pd.read_csv(root / "results/statistical_reliability/comparisons.csv")
    calibration = pd.read_csv(root / "results/calibration/metrics.csv")
    lines = [
        "# Robustness, statistical reliability, and forecast uncertainty",
        "",
        "## Executed scope",
        "",
        "The Phase 4/5 baseline was independently re-audited: 282 folds, 355,320 predictions, "
        "actual label endpoints, split/selection metadata, original hashes and recomputed errors. "
        "Its 210 missing legacy execution records remain missing. New inference and intervals "
        "reuse these forecasts; they do not establish the original training provenance.",
        "",
        "The fixed five-seed experiment retrains RF and LSTM on the earliest, middle, and latest "
        "folds in each index/horizon cell: 18 folds × 5 seeds = 90 full-budget tasks / 180 "
        "stochastic model-fold evaluations. Each seed covers 756 origins per cell. "
        "This is a balanced sensitivity subset, not a five-seed replication of all 282 folds. "
        "Deterministic forecasts are reused. All alternatives and seeds were declared in "
        "`configs/robustness.toml` and committed before execution.",
        "",
        "## Matched calendars",
        "",
        "Primary matching requires both origin and actual target endpoint dates; origin-only "
        "matching is a fixed sensitivity. They coincide on these saved files (zero endpoint "
        "disagreements). Each index/horizon retains 2,232 origins: 68.13% of NASDAQ's 3,276 "
        "and 10.93% of S&P's 20,412. This equalizes evaluation dates, not training histories "
        "or annual fold boundaries. Results are not independent markets "
        "or index transfer evidence.",
        "",
    ]
    lines += markdown_table(
        counts[
            (counts.model == "zero_return") & counts.population.isin(["native", "endpoint_matched"])
        ],
        ["dataset", "horizon", "population", "n", "removed"],
    )
    lines += ["", "Pooled RMSE winners (descriptive, selected from each reported population):", ""]
    winners = matched.loc[matched.groupby(["dataset", "horizon", "population"]).rmse.idxmin()]
    lines += markdown_table(
        winners[winners.population.isin(["native", "endpoint_matched"])],
        ["dataset", "horizon", "population", "model", "rmse", "skill_vs_zero"],
    )
    lines += [
        "",
        "All models' MAE/RMSE, both baseline skills and ranks are in "
        "`results/robustness/matched/metrics.csv`; membership records retain endpoint dates.",
        "",
        "## Optimization-seed sensitivity",
        "",
        "Ranges below summarize five seeds over the same three folds, not confidence "
        "intervals over independent market histories. Rank is among all five forecasts.",
        "",
    ]
    lines += markdown_table(
        seeds[seeds.model.isin(["random_forest", "lstm"])],
        [
            "dataset",
            "horizon",
            "model",
            "rmse_min",
            "rmse_max",
            "skill_min",
            "skill_max",
            "rank_min",
            "rank_max",
        ],
    )
    lines += [
        "",
        f"Fresh seed-171 predictions agree numerically at rtol=1e-10/atol=1e-12 "
        f"with the old forecasts in {int(replication.numerically_equal_at_1e_10.sum())}/"
        f"{len(replication)} selected model-fold comparisons. Maximum absolute prediction "
        f"difference: {replication.max_prediction_difference.max():.6g}. "
        "Agreement strengthens targeted numerical reproducibility; it does not reconstruct "
        "the old execution context or establish full historical replication.",
        "",
        "## Paired statistical comparisons",
        "",
        "Primary follow-ups use squared-error loss, native calendars, Bartlett HAC "
        "bandwidth 19, and one Holm family of 18 two-sided tests. Negative effects favor "
        "RF. The displayed CIs are marginal 95% moving-block percentile intervals "
        "(2,000 replicates, L=20), not simultaneous Holm-adjusted intervals. "
        "These same-history, retrospective comparisons rely on weak-dependence "
        "approximations and cannot supply independent confirmation.",
        "",
    ]
    primary = statistics[statistics.primary_family]
    lines += markdown_table(
        primary,
        [
            "dataset",
            "horizon",
            "second",
            "mean_loss_difference",
            "ci_lower",
            "ci_upper",
            "holm_p_value",
        ],
    )
    improvements = primary[
        (primary.second.isin(["zero_return", "historical_mean"]))
        & (primary.mean_loss_difference < 0)
        & (primary.holm_p_value < 0.05)
    ]
    lines += [
        "",
        f"RF has {len(improvements)} Holm-supported improvements over a simple baseline "
        "in the primary family. A nonsignificant difference is inconclusive, not equivalence. "
        "RF-versus-LSTM and adverse baseline differences must be interpreted separately. "
        "Full block-length 20/40/80 sensitivity, endpoint-matched comparisons, exploratory "
        "MAE results and regime diagnostics are persisted. A nominal block-equivalent "
        "count n/L is a heuristic, not effective independent sample size. COVID and the "
        "146-origin crisis slice do not meet the 200-origin inference threshold at L=20; "
        "they remain descriptive. Missing NASDAQ stress episodes remain unavailable.",
        "",
        "## Chronological interval quality",
        "",
        "Calibration uses only earlier OOS residuals whose actual target dates are strictly "
        "before the current origin. Fixed methods are trailing 504, expanding, and trailing "
        "volatility-scaled residuals, with minimum 252 mature scores. Levels 80%/90% use "
        "ceil((n+1)*coverage) order statistics. Intervals are empirical prequential "
        "procedure diagnostics, not exchangeable split-conformal guarantees. "
        "The same eligible origins are used for all methods: exclusions per model are "
        "253/257/272 for horizons 1/5/20. Bounds, score populations and cutoff dates are "
        "saved for every forecast, including explicit warmup rows.",
        "",
        "Representative zero-return procedure at nominal 90% (all five models and both "
        "levels are available in the machine-readable metrics):",
        "",
    ]
    overall = calibration[
        (calibration["slice"] == "overall")
        & (calibration.nominal == 0.9)
        & (calibration.model == "zero_return")
    ]
    lines += markdown_table(
        overall,
        [
            "dataset",
            "horizon",
            "method",
            "n",
            "coverage",
            "coverage_gap",
            "mean_width",
            "median_width",
            "interval_score",
        ],
    )
    lines += ["", "High-volatility diagnostics for the zero-return procedure at nominal 90%:", ""]
    high = calibration[
        (calibration["slice"] == "volatility_high")
        & (calibration.nominal == 0.9)
        & (calibration.model == "zero_return")
    ]
    lines += markdown_table(
        high, ["dataset", "horizon", "method", "n", "coverage", "mean_width", "interval_score"]
    )
    lines += ["", "COVID diagnostics at nominal 90% (52 origins per cell; descriptive only):", ""]
    covid = calibration[
        (calibration["slice"] == "covid_crash")
        & (calibration.nominal == 0.9)
        & (calibration.model == "zero_return")
    ]
    lines += markdown_table(
        covid, ["dataset", "horizon", "method", "n", "coverage", "mean_width", "interval_score"]
    )
    lines += [
        "",
        "Coverage close to 90% overall does not imply conditional coverage in high "
        "volatility or stress. Volatility scaling often changes coverage, width and score "
        "together; wider intervals are not automatically better. For example, compare "
        "the reported proper interval score rather than coverage alone. Expanding history "
        "can adapt slowly to shifts, while a trailing window has sampling variability. "
        "No alternative was selected by these final test results.",
        "",
        "## Figures and reproducibility",
        "",
        "Generate all figures with `python scripts/research_cycle.py figures`; exact captions "
        "are in `results/robustness/figures/captions.json`. The following figures use "
        "actual saved tables and interval records:",
        "",
    ]
    for name in (
        "matched_rmse",
        "seed_rmse",
        "paired_loss_intervals",
        "nominal_coverage",
        "volatility_coverage",
        "covid_intervals",
    ):
        lines += [f"![{name}](../results/robustness/figures/{name}.png)", ""]
    lines += [
        "See [methods](ROBUSTNESS_METHODOLOGY.md), "
        "[statistical reliability](STATISTICAL_RELIABILITY.md), "
        "[calibration](FORECAST_CALIBRATION.md), and "
        "[limitations](RESEARCH_LIMITATIONS.md). Restore/audit commands and artifacts "
        "are documented in `results/robustness/README.md`. No forecast, confidence "
        "interval, training time, or execution provenance has been invented.",
        "",
    ]
    (root / "docs/ROBUSTNESS_RESULTS.md").write_text("\n".join(lines))
