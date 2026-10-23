"""Generate the research report from the verified complete experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/walkforward/full"


def main():
    verified = json.loads((OUTPUT / "verification.json").read_text())
    assert verified["validated_folds"] == verified["planned_folds"] == 282
    pooled = pd.read_csv(OUTPUT / "pooled_metrics.csv")
    fold = pd.read_csv(OUTPUT / "fold_summary.csv")
    stress = pd.read_csv(OUTPUT / "stress_metrics.csv")
    regime = pd.read_csv(OUTPUT / "regime_metrics.csv")
    plan = json.loads((OUTPUT / "fold_plan.json").read_text())
    lines = [
        "# Walk-forward forecasting and market-regime results",
        "",
        "## Experimental setup and completion",
        "",
        "The complete native-calendar experiment contains 282 folds "
        "and 1,410 model-fold evaluations: "
        "both index datasets, horizons 1/5/20 trading sessions, and five forecasts. "
        "All model selection uses validation RMSE; test scores are reported once. "
        "Each fold has 252 test origins, and adjacent folds advance 252 eligible sessions. "
        "Training expands from at least 2,520 embargo-clean origins, with 252 validation origins.",
        "",
        f"The saved study contains {verified['prediction_rows']:,} forecast rows and "
        f"{verified['origin_horizon_pairs']:,} origin/horizon pairs across the six cells. "
        "The run recovered 210 existing folds and executed the remaining 72 S&P 500 "
        "twenty-session folds at the original full budgets. "
        "No smoke predictions enter these results. "
        f"Recorded training runtime for new folds totals "
        f"{verified['new_fold_runtime_seconds'] / 60:.2f} "
        "minutes, excluding recovery auditing and aggregation; legacy runtime was not recorded.",
        "",
        "The recovery code revision is `ce58c9263d5fabad2476b1cf46c07d7c61ae3cd9`. "
        "New folds record their generating revision and software context. "
        "The 210 recovered folds lack original revision/environment metadata: their internal "
        "consistency was checked, but their missing execution provenance cannot be established. "
        "They are not presented as 210 newly rerun experiments.",
        "",
        "See [methodology](WALKFORWARD_METHODOLOGY.md), [recovery](RECOVERY.md), and "
        "[artifact schemas and commands](../results/walkforward/README.md).",
        "",
        "## Data and test calendars",
        "",
        "NASDAQ-100: 6,289 price-index sessions, 2000-01-03–2024-12-31. "
        "S&P 500: 23,323 sessions, 1927-12-30–2020-11-04. "
        "Neither dataset establishes dividend-reinvested returns. "
        "Source retrieval metadata are incomplete; raw input hashes are in the manifest. "
        "The same model inputs and protocol are used, but native test calendars differ. "
        "These are separate index case studies, not a matched cross-index comparison.",
        "",
    ]
    for dataset in pooled.dataset.unique():
        for horizon in (1, 5, 20):
            items = [x for x in plan if x["dataset"] == dataset and x["horizon"] == horizon]
            first = items[0]["sample"]["test"]["start_date"]
            last = items[-1]["sample"]["test"]["end_date"]
            lines.append(
                f"- {dataset}, h={horizon}: {len(items)} folds; {first} through {last}; "
                f"{len(items) * 252:,} test origins per model."
            )
    lines += [
        "",
        "## Pooled forecast errors and baseline skill",
        "",
        "Errors are dimensionless log returns. Pooled RMSE is calculated after "
        "concatenating predictions, not by averaging fold RMSE. "
        "Skill is `1-MSE_model/MSE_baseline`; positive values favor the model. "
        "Model rankings here describe observed test results and are not a retuning rule.",
        "",
    ]
    for (dataset, horizon), block in pooled.groupby(["dataset", "horizon"], sort=True):
        lines += [f"### {dataset}, {horizon}-session horizon", ""]
        for row in block.sort_values("rank").itertuples():
            lines.append(
                f"- {row.model}: RMSE {row.rmse:.6f}; MAE {row.mae:.6f}; "
                f"skill vs zero {row.skill_vs_zero:+.4f}; "
                f"skill vs mean {row.skill_vs_historical_mean:+.4f}."
            )
        winner = block.loc[block.rmse.idxmin()]
        ranking = fold[(fold.dataset == dataset) & (fold.horizon == horizon)]
        best = ranking[ranking.model == winner.model].iloc[0]
        lines += [
            "",
            f"Pooled winner `{winner.model}` is best in {int(best.times_best)}/"
            f"{int(best.n_folds)} folds; mean fold rank {best.mean_rank:.2f}, "
            f"rank SD {best.rank_std:.2f}, and {int(best.rank_changes)} consecutive "
            "rank changes. A pooled win does not imply a stable annual ranking.",
            "",
        ]
    lstm = pooled[pooled.model == "lstm"]
    lines += [
        "## Observations and interpretation",
        "",
        f"On pooled RMSE, LSTM beats the zero baseline in {int((lstm.skill_vs_zero > 0).sum())}/6 "
        f"pooled cells and the historical mean in "
        f"{int((lstm.skill_vs_historical_mean > 0).sum())}/6. "
        "The historical motivation for neural short-horizon superiority is not "
        "automatically supported by stronger chronological testing. "
        "Raw RMSE grows with the horizon's target scale; skill is the appropriate "
        "baseline-relative comparison. Terminal-window successes must be assessed "
        "against these repeated-window results rather than carried forward as stable findings.",
        "",
        "## Historical stress slices",
        "",
        "Inclusive membership uses the forecast-origin date. Dot-com: "
        "2000-03-01–2002-12-31; financial-crisis slice: 2008-09-01–2009-03-31; "
        "COVID: 2020-02-15–2020-04-30. The financial-crisis slice does not cover "
        "the entire 2007–2009 crisis. Forecast labels can realize outside a slice. "
        "Unavailable means no valid out-of-sample prediction, not zero error.",
        "",
    ]
    for (dataset, horizon, name), block in stress[stress.slice != "outside_stress"].groupby(
        ["dataset", "horizon", "slice"], sort=True
    ):
        first = block.iloc[0]
        prefix = (
            f"- {dataset}, h={horizon}, {name}: {first.coverage}, "
            f"{int(first.n_tested)}/{int(first.n_eligible)} eligible origins tested"
        )
        if int(first.n_tested) == 0:
            lines.append(prefix + "; no score.")
        else:
            winner = block.loc[block.rmse.idxmin()]
            zero = block[block.model == "zero_return"].iloc[0]
            outside = stress[
                (stress.dataset == dataset)
                & (stress.horizon == horizon)
                & (stress.slice == "outside_stress")
                & (stress.model == "zero_return")
            ].iloc[0]
            lines.append(
                prefix + f"; lowest RMSE `{winner.model}` ({winner.rmse:.6f}, "
                f"skill vs zero {winner.skill_vs_zero:+.4f}); zero-baseline RMSE "
                f"{zero.rmse / outside.rmse:.2f}× its outside-stress value."
            )
    lines += [
        "",
        "NASDAQ-100's training minimum excludes dot-com and financial-crisis "
        "origins. S&P 500 supplies those episodes. Comparisons to outside-stress "
        "rows are descriptive and are confounded by changing target scale and history. "
        "COVID samples are small; these scores do not establish crash-prediction ability.",
        "",
        "## Volatility regimes",
        "",
        "Each fold uses frozen 1/3 and 2/3 quantiles of trailing 20-session "
        "volatility on training origins only. No future realized volatility or full-sample "
        "threshold enters classification. Counts below are per model. "
        "Full model errors/skills/ranks and ratios to low-volatility RMSE are saved "
        "in `regime_metrics.csv`.",
        "",
    ]
    for (dataset, horizon), block in regime.groupby(["dataset", "horizon"], sort=True):
        for name in ("low", "medium", "high"):
            part = block[block.regime == name]
            if part.empty:
                continue
            winner = part.loc[part.rmse.idxmin()]
            zero = part[part.model == "zero_return"].iloc[0]
            lines.append(
                f"- {dataset}, h={horizon}, {name}: n={int(winner.n):,}; "
                f"lowest RMSE `{winner.model}` ({winner.rmse:.6f}, "
                f"skill vs zero {winner.skill_vs_zero:+.4f}); "
                f"zero-baseline RMSE {zero.rmse_ratio_to_low:.2f}× low-volatility RMSE."
            )
    lines += [
        "",
        "Higher absolute error can follow higher target volatility without "
        "worse relative skill. Regime rankings and annual rankings answer different "
        "questions; neither is evidence of a universally superior model.",
        "",
        "## Figures",
        "",
        "Generated at 220 DPI from actual saved forecasts/metrics; no synthetic "
        "series contributes. Axis values are errors in log-return units or MSE-based skill.",
        "",
        "![Pooled skill](../results/walkforward/full/figures/pooled_skill.png)",
        "",
        "![NASDAQ annual RMSE](../results/walkforward/full/figures/rmse_by_fold_nasdaq-100.png)",
        "",
        "![S&P stress errors](../results/walkforward/full/figures/stress_rmse_sandp_500.png)",
        "",
        "![S&P volatility skill](../results/walkforward/full/figures/regime_skill_sandp_500.png)",
        "",
        "## Verification and reproducibility",
        "",
        "`python scripts/verify_walkforward.py` independently validates all 282 planned "
        "folds against source data, checks original recovered checksums, and recomputes "
        "pooled/stress/regime metrics. The full runner rejects corrupt or incompatible "
        "resume inputs. See `verification.json` for machine-readable completion. "
        "Use `python scripts/report_walkforward.py` to regenerate this report after auditing. "
        "Local Ruff lint/format checks and 67 pytest tests passed with PyTorch installed; "
        "CI additionally exercises neural smoke and the saved-study audit.",
        "",
        "New training ran on Apple M2 Pro (10 CPU cores, 16 GiB memory), CPU PyTorch "
        "with six intra-operation threads. Full software versions and platform are in "
        "the manifest, environment record, per-new-fold reports, and pinned recovery "
        "dependency inventory. Bitwise cross-platform neural reproduction is not promised. "
        "Original raw files and Phase 3 artifacts were not replaced.",
        "",
        "## Failure analysis and threats to validity",
        "",
        "No planned fold or required model is omitted from the completed native-calendar "
        "matrix. Legacy provenance remains incomplete. Only one random seed and small "
        "predeclared tuning grids were evaluated. Weight checkpoints are not serialized. "
        "Overlapping horizon labels, shared training history, volatility clustering, "
        "and market nonstationarity preclude iid significance claims. No formal "
        "significance test or uncertainty interval was run. Source acquisition metadata "
        "are incomplete, early S&P history predates the modern index, and aggregate "
        "indexes cannot measure point-in-time constituent/survivorship effects. "
        "Stress boundaries are fixed and narrow, episode counts are small, and no "
        "boundary-sensitivity study was performed. Different native index calendars "
        "do not demonstrate transfer to an unseen market. No transaction-cost, portfolio, "
        "or profitability evaluation is included.",
        "",
        "## Next combined research cycle",
        "",
        "Run a predeclared matched-calendar comparison with common fold dates and "
        "comparable training histories, then quantify model/seed uncertainty using "
        "multiple fixed seeds and dependence-aware block resampling. Compare observed "
        "skill and rank stability without choosing favorable seeds or episodes. "
        "Add chronological interval calibration and regime-specific coverage diagnostics "
        "as a separate uncertainty task. Preserve the current single-seed study and "
        "artifact hashes as the baseline; recover complete execution provenance through "
        "a separately budgeted fresh run rather than relabeling legacy results.",
        "",
    ]
    (ROOT / "docs/WALKFORWARD_RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
