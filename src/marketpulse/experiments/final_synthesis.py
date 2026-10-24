"""Manuscript evidence tables and SHA256-linked, executable quantitative claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json


def evaluate_claim(root: Path, claim: dict):
    path = root / claim["artifact"]
    if file_hash(path) != claim["sha256"]:
        raise ValueError(f"claim source changed: {claim['id']}")
    operation = claim["operation"]
    if operation.startswith("json"):
        value = json.loads(path.read_text())
        for key in claim.get("keys", []):
            value = value[key]
        if operation == "json_length":
            value = len(value)
        elif operation != "json_value":
            raise ValueError("unsupported JSON operation")
        return (
            value * claim.get("scale", 1)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else value
        )
    frame = pd.read_csv(path)
    for column, condition in claim.get("filters", {}).items():
        if isinstance(condition, dict):
            for operator, value in condition.items():
                if operator == "lt":
                    frame = frame[frame[column] < value]
                elif operator == "gt":
                    frame = frame[frame[column] > value]
                else:
                    raise ValueError("unsupported filter")
        elif isinstance(condition, list):
            frame = frame[frame[column].isin(condition)]
        else:
            frame = frame[frame[column] == condition]
    if operation == "count":
        return len(frame)
    if operation == "distinct_count":
        return len(frame.drop_duplicates(claim["columns"]))
    if operation == "sum":
        return float(frame[claim["column"]].sum())
    if operation == "minmax":
        return [float(frame[claim["column"]].min()), float(frame[claim["column"]].max())]
    if operation == "select":
        part = frame[claim["columns"]]
        return part.astype(object).where(pd.notna(part), None).to_dict("records")
    raise ValueError("unsupported evidence operation")


def verify_claim(root: Path, claim: dict) -> None:
    if evaluate_claim(root, claim) != claim["value"]:
        raise ValueError(f"claim value mismatch: {claim['id']}")


def build(root: Path) -> None:
    output = root / "results/phase12"
    if (output / "manifest.json").exists():
        raise ValueError("preserve previous synthesis")
    audits = json.loads((output / "audits/manifest.json").read_text())
    assert all(a["returncode"] == 0 for a in audits["audits"])
    claims = []

    def add(identifier, artifact, operation, classification="retrospective descriptive", **kwargs):
        claim = {
            "id": identifier,
            "artifact": artifact,
            "sha256": file_hash(root / artifact),
            "operation": operation,
            "classification": classification,
            **kwargs,
        }
        claim["value"] = evaluate_claim(root, claim)
        claims.append(claim)
        return claim["value"]

    historical = "results/walkforward/full/verification.json"
    add("C01", historical, "json_value", keys=["validated_folds"])
    add("C02", historical, "json_value", keys=["prediction_rows"])
    add("C03", historical, "json_value", keys=["legacy_folds"])
    add(
        "C04",
        "results/phase9/rehearsal/manifest.json",
        "json_value",
        classification="independent validation unavailable",
        keys=["independent_validation_completed"],
    )
    add(
        "C05",
        "results/phase10/analysis_manifest.json",
        "json_value",
        classification="retrospective sensitivity",
        keys=["new_tasks"],
    )
    add(
        "C06",
        "results/phase10/analysis_manifest.json",
        "json_value",
        keys=["temporal_origins_per_cell"],
    )
    seed_columns = [
        "dataset",
        "horizon",
        "model",
        "seed",
        "n",
        "rmse",
        "mae",
        "skill_vs_zero",
        "rank",
    ]
    seeds = add(
        "C07",
        "results/phase10/seed_metrics.csv",
        "select",
        classification="retrospective seed sensitivity",
        columns=seed_columns,
    )
    add(
        "C08",
        "results/phase10/comparisons.csv",
        "count",
        classification="retrospective approximate inference",
        filters={
            "primary": True,
            "second": ["zero_return", "historical_mean"],
            "mean_loss_difference": {"lt": 0},
            "holm_p": {"lt": 0.05},
        },
    )
    add(
        "C09",
        "results/statistical_reliability/comparisons.csv",
        "count",
        classification="retrospective approximate inference",
        filters={
            "primary_family": True,
            "second": ["zero_return", "historical_mean"],
            "mean_loss_difference": {"lt": 0},
            "holm_p_value": {"lt": 0.05},
        },
    )
    point_columns = [
        "dataset",
        "horizon",
        "model",
        "n",
        "mae",
        "rmse",
        "bias",
        "directional_accuracy",
        "skill_vs_zero",
        "rank",
    ]
    native = add(
        "C10", "results/walkforward/full/pooled_metrics.csv", "select", columns=point_columns
    )
    matched = add(
        "C11",
        "results/robustness/matched/metrics.csv",
        "select",
        filters={"population": "endpoint_matched"},
        columns=point_columns,
    )
    add(
        "C12",
        "results/phase10/seed_metrics.csv",
        "minmax",
        column="rank",
        filters={"dataset": "NASDAQ-100", "horizon": 20, "model": "lstm"},
    )
    add("C13", "results/phase11/audit.json", "json_value", keys=["records"])
    add("C14", "results/phase11/audit.json", "json_value", keys=["independent_bounds"])
    add(
        "C15",
        "results/phase11/score_comparisons.csv",
        "count",
        classification="retrospective approximate inference",
        filters={"primary": True, "mean_loss_difference": {"lt": 0}, "holm_p": {"lt": 0.05}},
    )
    adaptive_columns = [
        "dataset",
        "horizon",
        "model",
        "nominal",
        "mean_loss_difference",
        "ci_lower",
        "ci_upper",
        "holm_p",
    ]
    add(
        "C16",
        "results/phase11/score_comparisons.csv",
        "select",
        classification="retrospective approximate inference",
        filters={"primary": True, "mean_loss_difference": {"lt": 0}, "holm_p": {"lt": 0.05}},
        columns=adaptive_columns,
    )
    interval_columns = [
        "dataset",
        "horizon",
        "model",
        "method",
        "slice",
        "nominal",
        "n",
        "coverage",
        "mean_width",
        "interval_score",
    ]
    add(
        "C17",
        "results/phase11/all_methods_metrics.csv",
        "select",
        filters={
            "model": "zero_return",
            "nominal": 0.9,
            "slice": ["overall", "volatility_high"],
            "method": ["trailing", "volatility_scaled", "aci_w504_g001"],
        },
        columns=interval_columns,
    )
    calibration = add(
        "C18",
        "results/phase11/all_methods_metrics.csv",
        "select",
        filters={"slice": ["overall", "volatility_high"]},
        columns=interval_columns,
    )
    for identifier, variant, operation, column in [
        ("C19", "aci_w504_g001", "sum", "nesting_violations"),
        ("C20", "aci_w504_g001", "sum", "eligible"),
        ("C21", "aci_w504_g005", "sum", "nesting_violations"),
        ("C22", "aci_w252_g005", "sum", "nesting_violations"),
        ("C23", "aci_w252_g001", "sum", "nesting_violations"),
    ]:
        add(
            identifier,
            "results/phase11/diagnostics.csv",
            operation,
            classification="exploratory failure diagnostic",
            column=column,
            filters={"variant": variant},
        )
    add(
        "C24",
        "results/walkforward/full/pooled_metrics.csv",
        "minmax",
        column="rank",
        filters={"model": "lstm"},
    )
    add(
        "C25",
        "results/robustness/matched/metrics.csv",
        "select",
        columns=point_columns,
        filters={"population": "endpoint_matched", "dataset": "S&P 500", "horizon": 5, "rank": 1},
    )
    add(
        "C26",
        "results/statistical_reliability/comparisons.csv",
        "count",
        classification="retrospective approximate inference",
        filters={
            "primary_family": True,
            "first": "random_forest",
            "second": "lstm",
            "mean_loss_difference": {"lt": 0},
            "holm_p_value": {"lt": 0.05},
        },
    )
    add("C27", historical, "json_value", keys=["origin_horizon_pairs"])
    add(
        "C28",
        "results/phase9/rehearsal/metrics.csv",
        "count",
        classification="infrastructure rehearsal",
    )
    add(
        "C29",
        "results/phase9/rehearsal/metrics.csv",
        "distinct_count",
        classification="infrastructure rehearsal",
        columns=["dataset", "horizon", "model"],
    )
    add("C30", "results/phase10/common_calendar_counts.csv", "sum", column="shared_endpoints")
    add(
        "C31",
        "results/phase12/audits/manifest.json",
        "json_length",
        classification="reproducibility audit",
        keys=["audits"],
    )
    add(
        "C32",
        "results/phase12/audits/manifest.json",
        "json_value",
        classification="reproducibility audit",
        keys=["historical_tree_unchanged"],
    )
    statistical_columns = [
        "dataset",
        "horizon",
        "first",
        "second",
        "n",
        "mean_loss_difference",
        "ci_lower",
        "ci_upper",
        "holm_p",
    ]
    expanded = add(
        "C33",
        "results/phase10/comparisons.csv",
        "select",
        classification="retrospective approximate inference",
        filters={"primary": True},
        columns=statistical_columns,
    )
    native_statistics = add(
        "C34",
        "results/statistical_reliability/comparisons.csv",
        "select",
        classification="retrospective approximate inference",
        filters={"primary_family": True},
        columns=[c if c != "holm_p" else "holm_p_value" for c in statistical_columns],
    )
    adaptive_statistics = add(
        "C35",
        "results/phase11/score_comparisons.csv",
        "select",
        classification="retrospective approximate inference",
        filters={"primary": True},
        columns=[
            "dataset",
            "horizon",
            "model",
            "nominal",
            "n",
            "mean_loss_difference",
            "ci_lower",
            "ci_upper",
            "holm_p",
        ],
    )
    add("C36", "results/phase10/analysis_manifest.json", "json_value", keys=["combined_folds"])
    add(
        "C37",
        "results/phase10/analysis_manifest.json",
        "json_value",
        keys=["new_stochastic_predictions"],
    )
    add(
        "C38",
        "results/phase10/analysis_manifest.json",
        "json_value",
        keys=["new_training_seconds"],
        scale=1 / 60,
    )
    add(
        "C39",
        "results/statistical_reliability/comparisons.csv",
        "count",
        classification="retrospective approximate inference",
        filters={
            "primary_family": True,
            "first": "random_forest",
            "second": ["zero_return", "historical_mean"],
            "mean_loss_difference": {"gt": 0},
            "holm_p_value": {"lt": 0.05},
        },
    )
    add(
        "C40",
        "results/statistical_reliability/comparisons.csv",
        "select",
        classification="retrospective approximate inference",
        filters={
            "primary_family": True,
            "first": "random_forest",
            "second": ["zero_return", "historical_mean"],
            "mean_loss_difference": {"gt": 0},
            "holm_p_value": {"lt": 0.05},
        },
        columns=[
            "dataset",
            "horizon",
            "second",
            "mean_loss_difference",
            "ci_lower",
            "ci_upper",
            "holm_p_value",
        ],
    )
    assert len(native) == len(matched) == 30 and len(seeds) == 150
    assert len(calibration) == 840
    write_json(output / "claims.json", claims)
    points = []
    for name, rows, identifier in [("native", native, "C10"), ("endpoint_matched", matched, "C11")]:
        points.extend([{**r, "population": name, "source_claim": identifier} for r in rows])
    pd.DataFrame(points).to_csv(output / "point_comparisons.csv", index=False)
    seed_table = (
        pd.DataFrame(seeds)
        .groupby(["dataset", "horizon", "model"])
        .agg(
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
            skill_min=("skill_vs_zero", "min"),
            skill_max=("skill_vs_zero", "max"),
            skill_mean=("skill_vs_zero", "mean"),
            seeds=("seed", "nunique"),
            n_per_seed=("n", "first"),
        )
        .reset_index()
    )
    seed_table["source_claim"] = "C07"
    seed_table.to_csv(output / "seed_comparisons.csv", index=False)
    cal = pd.DataFrame(calibration)
    cal["source_claim"] = "C18"
    cal.to_csv(output / "calibration_comparisons.csv", index=False)
    tables = []
    for label, rows, identifier in [
        ("native_MSE", native_statistics, "C34"),
        ("expanded_seed_mean_MSE", expanded, "C33"),
        ("adaptive_interval_score", adaptive_statistics, "C35"),
    ]:
        frame = pd.DataFrame(rows).rename(columns={"holm_p_value": "holm_p"})
        frame["family"], frame["source_claim"] = label, identifier
        tables.append(frame)
    pd.concat(tables, ignore_index=True).to_csv(output / "statistical_summary.csv", index=False)
    write_figure(output, pd.DataFrame(native), seed_table, cal)
    write_reports(root, {c["id"]: c["value"] for c in claims})
    documents = {
        p: file_hash(root / p)
        for p in [
            "docs/FINAL_RESEARCH_RESULTS.md",
            "docs/RESEARCH_CONTRIBUTIONS.md",
            "docs/PHASE12_SCIENTIFIC_SYNTHESIS.md",
        ]
    }
    write_json(
        output / "manifest.json",
        {
            "classification": "retrospective synthesis; independent confirmation unavailable",
            "environment": environment_record(root, "synthesis"),
            "source_sha256": file_hash(Path(__file__)),
            "report_documents": documents,
            "claim_count": len(claims),
            "prior_audit_count": len(audits["audits"]),
        },
    )
    write_json(output / "checksums.json", inventory(output))
    verify(root)


def write_figure(output, native, seeds, calibration):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    cells = list(native.groupby(["dataset", "horizon"]).groups)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (dataset, horizon) in enumerate(cells):
        old = native[
            (native.dataset == dataset)
            & (native.horizon == horizon)
            & (native.model == "random_forest")
        ].iloc[0]
        selected = seeds[
            (seeds.dataset == dataset)
            & (seeds.horizon == horizon)
            & (seeds.model == "random_forest")
        ].iloc[0]
        axes[0].plot(
            [100 * selected.skill_min, 100 * selected.skill_max],
            [i + 0.08, i + 0.08],
            color="orange",
        )
        axes[0].scatter(
            100 * selected.skill_mean,
            i + 0.08,
            color="orange",
            s=25,
            label="Selected 5-fold sample: seed range" if i == 0 else None,
        )
        axes[0].scatter(
            100 * old.skill_vs_zero,
            i - 0.08,
            color="navy",
            s=25,
            label="Full history: seed 171" if i == 0 else None,
        )
        part = calibration[
            (calibration.dataset == dataset)
            & (calibration.horizon == horizon)
            & (calibration.model == "zero_return")
            & calibration.nominal.eq(0.9)
            & calibration.slice.eq("volatility_high")
        ]
        for offset, method in enumerate(["trailing", "aci_w504_g001", "volatility_scaled"]):
            row = part[part.method == method].iloc[0]
            axes[1].scatter(
                row.coverage,
                i + (offset - 1) * 0.12,
                label=method if i == 0 else None,
                color={
                    "trailing": "tab:blue",
                    "aci_w504_g001": "tab:orange",
                    "volatility_scaled": "tab:green",
                }[method],
                s=25,
            )
    for ax in axes:
        ax.set_yticks(range(len(cells)), [f"{d} h{h}" for d, h in cells])
        ax.legend(fontsize=8, loc="best")
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("RF MSE reduction vs zero (%)")
    axes[0].set_title("Period/seed sensitivity; orange ranges are not CIs")
    axes[1].axvline(0.9, color="black", linestyle="--", linewidth=0.7)
    axes[1].set_xlabel("High-volatility coverage; nominal 90%")
    axes[1].set_title("Zero-return intervals; coverage points are not guarantees")
    fig.suptitle("MarketPulse: retrospective evidence, distinct evaluation populations")
    fig.tight_layout()
    fig.savefig(directory / "research_summary.png", dpi=220)
    plt.close(fig)
    write_json(
        directory / "caption.json",
        {
            "figure": "research_summary.png",
            "left_sources": ["point_comparisons.csv (native)", "seed_comparisons.csv"],
            "right_source": "calibration_comparisons.csv (zero, 90%, high volatility)",
            "caveat": (
                "five separated selected years versus full history; seed ranges are "
                "not inferential CIs; no independently validated gain"
            ),
        },
    )


def write_reports(root: Path, c: dict) -> None:
    text = f"""# Final MarketPulse research results

The saved evidence does not establish consistent superiority of complex ML over
simple return baselines. Some RF point estimates improve on the baselines, but
those gains do not survive the declared corrected comparisons. The evidence is
retrospective; independently unseen generalization remains unresolved.

## Evidence populations and research status

The historical study contains {c["C01"]} folds, {c["C02"]:,} model forecasts and
{c["C27"]:,} origin/horizon pairs [C01] [C02] [C27]. The canonical comparison is
zero return, historical mean, linear regression, RF and one LSTM family on two US
price indexes at horizons 1/5/20. These are price-return forecasts, not realized
trading strategies. Earlier course notebooks and terminal splits remain historical
or exploratory snapshots, not additional independent replications.

The Phase 9 independent-completion flag is `{c["C04"]}` [C04]. Its frozen
prospective protocol is available, with {c["C29"]} canonical model evaluations and
{c["C28"]} interval metric rows in the historical infrastructure rehearsal [C29]
[C28]. Baselines are procedures, so “model evaluations” must not be read as that
many independently fitted estimators. The older audit key `rehearsal_model_fits`
uses this evaluation count. Rehearsal scores are excluded from generalization claims.

Phase 10 adds {c["C05"]} full-budget stochastic runs and {c["C37"]:,} forecasts,
with {c["C38"]:.2f} summed training minutes [C05] [C37] [C38]. Combined sensitivity
covers {c["C36"]} selected folds and {c["C06"]:,} chronological origins per cell
[C36] [C06], conditional on the fixed seeds [C07]. This is not a full-history
multi-seed replication. The selected cross-index seed sample has zero shared
endpoints [C30]; its cross-index comparison is unavailable. The separate historical
endpoint-matched population is retained [C11].

## Do complex models consistently improve point forecasts?

Native pooled winners below minimize concatenated-origin RMSE, not mean fold RMSE.
All scores, biases, MAE and directional accuracy remain in the consolidated point
table [C10]. Rankings describe a fixed retrospective sample, not forecast guarantees.

"""
    native = pd.DataFrame(c["C10"])
    for (dataset, horizon), part in native.groupby(["dataset", "horizon"]):
        winner = part.sort_values("rmse").iloc[0]
        rf = part[part.model == "random_forest"].iloc[0]
        text += (
            f"- {dataset}, h{horizon}: {winner.model} has minimum RMSE "
            f"{winner.rmse:.6f}; RF MSE skill versus zero {rf.skill_vs_zero:.4f}; "
            f"n={winner.n:,} origins [C10].\n"
        )
    text += f"""
The original LSTM pooled rank is {c["C24"][0]:.0f} in every native cell [C24].
RF beats LSTM in {c["C26"]} corrected original comparisons [C26], but **zero** RF
improvements over either simple baseline survive the original primary family
[C09]. Zero expanded ML improvements over those baselines survive its separate
family [C08] [C33]. Nonsignificance is not an equivalence result or proof that a
model can never help. The original RF S&P h20 comparisons show harm versus both
simple baselines: {c["C39"]} corrected tests [C39] [C40]. Do not pool these
families, observations or p-values.

## What changes with population, horizon, regime and seed?

Matching actual origins AND endpoint dates changes the S&P h5 winner to linear
regression [C25], whereas its native winner is historical mean [C10]. Equal test
dates do not equalize training-history length, data conventions or index membership.
The expanded NASDAQ h20 LSTM seed ranks remain {c["C12"][0]:.0f}–{c["C12"][1]:.0f}
[C12], despite its full-history last-place ranking. The complete seed ranges and
per-model point scores are preserved [C07]; selected-year outcomes do not overturn
full-history outcomes. Favorable seed selection is not evidence of consistency.

Volatility regimes use causal trailing volatility and training-only thresholds.
Conditional loss estimates retain the original timeline; separated regime days
are not compressed into adjacent observations. Short stress episodes remain
primarily descriptive. NASDAQ's forecast evaluation does not cover the earlier
dot-com or financial-crisis windows, while S&P's earlier series predates the modern
index. A crisis narrative must respect that unequal observation support.

## Are uncertainty estimates reliable?

The adaptive extension stores {c["C13"]:,} method-origin records and independently
checks {c["C14"]:,} eligible level/bound pairs [C13] [C14]. These repeat the same
underlying point forecasts; they are not additional independent market outcomes.
All methods use only strictly mature actual target endpoints. Coverage, width and
proper interval score for all methods/models and both nominal levels are in the
consolidated calibration table [C18].

The designated delayed adaptive variant has {c["C15"]} Holm-adjusted proper-score
improvements [C15]: S&P horizons 1/5, all canonical point models and both levels
[C16]. These approximate retrospective tests compare it with trailing calibration,
not with the best possible method. No h20 or NASDAQ primary gain survives [C16]
[C35]. The longer-bandwidth sensitivity is exploratory, not another confirmation.
Volatility scaling can score better than the designated adaptive method [C17].
"""
    examples = pd.DataFrame(c["C17"])
    for dataset, horizon in [("S&P 500", 20), ("NASDAQ-100", 20)]:
        rows = examples[
            (examples.dataset == dataset)
            & (examples.horizon == horizon)
            & examples.slice.eq("overall")
        ]
        a = rows[rows.method == "aci_w504_g001"].iloc[0]
        b = rows[rows.method == "trailing"].iloc[0]
        text += (
            f"\nFor zero-return {dataset} h{horizon} 90% intervals, coverage changes "
            f"from {b.coverage:.6f} to {a.coverage:.6f}, while the interval score "
            f"changes from {b.interval_score:.6f} to {a.interval_score:.6f} [C17]. "
            "A smaller proper score therefore need not mean closer nominal coverage.\n"
        )
    text += f"""
The designated adaptive variant has {int(c["C19"]):,}/{int(c["C20"]):,} eligible
interval nesting violations [C19] [C20]. Faster adaptation has
{int(c["C21"]):,} (longer window) and {int(c["C22"]):,} (shorter window); slower
short-window adaptation has {int(c["C23"]):,} [C21] [C22] [C23]. Independently
adapted levels can cross; alpha clipping also changes the theoretical procedure.
These failures were retained, not repaired after observing outcomes. No finite
sample, arbitrary-dependence or conditional-regime coverage guarantee follows.

## Statistical interpretation and limitations

Paired moving blocks preserve actual session continuity and target endpoints.
Bartlett HAC and Holm use the declared families; bootstrap intervals are marginal,
not simultaneous. Inference approximates historical average effects under weak
dependence; structural shifts, long-memory dependence and overlapping targets can
undermine that approximation. Block-length sensitivity is not an estimate of the
true effective sample size. Fixed-seed loss averaging conditions on those optimizer
realizations and does not create more temporal observations or evaluate an ensemble.

All quantitative comparisons are retrospective or sensitivity analyses. A protocol
frozen before new execution does not erase earlier model/method exposure. No
confirmatory outcomes exist [C04]. NASDAQ and S&P are correlated US indexes with
unequal training histories, overlapping coverage and changing construction; these
are not independent samples of all financial markets. Raw acquisition timestamps,
precise source versions and licensing remain incomplete. Price closes are not
dividend-adjusted total returns. The {c["C03"]} recovered original folds still lack
original execution revisions/environments [C03]; independent audits do not repair
that missing history. Local numerical reruns are not cross-platform or weight-level
reproducibility. Transaction costs, turnover, investable execution and economic
utility were not tested. Other architectures, markets and feature families remain
outside this canonical comparison.

## Evidence and manuscript readiness

Consolidated CSVs are `point_comparisons.csv`, `seed_comparisons.csv`,
`calibration_comparisons.csv` and `statistical_summary.csv` under `results/phase12`.
They retain source-claim IDs and distinct populations/loss units. The executable
registry `claims.json` pins source SHA256, selectors, operations and expected values.
Every measured conclusion above cites its registry IDs. The synthesis audit checks
all claims and report hashes. Earlier independent arithmetic audits remain the
basis for scientific calculations, not merely this traceability layer.

All {c["C31"]} cross-phase audits pass and historical trees remain unchanged
[C31] [C32]. The study is ready for a carefully qualified retrospective empirical
manuscript. Independent chronological confirmation is a required future experiment
for claims of unseen generalization; it is not required to honestly report these
negative/inconclusive comparisons and calibration failure cases. Phase 13 may
proceed with those limits, without claims of profitable trading, universal ML
inferiority, theoretical coverage guarantees or independently validated adaptive gains.

![Consolidated research evidence](../results/phase12/figures/research_summary.png)

Reproduce with `python scripts/final_synthesis.py verify`; run the individual saved
study audits listed in `results/phase12/audits/manifest.json`. A fresh comprehensive
logged audit uses `python scripts/audit_all_research.py` in a new unpopulated output
state. Existing logs are preserved. Future intake is described in
`docs/PROSPECTIVE_INTAKE.md`; its unavailable observations are not synthesized here.
"""
    (root / "docs/FINAL_RESEARCH_RESULTS.md").write_text(text)
    contributions = f"""# Research contributions and boundaries

MarketPulse contributes an auditable retrospective comparison, not a claim that
complex forecasting models always fail or a new theorem for dependent conformal
inference. The supported contributions are:

- A shared causal forecasting protocol across the canonical procedures, two US
  price-index histories and multiple session horizons; {c["C01"]} walk-forward
  folds and {c["C02"]:,} stored model forecasts [C01] [C02]. Recovered legacy
  execution metadata remains explicitly incomplete [C03].
- A demonstration that point-estimate wins, corrected loss evidence, selected
  calendar populations and seed sensitivity answer different questions. Neither
  the original nor the expanded primary family establishes a corrected ML gain
  over simple baselines [C09] [C08]; full-history and matched tables remain distinct
  [C10] [C11]. This does not establish statistical equivalence or universal inferiority.
- A full-budget temporal expansion adding {c["C05"]} stochastic runs [C05],
  conditional on the fixed seed design [C07]. The NASDAQ long-horizon rank
  instability persists [C12], and absent cross-index seed overlap is reported
  rather than disguised as calendar matching [C30].
- A delayed-outcome uncertainty experiment with independently checked interval
  populations, quantiles and feedback states [C13] [C14]. The designated variant's
  corrected score gains are limited to the documented comparisons [C15] [C16];
  coverage/width trade-offs and nesting failures remain visible [C17] [C19] [C21].
- A prospective acquisition/validation contract whose execution is explicitly
  pending unexposed data [C04], and an executable evidence-to-claim registry with
  preserved historical result trees [C31] [C32].

These are empirical and engineering contributions within this protocol, not claims
of novelty over the conformal literature or comprehensive financial forecasting.
The original course work and subsequent empirical study remain distinguishable.
No trading profitability, new-market generalization, arbitrary-dependence guarantee
or independently confirmed adaptive superiority is established. A manuscript may
advance as a transparent retrospective study; those stronger questions remain open.
"""
    (root / "docs/RESEARCH_CONTRIBUTIONS.md").write_text(contributions)


def verify(root: Path) -> None:
    output = root / "results/phase12"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"] == file_hash(Path(__file__))
    claims = json.loads((output / "claims.json").read_text())
    ids = {c["id"] for c in claims}
    assert len(ids) == len(claims) == manifest["claim_count"]
    for claim in claims:
        verify_claim(root, claim)
    for path, sha in manifest["report_documents"].items():
        assert file_hash(root / path) == sha, path
        used = set(re.findall(r"\[(C\d+)\]", (root / path).read_text()))
        assert used <= ids, used - ids
    audits = json.loads((output / "audits/manifest.json").read_text())
    assert audits["historical_tree_unchanged"]
    for audit in audits["audits"]:
        assert audit["returncode"] == 0
        assert file_hash(output / "audits" / audit["log"]) == audit["sha256"]
    assert json.loads((output / "checksums.json").read_text()) == inventory(output)
    print(
        json.dumps(
            {
                "status": "passed",
                "quantitative_claims": len(claims),
                "prior_audits": len(audits["audits"]),
                "independent_confirmation": False,
            }
        )
    )
