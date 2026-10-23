"""Publication figures directly from the saved Phase 6–8 metric tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_figures(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    destination = root / "results/robustness/figures"
    destination.mkdir(exist_ok=True)
    matched = pd.read_csv(root / "results/robustness/matched/metrics.csv")
    seeds = pd.read_csv(root / "results/robustness/seed_metrics.csv")
    statistics = pd.read_csv(root / "results/statistical_reliability/comparisons.csv")
    intervals = pd.read_csv(root / "results/calibration/metrics.csv")
    datasets = ["NASDAQ-100", "S&P 500"]
    horizons = [1, 5, 20]
    models = ["zero_return", "historical_mean", "linear_regression", "random_forest", "lstm"]
    colors = dict(zip(models, plt.get_cmap("tab10").colors[:5], strict=True))
    captions = {}

    def save(figure, name, caption):
        figure.savefig(destination / f"{name}.png", dpi=220, bbox_inches="tight")
        plt.close(figure)
        captions[name] = caption

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            axis = axes[i, j]
            group = matched[(matched.dataset == dataset) & (matched.horizon == horizon)]
            for k, population in enumerate(("native", "endpoint_matched")):
                values = group[group.population == population].set_index("model").loc[models]
                axis.barh(
                    np.arange(5) + (k - 0.5) * 0.34,
                    values.rmse,
                    height=0.34,
                    label=population,
                    color=(".65", "tab:blue")[k],
                )
            axis.set_yticks(np.arange(5), models)
            axis.set_xlabel("Pooled RMSE (log return)")
            axis.set_title(f"{dataset}, {horizon} sessions")
    axes[0, 0].legend(fontsize=8)
    save(
        fig,
        "matched_rmse",
        "Native and shared origin+endpoint populations; unchanged fitted forecasts. "
        "Different training histories remain. Source: matched/metrics.csv.",
    )

    membership = pd.read_csv(root / "results/robustness/matched/membership.csv.gz")
    fig, axis = plt.subplots(figsize=(12, 4))
    labels = []
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            y = i * 3 + j
            sub = membership[membership.horizon == horizon]
            endpoint = "endpoint_a" if dataset == datasets[0] else "endpoint_b"
            full = sub[sub[endpoint].notna()]
            selected = sub[sub.endpoint_match]
            axis.plot(pd.to_datetime(full.date), np.full(len(full), y), color=".75", linewidth=5)
            axis.plot(
                pd.to_datetime(selected.date),
                np.full(len(selected), y),
                color="tab:blue",
                linewidth=5,
            )
            labels.append(f"{dataset} h={horizon}")
    axis.set_yticks(np.arange(6), labels)
    axis.set_xlabel("Forecast-origin date")
    axis.set_title("Native evaluation coverage (gray) and endpoint-matched dates (blue)")
    save(
        fig,
        "calendar_coverage",
        "Actual evaluated origins, not raw data coverage. Source: membership.csv.gz.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    fig_rank, rank_axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            part = seeds[(seeds.dataset == dataset) & (seeds.horizon == horizon)]
            for k, model in enumerate(("random_forest", "lstm")):
                data = part[part.model == model]
                axes[i, j].plot(np.full(5, k), data.rmse, "o", color=colors[model], alpha=0.7)
                rank_axes[i, j].plot(
                    np.full(5, k), data["rank"], "o", color=colors[model], alpha=0.7
                )
            for model, style in (("zero_return", "--"), ("historical_mean", ":")):
                axes[i, j].axhline(
                    part[part.model == model].rmse.iloc[0], color=".4", linestyle=style, label=model
                )
            for axis in (axes[i, j], rank_axes[i, j]):
                axis.set_xticks([0, 1], ["Random forest", "LSTM"])
                axis.set_title(f"{dataset}, {horizon} sessions; 756 origins")
            axes[i, j].set_ylabel("Pooled RMSE (log return)")
            rank_axes[i, j].set_ylabel("RMSE rank among five forecasts")
            rank_axes[i, j].set_ylim(0.7, 5.3)
    axes[0, 0].legend(fontsize=8)
    save(
        fig,
        "seed_rmse",
        "Five fixed seeds on three time-stratified folds per cell; points can overlap. "
        "Horizontal lines are unchanged deterministic baselines. Source: seed_metrics.csv.",
    )
    save(
        fig_rank,
        "seed_ranks",
        "Five optimization seeds, not independent historical samples. Source: seed_metrics.csv.",
    )

    primary = statistics[statistics.primary_family].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(10, 8))
    center = primary.mean_loss_difference
    axis.errorbar(
        center,
        np.arange(18),
        xerr=np.vstack([center - primary.ci_lower, primary.ci_upper - center]),
        fmt="o",
        color="tab:blue",
        capsize=2,
    )
    axis.set_yticks(
        np.arange(18), [f"{r.dataset} h={r.horizon}: RF − {r.second}" for r in primary.itertuples()]
    )
    axis.axvline(0, color=".4", linewidth=1)
    axis.set_xlabel("Mean squared-error difference (squared log return); negative favors RF")
    axis.set_title("Paired loss effects and marginal 95% moving-block intervals (L=20)")
    save(
        fig,
        "paired_loss_intervals",
        "Marginal percentile intervals are not Holm-simultaneous intervals. "
        "Adjusted HAC p-values are in comparisons.csv; these are retrospective follow-ups.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            group = statistics[
                (statistics.dataset == dataset)
                & (statistics.horizon == horizon)
                & (statistics.population == "native")
                & (statistics.loss == "squared_error")
                & (statistics.second == "zero_return")
            ]
            axis = axes[i, j]
            axis.errorbar(
                group.mean_loss_difference,
                group.block_length,
                xerr=np.vstack(
                    [
                        group.mean_loss_difference - group.ci_lower,
                        group.ci_upper - group.mean_loss_difference,
                    ]
                ),
                fmt="o",
                color="tab:blue",
                capsize=2,
            )
            axis.axvline(0, color=".5", linewidth=0.8)
            axis.set_title(f"{dataset}, {horizon} sessions")
            axis.set_ylabel("Block length (sessions)")
            axis.set_xlabel("RF − zero mean squared-error loss")
    save(
        fig,
        "dependence_sensitivity",
        "RF versus zero, block lengths 20/40/80; 2,000 paired replicates per point.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    fig_vol, vol_axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    fig_trade, trade_axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for i, dataset in enumerate(datasets):
        for j, horizon in enumerate(horizons):
            part = intervals[(intervals.dataset == dataset) & (intervals.horizon == horizon)]
            for model in models:
                group = part[
                    (part.model == model)
                    & (part.method == "trailing")
                    & (part["slice"] == "overall")
                ].sort_values("nominal")
                axes[i, j].plot(
                    group.nominal, group.coverage, "o-", label=model, color=colors[model]
                )
                volatility = part[
                    (part.model == model) & (part.method == "trailing") & (part.nominal == 0.9)
                ].set_index("slice")
                vol_axes[i, j].plot(
                    [0, 1, 2],
                    [
                        volatility.loc[f"volatility_{r}", "coverage"]
                        for r in ("low", "medium", "high")
                    ],
                    "o-",
                    label=model,
                    color=colors[model],
                )
                for method, marker in (
                    ("trailing", "o"),
                    ("expanding", "s"),
                    ("volatility_scaled", "^"),
                ):
                    point = part[
                        (part.model == model)
                        & (part.method == method)
                        & (part.nominal == 0.9)
                        & (part["slice"] == "overall")
                    ].iloc[0]
                    trade_axes[i, j].scatter(
                        point.mean_width,
                        point.coverage,
                        marker=marker,
                        color=colors[model],
                        label=f"{model}, {method}",
                    )
            axes[i, j].plot([0.8, 0.9], [0.8, 0.9], "--", color=".4", label="nominal")
            axes[i, j].set_xlabel("Nominal coverage")
            axes[i, j].set_ylabel("Empirical marginal coverage")
            vol_axes[i, j].set_xticks([0, 1, 2], ["Low", "Medium", "High"])
            vol_axes[i, j].axhline(0.9, color=".4", linestyle="--")
            vol_axes[i, j].set_ylabel("Empirical coverage at nominal 90%")
            vol_axes[i, j].set_xlabel("Training-defined volatility regime")
            trade_axes[i, j].axhline(0.9, color=".4", linestyle="--")
            trade_axes[i, j].set_xlabel("Mean interval width (log return)")
            trade_axes[i, j].set_ylabel("Empirical coverage at nominal 90%")
            for axis in (axes[i, j], vol_axes[i, j], trade_axes[i, j]):
                axis.set_title(f"{dataset}, {horizon} sessions")
    axes[0, 0].legend(fontsize=6)
    vol_axes[0, 0].legend(fontsize=6)
    save(
        fig,
        "nominal_coverage",
        "Trailing 504 mature residuals, minimum 252; exclusions counted in metrics.csv. "
        "No time-series coverage guarantee is asserted.",
    )
    save(
        fig_vol,
        "volatility_coverage",
        "Trailing intervals by training-defined regimes; diagnostic conditional coverage.",
    )
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=colors[m], label=m) for m in models]
    handles += [
        Line2D([0], [0], color=".3", marker=s, linestyle="", label=m)
        for m, s in (("trailing", "o"), ("expanding", "s"), ("volatility_scaled", "^"))
    ]
    fig_trade.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8
    )
    save(
        fig_trade,
        "width_coverage",
        "Circles: trailing; squares: expanding; triangles: volatility scaled. "
        "Colors identify models. Proper interval scores are also reported; "
        "wider is not automatically better.",
    )

    records = root / "results/calibration/records"
    raw = pd.read_csv(records / "sandp_500_h20_zero_return_trailing.csv.gz")
    scaled = pd.read_csv(records / "sandp_500_h20_zero_return_volatility_scaled.csv.gz")
    selected = raw.stress.eq("covid_crash")
    dates = pd.to_datetime(raw.loc[selected, "date"])
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.fill_between(
        dates,
        raw.loc[selected, "lower_90"],
        raw.loc[selected, "upper_90"],
        color="tab:blue",
        alpha=0.2,
        label="Trailing 90% interval",
    )
    axis.plot(
        dates,
        scaled.loc[selected, "lower_90"],
        color="tab:orange",
        linestyle="--",
        label="Volatility-scaled 90% bounds",
    )
    axis.plot(dates, scaled.loc[selected, "upper_90"], color="tab:orange", linestyle="--")
    axis.plot(
        dates, raw.loc[selected, "y_true"], color=".15", label="Realized 20-session log return"
    )
    axis.set_xlabel("Forecast-origin date")
    axis.set_ylabel("Twenty-session log return")
    axis.set_title("S&P 500 zero-return procedure during the predeclared COVID slice")
    axis.legend(fontsize=8)
    save(
        fig,
        "covid_intervals",
        "Bounds use earlier matured OOS residuals. Returns realize 20 sessions after "
        "each plotted origin; overlapping targets are dependent.",
    )
    from marketpulse.experiments.robustness_data import write_json

    write_json(destination / "captions.json", captions)
