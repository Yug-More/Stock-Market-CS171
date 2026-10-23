"""Figures for the walk-forward and regime results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_ORDER = (
    "zero_return",
    "historical_mean",
    "linear_regression",
    "random_forest",
    "lstm",
)
_REGIME_ORDER = ("low", "medium", "high")


def write_walkforward_figures(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    summary: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    stress_metrics: pd.DataFrame,
) -> None:
    """Write one figure per comparison. A missing matplotlib install skips the files."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    test = fold_metrics.loc[fold_metrics["partition"] == "test"].copy()
    for dataset, frame in test.groupby("dataset", sort=False):
        _rmse_by_fold(
            plt, figure_dir / f"rmse_by_fold_{_slug(str(dataset))}.png", str(dataset), frame
        )
        _regime_bars(
            plt,
            figure_dir / f"regime_rmse_{_slug(str(dataset))}.png",
            str(dataset),
            regime_metrics,
            "rmse",
            "Test RMSE",
        )
        _regime_bars(
            plt,
            figure_dir / f"regime_skill_{_slug(str(dataset))}.png",
            str(dataset),
            regime_metrics,
            "skill_vs_zero",
            "Skill against the zero return",
        )
        _stress_bars(
            plt, figure_dir / f"stress_rmse_{_slug(str(dataset))}.png", str(dataset), stress_metrics
        )
    _pooled_bars(plt, figure_dir / "pooled_rmse.png", pooled, "rmse", "Pooled test RMSE")
    _pooled_bars(
        plt, figure_dir / "pooled_skill.png", pooled, "skill_vs_zero", "Pooled skill against zero"
    )
    _rank_bars(plt, figure_dir / "rank_stability.png", summary)


def _rmse_by_fold(plt: object, path: Path, dataset: str, frame: pd.DataFrame) -> None:
    horizons = sorted(int(value) for value in frame["horizon"].unique())
    figure, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 4.2), squeeze=False)
    for axis, horizon in zip(axes[0], horizons, strict=True):
        subset = frame.loc[frame["horizon"] == horizon]
        for model in _present(subset["model"]):
            ordered = subset.loc[subset["model"] == model].sort_values("fold")
            axis.plot(ordered["test_end"], ordered["rmse"], marker="o", linewidth=1.4, label=model)
        axis.set_title(f"{dataset}, horizon {horizon}")
        axis.set_xlabel("Test-window end")
        axis.set_ylabel("Test RMSE")
        axis.tick_params(axis="x", labelrotation=45)
        n_folds = int(subset["fold"].nunique())
        n_test = int(subset.loc[subset["model"] == subset["model"].iloc[0], "n"].sum())
        axis.text(
            0.02, 0.98, f"{n_folds} folds, {n_test} origins", transform=axis.transAxes, va="top"
        )
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _pooled_bars(plt: object, path: Path, pooled: pd.DataFrame, column: str, ylabel: str) -> None:
    datasets = list(dict.fromkeys(pooled["dataset"]))
    horizons = sorted(int(value) for value in pooled["horizon"].unique())
    figure, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets, strict=True):
        subset = pooled.loc[pooled["dataset"] == dataset]
        models = _present(subset["model"])
        x = np.arange(len(horizons))
        width = 0.15
        for offset, model in enumerate(models):
            heights = [_lookup(subset, horizon, model, column) for horizon in horizons]
            axis.bar(
                x + (offset - len(models) / 2) * width + width / 2, heights, width, label=model
            )
        axis.set_xticks(x, [str(horizon) for horizon in horizons])
        axis.set_xlabel("Horizon (trading sessions)")
        axis.set_ylabel(ylabel)
        axis.set_title(dataset)
        axis.axhline(0.0, color="black", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _rank_bars(plt: object, path: Path, summary: pd.DataFrame) -> None:
    datasets = list(dict.fromkeys(summary["dataset"]))
    horizons = sorted(int(value) for value in summary["horizon"].unique())
    figure, axes = plt.subplots(len(datasets), 1, figsize=(8.5, 3.6 * len(datasets)), squeeze=False)
    for axis, dataset in zip(axes[:, 0], datasets, strict=True):
        subset = summary.loc[summary["dataset"] == dataset]
        models = _present(subset["model"])
        x = np.arange(len(horizons))
        width = 0.15
        for offset, model in enumerate(models):
            shares = []
            for horizon in horizons:
                row = subset.loc[(subset["horizon"] == horizon) & (subset["model"] == model)]
                if row.empty:
                    shares.append(0.0)
                else:
                    shares.append(float(row["times_best"].iloc[0]) / float(row["n_folds"].iloc[0]))
            axis.bar(x + (offset - len(models) / 2) * width + width / 2, shares, width, label=model)
        axis.set_xticks(x, [str(horizon) for horizon in horizons])
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Horizon (trading sessions)")
        axis.set_ylabel("Share of folds with the best RMSE")
        axis.set_title(dataset)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _regime_bars(
    plt: object,
    path: Path,
    dataset: str,
    regime_metrics: pd.DataFrame,
    column: str,
    ylabel: str,
) -> None:
    subset = regime_metrics.loc[regime_metrics["dataset"] == dataset]
    if subset.empty:
        return
    horizons = sorted(int(value) for value in subset["horizon"].unique())
    figure, axes = plt.subplots(1, len(horizons), figsize=(5.4 * len(horizons), 4.3), squeeze=False)
    for axis, horizon in zip(axes[0], horizons, strict=True):
        block = subset.loc[subset["horizon"] == horizon]
        models = _present(block["model"])
        x = np.arange(len(_REGIME_ORDER))
        width = 0.15
        for offset, model in enumerate(models):
            heights = []
            for regime in _REGIME_ORDER:
                row = block.loc[(block["regime"] == regime) & (block["model"] == model)]
                heights.append(float(row[column].iloc[0]) if not row.empty else np.nan)
            axis.bar(
                x + (offset - len(models) / 2) * width + width / 2, heights, width, label=model
            )
        counts = [
            int(block.loc[block["regime"] == regime, "n"].max())
            if (block["regime"] == regime).any()
            else 0
            for regime in _REGIME_ORDER
        ]
        axis.set_xticks(
            x, [f"{regime}\nn={count}" for regime, count in zip(_REGIME_ORDER, counts, strict=True)]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(f"{dataset}, horizon {horizon}")
        axis.axhline(0.0, color="black", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _stress_bars(plt: object, path: Path, dataset: str, stress_metrics: pd.DataFrame) -> None:
    subset = stress_metrics.loc[
        (stress_metrics["dataset"] == dataset) & stress_metrics["model"].notna()
    ]
    if subset.empty:
        return
    horizons = sorted(int(value) for value in subset["horizon"].unique())
    slices = list(dict.fromkeys(subset["slice"]))
    figure, axes = plt.subplots(1, len(horizons), figsize=(5.6 * len(horizons), 4.4), squeeze=False)
    for axis, horizon in zip(axes[0], horizons, strict=True):
        block = subset.loc[subset["horizon"] == horizon]
        models = _present(block["model"])
        x = np.arange(len(slices))
        width = 0.15
        for offset, model in enumerate(models):
            heights = []
            for key in slices:
                row = block.loc[(block["slice"] == key) & (block["model"] == model)]
                heights.append(
                    float(row["rmse"].iloc[0]) if not row.empty and "rmse" in row else np.nan
                )
            axis.bar(
                x + (offset - len(models) / 2) * width + width / 2, heights, width, label=model
            )
        labels = []
        for key in slices:
            row = block.loc[block["slice"] == key]
            count = int(row["n_tested"].iloc[0]) if not row.empty else 0
            coverage = str(row["coverage"].iloc[0]) if not row.empty else "unavailable"
            labels.append(f"{key}\n{coverage}, n={count}")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Pooled test RMSE")
        axis.set_title(f"{dataset}, horizon {horizon}")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _present(models: pd.Series) -> list[str]:
    found = set(models.dropna().astype(str))
    return [model for model in _MODEL_ORDER if model in found]


def _lookup(frame: pd.DataFrame, horizon: int, model: str, column: str) -> float:
    row = frame.loc[(frame["horizon"] == horizon) & (frame["model"] == model)]
    if row.empty or pd.isna(row[column].iloc[0]):
        return np.nan
    return float(row[column].iloc[0])


def _slug(dataset: str) -> str:
    return dataset.lower().replace(" ", "_").replace("&", "and")
