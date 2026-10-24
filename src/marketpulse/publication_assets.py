"""Reconstruct manuscript tables and figures from immutable saved-study outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.experiments.recovery import file_hash
from marketpulse.publication import LABELS, MODELS

TABLES = {
    "native_metrics": "results/walkforward/full/pooled_metrics.csv",
    "matched_metrics": "results/robustness/matched/metrics.csv",
    "seed_metrics": "results/phase10/seed_metrics.csv",
    "original_comparisons": "results/statistical_reliability/comparisons.csv",
    "expanded_comparisons": "results/phase10/comparisons.csv",
    "adaptive_comparisons": "results/phase11/score_comparisons.csv",
    "interval_metrics": "results/phase11/all_methods_metrics.csv",
    "coverage_intervals": "results/phase11/coverage_uncertainty.csv",
    "nesting_diagnostics": "results/phase11/diagnostics.csv",
}
COLORS = dict(zip(MODELS, ["#222222", "#0072B2", "#009E73", "#D55E00", "#CC79A7"], strict=True))
CELLS = [(d, h) for d in ["NASDAQ-100", "S&P 500"] for h in [1, 5, 20]]
METHODS = {
    "trailing": ("Trailing 504", "#222222"),
    "expanding": ("Expanding", "#0072B2"),
    "volatility_scaled": ("Volatility-scaled", "#009E73"),
    "aci_w504_g001": ("Adaptive 504/.01", "#D55E00"),
    "aci_w252_g001": ("Adaptive 252/.01", "#CC79A7"),
    "aci_w504_g005": ("Adaptive 504/.05", "#56B4E9"),
    "aci_w252_g005": ("Adaptive 252/.05", "#8B7500"),
}


def tables(root: Path) -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(root / path) for name, path in TABLES.items()}
    for name, selector, expected in [
        ("original", "primary_family", 18),
        ("expanded", "primary", 30),
        ("adaptive", "primary", 60),
    ]:
        part = frames[name + "_comparisons"]
        selected = part[part[selector]].reset_index(drop=True)
        assert len(selected) == expected
        frames[name + "_primary"] = selected
    return frames


def figures(frames: dict[str, pd.DataFrame], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output / f"{name}.png", bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    all_skill = []
    for ax, name, title in zip(
        axes,
        ["native_metrics", "matched_metrics"],
        ["Native full-history origins", "Common-calendar endpoints: 2,232 per cell"],
        strict=True,
    ):
        frame = frames[name]
        if "population" in frame:
            frame = frame[frame.population.eq("endpoint_matched")]
        assert len(frame) == 30
        for j, model in enumerate(MODELS):
            part = frame[frame.model.eq(model)].set_index(["dataset", "horizon"]).loc[CELLS]
            values = part.skill_vs_zero.to_numpy() * 100
            all_skill.extend(values)
            ax.scatter(
                values,
                np.arange(6) + (j - 2) * 0.12,
                label=LABELS[model],
                s=25,
                color=COLORS[model],
            )
        ax.axvline(0, color=".6", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("MSE reduction versus zero return (%)")
        ax.grid(axis="x", alpha=0.2)
    limits = (min(all_skill) - 2, max(all_skill) + 2)
    axes[0].set_xlim(limits)
    axes[0].set_yticks(range(6), [f"{d}, h={h}" for d, h in CELLS])
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5)
    fig.suptitle("All five models: descriptive pooled skill, no inferential intervals")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "point_skill")

    frame = frames["original_primary"]
    frame = frame[frame.second.isin(["zero_return", "historical_mean"])].copy()
    assert len(frame) == 12
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, row in enumerate(frame.itertuples()):
        color = "#D55E00" if row.holm_p_value < 0.05 else "#0072B2"
        ax.plot([row.ci_lower * 1e4, row.ci_upper * 1e4], [i, i], color=color)
        ax.scatter(row.mean_loss_difference * 1e4, i, color=color, s=25)
        ax.text(
            1.02,
            i,
            f"{row.p_value:.3g} / {row.holm_p_value:.3g}",
            transform=ax.get_yaxis_transform(),
            va="center",
        )
    ax.set_yticks(
        range(len(frame)),
        [f"{r.dataset}, h={r.horizon}, RF minus {LABELS[r.second]}" for r in frame.itertuples()],
    )
    ax.invert_yaxis()
    ax.axvline(0, color=".5", linewidth=0.8)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("Mean squared-loss difference (squared log-return × 10,000); left favors RF")
    ax.set_title(
        "RF versus simple baselines: all 12 contrasts in the original 18-test family\n"
        "Bars: marginal 95% moving-block intervals; right: nominal p / Holm p"
    )
    fig.subplots_adjust(left=0.30, right=0.78, bottom=0.12, top=0.88)
    save(fig, "baseline_effects")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    frame = frames["interval_metrics"]
    for row_i, scope in enumerate(["overall", "volatility_high"]):
        selected = frame[
            frame.model.eq("zero_return") & frame.nominal.eq(0.9) & frame.slice.eq(scope)
        ]
        for j, (method, (label, color)) in enumerate(METHODS.items()):
            part = selected[selected.method.eq(method)].set_index(["dataset", "horizon"]).loc[CELLS]
            for col, metric in enumerate(["coverage", "mean_width"]):
                axes[row_i, col].scatter(
                    part[metric], np.arange(6) + (j - 3) * 0.10, s=20, label=label, color=color
                )
        axes[row_i, 0].axvline(0.9, color=".5", linestyle="--", linewidth=1)
        axes[row_i, 0].set_xlim(0, 1)
        axes[row_i, 1].set_xlim(left=0)
        axes[row_i, 0].set_title(
            f"{'All eligible' if scope == 'overall' else 'High-volatility'} origins"
        )
        for col in range(2):
            axes[row_i, col].set_yticks(range(6), [f"{d}, h={h}" for d, h in CELLS])
            axes[row_i, col].grid(axis="x", alpha=0.2)
    axes[0, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Coverage fraction; nominal 0.90")
    axes[1, 1].set_xlabel("Mean interval width (log-return units)")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=8)
    fig.suptitle(
        "Zero-return 90% intervals: all seven methods, descriptive points\n"
        "No coverage guarantee; marginal adaptive coverage intervals supplied separately"
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "interval_tradeoffs")


def build(root: Path) -> None:
    output = root / "paper"
    frames = tables(root)
    table_dir = output / "tables"
    table_dir.mkdir(exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    figures(frames, output / "figures")
    inputs = [
        *TABLES.values(),
        "src/marketpulse/publication_assets.py",
        "scripts/publication_assets.py",
    ]
    manifest = {
        "inputs": {p: file_hash(root / p) for p in inputs},
        "outputs": {
            str(p.relative_to(root)): file_hash(p)
            for folder in ["tables", "figures"]
            for p in sorted((output / folder).glob("*"))
        },
        "table_rows": {name: len(frame) for name, frame in frames.items()},
        "figure_sources": {
            "point_skill": [TABLES["native_metrics"], TABLES["matched_metrics"]],
            "baseline_effects": [TABLES["original_comparisons"]],
            "interval_tradeoffs": [TABLES["interval_metrics"]],
        },
        "notes": "No fitting or artifact rewriting. Precision retained in CSV exports. "
        "Figure effects use source CIs; p columns retain the complete original family. "
        "PDF metadata timestamps excluded; binary graphics can vary by library/platform.",
    }
    (output / "evidence/assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(root)


def verify(root: Path) -> None:
    manifest = json.loads((root / "paper/evidence/assets_manifest.json").read_text())
    for path, expected in {**manifest["inputs"], **manifest["outputs"]}.items():
        assert file_hash(root / path) == expected, path
    for name, frame in tables(root).items():
        actual = pd.read_csv(root / f"paper/tables/{name}.csv")
        pd.testing.assert_frame_equal(actual, frame, check_exact=False, rtol=1e-13, atol=1e-15)
    print(json.dumps({"status": "passed", "tables": len(manifest["table_rows"]), "figures": 3}))
