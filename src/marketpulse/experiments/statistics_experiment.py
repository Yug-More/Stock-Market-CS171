"""Execute the declared native/matched and regime paired-loss comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketpulse.evaluation.regimes import stress_periods
from marketpulse.evaluation.reliability import compare_losses, holm_adjust, paired_losses
from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.robustness_data import (
    load_design,
    load_predictions,
    matched_membership,
    select_matched,
    write_json,
)


def run_statistics(root: Path, output: Path) -> None:
    design = load_design(root)
    frame = load_predictions(root)
    membership = matched_membership(frame)
    output.mkdir(parents=True, exist_ok=True)
    registry = [
        {
            "dataset": d,
            "horizon": h,
            "first": pair[0],
            "second": pair[1],
            "loss": "squared_error",
            "population": "native",
            "block_length": 20,
            "family": "18 retrospective primary follow-up comparisons",
        }
        for d in sorted(frame.dataset.unique())
        for h in (1, 5, 20)
        for pair in design["primary_pairs"]
    ]
    write_json(output / "primary_registry.json", registry)
    rows, regimes = [], []
    populations = (
        ("native", frame),
        ("endpoint_matched", select_matched(frame, membership, endpoints=True)),
    )
    for population, selected in populations:
        for (dataset, horizon), group in selected.groupby(["dataset", "horizon"], sort=True):
            for first, second in design["primary_pairs"]:
                for loss in ("squared_error", "absolute_error"):
                    timeline, a, b = paired_losses(
                        group, first, second, absolute=loss == "absolute_error"
                    )
                    for length in design["block_lengths"]:
                        result = compare_losses(
                            a,
                            b,
                            timeline.origin_position.to_numpy(),
                            length=length,
                            replicates=design["bootstrap_replicates"],
                            seed=design["bootstrap_seed"],
                            confidence=design["confidence"],
                            minimum_blocks=design["minimum_block_equivalents"],
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "horizon": int(horizon),
                                "first": first,
                                "second": second,
                                "loss": loss,
                                "population": population,
                                **result,
                            }
                        )
                    if population == "native" and loss == "squared_error":
                        masks = {
                            f"volatility_{r}": timeline.regime.eq(r).to_numpy()
                            for r in ("low", "medium", "high")
                        }
                        masks.update(
                            {p.key: timeline.stress.eq(p.key).to_numpy() for p in stress_periods()}
                        )
                        for name, mask in masks.items():
                            result = compare_losses(
                                a,
                                b,
                                timeline.origin_position.to_numpy(),
                                length=20,
                                replicates=design["bootstrap_replicates"],
                                seed=design["bootstrap_seed"],
                                confidence=design["confidence"],
                                minimum_blocks=design["minimum_block_equivalents"],
                                mask=mask,
                            )
                            regimes.append(
                                {
                                    "dataset": dataset,
                                    "horizon": int(horizon),
                                    "slice": name,
                                    "first": first,
                                    "second": second,
                                    **result,
                                }
                            )
    result = pd.DataFrame(rows)
    primary = (
        result.population.eq("native")
        & result.loss.eq("squared_error")
        & result.block_length.eq(20)
    )
    assert primary.sum() == 18
    result["primary_family"] = primary
    result.loc[primary, "holm_p_value"] = holm_adjust(result.loc[primary, "p_value"].to_numpy())
    result.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(regimes).to_csv(output / "regime_comparisons.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "environment": environment_record(root, "statistics"),
            "configuration": design,
            "primary_family_size": 18,
            "interpretation": "retrospective weak-dependence approximations; no independent replication",
        },
    )
