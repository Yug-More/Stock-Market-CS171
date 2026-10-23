"""Materialize interval records and coverage diagnostics without refitting point models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketpulse.evaluation.calibration import chronological_intervals, interval_metrics
from marketpulse.evaluation.regimes import stress_periods
from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import load_design, load_predictions, write_json
from marketpulse.experiments.walkforward import _slug


def run_calibration(root: Path, output: Path) -> None:
    design = load_design(root)
    frame = load_predictions(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(exist_ok=True)
    rows, files = [], []
    for (dataset, horizon, model), group in frame.groupby(
        ["dataset", "horizon", "model"], sort=True
    ):
        eligible_reference = None
        for method in design["calibration_methods"]:
            print(f"calibrate {dataset} h{horizon} {model} {method}", flush=True)
            records = chronological_intervals(
                group,
                method=method,
                levels=design["coverage_levels"],
                minimum=design["calibration_minimum"],
                window=design["calibration_trailing"],
            )
            if eligible_reference is None:
                eligible_reference = records.eligible.to_numpy()
            elif not (eligible_reference == records.eligible.to_numpy()).all():
                raise AssertionError("method comparison populations differ")
            path = output / "records" / f"{_slug(dataset)}_h{horizon}_{model}_{method}.csv.gz"
            temporary = path.with_suffix(".tmp")
            records.to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
            temporary.replace(path)
            files.append(
                {
                    "dataset": dataset,
                    "horizon": int(horizon),
                    "model": model,
                    "method": method,
                    "path": str(path.relative_to(output)),
                    "sha256": file_hash(path),
                    "rows": len(records),
                }
            )
            slices = {"overall": records}
            slices.update(
                {f"volatility_{r}": records[records.regime == r] for r in ("low", "medium", "high")}
            )
            slices.update({p.key: records[records.stress == p.key] for p in stress_periods()})
            for name, part in slices.items():
                for level in design["coverage_levels"]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "horizon": int(horizon),
                            "model": model,
                            "method": method,
                            "slice": name,
                            "status": "available" if len(part) else "unavailable",
                            **interval_metrics(part, level),
                        }
                    )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "configuration": design,
            "files": files,
            "environment": environment_record(root, "calibration"),
            "evaluation": "same eligible origins; online matured OOS residuals",
            "coverage_guarantee": "none under market dependence and distribution shift",
        },
    )
