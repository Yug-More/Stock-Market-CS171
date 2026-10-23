"""Small dispatcher for the independently executable Phase 6–8 research steps."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json


def run_step(root: Path, step: str, output: Path | None = None) -> None:
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    if step == "audit":
        from marketpulse.experiments.robustness_analysis import audit_historical

        destination = output or root / "results/robustness/audit"
        audit_historical(root, destination)
    elif step == "matched":
        from marketpulse.experiments.robustness_analysis import matched_analysis

        destination = output or root / "results/robustness/matched"
        matched_analysis(root, destination)
    elif step == "statistics":
        from marketpulse.experiments.statistics_experiment import run_statistics

        destination = output or root / "results/statistical_reliability"
        run_statistics(root, destination)
    elif step == "calibration":
        from marketpulse.experiments.calibration_experiment import run_calibration

        destination = output or root / "results/calibration"
        run_calibration(root, destination)
    elif step == "verify":
        from marketpulse.experiments.research_audit import audit_cycle

        audit_cycle(root)
        return
    elif step == "figures":
        from marketpulse.experiments.research_figures import write_figures

        write_figures(root)
        return
    elif step == "report":
        from marketpulse.experiments.research_report import write_report

        write_report(root)
        return
    else:
        raise ValueError(f"unknown step: {step}")
    paths = sorted((root / "src/marketpulse").rglob("*.py"))
    paths += sorted((root / "configs").glob("*.toml"))
    write_json(
        destination / "execution.json",
        {
            "step": step,
            "started_at_utc": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "environment": environment_record(root, step),
            "source_hashes": {str(p.relative_to(root)): file_hash(p) for p in paths},
            "source_revision_note": "Hashes identify all execution source files",
        },
    )
    write_json(destination / "checksums.json", inventory(destination))
