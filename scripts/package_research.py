"""Inventory the completed research bundle without changing scientific outputs."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from marketpulse.data.catalog import load_catalog
from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import inventory, write_json


def main() -> None:
    import torch

    root = Path(__file__).resolve().parents[1]
    folders = [
        root / "results" / name for name in ("robustness", "statistical_reliability", "calibration")
    ]
    runs = [json.loads(path.read_text()) for path in (folders[0] / "multiseed/runs").glob("*.json")]
    if len(runs) != 90:
        raise ValueError("complete the declared 90 seed tasks before packaging")
    inputs = {
        "recorded_during": "post-execution independent audit, not retroactive run metadata",
        "predictions": {
            "path": "results/walkforward/full/predictions.csv.gz",
            "sha256": file_hash(root / "results/walkforward/full/predictions.csv.gz"),
        },
        "datasets": {entry.path: file_hash(root / entry.path) for entry in load_catalog(root)},
        "configuration_sha256": file_hash(root / "configs/robustness.toml"),
    }
    write_json(
        folders[0] / "study_environment.json",
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "environment": environment_record(root, "phase678_delivery"),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "workstation_reference": json.loads(
                (root / "results/walkforward/full/recovery_hardware.json").read_text()
            ),
            "seed_tasks": len(runs),
            "training_runtime_seconds": sum(r["execution"]["elapsed_seconds"] for r in runs),
            "training_revisions": sorted({r["execution"]["git_sha"] for r in runs}),
            "input_references": inputs,
        },
    )
    for name in (
        "robustness/audit",
        "robustness/matched",
        "statistical_reliability",
        "calibration",
    ):
        directory = root / "results" / name
        write_json(directory / "audited_inputs.json", inputs)
        write_json(directory / "checksums.json", inventory(directory))
    files = [
        path
        for folder in folders
        for path in folder.rglob("*")
        if path.is_file() and path.name != "study_checksums.json"
    ]
    checksums = {str(path.relative_to(root)): file_hash(path) for path in sorted(files)}
    write_json(folders[0] / "study_checksums.json", checksums)
    print(f"Inventoried {len(checksums)} completed research files")


if __name__ == "__main__":
    main()
