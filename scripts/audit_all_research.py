"""Re-audit all saved studies, preserving logs in a new Phase 12 audit directory."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from marketpulse.experiments.benchmark import environment_record
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results/phase12/audits"
    if output.exists():
        raise ValueError("preserve previous audit logs")
    output.mkdir(parents=True)
    historical = root / "results/walkforward/full"
    with tarfile.open(historical / "fold_predictions.tar.gz") as archive:
        archive.extractall(historical, filter="data")
    commands = [
        ["scripts/verify_walkforward.py"],
        ["scripts/research_cycle.py", "verify"],
        ["scripts/audit_phase9.py"],
        ["scripts/phase10.py", "verify"],
        ["scripts/phase11.py", "verify"],
        ["scripts/phase11_diagnostics.py", "--verify"],
    ]
    records = []
    for i, args in enumerate(commands):
        print("audit", *args, flush=True)
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, *args], cwd=root, text=True, capture_output=True, check=False
        )
        log = output / f"audit_{i + 1}.log"
        log.write_text(result.stdout + result.stderr)
        records.append(
            {
                "command": ["python", *args],
                "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "log": log.name,
                "sha256": file_hash(log),
            }
        )
        if result.returncode:
            write_json(output / "failure.json", records)
            raise RuntimeError(f"audit failed: {args}; inspect {log}")
    preserved = [
        "data",
        "notebooks",
        "results/benchmarks",
        "results/walkforward",
        "results/robustness",
        "results/calibration",
        "results/statistical_reliability",
    ]
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "1e4b2f373a7b12a983ca457f32d1e18c870c21e9",
            "--",
            *preserved,
        ],
        cwd=root,
        text=True,
    )
    assert not diff, diff
    write_json(
        output / "manifest.json",
        {
            "environment": environment_record(root, "synthesis_audit"),
            "audits": records,
            "preserved_paths": preserved,
            "baseline": "1e4b2f373a7b12a983ca457f32d1e18c870c21e9",
            "historical_tree_unchanged": True,
            "source_sha256": file_hash(Path(__file__)),
        },
    )
    print(json.dumps({"passed_audits": len(records), "historical_tree_unchanged": True}))


if __name__ == "__main__":
    main()
