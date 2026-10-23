"""Phase 10: full-budget intervening-fold expansion; original evidence is immutable."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from marketpulse.data.catalog import load_catalog
from marketpulse.experiments.benchmark import (
    environment_record,
    load_benchmark_settings,
    score_split,
)
from marketpulse.experiments.recovery import file_hash
from marketpulse.experiments.robustness_data import write_json
from marketpulse.experiments.walkforward import _plan, _slug, _with_regimes
from marketpulse.protocol import load_forecasting_protocol


def selected_fold_indices(count: int, quantiles: list[float]) -> list[int]:
    if count < 1 or not quantiles or any(q < 0 or q > 1 for q in quantiles):
        raise ValueError("invalid fold quantiles")
    return sorted({int(np.floor(q * (count - 1))) for q in quantiles})


def run_expansion(root: Path, output: Path, *, smoke: bool = False) -> None:
    design = json.loads((root / "configs/phase10.json").read_text())
    protocol = load_forecasting_protocol(root)
    settings = load_benchmark_settings(root)
    if smoke:
        settings = settings.for_smoke()
    tasks = _plan(root, load_catalog(root), protocol.horizons, protocol, settings, None)
    tasks = [
        task
        for task in tasks
        if task["fold_index"] in selected_fold_indices(task["n_folds"], design["fold_quantiles"])
    ]
    seeds = design["seeds"]
    if smoke:
        tasks, seeds = tasks[:1], seeds[:1]
    output.mkdir(parents=True, exist_ok=True)
    historical = json.loads((root / "results/walkforward/full/manifest.json").read_text())
    paths = list(historical["identity"]["inputs"])
    paths += [
        "configs/phase10.json",
        "src/marketpulse/experiments/expanded_seeds.py",
        "src/marketpulse/experiments/robustness_data.py",
    ]
    identity = {
        "inputs": {path: file_hash(root / path) for path in sorted(paths)},
        "mode": "smoke" if smoke else "full",
        "seeds": seeds,
        "tasks": [
            {"dataset": t["entry"].name, "horizon": t["horizon"], "fold": t["fold_index"]}
            for t in tasks
        ],
    }
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("multiseed inputs changed; use a new output directory")
    else:
        write_json(
            manifest,
            {
                "identity": identity,
                "environment": environment_record(root, "smoke" if smoke else "full"),
                "declared_at_utc": datetime.now(UTC).isoformat(),
                "expected_tasks": len(tasks) * len(seeds),
            },
        )
    for task in tasks:
        for seed in seeds:
            name = (
                f"{_slug(task['entry'].name)}_h{task['horizon']}_f{task['fold_index']:03d}_s{seed}"
            )
            marker = output / "runs" / f"{name}.json"
            csv = marker.with_suffix(".csv.gz")
            if marker.exists() or csv.exists():
                if not marker.exists() or not csv.exists():
                    raise ValueError(f"partial run preserved: {marker}")
                payload = json.loads(marker.read_text())
                if file_hash(csv) != payload["predictions_sha256"]:
                    raise ValueError(f"corrupt run: {csv}")
                print(f"reuse {name}", flush=True)
                continue
            started_at = datetime.now(UTC).isoformat()
            started = time.monotonic()
            print(f"train {name}", flush=True)
            features = task["features"]
            result = score_split(
                dataset_name=task["entry"].name,
                index_id=task["entry"].index_id,
                horizon=task["horizon"],
                feature_matrix=features.to_numpy(dtype=float),
                target_values=task["target"].to_numpy(dtype=float),
                target=task["target"],
                features=features,
                dates=pd.to_datetime(task["market"]["date"]),
                split=task["split"],
                settings=settings,
                seed=seed,
                models=tuple(design["models"]),
                fold_index=task["fold_index"],
                n_folds=task["n_folds"],
            )
            frame = _with_regimes(pd.DataFrame(result.pop("predictions")), features, task["split"])
            if len(frame) != 2 * len(task["split"].test):
                raise AssertionError("incomplete stochastic forecast set")
            frame["seed"] = seed
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = csv.with_suffix(".tmp")
            frame.to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
            temporary.replace(csv)
            result["execution"] = {
                **environment_record(root, "smoke" if smoke else "full"),
                "started_at_utc": started_at,
                "elapsed_seconds": time.monotonic() - started,
                "settings": asdict(settings),
            }
            result["predictions_sha256"] = file_hash(csv)
            write_json(marker, result)
    print(f"completed {len(tasks) * len(seeds)} tasks", flush=True)
