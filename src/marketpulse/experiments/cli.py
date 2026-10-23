"""Command line for the terminal-split forecasting benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marketpulse.data.catalog import find_project_root, load_catalog
from marketpulse.data.loader import load_market_csv
from marketpulse.experiments.benchmark import (
    ALL_MODELS,
    environment_record,
    load_benchmark_settings,
    run_index_horizon,
    write_benchmark_artifacts,
)
from marketpulse.protocol import ForecastingProtocol, load_forecasting_protocol


def main(argv: list[str] | None = None) -> int:
    """Run the smoke check or the full terminal-split benchmark."""

    parser = argparse.ArgumentParser(
        prog="marketpulse-benchmark",
        description=(
            "Evaluate return forecasts on a terminal chronological split. "
            "The smoke mode is a short wiring check. It is not a benchmark result."
        ),
    )
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = find_project_root(args.root)
    protocol = load_forecasting_protocol(root)
    settings = load_benchmark_settings(root)
    catalogs = load_catalog(root)
    if args.mode == "smoke":
        settings = settings.for_smoke()
        protocol = ForecastingProtocol(
            schema_version=protocol.schema_version,
            primary_target=protocol.primary_target,
            horizons=(protocol.horizons[0],),
            seed=protocol.seed,
            minimum_training_origins=protocol.minimum_training_origins,
            validation_window=protocol.validation_window,
            test_window=protocol.test_window,
            step=protocol.step,
            nominal_coverage=protocol.nominal_coverage,
        )
        catalogs = catalogs[:1]

    results = []
    for entry in catalogs:
        frame = load_market_csv(root / entry.path, dataset_name=entry.name)
        for horizon in protocol.horizons:
            print(f"{args.mode}: {entry.name} horizon {horizon}", flush=True)
            results.append(
                run_index_horizon(
                    frame,
                    dataset_name=entry.name,
                    index_id=entry.index_id,
                    horizon=horizon,
                    protocol=protocol,
                    settings=settings,
                    models=ALL_MODELS,
                )
            )
    payload = {
        "mode": args.mode,
        "environment": environment_record(root, args.mode),
        "protocol": {
            "horizons": list(protocol.horizons),
            "seed": protocol.seed,
            "minimum_training_origins": protocol.minimum_training_origins,
            "validation_window": protocol.validation_window,
            "test_window": protocol.test_window,
            "lookback": settings.lookback,
        },
        "results": [
            {key: value for key, value in result.items() if key != "predictions"}
            for result in results
        ],
        "predictions": [row for result in results for row in result["predictions"]],
    }
    write_benchmark_artifacts(args.output, payload)
    (args.output / "environment.json").write_text(
        json.dumps(payload["environment"], indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)
    return 0
