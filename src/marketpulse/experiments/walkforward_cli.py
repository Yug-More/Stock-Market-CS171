"""Command line for the expanding-window walk-forward study."""

from __future__ import annotations

import argparse
from pathlib import Path

from marketpulse.data.catalog import find_project_root
from marketpulse.experiments.benchmark import load_benchmark_settings
from marketpulse.experiments.walkforward import run_walk_forward
from marketpulse.protocol import load_forecasting_protocol


def main(argv: list[str] | None = None) -> int:
    """Run the smoke check or the full walk-forward and regime study."""

    parser = argparse.ArgumentParser(
        prog="marketpulse-walkforward",
        description=(
            "Refit the five forecasts on every expanding window. "
            "Smoke mode scores two NASDAQ-100 one-session folds and is not a result."
        ),
    )
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = find_project_root(args.root)
    run_walk_forward(
        root,
        mode=args.mode,
        output_dir=args.output,
        protocol=load_forecasting_protocol(root),
        settings=load_benchmark_settings(root),
    )
    return 0
