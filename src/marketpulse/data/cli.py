"""Command-line validation for the catalogued market datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marketpulse.data.catalog import build_catalog_report, validate_catalog
from marketpulse.data.exceptions import MarketDataError
from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import DatasetSummary, format_dataset_summary, validate_market_frame


def main(argv: list[str] | None = None) -> int:
    """Validate catalog datasets, or the CSV paths given on the command line."""

    parser = argparse.ArgumentParser(
        prog="marketpulse-validate",
        description=(
            "Validate historical market CSV files. With no paths, validate the "
            "datasets declared in configs/datasets.toml."
        ),
    )
    parser.add_argument("paths", nargs="*", type=Path, help="CSV files to validate.")
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Dataset name for a positional path. Repeat once per path, in order.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the directory that contains configs/datasets.toml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a JSON report to this path.",
    )
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except MarketDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.paths:
        summaries = _validate_paths(args.paths, args.name)
        report = {
            "datasets": [
                {"name": summary.dataset_name, "path": str(path), "summary": summary.to_dict()}
                for path, summary in zip(args.paths, summaries, strict=True)
            ]
        }
        blocks = [format_dataset_summary(summary) for summary in summaries]
    else:
        results = validate_catalog(args.root)
        report = build_catalog_report(args.root)
        blocks = []
        for result in results:
            blocks.append(format_dataset_summary(result.summary))
            if result.entry.documented_limitations:
                limitation_lines = "\n".join(
                    f"  - {item}" for item in result.entry.documented_limitations
                )
                blocks[-1] = (
                    f"{blocks[-1]}\nDocumented limitations:\n{limitation_lines}"
                )

    print("\n\n".join(blocks))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = any(
        dataset["summary"]["validation_status"] == "failed" for dataset in report["datasets"]
    )
    return 1 if failed else 0


def _validate_paths(paths: list[Path], names: list[str] | None) -> list[DatasetSummary]:
    if names is not None and len(names) != len(paths):
        raise SystemExit("--name must be supplied once for every positional CSV path.")
    summaries: list[DatasetSummary] = []
    for index, path in enumerate(paths):
        dataset_name = names[index] if names is not None else path.stem
        frame = load_market_csv(path, dataset_name=dataset_name)
        summaries.append(validate_market_frame(frame, dataset_name))
    return summaries
