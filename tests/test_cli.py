"""Command-line validation tests."""

from __future__ import annotations

import json
from pathlib import Path

from marketpulse.data.cli import main
from tests.helpers import write_csv


def test_cli_validates_catalog(repo_root: Path, capsys) -> None:
    exit_code = main(["--root", str(repo_root)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dataset: NASDAQ-100" in captured.out
    assert "Dataset: S&P 500" in captured.out
    assert "Validation: passed" in captured.out
    assert "Validation: passed_with_warnings" in captured.out
    assert "Rows: 6289" in captured.out
    assert "Rows: 23323" in captured.out


def test_cli_writes_catalog_report(repo_root: Path, tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    exit_code = main(["--root", str(repo_root), "--output", str(destination)])
    report = json.loads(destination.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["package_version"] == "0.1.0"
    assert [item["name"] for item in report["datasets"]] == ["NASDAQ-100", "S&P 500"]
    assert report["datasets"][0]["summary"]["row_count"] == 6289
    assert report["datasets"][1]["summary"]["zero_volume_count"] == 5496


def test_cli_returns_failure_for_a_bad_file(tmp_path: Path, capsys) -> None:
    path = write_csv(tmp_path / "empty.csv", "")
    exit_code = main([str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "empty" in captured.err.lower()


def test_committed_validation_report_matches_catalog(repo_root: Path) -> None:
    from marketpulse.data.catalog import build_catalog_report

    committed = json.loads(
        (repo_root / "results" / "data_validation.json").read_text(encoding="utf-8")
    )
    assert committed == build_catalog_report(repo_root)
