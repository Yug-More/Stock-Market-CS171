"""Checks against the committed market files.

These tests lock the loader and validator to the files in the repository.
They are not forecasting experiments.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import validate_market_frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_import_and_version() -> None:
    import marketpulse

    assert marketpulse.__version__ == "0.1.0"


def test_nasdaq100_file(repo_root: Path) -> None:
    path = repo_root / "data" / "nasdaq100.csv"
    digest = _sha256(path)
    frame = load_market_csv(path, dataset_name="NASDAQ-100")
    summary = validate_market_frame(frame, "NASDAQ-100")

    assert _sha256(path) == digest
    assert summary.row_count == 6289
    assert summary.start_date.isoformat() == "2000-01-03"
    assert summary.end_date.isoformat() == "2024-12-31"
    assert summary.columns == ("date", "open", "high", "low", "close", "volume")
    assert all(count == 0 for count in summary.missing_value_counts.values())
    assert summary.duplicate_timestamp_count == 0
    assert summary.duplicate_row_count == 0
    assert summary.zero_volume_count == 0
    assert summary.validation_status == "passed"
    assert [
        (gap.previous_date.isoformat(), gap.next_date.isoformat(), gap.calendar_days)
        for gap in summary.calendar_gaps
    ] == [
        ("2001-09-10", "2001-09-17", 7),
        ("2006-12-29", "2007-01-03", 5),
        ("2012-10-26", "2012-10-31", 5),
    ]


def test_sp500_file(repo_root: Path) -> None:
    path = repo_root / "data" / "SPX.csv"
    digest = _sha256(path)
    frame = load_market_csv(path, dataset_name="S&P 500")
    summary = validate_market_frame(frame, "S&P 500")

    assert _sha256(path) == digest
    assert summary.row_count == 23323
    assert summary.start_date.isoformat() == "1927-12-30"
    assert summary.end_date.isoformat() == "2020-11-04"
    assert "adj_close" in summary.columns
    assert all(count == 0 for count in summary.missing_value_counts.values())
    assert summary.duplicate_timestamp_count == 0
    assert summary.zero_volume_count == 5496
    assert summary.first_nonzero_volume_date.isoformat() == "1950-01-03"
    assert summary.validation_status == "passed_with_warnings"
    assert any(issue.code == "zero_volume" for issue in summary.issues)
    assert summary.calendar_gaps[0].previous_date.isoformat() == "1929-11-27"
    assert summary.calendar_gaps[0].calendar_days == 5
    assert max(gap.calendar_days for gap in summary.calendar_gaps) == 12
    assert len(summary.calendar_gaps) == 10
    assert (frame["close"] - frame["adj_close"]).abs().max() == pytest.approx(0)


def test_processed_nasdaq_matches_raw_slice(repo_root: Path) -> None:
    raw = load_market_csv(repo_root / "data" / "nasdaq100.csv")
    processed = load_market_csv(repo_root / "notebooks" / "nasdaq100" / "nasdaq_processed.csv")
    raw_dates = [value.date().isoformat() for value in raw["date"]]
    processed_dates = [value.date().isoformat() for value in processed["date"]]
    start = raw_dates.index(processed_dates[0])
    stop = raw_dates.index(processed_dates[-1]) + 1

    assert processed_dates == raw_dates[start:stop]
    for column in ("open", "high", "low", "close", "volume"):
        assert processed[column].tolist() == pytest.approx(raw[column].iloc[start:stop].tolist())
