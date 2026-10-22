"""Validation tests use synthetic tables and do not alter source files."""

from __future__ import annotations

from pathlib import Path

from marketpulse.data.loader import load_market_csv
from marketpulse.data.validation import validate_market_frame
from tests.helpers import write_csv

_HEADER = "Date,Open,High,Low,Close,Volume\n"


def _frame(tmp_path: Path, body: str, name: str = "fixture.csv"):
    return load_market_csv(write_csv(tmp_path / name, _HEADER + body))


def test_summary_for_valid_rows(tmp_path: Path) -> None:
    frame = _frame(
        tmp_path,
        "2020-01-02,10,11,9,10,5\n2020-01-03,10,12,10,11,6\n2020-01-06,11,12,10,11.5,7\n",
    )
    summary = validate_market_frame(frame, "Fixture")

    assert summary.dataset_name == "Fixture"
    assert summary.row_count == 3
    assert summary.start_date.isoformat() == "2020-01-02"
    assert summary.end_date.isoformat() == "2020-01-06"
    assert summary.columns == ("date", "open", "high", "low", "close", "volume")
    assert summary.missing_value_counts == {
        "date": 0,
        "open": 0,
        "high": 0,
        "low": 0,
        "close": 0,
        "volume": 0,
    }
    assert summary.duplicate_timestamp_count == 0
    assert summary.duplicate_row_count == 0
    assert summary.validation_status == "passed"
    assert summary.issues == ()
    assert summary.calendar_gaps == ()
    assert summary.zero_volume_count == 0
    payload = summary.to_dict()
    assert payload["start_date"] == "2020-01-02"
    assert payload["validation_status"] == "passed"


def test_duplicate_timestamps(tmp_path: Path) -> None:
    frame = _frame(
        tmp_path,
        "2020-01-02,1,2,1,1,1\n2020-01-02,1,2,1,1.5,2\n2020-01-03,1,2,1,1.6,3\n",
    )
    summary = validate_market_frame(frame, "duplicates")
    assert summary.duplicate_timestamp_count == 1
    assert summary.duplicate_row_count == 1
    assert summary.validation_status == "failed"
    assert any(issue.code == "duplicate_timestamps" for issue in summary.issues)


def test_incorrect_chronological_order_is_preserved_and_reported(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-03,1,2,1,1,1\n2020-01-02,1,2,1,1,1\n")
    assert str(frame.loc[0, "date"].date()) == "2020-01-03"
    summary = validate_market_frame(frame, "unsorted")
    assert summary.validation_status == "failed"
    assert any(issue.code == "chronological_order" for issue in summary.issues)
    assert summary.duplicate_timestamp_count == 0


def test_missing_numerical_values(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-02,1,2,,1,1\n")
    summary = validate_market_frame(frame, "missing")
    assert len(frame) == 1
    assert summary.missing_value_counts["low"] == 1
    assert summary.validation_status == "failed"
    assert any(issue.code == "missing_values" for issue in summary.issues)


def test_nonpositive_prices_are_retained(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-02,1,2,1,0,1\n2020-01-03,1,2,0.5,-1,1\n")
    summary = validate_market_frame(frame, "nonpositive")
    assert len(frame) == 2
    assert summary.validation_status == "failed"
    issue = next(issue for issue in summary.issues if issue.code == "nonpositive_prices")
    assert issue.count == 2
    assert "close=2" in issue.message


def test_infinite_values(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-02,1,2,1,inf,1\n")
    summary = validate_market_frame(frame, "infinite")
    assert summary.validation_status == "failed"
    assert any(issue.code == "infinite_values" and issue.count == 1 for issue in summary.issues)


def test_zero_volume_is_a_warning(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-02,1,2,1,1,0\n2020-01-03,1,2,1,1,5\n")
    summary = validate_market_frame(frame, "volume")
    assert summary.validation_status == "passed_with_warnings"
    assert summary.zero_volume_count == 1
    assert summary.first_nonzero_volume_date.isoformat() == "2020-01-03"
    assert any(
        issue.code == "zero_volume" and issue.severity == "warning" for issue in summary.issues
    )


def test_long_calendar_gap_is_reported_without_failing(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-02,1,2,1,1,1\n2020-01-10,1,2,1,1,1\n")
    summary = validate_market_frame(frame, "gap")
    assert summary.validation_status == "passed"
    assert len(summary.calendar_gaps) == 1
    gap = summary.calendar_gaps[0]
    assert gap.previous_date.isoformat() == "2020-01-02"
    assert gap.next_date.isoformat() == "2020-01-10"
    assert gap.calendar_days == 8


def test_weekend_is_not_a_quality_gap(tmp_path: Path) -> None:
    frame = _frame(tmp_path, "2020-01-03,1,2,1,1,1\n2020-01-06,1,2,1,1,1\n")
    summary = validate_market_frame(frame, "weekend")
    assert summary.calendar_gaps == ()
    assert summary.validation_status == "passed"
