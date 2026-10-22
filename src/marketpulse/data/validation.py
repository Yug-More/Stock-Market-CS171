"""Quality checks for canonical historical market tables.

Validation reports problems. It does not drop, fill, or rewrite observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from marketpulse.data.exceptions import MissingColumnError
from marketpulse.data.loader import REQUIRED_COLUMNS

_PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
_OHLC_TOLERANCE = 1e-4


@dataclass(frozen=True)
class QualityIssue:
    """One data-quality finding."""

    code: str
    severity: str
    message: str
    count: int


@dataclass(frozen=True)
class CalendarGap:
    """A break between consecutive rows longer than a normal weekend."""

    previous_date: date
    next_date: date
    calendar_days: int


@dataclass(frozen=True)
class DatasetSummary:
    """Measured summary of one loaded market table."""

    dataset_name: str
    row_count: int
    start_date: date | None
    end_date: date | None
    columns: tuple[str, ...]
    missing_value_counts: dict[str, int]
    duplicate_timestamp_count: int
    duplicate_row_count: int
    validation_status: str
    issues: tuple[QualityIssue, ...]
    calendar_gaps: tuple[CalendarGap, ...]
    zero_volume_count: int
    first_nonzero_volume_date: date | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "start_date": _iso_or_none(self.start_date),
            "end_date": _iso_or_none(self.end_date),
            "columns": list(self.columns),
            "missing_value_counts": dict(self.missing_value_counts),
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "duplicate_row_count": self.duplicate_row_count,
            "validation_status": self.validation_status,
            "zero_volume_count": self.zero_volume_count,
            "first_nonzero_volume_date": _iso_or_none(self.first_nonzero_volume_date),
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "count": issue.count,
                }
                for issue in self.issues
            ],
            "calendar_gaps": [
                {
                    "previous_date": gap.previous_date.isoformat(),
                    "next_date": gap.next_date.isoformat(),
                    "calendar_days": gap.calendar_days,
                }
                for gap in self.calendar_gaps
            ],
        }


def validate_market_frame(frame: pd.DataFrame, dataset_name: str) -> DatasetSummary:
    """Validate a frame returned by :func:`marketpulse.data.loader.load_market_csv`.

    ``duplicate_timestamp_count`` is the number of distinct timestamps that occur
    more than once. ``duplicate_row_count`` is the number of rows beyond the
    first copy of those timestamps. Status is ``passed`` when there are no
    issues, ``passed_with_warnings`` when every issue is a warning, and
    ``failed`` when any issue is an error.
    """

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise MissingColumnError(missing_columns)
    if not dataset_name.strip():
        raise ValueError("dataset_name must be non-empty")

    table = frame.copy()
    issues: list[QualityIssue] = []
    row_count = int(len(table))
    if row_count == 0:
        issues.append(QualityIssue("empty_dataset", "error", "The table has no observations.", 0))

    duplicate_timestamp_count, duplicate_row_count = _duplicate_counts(table)
    if duplicate_timestamp_count:
        issues.append(
            QualityIssue(
                "duplicate_timestamps",
                "error",
                (
                    f"{duplicate_timestamp_count} timestamp(s) occur more than once "
                    f"({duplicate_row_count} extra row(s))."
                ),
                duplicate_timestamp_count,
            )
        )

    if row_count > 1 and not _is_strictly_increasing(table["date"]):
        issues.append(
            QualityIssue(
                "chronological_order",
                "error",
                "Timestamps are not strictly increasing in file order.",
                row_count,
            )
        )

    checked_columns = [
        column for column in (*_PRICE_COLUMNS, "volume") if column in table.columns
    ]
    missing_counts = {
        column: int(table[column].isna().sum()) for column in ("date", *checked_columns)
    }
    missing_total = sum(missing_counts.values())
    if missing_total:
        detail = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items() if count
        )
        issues.append(
            QualityIssue(
                "missing_values",
                "error",
                f"Missing values: {detail}.",
                missing_total,
            )
        )

    infinite_counts = _infinite_counts(table, checked_columns)
    infinite_total = sum(infinite_counts.values())
    if infinite_total:
        detail = ", ".join(f"{column}={count}" for column, count in infinite_counts.items() if count)
        issues.append(
            QualityIssue(
                "infinite_values",
                "error",
                f"Infinite values: {detail}.",
                infinite_total,
            )
        )

    nonpositive_counts = _nonpositive_counts(table)
    nonpositive_total = sum(nonpositive_counts.values())
    if nonpositive_total:
        detail = ", ".join(
            f"{column}={count}" for column, count in nonpositive_counts.items() if count
        )
        issues.append(
            QualityIssue(
                "nonpositive_prices",
                "error",
                f"Nonpositive prices: {detail}.",
                nonpositive_total,
            )
        )

    if "volume" in table.columns:
        negative_volume = int((table["volume"] < 0).sum())
        if negative_volume:
            issues.append(
                QualityIssue(
                    "negative_volume",
                    "error",
                    f"Volume is negative on {negative_volume} row(s).",
                    negative_volume,
                )
            )

    if {"high", "low"}.issubset(table.columns):
        comparable = table["high"].notna() & table["low"].notna()
        high_below_low = int((table.loc[comparable, "high"] < table.loc[comparable, "low"]).sum())
        if high_below_low:
            issues.append(
                QualityIssue(
                    "high_below_low",
                    "error",
                    f"High is below low on {high_below_low} row(s).",
                    high_below_low,
                )
            )

    if {"close", "high", "low"}.issubset(table.columns):
        outside = _close_outside_range(table)
        if outside:
            issues.append(
                QualityIssue(
                    "close_outside_range",
                    "error",
                    f"Close is outside the high-low range on {outside} row(s).",
                    outside,
                )
            )

    zero_volume_count = 0
    if "volume" in table.columns:
        zero_volume_count = int((table["volume"] == 0).sum())
        if zero_volume_count:
            issues.append(
                QualityIssue(
                    "zero_volume",
                    "warning",
                    f"Volume is zero on {zero_volume_count} row(s).",
                    zero_volume_count,
                )
            )

    strictly_increasing = row_count > 0 and _is_strictly_increasing(table["date"])
    gaps = _calendar_gaps(table) if strictly_increasing else ()
    start_date, end_date = _endpoints(table)
    return DatasetSummary(
        dataset_name=dataset_name,
        row_count=row_count,
        start_date=start_date,
        end_date=end_date,
        columns=tuple(str(column) for column in table.columns),
        missing_value_counts=missing_counts,
        duplicate_timestamp_count=duplicate_timestamp_count,
        duplicate_row_count=duplicate_row_count,
        validation_status=_status(issues),
        issues=tuple(issues),
        calendar_gaps=gaps,
        zero_volume_count=zero_volume_count,
        first_nonzero_volume_date=_first_nonzero_volume_date(table),
    )


def format_dataset_summary(summary: DatasetSummary) -> str:
    """Render a summary for the command-line report."""

    missing = summary.missing_value_counts
    missing_text = (
        "none"
        if not any(missing.values())
        else ", ".join(f"{column}={count}" for column, count in missing.items() if count)
    )
    lines = [
        f"Dataset: {summary.dataset_name}",
        f"Rows: {summary.row_count}",
        f"Start: {_iso(summary.start_date)}",
        f"End: {_iso(summary.end_date)}",
        f"Columns: {', '.join(summary.columns)}",
        f"Missing values: {missing_text}",
        f"Duplicate timestamps: {summary.duplicate_timestamp_count}",
        f"Duplicate extra rows: {summary.duplicate_row_count}",
        f"Zero-volume rows: {summary.zero_volume_count}",
        f"First nonzero volume: {_iso(summary.first_nonzero_volume_date)}",
        f"Validation: {summary.validation_status}",
    ]
    if summary.issues:
        lines.append("Issues:")
        lines.extend(
            f"  [{issue.severity}] {issue.code}: {issue.message}" for issue in summary.issues
        )
    else:
        lines.append("Issues: none")
    lines.append(f"Calendar gaps over 4 days: {len(summary.calendar_gaps)}")
    lines.extend(
        (
            f"  {gap.previous_date.isoformat()} -> {gap.next_date.isoformat()} "
            f"({gap.calendar_days} calendar days)"
        )
        for gap in summary.calendar_gaps
    )
    return "\n".join(lines)


def _duplicate_counts(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty:
        return 0, 0
    duplicate_timestamp_count = int(frame.loc[frame["date"].duplicated(keep=False), "date"].nunique())
    duplicate_row_count = int(frame["date"].duplicated(keep="first").sum())
    return duplicate_timestamp_count, duplicate_row_count


def _is_strictly_increasing(values: pd.Series) -> bool:
    if len(values) <= 1:
        return True
    timestamps = values.to_numpy()
    return bool(np.all(timestamps[1:] > timestamps[:-1]))


def _infinite_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        numeric = frame[column].to_numpy(dtype=float)
        counts[column] = int(np.isinf(numeric).sum())
    return counts


def _nonpositive_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in _PRICE_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column]
        counts[column] = int(((values.notna()) & (values <= 0)).sum())
    return counts


def _close_outside_range(frame: pd.DataFrame) -> int:
    comparable = frame["close"].notna() & frame["high"].notna() & frame["low"].notna()
    subset = frame.loc[comparable]
    outside = (subset["close"] > subset["high"] + _OHLC_TOLERANCE) | (
        subset["close"] < subset["low"] - _OHLC_TOLERANCE
    )
    return int(outside.sum())


def _calendar_gaps(frame: pd.DataFrame) -> tuple[CalendarGap, ...]:
    if len(frame) < 2:
        return ()
    dates = [_as_date(value) for value in frame["date"]]
    gaps: list[CalendarGap] = []
    for index in range(len(dates) - 1):
        previous = dates[index]
        current = dates[index + 1]
        calendar_days = (current - previous).days
        if calendar_days > 4:
            gaps.append(CalendarGap(previous, current, calendar_days))
    return tuple(gaps)


def _endpoints(frame: pd.DataFrame) -> tuple[date | None, date | None]:
    if frame.empty:
        return None, None
    first = frame["date"].iloc[0]
    last = frame["date"].iloc[-1]
    if pd.isna(first) or pd.isna(last):
        return None, None
    return _as_date(first), _as_date(last)


def _first_nonzero_volume_date(frame: pd.DataFrame) -> date | None:
    if frame.empty or "volume" not in frame.columns:
        return None
    positive = frame["volume"].notna() & (frame["volume"] > 0)
    if not bool(positive.any()):
        return None
    return _as_date(frame.loc[positive, "date"].iloc[0])


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    timestamp = pd.Timestamp(value)
    return timestamp.date()


def _iso(value: date | None) -> str:
    if value is None:
        return "none"
    return value.isoformat()


def _iso_or_none(value: date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _status(issues: list[QualityIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "failed"
    if issues:
        return "passed_with_warnings"
    return "passed"
