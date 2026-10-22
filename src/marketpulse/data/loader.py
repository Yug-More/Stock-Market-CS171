"""Load historical OHLCV CSV files without altering the source data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from marketpulse.data.exceptions import MarketDataFormatError, MissingColumnError, TimestampParseError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")
OPTIONAL_COLUMNS = ("adj_close", "volume")
CANONICAL_ORDER = ("date", "open", "high", "low", "close", "adj_close", "volume")
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
_ALIASES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adj close": "adj_close",
}


def load_market_csv(path: str | Path, *, dataset_name: str | None = None) -> pd.DataFrame:
    """Load a market CSV into a canonical table.

    Rows are not dropped, sorted, or interpolated. The returned frame preserves
    file order. ``dataset_name`` is accepted so callers can log a stable name;
    it does not change parsing.

    Two layouts are recognized. A standard file has one header row. A yfinance
    export has a ticker row and a second ``Date`` row under the column labels,
    and may store the date index under the name ``Price``.
    """

    source = Path(path)
    if dataset_name is not None:
        logger.info("Loading %s from %s", dataset_name, source)
    frame = _read_table(source)
    frame = _canonicalize_columns(frame)
    _require_columns(frame)
    frame = _parse_timestamps(frame)
    frame = _parse_numeric_columns(frame)
    ordered = [column for column in CANONICAL_ORDER if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *extras]].reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MarketDataFormatError(f"Market data file does not exist: {path}")
    try:
        header = _header_lines(path)
    except UnicodeDecodeError as exc:
        raise MarketDataFormatError(f"Could not read {path} as UTF-8 text") from exc

    skiprows = [1, 2] if _is_yfinance_multiheader(header) else None
    try:
        frame = pd.read_csv(path, skiprows=skiprows)
    except pd.errors.EmptyDataError as exc:
        raise MarketDataFormatError(f"CSV {path} is empty") from exc
    except pd.errors.ParserError as exc:
        raise MarketDataFormatError(f"Could not parse CSV {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise MarketDataFormatError(f"Could not read {path} as UTF-8 text") from exc

    if frame.empty or len(frame.columns) == 0:
        raise MarketDataFormatError(f"CSV {path} contains no observations")

    if skiprows is not None and len(frame.columns) > 0:
        columns = [str(column) for column in frame.columns]
        if columns[0] == "Price" and "Date" not in columns:
            columns[0] = "Date"
            frame.columns = columns
        logger.info("Parsed yfinance multi-row header in %s", path)
    return frame


def _header_lines(path: Path, count: int = 3) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for _ in range(count):
            line = handle.readline()
            if line == "":
                break
            lines.append(line.rstrip("\r\n"))
    return lines


def _is_yfinance_multiheader(lines: list[str]) -> bool:
    if len(lines) < 3:
        return False
    return lines[1].startswith("Ticker,") and lines[2].startswith("Date,")


def _canonical_name(column: object) -> str:
    text = " ".join(str(column).strip().lower().replace("_", " ").split())
    return _ALIASES.get(text, text.replace(" ", "_"))


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    names = [_canonical_name(column) for column in frame.columns]
    if len(names) != len(set(names)):
        raise MarketDataFormatError(f"Duplicate column names after normalization: {names}")
    renamed = frame.copy()
    renamed.columns = names
    return renamed


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise MissingColumnError(missing)


def _parse_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    date_text = frame["date"].astype(str).str.strip()
    parsed = pd.to_datetime(date_text, errors="coerce", format="%Y-%m-%d")
    invalid = parsed.isna()
    if invalid.any():
        examples = date_text.loc[invalid].head(5).tolist()
        raise TimestampParseError(int(invalid.sum()), examples)
    parsed_frame = frame.copy()
    parsed_frame["date"] = parsed
    return parsed_frame


def _parse_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.copy()
    for column in _NUMERIC_COLUMNS:
        if column not in parsed.columns:
            continue
        parsed[column] = _coerce_numeric(parsed[column], column)
    return parsed


def _coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    text = series.astype(str).str.strip()
    missing = series.isna() | text.eq("") | text.str.lower().eq("nan")
    converted = pd.to_numeric(text.mask(missing), errors="coerce")
    failed = ~missing & converted.isna()
    if bool(failed.any()):
        examples = text.loc[failed].head(5).tolist()
        raise MarketDataFormatError(
            f"Column '{column}' contains non-numeric values: {examples}"
        )
    return converted.astype("float64")
