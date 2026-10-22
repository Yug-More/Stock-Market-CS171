"""Loader tests use synthetic CSV fixtures only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from marketpulse.data.exceptions import (
    MarketDataFormatError,
    MissingColumnError,
    TimestampParseError,
)
from marketpulse.data.loader import load_market_csv
from tests.helpers import write_csv

_HEADER = "Date,Open,High,Low,Close,Volume\n"


def test_loads_standard_ohlcv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "prices.csv",
        _HEADER + "2020-01-02,10,11,9,10.5,100\n2020-01-03,10.5,12,10,11,110\n",
    )
    frame = load_market_csv(path, dataset_name="fixture")

    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame.loc[0, "close"] == pytest.approx(10.5)
    assert frame.loc[1, "volume"] == pytest.approx(110)
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_loads_yfinance_multiheader(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "nasdaq.csv",
        "Price,Close,High,Low,Open,Volume\n"
        "Ticker,^NDX,^NDX,^NDX,^NDX,^NDX\n"
        "Date,,,,,\n"
        "2020-01-02,100,101,99,100,1000\n"
        "2020-01-03,101,102,100,100.5,1100\n",
    )
    frame = load_market_csv(path)

    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert frame.loc[0, "open"] == pytest.approx(100)
    assert frame.loc[0, "close"] == pytest.approx(100)
    assert frame.loc[1, "high"] == pytest.approx(102)
    assert str(frame.loc[0, "date"].date()) == "2020-01-02"


def test_keeps_adjusted_close(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "adjusted.csv",
        "Date,Open,High,Low,Close,Adj Close,Volume\n2020-01-02,1,1,1,1,1.5,5\n",
    )
    frame = load_market_csv(path)
    assert "adj_close" in frame.columns
    assert frame.loc[0, "adj_close"] == pytest.approx(1.5)


def test_empty_file_is_malformed(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "empty.csv", "")
    with pytest.raises(MarketDataFormatError, match="empty"):
        load_market_csv(path)


def test_unreadable_csv_is_malformed(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "broken.csv", 'Date,Open,High,Low,Close\n"unterminated,1,1,1,1\n')
    with pytest.raises(MarketDataFormatError):
        load_market_csv(path)


def test_missing_file_is_malformed(tmp_path: Path) -> None:
    with pytest.raises(MarketDataFormatError, match="does not exist"):
        load_market_csv(tmp_path / "absent.csv")


def test_missing_required_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "partial.csv", "Date,Close\n2020-01-02,10\n")
    with pytest.raises(MissingColumnError) as caught:
        load_market_csv(path)
    assert "open" in caught.value.missing
    assert "high" in caught.value.missing
    assert "low" in caught.value.missing


def test_invalid_timestamps(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "bad_dates.csv",
        _HEADER + "not-a-date,1,1,1,1,1\n",
    )
    with pytest.raises(TimestampParseError) as caught:
        load_market_csv(path)
    assert caught.value.invalid_count == 1
    assert "not-a-date" in caught.value.examples


def test_non_numeric_price_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "text.csv", _HEADER + "2020-01-02,1,1,1,abc,1\n")
    with pytest.raises(MarketDataFormatError, match="non-numeric"):
        load_market_csv(path)


def test_does_not_reorder_or_drop_rows(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "order.csv",
        _HEADER + "2020-01-03,1,1,1,1,1\n2020-01-02,2,2,2,0,2\n",
    )
    frame = load_market_csv(path)
    assert len(frame) == 2
    assert str(frame.loc[0, "date"].date()) == "2020-01-03"
    assert frame.loc[1, "close"] == pytest.approx(0)
