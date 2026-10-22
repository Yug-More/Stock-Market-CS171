# Data

This directory holds the raw historical files used by MarketPulse. The loader reads them and does not modify them. Validation does not drop, fill, or reorder rows.

Derived features used by the original notebooks live under `notebooks/` and are documented at the bottom of this file. They are not a second raw source.

## Files

| File | Series | Rows | First session | Last session | Index id |
| --- | --- | --- | --- | --- | --- |
| `nasdaq100.csv` | NASDAQ-100 | 6,289 | 2000-01-03 | 2024-12-31 | `^NDX` |
| `SPX.csv` | S&P 500 | 23,323 | 1927-12-30 | 2020-11-04 | `^GSPC` |

The row counts and dates are measurements of these files. Both series are strictly increasing, have no duplicate timestamps, no missing OHLC values, no nonpositive prices, and no infinite values. Close is inside the high-low range on every row. There are no weekend rows.

The series do not cover the same years. Their overlap is 2000-01-03 through 2020-11-04. A comparison across indexes has to be limited to that overlap and has to use the same forecasting protocol. That comparison has not been run.

## NASDAQ-100

`nasdaq100.csv` is a yfinance-style export. The first three lines are

```text
Price,Close,High,Low,Open,Volume
Ticker,^NDX,^NDX,^NDX,^NDX,^NDX
Date,,,,,
```

The date is in the first field. The column name on that field is `Price`. `marketpulse.data.load_market_csv` detects this header, renames the field, and returns `date`, `open`, `high`, `low`, `close`, and `volume`.

The preprocessing notebook records:

```python
yf.download("^NDX", start="2000-01-01", end="2024-12-31")
```

The ticker row is `^NDX`, which agrees with that call. The repository does not store the download time or the yfinance version. Current yfinance treats `end` as exclusive, and this file includes the 2024-12-31 session, so running the recorded call again may not reproduce the file byte for byte. There is no adjusted-close column.

Calendar gaps longer than four days:

| Previous session | Next session | Calendar days |
| --- | --- | --- |
| 2001-09-10 | 2001-09-17 | 7 |
| 2006-12-29 | 2007-01-03 | 5 |
| 2012-10-26 | 2012-10-31 | 5 |

The seven-day gap is consistent with the post-September 11 closure. The 2012 gap is consistent with the Hurricane Sandy closure. A Friday-to-Monday weekend is three calendar days and is not listed. Volume is positive on every row. The close ranges from about 804.64 to about 22,096.66.

## S&P 500

`SPX.csv` has one header row: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.

The README and the S&P preprocessing notebook describe the file as a Kaggle download. No URL, version, or retrieval date is stored here, so that description cannot be checked from the repository. `Adj Close` equals `Close` on all 23,323 rows. The file therefore does not contain a separate dividend-adjusted series. For an index, `Close` is the index level in the file. This project does not claim a total-return adjustment that the file does not encode.

Volume is zero on 5,496 rows. The first positive volume is 1950-01-03. Zero volume is reported as a warning, not as an invalid price. Negative volume would be an error. The close ranges from 4.4 to about 3,580.84.

Ten gaps are longer than four calendar days. The longest is 1933-03-03 to 1933-03-15 (12 days), consistent with the 1933 bank holiday. The file also contains the September 2001 closure and the October 2012 closure. The other long gaps are five calendar days. They were not each matched to an exchange holiday calendar.

## Column definitions

| Canonical name | Meaning in these files |
| --- | --- |
| `date` | Session date, `YYYY-MM-DD`, with no timezone. |
| `open`, `high`, `low`, `close` | Index levels for that session. |
| `adj_close` | Present only for the S&P 500 file, and identical to `close` there. |
| `volume` | Reported volume. Zero in the early S&P 500 history. |

The loader also accepts a normal one-header CSV. It rejects a file that is missing `date`, `open`, `high`, `low`, or `close`, a date that is not `YYYY-MM-DD`, and a price field that is not numeric. Empty numeric fields become missing values and are reported by the validator. They are not filled.

Validation status is `passed` when there are no issues, `passed_with_warnings` when every issue is a warning, and `failed` when any issue is an error. Duplicate timestamps, unsorted timestamps, missing values, infinite values, and nonpositive prices are errors. Zero volume is a warning. A gap of more than four calendar days is recorded and does not, by itself, fail validation. Market closures are real.

`duplicate_timestamp_count` is the number of distinct timestamps that appear more than once. `duplicate_row_count` is the number of extra copies.

## How to reproduce the validation

From the repository root, after installing the package:

```bash
marketpulse-validate
```

That command reads `configs/datasets.toml`, prints a measured summary, and repeats the documented limitations from the catalog. The limitations are statements about provenance. They are not something the validator infers by guessing a vendor.

To write the same measurements to JSON:

```bash
marketpulse-validate --output results/data_validation.json
```

`results/data_validation.json` is the report generated from these files. `tests/test_cli.py` checks that the committed JSON matches a fresh run. The report has no wall-clock timestamp, so the same files produce the same JSON.

The original files are not rewritten by either command. Tests hash `nasdaq100.csv` and `SPX.csv` before and after loading.

## Acquisition rules for any later download

A new raw file needs all of the following in the catalog and in this document:

- The source and the series identifier, such as a ticker.
- The retrieval date.
- The client and version, or the vendor export name.
- The request parameters, including the date range and whether the vendor's end date is inclusive.
- A statement of whether the prices are adjusted.

No download is part of the current validation. Do not replace these two files in order to extend the sample until that metadata exists and the change is reviewed. The S&P 500 file ending in 2020 and the NASDAQ-100 file ending in 2024 is a real limitation, not a defect to be patched silently.

## Historical derived file

`notebooks/nasdaq100/nasdaq_processed.csv` was produced for the original notebooks. It has 6,263 rows, from 2000-02-01 through 2024-12-20, with log returns, moving averages, 20-day return volatility, RSI, a next-session return, a five-session return, and a direction label.

On the dates it contains, open, high, low, close, and volume match `nasdaq100.csv`. Those dates are one contiguous slice of the raw file. The slice ends on 2024-12-20 even though the raw file continues through 2024-12-31. Notebook 1 drops rows with missing indicators or forward targets. On a series that includes 2024-12-31, the five-session target is still defined on 2024-12-23. That row is not in the processed file. The processed tail matches the notebook rule applied to a series that ends on 2024-12-30. Use the raw file for new work.

The original README mentions `lag_1`, `lag_2`, and `lag_5`. Those columns are not in the processed file.

## Adjustment and quality limits

- Neither file documents a total-return adjustment. The S&P 500 adjusted close is a copy of the close.
- Early S&P 500 volume cannot be used as a feature without an explicit rule for the zeros.
- Gaps longer than a long weekend are consistent with known closures in the cases named above. They are not deleted.
- The two files must not be concatenated or compared as if they were the same sample.
