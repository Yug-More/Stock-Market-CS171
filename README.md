# MarketPulse

MarketPulse evaluates how reliably machine learning models forecast financial time series when the horizon, the market regime, and the data distribution change.

The project uses historical NASDAQ-100 and S&P 500 levels. The forecasting target is a future log return, not the price level itself. Models are compared with simple baselines under chronological validation. This repository does not claim that a model predicts the market better than those baselines. That comparison has not been rerun under the protocol in `docs/RESEARCH_DESIGN.md`.

## Research questions

1. Do linear regression, random forest, or a sequence LSTM beat a zero-return forecast and a historical-mean forecast for the next trading session?
2. Does skill relative to the zero-return forecast change at 5 and 20 trading sessions?
3. Does the ranking of models and baselines stay the same across walk-forward windows and volatility regimes?
4. What empirical coverage and width do chronological prediction intervals have on a later test period?

The hypotheses and the leakage rules are in `docs/RESEARCH_DESIGN.md`. The answers are not predetermined.

## Motivation

Short-horizon index returns are noisy. A chart of predicted prices can look accurate because the price level is persistent, even when the model is no better than predicting that the price will not change. Error also grows with the horizon because the target itself is larger, so a larger mean squared error is not evidence that a longer horizon is harder to forecast. MarketPulse separates those effects by scoring log returns against baselines, on chronological folds, with preprocessing fit only on past data.

## Original project

The repository began as a collaborative course project.

- **Yug More** built the NASDAQ-100 preprocessing, return-model, and price-prediction notebooks, and contributed the NASDAQ-100 file.
- **Srihan Cheemangunta** built the S&P 500 preprocessing and model notebooks, the follow-up analysis notebook, and contributed the S&P 500 file.

Both contributors wrote parts of the original README. The original notebooks are still in `notebooks/`. They studied linear regression, random forest, a length-1 LSTM, a 60-day LSTM, and a date-based gradient-boosting fit. The original write-up reported that more complex models forecast better than linear regression and that the S&P 500 was easier to predict than the NASDAQ-100.

Those statements are the course project's conclusions. `docs/REPOSITORY_AUDIT.md` explains why they are not treated as established results: the two indexes were not evaluated on the same target or the same dates, the LSTM test error was not stored, and the stored return errors were not compared with a zero-return forecast. This revision does not replace that write-up with a new accuracy claim.

## Current extension

The current code is a research foundation:

- A `marketpulse` package that loads the committed CSV files, including the yfinance header on the NASDAQ-100 file.
- Validation that reports duplicate timestamps, bad prices, missing and infinite values, and chronological order without deleting rows.
- Tested definitions of the forward log return, point-forecast metrics, and the training embargo for multi-day labels.
- An initial walk-forward protocol in `configs/forecasting.toml`.
- Tests, Ruff, and a GitHub Actions workflow that does not train models.

Training, walk-forward runs, regime analysis, and interval calibration are later phases. They are not implemented here.

## Data

| File | Series | Sessions | Range |
| --- | --- | --- | --- |
| `data/nasdaq100.csv` | NASDAQ-100 (`^NDX`) | 6,289 | 2000-01-03 to 2024-12-31 |
| `data/SPX.csv` | S&P 500 (`^GSPC`) | 23,323 | 1927-12-30 to 2020-11-04 |

The NASDAQ-100 file matches a yfinance export recorded in the preprocessing notebook. The S&P 500 file is described in the original notebooks as a Kaggle download. The repository does not contain a Kaggle URL or a retrieval timestamp. `Adj Close` in the S&P 500 file equals `Close` on every row, and 5,496 early volume values are zero. Details, gap lists, and the limits of those sources are in `data/README.md`.

The files are not interchangeable. Shared dates run from 2000-01-03 through 2020-11-04 only.

## Planned models and baselines

Baselines for the return task:

- Predict zero, which is the random-walk or persistence forecast in price space.
- Predict the mean return of the training window.

Models planned for the first benchmark:

- Linear regression.
- Random forest.
- A sequence LSTM whose window ends at the forecast origin.

Features will be built from information available at that origin. Raw price moving averages are not the default, because the audit found a large level shift between the original training and test periods. The planned features include lagged returns, rolling volatility, RSI, and scale-free moving-average distances.

## Planned evaluation

- Chronological walk-forward folds with an expanding training window.
- A validation window for hyperparameters and a later test window that is not used for model selection.
- Scalers fit on the training window only.
- Training labels whose future close falls inside the evaluation window are excluded.
- Horizons of 1, 5, and 20 trading sessions.
- MAE, RMSE, forecast bias, directional accuracy, and skill relative to the baselines.
- Volatility regimes defined from trailing volatility known at the forecast origin.
- Prediction intervals fit on chronological calibration data, with empirical coverage and width reported. The design does not claim an exchangeable conformal guarantee.

The fold lengths and the nominal coverage are in `configs/forecasting.toml`. The reasoning is in `docs/RESEARCH_DESIGN.md`.

## Repository layout

```text
configs/                 Dataset catalog and forecasting protocol
data/                    Raw NASDAQ-100 and S&P 500 files
docs/                    Audit and research design
notebooks/               Original course notebooks, unchanged
results/                 Measured validation report
src/marketpulse/         Research package
tests/                   Unit tests and checks on the real files
.github/workflows/       Lint, format, and test
```

`src/marketpulse/data` loads and validates prices. `targets`, `evaluation`, and `protocol` define the forecasting contract used by later experiments. There is no training loop yet, and there is no web application.

## Installation

Python 3.11 or newer is required. Use a virtual environment so the install does not change the system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra installs pytest and Ruff. It is enough to validate data and run the tests. The `research` extra adds scikit-learn, matplotlib, and PyTorch for later CPU training:

```bash
python -m pip install -e ".[research,dev]"
```

PyTorch is not imported by the current package and is not installed in CI.

## Validate the data

```bash
marketpulse-validate
```

The command prints, for each catalogued file, the row count, date range, columns, missing-value counts, duplicate counts, validation status, warnings, and calendar gaps longer than four days. It also prints the documented limitations from `configs/datasets.toml`.

```bash
marketpulse-validate --output results/data_validation.json
```

regenerates the committed JSON report. The same report is produced by `python -m marketpulse.data`.

On the current files the NASDAQ-100 validation status is `passed`. The S&P 500 status is `passed_with_warnings` because of the zero-volume rows. Neither file is modified.

## Tests and lint

```bash
ruff check .
ruff format --check .
pytest
```

Synthetic CSV fixtures cover malformed files, missing columns, bad timestamps, duplicate and unsorted dates, missing and infinite values, and nonpositive prices. Separate tests load the two real files and check the measured counts above. CI runs the same three commands on Python 3.12. It does not fit a model.

## Development status

| Item | Status |
| --- | --- |
| Audit of the original notebooks and claims | Written in `docs/REPOSITORY_AUDIT.md` |
| Research protocol | Written, not executed |
| Package install, data validation, tests, lint, CI | Implemented |
| Baseline and model training | Not run |
| Walk-forward evaluation | Not implemented |
| Regime and distribution-shift experiments | Not implemented |
| Prediction intervals | Not implemented |
| Results-driven research report | Not written |

No accuracy number in the original README should be read as a result of this extension.

## Later phases

**Phase 2.** Build causal features, fit the baselines and the three models under the walk-forward protocol, and save fold-level metrics with the git revision and seed.

**Phase 3.** Slice those results by volatility regime and by the predeclared historical windows, test whether rankings move, and evaluate chronological prediction intervals.

**Reporting.** Revise this README and write the research report after those runs exist. A negative result, including a baseline that wins, stays in the report.

## Contributors

- Yug More, yugmore20@gmail.com. NASDAQ-100 course notebooks and data. Research-extension design and engineering.
- Srihan Cheemangunta. S&P 500 course notebooks and data.
