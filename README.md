# MarketPulse

MarketPulse measures how reliably machine learning models forecast financial time series when the horizon, the market period, and the data distribution change.

The forecasting target is a forward log return, `log(Close[t + h] / Close[t])`, at horizons of 1, 5, and 20 trading sessions. Models are compared with a zero-return forecast and a historical-mean forecast on the same origins. This repository reports one terminal chronological test for each series and horizon. It does not claim a trading strategy, and it is not a peer-reviewed study.

## Research objective

The question is whether a model forecast is more accurate than simple baselines when the horizon and the evaluation window change, under a protocol that keeps future prices out of training. The hypotheses and the later walk-forward, regime, and interval designs are in `docs/RESEARCH_DESIGN.md`. The terminal-split evidence now available is in `docs/BENCHMARK_RESULTS.md`.

## Original project

The repository began as a collaborative course project.

- **Yug More** built the NASDAQ-100 preprocessing, return-model, and price-prediction notebooks, and contributed the NASDAQ-100 file.
- **Srihan Cheemangunta** built the S&P 500 preprocessing and model notebooks, the follow-up analysis notebook, and contributed the S&P 500 file.

Both contributors wrote parts of the original README. The original notebooks remain in `notebooks/`. They studied linear regression, random forest, a length-1 LSTM, a 60-day LSTM, and a date-based gradient-boosting fit. The audit of those notebooks is `docs/REPOSITORY_AUDIT.md`. Stored notebook errors are not results of the benchmark in this README.

## Data

| File | Series | Sessions | Range |
| --- | --- | ---: | --- |
| `data/nasdaq100.csv` | NASDAQ-100 (`^NDX`) | 6,289 | 2000-01-03 to 2024-12-31 |
| `data/SPX.csv` | S&P 500 (`^GSPC`) | 23,323 | 1927-12-30 to 2020-11-04 |

The NASDAQ-100 file matches a yfinance export recorded in the preprocessing notebook. The recorded call has no retrieval timestamp, and current yfinance treats the end date as exclusive, so a literal rerun may stop on 2024-12-30. The file has no adjusted close. The S&P 500 file is described in the original notebooks as a Kaggle download. The repository has no Kaggle URL or retrieval timestamp. `Adj Close` equals `Close` on every row, and 5,496 early volume values are zero. Those zeros are not treated as measured trading inactivity. Shared dates run from 2000-01-03 through 2020-11-04. Details are in `data/README.md`.

The raw files and the original notebooks are left unchanged. The benchmark reads the raw files through `marketpulse.data`. It does not fill or delete observations. Rows without a complete feature history or a future close are excluded, and the counts are stored with the results.

## Forecasting method

A forecast at origin `t` is made after the close at `t` is known. Horizons count trading sessions in file order. The feature row at `t` uses closes through `t` only. The ten features are lagged log returns, 5- and 20-session momentum, trailing mean returns, 20-session return volatility, the log distance from the trailing 20-session price mean, and a 14-session RSI. Volume is excluded from both indexes because the S&P 500 zeros do not have a documented meaning. Definitions, the embargo, and the model settings are in `docs/FORECASTING_PIPELINE.md`.

The benchmark uses the window lengths in `configs/forecasting.toml`: 2,520 minimum training origins, then 252 validation origins and 252 test origins. The test block is the latest eligible window. Labels that would read a price inside a later block are purged. Scaling is fit on the training origins only. This is one split per series and horizon. The repeated walk-forward loop in the research design has not been run.

## Models

All five forecasts are scored on the same test origins.

- Zero return: predict 0.
- Historical mean: predict the mean of the embargoed training targets, held fixed through the test window.
- Linear regression: training-only standardization, then ordinary least squares.
- Random forest: the same standardization, 200 trees, minimum leaf 5, and depth chosen from `{4, 8}` by validation RMSE. Seed 171.
- Sequence LSTM: a one-layer PyTorch network on a 20-session feature window ending at `t`, with hidden size chosen from `{16, 32}` by validation RMSE. Training uses Adam, at most 25 epochs, and early stopping after 5 epochs without a better validation loss. The best validation weights are restored. CPU bitwise reproducibility across PyTorch versions is not guaranteed. The run records the versions it used.

The primary scores are MAE, RMSE, forecast bias, directional accuracy, and skill `1 - MSE_model / MSE_baseline` against the zero forecast and the historical mean. A zero prediction does not count as a matching direction.

## Installation

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[research,dev]"
```

The `research` extra installs PyTorch and matplotlib. scikit-learn is a core dependency, so the classical tests run without PyTorch. CI installs the `dev` extra only and does not train the benchmark.

## Commands

Validate the raw files:

```bash
marketpulse-validate
```

Run the tests and lint:

```bash
ruff check .
ruff format --check .
pytest
```

Run the benchmark from the repository root:

```bash
marketpulse-benchmark --mode smoke --output results/benchmarks/smoke
marketpulse-benchmark --mode full --output results/benchmarks/full
```

Smoke mode trains the NASDAQ-100 one-session case with a smaller forest, one LSTM size, and two epochs. It checks that the command runs. It is not a benchmark result, and its output directory is gitignored. Full mode runs both indexes, all three horizons, and all five models. It writes `metrics.csv`, `metrics.json`, `predictions.csv`, `environment.json`, and `test_rmse.png`.

## Results of the full run

The full run completed for both indexes and all three horizons. Every test cell has 252 origins. The NASDAQ-100 test windows fall in late 2023 through 2024. The S&P 500 test windows fall in late 2019 through 2020 and include the early-2020 decline. The tables below are test RMSE. Skill against the zero forecast is in parentheses. Unrounded metrics, biases, and sample dates are in `docs/BENCHMARK_RESULTS.md`.

| Model | NASDAQ-100 h=1 | h=5 | h=20 | S&P 500 h=1 | h=5 | h=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero return | 0.011510 (0.000) | 0.024820 (0.000) | 0.044044 (0.000) | 0.021723 (0.000) | 0.040743 (0.000) | 0.089691 (0.000) |
| Historical mean | 0.011498 (0.002) | 0.024669 (0.012) | 0.042562 (0.066) | 0.021719 (0.000) | 0.040710 (0.002) | 0.089295 (0.009) |
| Linear regression | 0.011541 (-0.005) | 0.024657 (0.013) | 0.041564 (0.109) | 0.021757 (-0.003) | 0.040496 (0.012) | 0.089865 (-0.004) |
| Random forest | 0.011508 (0.000) | 0.024260 (0.045) | 0.040842 (0.140) | 0.020852 (0.079) | 0.041317 (-0.028) | 0.089839 (-0.003) |
| LSTM | 0.011554 (-0.008) | 0.024508 (0.025) | 0.042515 (0.068) | 0.021727 (0.000) | 0.041460 (-0.036) | 0.097422 (-0.180) |

On the NASDAQ-100 one-session window, the RMSE values sit within 0.011498 and 0.011554. That spread is too small to read as a useful forecast. At the longer NASDAQ-100 horizons, the random forest has the lowest RMSE. Part of the gap versus a zero forecast is the positive average of the overlapping test returns; the historical mean also beats zero there. On the S&P 500 window, the one-session random forest has skill 0.079 against zero, while at five and twenty sessions the learned models do not beat the historical mean. The twenty-session LSTM has larger error than the zero forecast. Validation rankings and test rankings disagree in several cells. One window does not show that a ranking will hold on another window.

## Reproducibility

The saved `environment.json` records the package versions and `git rev-parse HEAD` for the code that produced the files. The protocol seed is 171. Forest and LSTM settings are in `configs/benchmark.toml`. PyTorch seeding is set for Python, NumPy, and PyTorch. A later PyTorch build can still change the LSTM numbers. The classical fits use `n_jobs=1` and the protocol seed.

To reproduce the tables, install the `research` extra and run the full command above. Compare `metrics.csv` with the committed file.

## Current limitations

The published comparison is one terminal window per series and horizon. It does not estimate whether a ranking is stable across years. Horizon 5 and horizon 20 labels overlap, so the test errors in those cells are dependent. The two indexes are scored on different calendars. Directional accuracy of the zero forecast is 0 under the metric definition, and directional accuracy of the positive historical mean equals the share of positive outcomes in that window. No prediction interval has been calibrated. No result here is a statement about investment returns.

## Development status

| Item | Status |
| --- | --- |
| Audit and research protocol | Written |
| Data validation, tests, lint, CI | Implemented; CI does not train models |
| Causal features, embargoed terminal split, five forecasts | Implemented and run on the raw files |
| Walk-forward evaluation | Designed, not run |
| Regime and distribution-shift slices | Designed, not run |
| Prediction intervals | Designed, not run |

## Later work

The next experiment is the repeated walk-forward evaluation from `docs/RESEARCH_DESIGN.md`: expanding training, a step of 252 sessions, and a check of whether the RMSE ranking stays the same. After that come the predeclared historical slices, volatility terciles fit on training data only, and chronological prediction intervals at the nominal coverage of 0.9. Empirical coverage in that later study would be a diagnostic. The design does not treat overlapping returns as exchangeable.

## Contributors

- Yug More, yugmore20@gmail.com. NASDAQ-100 course notebooks and data. Research-extension design and engineering.
- Srihan Cheemangunta. S&P 500 course notebooks and data.
