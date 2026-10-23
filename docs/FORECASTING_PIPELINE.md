# Forecasting pipeline

This document describes the benchmark implemented for the first empirical comparison. It is the terminal chronological split from the research design, not the repeated walk-forward study. No trading rule is evaluated.

## Forecast time

A forecast at origin `t` is made after the closing price of trading session `t` is known. Horizons count rows in the file, which are trading sessions, not calendar days. The target is

```text
y_t(h) = log(Close[t + h] / Close[t])
```

The label is stored on row `t`. A model that sees a window ending at `t` predicts this same label. It does not predict the label of row `t + 1`.

The last `h` rows have no target. They are excluded and the count is stored in the benchmark sample record. Prices are not interpolated.

## Features

The feature row at `t` uses closes through `t` only. Rolling windows are trailing (`center=False`). The columns are:

| Column | Definition at origin t |
| --- | --- |
| `ret_0` | `log(Close[t] / Close[t - 1])` |
| `ret_1` | `ret_0` at `t - 1` |
| `ret_5` | `ret_0` at `t - 5` |
| `mom_5` | `log(Close[t] / Close[t - 5])` |
| `mom_20` | `log(Close[t] / Close[t - 20])` |
| `roll_mean_5` | Mean of the five one-session log returns ending at `t` |
| `roll_mean_20` | Mean of the twenty one-session log returns ending at `t` |
| `roll_vol_20` | Standard deviation of those twenty returns, with divisor `n - 1` |
| `ma_distance_20` | `log(Close[t] / mean(Close[t - 19 : t]))` |
| `rsi_14` | 14-session RSI from trailing average gains and losses |

RSI is 100 when the trailing loss is zero and the trailing gain is positive. It is missing when both are zero, and that origin is then excluded if the rest of the row is not finite. Raw price moving averages are not used as levels. The audit showed that those levels shift between earlier and later years.

Volume is not a feature. The S&P 500 file has 5,496 leading zeros, and nothing in the file shows that those zeros are traded volume rather than missing history. Using volume on the NASDAQ-100 only would make the two benchmarks different experiments. Both indexes therefore use the same ten features.

## Eligible origins and the split

An origin is eligible when its target is finite and the `lookback` feature rows ending at that origin are all finite. The benchmark lookback is 20, set in `configs/benchmark.toml` before the test scores are computed. It is not chosen on the test set. Tabular models use the feature row at the origin. The LSTM uses the 20 feature rows ending at the same origin. Every model is scored on the same eligible origins.

`terminal_chronological_split` places windows from `configs/forecasting.toml` at the end of the eligible origins:

- The test block is the last 252 eligible origins.
- The validation block is the last 252 eligible origins whose labels finish strictly before the test block.
- Training is every earlier eligible origin whose label finishes strictly before the validation block.
- Origins in the gaps are purged. A label at `t` reads `Close[t + h]`, so the gap before a block that starts at `s` is the origins with `t + h >= s`.

The same rule is the existing `exclusive_training_end` helper. Training must contain at least 2,520 origins after that embargo. Scaling is fit on those training rows and applied to validation and test. The historical-mean baseline is the mean of the training targets only. It is not updated inside the test window.

This is one split per series and horizon. It is not a walk-forward average, and the NASDAQ-100 and S&P 500 test calendars are not the same. The S&P 500 file ends in 2020. The NASDAQ-100 file ends in 2024.

## Models

All five forecasts use the seed `171` where a seed applies.

- **Zero return.** Predict 0. In price space this is the forecast that the index does not change.
- **Historical mean.** Predict the training-target mean.
- **Linear regression.** `StandardScaler` and `LinearRegression` in one training pipeline.
- **Random forest.** The same training-only scaler, 200 trees, minimum leaf size 5, and `max_depth` in `{4, 8}`. The depth with the lower validation RMSE is kept. A tie keeps the smaller depth. The test set is not an input to that choice.
- **LSTM.** One PyTorch LSTM layer and a linear readout of the last timestep. Hidden size is chosen from `{16, 32}` by validation RMSE. Training uses Adam with learning rate `0.001`, batch size 256, at most 25 epochs, and early stopping after 5 epochs without a better validation MSE. The weights from the best validation epoch are restored. The network is trained on CPU.

Directional accuracy follows the existing definition: a zero prediction does not match a nonzero outcome, and zero outcomes are excluded. The zero-return baseline therefore has directional accuracy 0 whenever every test outcome is nonzero. That number is a property of the definition, not evidence that the baseline misses a tradable direction.

Skill is `1 - MSE_model / MSE_baseline`. The baseline MSE must be positive; otherwise skill is recorded as missing. The baselines are the zero forecast and the historical mean.

## What this run does not estimate

Overlapping labels at horizons 5 and 20 make the test errors dependent. The reported RMSE has no iid standard error. One terminal window does not show whether a ranking is stable. No volatility regime, historical-crisis slice, or prediction interval is computed here. Those questions remain later phases.

PyTorch seeding does not promise bitwise identical LSTM results across versions or hardware. The saved artifacts record the versions that produced them.

## Commands

```bash
python -m pip install -e ".[research,dev]"
marketpulse-benchmark --mode smoke --output results/benchmarks/smoke
marketpulse-benchmark --mode full --output results/benchmarks/full
```

Smoke mode runs the NASDAQ-100 file at the first configured horizon with one forest depth, one LSTM size, and two epochs. Its output is a wiring check. The research result is the full mode. The full run saved under `results/benchmarks/full/` is interpreted in `docs/BENCHMARK_RESULTS.md`.
