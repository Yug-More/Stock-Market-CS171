# Terminal-split benchmark

This file reports the full run of `marketpulse-benchmark --mode full`. It is one chronological test window per series and horizon. It is not a walk-forward study, a regime study, or a trading backtest, and it is not a peer-reviewed result.

The code revision is `6b05c494ba51d30102a699ba23a29f699904e49f`. The machine record is `results/benchmarks/full/environment.json`: Python 3.13.2, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, and PyTorch 2.14.1 on macOS arm64. Every model in a dataset-horizon cell uses the same 252 test origins, the same dates, and the same target values. The exact rows are in `results/benchmarks/full/predictions.csv`. The figure is `results/benchmarks/full/test_rmse.png`.

## Sample

Feature warmup leaves the first 20 rows without a complete feature vector. The last `h` rows have no target. Those rows are excluded and counted below. Volume is unused on both indexes.

The historical mean is the mean of the embargoed training targets, held fixed. Forest depth is chosen from `{4, 8}` by validation RMSE. LSTM hidden size is chosen from `{16, 32}` the same way. The lookback is 20 for every model and was fixed before scoring the test window. Early stopping restores the weights from the best validation epoch. Training stopped before 25 epochs in every LSTM run because validation error stopped improving.

| Series | h | Train origins | Train dates | Validation dates | Test dates | Purged before test |
| --- | ---: | ---: | --- | --- | --- | ---: |
| NASDAQ-100 | 1 | 5,743 | 2000-02-29 to 2022-12-22 | 2022-12-27 to 2023-12-27 | 2023-12-29 to 2024-12-30 | 1 |
| NASDAQ-100 | 5 | 5,731 | 2000-02-29 to 2022-12-06 | 2022-12-14 to 2023-12-14 | 2023-12-22 to 2024-12-23 | 5 |
| NASDAQ-100 | 20 | 5,686 | 2000-02-29 to 2022-10-03 | 2022-11-01 to 2023-11-01 | 2023-12-01 to 2024-12-02 | 20 |
| S&P 500 | 1 | 22,777 | 1928-02-28 to 2018-10-30 | 2018-11-01 to 2019-11-01 | 2019-11-05 to 2020-11-03 | 1 |
| S&P 500 | 5 | 22,765 | 1928-02-28 to 2018-10-12 | 2018-10-22 to 2019-10-22 | 2019-10-30 to 2020-10-28 | 5 |
| S&P 500 | 20 | 22,720 | 1928-02-28 to 2018-08-09 | 2018-09-10 to 2019-09-10 | 2019-10-09 to 2020-10-07 | 20 |

Each test cell has 252 origins. The two indexes end in different years, so their test calendars are different experiments. The S&P 500 window includes February through April 2020. Horizon 5 and horizon 20 labels overlap, so those 252 errors are dependent.

Selected settings, all at seed 171: random forest used depth 8 for the NASDAQ-100 one-session model and depth 4 otherwise, with 200 trees and a minimum leaf of 5. Every LSTM selected hidden size 32. The NASDAQ-100 20-session LSTM restored epoch 1. The other LSTM runs restored an epoch between 4 and 10.

## Test metrics

Skill is `1 - MSE_model / MSE_baseline`. Positive skill is a smaller squared error than that baseline. Bias is the mean of prediction minus outcome. Directional accuracy uses the project definition: a zero prediction does not match a nonzero outcome, and zero outcomes are excluded. In this run every test outcome is nonzero, so the zero-return forecast has directional accuracy 0 by that definition.

The training mean is positive in every cell. For the historical-mean forecast, directional accuracy therefore equals the share of positive test outcomes. That share is 0.575, 0.615, and 0.782 on the NASDAQ-100 at horizons 1, 5, and 20, and 0.595, 0.655, and 0.754 on the S&P 500. A higher directional accuracy at the long horizon is the rising overlap of positive returns, which is a property of a positive constant forecast.

### NASDAQ-100

| h | Model | RMSE | Skill vs zero | Skill vs mean | Bias | Dir. acc. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Zero return | 0.011510 | 0.0000 | -0.0020 | -0.000882 | 0.000 |
| 1 | Historical mean | 0.011498 | 0.0020 | 0.0000 | -0.000717 | 0.575 |
| 1 | Linear regression | 0.011541 | -0.0054 | -0.0074 | -0.000655 | 0.536 |
| 1 | Random forest | 0.011508 | 0.0004 | -0.0016 | -0.000589 | 0.532 |
| 1 | LSTM | 0.011554 | -0.0077 | -0.0097 | 0.000181 | 0.556 |
| 5 | Zero return | 0.024820 | 0.0000 | -0.0123 | -0.004778 | 0.000 |
| 5 | Historical mean | 0.024669 | 0.0122 | 0.0000 | -0.003917 | 0.615 |
| 5 | Linear regression | 0.024657 | 0.0131 | 0.0010 | -0.003336 | 0.508 |
| 5 | Random forest | 0.024260 | 0.0446 | 0.0328 | -0.002252 | 0.607 |
| 5 | LSTM | 0.024508 | 0.0250 | 0.0130 | -0.000537 | 0.579 |
| 20 | Zero return | 0.044044 | 0.0000 | -0.0708 | -0.021287 | 0.000 |
| 20 | Historical mean | 0.042562 | 0.0662 | 0.0000 | -0.018022 | 0.782 |
| 20 | Linear regression | 0.041564 | 0.1095 | 0.0464 | -0.015125 | 0.750 |
| 20 | Random forest | 0.040842 | 0.1401 | 0.0792 | -0.013479 | 0.754 |
| 20 | LSTM | 0.042515 | 0.0682 | 0.0022 | 0.002772 | 0.726 |

### S&P 500

| h | Model | RMSE | Skill vs zero | Skill vs mean | Bias | Dir. acc. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Zero return | 0.021723 | 0.0000 | -0.0003 | -0.000450 | 0.000 |
| 1 | Historical mean | 0.021719 | 0.0003 | 0.0000 | -0.000227 | 0.595 |
| 1 | Linear regression | 0.021757 | -0.0032 | -0.0035 | -0.000028 | 0.536 |
| 1 | Random forest | 0.020852 | 0.0785 | 0.0782 | -0.000216 | 0.595 |
| 1 | LSTM | 0.021727 | -0.0004 | -0.0007 | -0.000134 | 0.484 |
| 5 | Zero return | 0.040743 | 0.0000 | -0.0016 | -0.001732 | 0.000 |
| 5 | Historical mean | 0.040710 | 0.0016 | 0.0000 | -0.000617 | 0.655 |
| 5 | Linear regression | 0.040496 | 0.0121 | 0.0105 | -0.000570 | 0.671 |
| 5 | Random forest | 0.041317 | -0.0284 | -0.0300 | -0.000992 | 0.631 |
| 5 | LSTM | 0.041460 | -0.0355 | -0.0371 | -0.002030 | 0.512 |
| 20 | Zero return | 0.089691 | 0.0000 | -0.0089 | -0.010162 | 0.000 |
| 20 | Historical mean | 0.089295 | 0.0088 | 0.0000 | -0.005697 | 0.754 |
| 20 | Linear regression | 0.089865 | -0.0039 | -0.0128 | -0.005843 | 0.730 |
| 20 | Random forest | 0.089839 | -0.0033 | -0.0122 | -0.006815 | 0.516 |
| 20 | LSTM | 0.097422 | -0.1798 | -0.1903 | -0.014343 | 0.540 |

RMSE values are rounded from `metrics.csv`. The unrounded metrics, including the validation partition, are in that file and in `metrics.json`.

## Reading the table

On the NASDAQ-100 one-session window, the five RMSE values span 0.011498 to 0.011554. The random-forest skill against zero is 0.0004. That gap is too small to treat as evidence of predictive power. Linear regression and the LSTM are slightly worse than the zero forecast on that window.

At NASDAQ-100 horizons 5 and 20, the zero forecast has a negative bias: the overlapping test returns were positive on average, and a forecast of zero sat below them. The historical mean captures part of that gap because the training mean is positive. The random forest has the lowest RMSE in both of those cells, with skill against the historical mean of 0.033 at five sessions and 0.079 at twenty. Those labels overlap, and each cell is one year. The same ordering is not a result about other years.

On the S&P 500 one-session window, the random forest has skill 0.079 against zero and against the historical mean. Linear regression and the LSTM are within a small margin of the zero forecast, on the side of larger error. At five and twenty sessions, the learned models do not improve on the historical mean. The twenty-session LSTM has skill -0.180 against zero, so its squared error is larger. The twenty-session LSTM also had positive validation skill against zero, about 0.010, before the test window was scored. Validation error and test error disagree in several other cells as well. A model selected on the validation window is not thereby shown to be stable.

The figure shows the same RMSE values. Its axis starts at zero, so the one-session bars look almost equal. That picture matches the table: the level of RMSE grows with the horizon because the target itself is a larger return, and the model-to-baseline gaps are small beside that level.

These numbers are from this protocol only. The stored errors in the original notebooks come from different targets, splits, and code paths, and they are not results of this run.
