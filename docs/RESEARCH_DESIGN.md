# Research design

> Implementation status: the terminal study is preserved in
> [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). The completed Phase 4/5 study
> is documented in [WALKFORWARD_RESULTS.md](WALKFORWARD_RESULTS.md) and
> [WALKFORWARD_METHODOLOGY.md](WALKFORWARD_METHODOLOGY.md). This document
> preserves the original research design. Actual execution fixes lookback
> at 20 rather than tuning it, uses one seed rather than repeated seeds,
> and evaluates each index on its native calendar rather than a matched
> cross-index calendar. Prediction intervals remain unimplemented. Legacy
> fold execution provenance is incomplete and explicitly disclosed.


This document defines the forecasting protocol. Initial numeric choices also live in `configs/forecasting.toml`. If those choices change, the config file and this document must change together.

`docs/BENCHMARK_RESULTS.md` reports one terminal chronological split for each series and horizon. The repeated walk-forward evaluation and regime slices are now reported separately; the prediction-interval study remains unrun.

The primary series are the committed raw files, read by `marketpulse.data`. `notebooks/nasdaq100/nasdaq_processed.csv` is a historical feature table. It is not the input to the new experiments.

## Primary forecasting task

At forecast origin t, the close C(t) is known. The target at horizon h is

```text
r(t, h) = log(C(t + h) / C(t))
```

`forward_log_return` implements that definition.

- **Forecast origin.** The close of trading row t. The decision time is that close, not the next open.
- **Information set.** Prices, volume, and features that can be computed from rows at or before t. A rolling window that includes row t is allowed. A window that includes row t+1 is not.
- **Horizon.** h is a count of rows in the cleaned trading-day series, not a count of calendar days. The initial horizons are 1, 5, and 20.
- **Alignment.** The label for origin t is stored on row t. It is not stored on row t+h. A sequence model whose window ends at t predicts r(t, h), not r(t+1, h).
- **Indexing.** After validation, rows are the observed sessions in file order. Missing sessions are not inserted. A horizon of 5 means five sessions in the file, which can span more than five calendar days when a holiday falls inside the window.
- **Missing values.** Leading rows whose indicators are undefined, and trailing rows whose future close is unknown, stay missing. They are excluded from supervised learning and the exclusion count is recorded. Prices are not interpolated, and rows are not dropped without that record.

Each index is modeled separately. Volume is not a feature until the S&P 500 zero-volume history is handled explicitly. The NASDAQ-100 file has no zero-volume rows. The S&P 500 file has 5,496 leading zero-volume rows and ends on 2020-11-04.

## Why this target

The existing price-level charts track the level of the index. A no-change forecast already does that. Log returns ask whether the model improves on that no-change forecast. The same definition also makes horizons comparable once errors are judged against a baseline, which raw MSE does not do.

## Horizons

The initial set is 1, 5, and 20 trading sessions. These stand for a session, a week, and about a month of trading days. They are research choices, not estimates of an optimal horizon. A horizon can be dropped or added if a series is too short to support the walk-forward windows below. That change has to be made before looking at test skill.

## Baselines

Every return experiment includes both of these forecasts:

- **Zero return.** Predict 0 at every origin. For a price-level reading of the same task, this is the persistence forecast C(t+h) = C(t). It is not a second return baseline.
- **Historical mean.** Predict the mean of the training-window targets. The mean is computed after the embargo below, and it is not computed from validation or test rows.

A model that does not beat both baselines on the predeclared metric has not shown forecasting value, even if its error looks small. Gradient boosting is not part of this required baseline set. The original S&P gradient-boosting fit mapped a date ordinal to a price level. That is a different task.

## Models

The first benchmark is:

- Linear regression.
- Random forest.
- A sequence LSTM whose input window ends at the forecast origin and whose label is r(t, h).

The sequence length is greater than one. It is chosen on the validation window, together with the other hyperparameters, from a small set written down before the test window is scored. The length-1 LSTM in the original notebook is not evidence about sequence learning and will not be reported as a sequence model.

No further architecture is added until these three have been scored against the baselines. Stochastic gradient descent and tree models use the seed in `configs/forecasting.toml` (`171`) through NumPy, Python's `random`, PyTorch, and scikit-learn. Training is specified for CPU. A GPU run is not required and is not comparable unless its nondeterminism is recorded.

## Features

Candidate features, all known at origin t:

- Lagged daily log returns, including the return realized at t, `log(C(t) / C(t-1))`.
- Rolling means of daily log returns.
- Rolling standard deviation of daily log returns.
- Scale-free moving-average distance, `log(C(t) / MA_w(t))`, instead of the raw moving-average level. The audit found that raw `ma5` and `ma20` in the original test period sit far above the training range.
- RSI computed through t.
- Momentum, `log(C(t) / C(t-k))`, for a small set of lags k.

Indicators are not copied from the original processed file. They are recomputed inside each training window's information set. Any warmup rows that an indicator leaves missing are counted and removed from that fold's supervised rows. They are not filled.

## Evaluation

### Folds

The initial walk-forward scheme, recorded in `configs/forecasting.toml`:

- Expanding training history.
- At least 2,520 training origins before the first evaluation window, about ten trading years.
- Next 252 sessions for validation.
- Next 252 sessions for test.
- Step forward 252 sessions and repeat.
- Refit the model and the preprocessor on each training window.

The implemented folds count those windows in eligible forecast origins and anchor them on the latest origin. The final fold is the terminal split already reported in `docs/BENCHMARK_RESULTS.md`. Each earlier fold steps back by 252 origins, which is 252 trading sessions when eligible origins are consecutive sessions. A fold is kept only when the embargoed training sample still has at least 2,520 origins, so the grid can leave a block of early origins unscored. Test origins are not repeated. The step is not allowed to be shorter than the test window.

The NASDAQ-100 file has 6,289 rows and the S&P 500 file has 23,323, so a 2,520-row minimum is feasible on both. If a later feature warmup makes the first fold too short, the minimum is revised in the config before test scores are read.

Hyperparameters are chosen by validation RMSE inside each fold, or by one validation rule fixed before the run. The test window is scored once with those choices. It is not used to pick the model that will be written up.

### Embargo

A label r(t, h) uses C(t+h). If an evaluation window starts at index `evaluation_start`, every origin with `t + h >= evaluation_start` sees a close inside that window. `exclusive_training_end` returns `evaluation_start - h` as the exclusive end of the allowable training origins. Training rows are `range(exclusive_training_end)`. The same rule applies between training and validation, and between validation and test when a calibration block is inserted.

Overlapping multi-day labels inside the test window are still dependent. The embargo stops training labels from reading evaluation prices. It does not make test errors independent. Reports will not treat those errors as an independent sample, and they will not quote iid standard errors as if the overlap were absent.

### Preprocessing

Scalers, feature selection, and any imputation rule are fit on the embargoed training rows of the current fold only. Validation and test rows are transformed with that fit. The original Notebook 2 scaler followed this rule for its single split. The new protocol keeps it for every fold.

### What is compared

Within a fold, every model and both baselines are scored on the same origins, the same horizon, and the same index. Daily and weekly errors are compared through skill against the zero-return baseline, not through raw MSE. Cross-index tables are produced only on the date intersection, 2000-01-03 through 2020-11-04, with the same horizons and the same fold calendar. Until that table exists, the indexes are not ranked against each other.

### Seeds and repetition

The protocol seed is 171. Where a model remains stochastic on CPU, the fold is repeated for a small number of seeds declared with the run, and the report shows the spread. Repeats are not added after seeing which seed looks best.

## Metrics

For each model, baseline, horizon, and test window:

- MAE.
- RMSE.
- Forecast bias, defined as the mean of prediction minus outcome. Positive bias means the forecast is too high.
- Directional accuracy on nonzero outcomes. A zero prediction does not match a nonzero outcome. Zero outcomes are excluded. If a window has no nonzero outcome, the metric is reported as undefined rather than as zero.
- Skill, `1 - MSE_model / MSE_baseline`, against each baseline.

MAPE is not a primary metric. Returns are often near zero, and MAPE divides by those outcomes.

The original notebooks' price-level MSE can be recomputed as a descriptive check. It is not the primary score.

## Market regimes

Regimes are a way to slice results, not a set of future labels.

The initial definition is the trailing 20-session standard deviation of daily log returns, using rows at or before the forecast origin. That series is the `roll_vol_20` feature. Low, medium, and high are the terciles of that value. Thresholds are the linear 1/3 and 2/3 quantiles of the embargoed training origins in the current fold, then applied to that fold's test origins. A test value on the lower threshold is low. A value on the upper threshold is medium. A value above the upper threshold is high. When the two quantiles are equal, the medium bin is empty. A threshold estimated on the test outcomes is not an operational regime.

A separate retrospective description may assign terciles using only past volatility at each origin and report performance by those bins. That description is labeled retrospective. It is not fed back into the model as a feature unless the feature itself is causal at t.

The bins are not defined with a future realized volatility that includes sessions after t.

## Distribution shift

The reports planned for Phase 3 are:

- RMSE and skill by walk-forward test window.
- Whether the ranking of the three models and two baselines changes across windows.
- The validation-window volatility compared with the test-window volatility.
- Predeclared historical slices, used only as reporting groups: the dot-com drawdown (2000-03-01 through 2002-12-31), the financial-crisis window (2008-09-01 through 2009-03-31), and the COVID crash window (2020-02-15 through 2020-04-30). These dates are fixed here, before the experiments. They are not model inputs. The COVID window is inside the NASDAQ-100 file and at the end of the S&P 500 file. A slice that a series does not cover is omitted for that series, not filled.

A single full-sample test score will not be the only published number.

## Uncertainty

Phase 3 will build prediction intervals on a chronological calibration block placed after training and before the final test block. The test block is not used to choose the interval width.

If the method is split conformal:

- Calibration scores are absolute residuals, or another score declared before the run.
- The interval is formed from the calibration scores and applied to the later test origins.
- The report gives empirical coverage and mean interval width at the nominal level in the config, initially 0.9.
- Coverage is reported overall and by walk-forward window.

Exchangeability does not hold. Returns overlap at h greater than 1, volatility clusters, and the test period can shift away from the calibration period. The experiment therefore does not claim the finite-sample marginal coverage guarantee of exchangeable conformal prediction. Empirical coverage is a diagnostic. Block or weighted conformal methods can be added later. They will not be described as already providing that guarantee.

## Research questions and hypotheses

The experiments test the following questions. The hypotheses are null statements, not expected conclusions.

**RQ1.** On the walk-forward protocol above, does linear regression, random forest, or the sequence LSTM have lower test RMSE than both the zero-return and historical-mean baselines for one-session log returns?

**H1.** For each model, one-session test skill against the zero-return baseline is zero.

**RQ2.** Does skill against the zero-return baseline change from horizon 1 to horizon 5 to horizon 20?

**H2.** For each model, skill is the same at all three horizons.

**RQ3.** Is the RMSE ranking of the models and baselines the same in every walk-forward test window and in every volatility regime?

**H3.** The ranking is stable across windows and across the three volatility bins.

**RQ4.** On a later test block, what empirical coverage and mean width do the chronological prediction intervals have at the nominal level?

**H4.** Empirical coverage equals the nominal coverage. Rejection in either direction is reported. A failure to match the nominal level is an expected possibility under dependence and shift, not a reason to retune the interval on the test block.

## What will be saved

Each future run will save the git revision, the protocol file, the seed, the fold date ranges, the embargo length, the selected hyperparameters, and the metrics above. A result that is not tied to a revision and a config is not a result of this project.

## Integrity constraints

- No synthetic market series enters a reported experiment. Synthetic rows are for unit tests of the code.
- No missing close is invented.
- A model is not described as better than another without a comparison on the same origins, horizon, and index.
- Implementing a model is not the same as having run it.
- The original README ranking is background. It is not carried forward as a finding.
- Baseline wins and unstable rankings will be reported if that is what the runs produce.
