# Repository audit

This audit describes the MarketPulse repository as it existed on `main` at commit `87011f0` (`Update README.md`), before the research-foundation revision. Measurements below were taken from the committed files and from stored notebook outputs. Models were not retrained for this audit.

The engineering items in Phase 1 of the roadmap are implemented in the research-foundation revision. That revision does not rerun the forecasting experiments, and it does not replace the original claims with a new performance result.

## 1. Executive summary

MarketPulse is a collaborative course project that compares forecasting on the NASDAQ-100 and the S&P 500. The committed material is a set of notebooks, two raw price files, one derived NASDAQ feature file, and a README. There is no installable package, test suite, dependency lock, or continuous-integration workflow on the audited `main` commit.

The NASDAQ notebooks forecast log returns with a chronological split, and they fit scaling on the training rows only. Those are sound starting points. The S&P notebooks forecast the closing level from the calendar date alone. The two efforts do not share a target, a feature set, a sample period, or a split, so the README's cross-index ranking is not a valid comparison.

Stored NASDAQ test errors do not establish the README's claim that more complex models are reliably better. On the same 80/20 split used by the modeling notebook, the stored linear-regression error is slightly worse than a zero-return forecast, and the stored random-forest error is only slightly better. The stored weekly random-forest error is worse than a zero-return forecast. Neither LSTM has a stored test error. The simple LSTM is a one-step model, which the notebook itself describes as having no real sequence. The deep LSTM uses a 60-day window, but its label is one row later than the label used for the linear and random-forest models, and the saved outputs never score it on the test set.

These observations are limitations of the existing evidence. They are not a new experiment and they are not a claim that every model will fail under a corrected protocol.

## 2. Original repository architecture

Tracked layout on the audited commit:

```text
README.md
requirements.txt          # empty
.gitignore                # empty
data/nasdaq100.csv
data/SPX.csv
notebooks/nasdaq100/01_preprocessing_nasdaq100.ipynb
notebooks/nasdaq100/02_classical_models_nasdaq100.ipynb
notebooks/nasdaq100/03_price_prediction_nasdaq100.ipynb
notebooks/nasdaq100/nasdaq_processed.csv
notebooks/sp500/S&P 500 Historical Data (Data Pre-processing).ipynb
notebooks/sp500/S&P 500 Historical Data (Model Construction).ipynb
notebooks/sp500/temp.java
notebooks/analysis/ModelAnalysis.ipynb
notebooks/analysis/temp1.java
```

`.DS_Store` files are also tracked. There is no `src/` tree, no `tests/`, no `pyproject.toml`, and no `.github/workflows/`.

Git history attributes the NASDAQ notebooks, the NASDAQ raw file, and most README revisions to Yug More. It attributes the S&P 500 file and the S&P notebooks to Srihan Cheemangunta (`srihanrc`). Both people edited the README. This audit does not reassign that work.

`notebooks/sp500/temp.java` contains an empty Java class. `notebooks/analysis/temp1.java` is an empty file. Neither participates in the research.

## 3. Dataset inventory

### NASDAQ-100 raw file

| Item | Observation |
| --- | --- |
| Path | `data/nasdaq100.csv` |
| Rows | 6,289 |
| First session | 2000-01-03 |
| Last session | 2024-12-31 |
| Columns after parsing | date, open, high, low, close, volume |
| Duplicate timestamps | 0 |
| Missing values | 0 |
| Nonpositive prices | 0 |
| Infinite values | 0 |
| Weekend rows | 0 |
| High below low | 0 |
| Close outside high-low | 0 |
| Zero-volume rows | 0 |
| Close range | 804.6400146484375 to 22096.66015625 |

The file has three header rows:

```text
Price,Close,High,Low,Open,Volume
Ticker,^NDX,^NDX,^NDX,^NDX,^NDX
Date,,,,,
```

The first field of each data row is the session date, stored under the name `Price`. That layout matches a recent yfinance export. The preprocessing notebook records `yf.download("^NDX", start="2000-01-01", end="2024-12-31")`. Current yfinance treats `end` as exclusive, so a literal re-run of that call can stop on 2024-12-30, while this file includes 2024-12-31. The repository does not store an acquisition time or a yfinance version. Provenance is therefore supported but not byte-reproducible from the notebook alone.

Three calendar gaps are longer than four days:

- 2001-09-10 to 2001-09-17 (7 days), consistent with the post-September 11 market closure.
- 2006-12-29 to 2007-01-03 (5 days).
- 2012-10-26 to 2012-10-31 (5 days), consistent with the Hurricane Sandy closure.

No separate adjusted-close series is present.

### S&P 500 raw file

| Item | Observation |
| --- | --- |
| Path | `data/SPX.csv` |
| Rows | 23,323 |
| First session | 1927-12-30 |
| Last session | 2020-11-04 |
| Columns | Date, Open, High, Low, Close, Adj Close, Volume |
| Duplicate timestamps | 0 |
| Missing values | 0 |
| Nonpositive OHLC prices | 0 |
| Infinite values | 0 |
| Weekend rows | 0 |
| High below low | 0 |
| Close outside high-low | 0 |
| Close range | 4.4 to 3580.840088 |
| Rows where Adj Close differs from Close | 0 |
| Zero-volume rows | 5,496 |
| First positive volume | 1950-01-03 |

The README and the S&P preprocessing notebook call this a Kaggle download. The repository stores no Kaggle URL, dataset version, or retrieval timestamp, so that provenance cannot be verified from the files alone. Because `Adj Close` equals `Close` on every row, the file does not encode a separate dividend adjustment.

Ten calendar gaps are longer than four days. The longest is 1933-03-03 to 1933-03-15 (12 days), consistent with the 1933 bank holiday. The file also contains the 2001-09-10 to 2001-09-17 closure and the 2012-10-26 to 2012-10-31 closure. The other gaps are five calendar days and are consistent with holidays adjacent to weekends. This audit did not match every gap to an exchange calendar.

The two raw files do not cover the same history. Their overlap is 2000-01-03 through 2020-11-04. A cross-index comparison that ignores that fact mixes different eras.

### Derived NASDAQ file

`notebooks/nasdaq100/nasdaq_processed.csv` has 6,263 rows, from 2000-02-01 through 2024-12-20. Columns are `index`, `date`, `close`, `high`, `low`, `open`, `volume`, `log_return`, `ma5`, `ma20`, `volatility`, `rsi`, `target_next_day_return`, `target_next_week_return`, and `target_direction`.

Every processed OHLCV row matches the raw NASDAQ file. The processed dates are the contiguous raw slice from index 20 through 2024-12-20. On the current raw file, which includes 2024-12-31, a five-session forward target is still defined on 2024-12-23. That session is absent from the processed file. The processed tail is what Notebook 1's `dropna()` produces if the input ends on 2024-12-30. The two committed NASDAQ files are therefore not an exact input and output pair of the current raw file. The overlapping rows do match.

Checked against the processed closes:

- `log_return` at row t equals `log(close_t / close_{t-1})` on the checked rows.
- `target_next_day_return` at row t equals the next row's `log_return` on all 6,262 consecutive pairs.
- `target_next_week_return` at row t equals `log(close_{t+5} / close_t)` on all 6,258 pairs that have five following rows.
- `target_direction` is 1 when the next-day target is positive and 0 otherwise.

The README lists `lag_1`, `lag_2`, and `lag_5`. Those columns are not in this file, and Notebook 1 does not create them.

## 4. Existing model inventory

### Linear regression and random forest, NASDAQ returns

Notebook 2 (`02_classical_models_nasdaq100.ipynb`) loads `nasdaq_processed.csv`. Features are `log_return`, `ma5`, `ma20`, `volatility`, and `rsi`. The first target is `target_next_day_return`. The split is `split_idx = int(len(X) * 0.8)`, which is index 5,010 on the current file: training 2000-02-01 through 2019-12-30 (5,010 rows) and test 2019-12-31 through 2024-12-20 (1,253 rows). `StandardScaler` is fit on the training block and applied to the test block.

`LinearRegression` and `RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)` are fit on the scaled training rows. The stored test MSEs are:

- Linear regression, next-day: `0.0002733094990144006`
- Random forest, next-day: `0.00026314561242952284`

The same features and split are reused for `target_next_week_return`. The stored random-forest weekly MSE is `0.0011371821774815668`. There is no validation block. The forest hyperparameters are fixed in the notebook, not selected on a held-out validation window.

### Simple LSTM and deep LSTM

The simple LSTM receives tensors with a sequence axis of length 1 (`unsqueeze(1)`). The notebook text says this adds the dimension PyTorch requires and that there is no real sequence in the data. Each row is an independent length-1 sequence, so the recurrence does not connect one trading day to the next. The hidden state returned by the LSTM is discarded. Training is full-batch Adam for 50 epochs with MSE loss. There is no `torch.manual_seed`, no validation split, and no early stopping. The saved outputs contain training loss for the deep model only. They do not contain a test MSE for either LSTM.

The deep LSTM builds windows of 60 rows:

```python
for i in range(len(X) - seq_len):
    X_seq.append(X[i : i + seq_len])
    y_seq.append(y[i + seq_len])
```

The window ends at row `i + 59`. The label is `y[i + 60]`. In this notebook, `y[t]` is the return from close t to close t+1. The classical models score `y[t]` using features at t. The deep model scores `y[t+1]` using features only through t. Relative to the last feature row, that is the return from the next close to the close after that, not the next-day return from the forecast origin. The architecture is a two-layer LSTM with hidden size 64, dropout 0.2, and a linear head. Stored training losses are not monotone: epoch 15 prints `0.000455` and epoch 20 prints `0.000593`. No test MSE is stored.

The weekly LSTM repeats the simple length-1 model. It does not use the 60-day deep model. The horizon comparison therefore mixes architectures, and it never scores the deep model on the weekly target.

### NASDAQ price notebook

Notebook 3 constructs `lag1`, `ma3`, `ma7`, `ma14`, and `next_day_price = close.shift(-1)`. It imports `train_test_split` but splits with `iloc` at 80 percent. The code fits `LinearRegression` only. The introductory markdown and the README also name a gradient-boosting regressor. That model is not trained in the notebook.

Stored errors are training MSE `3089.9683400922836` and test MSE `57225.39275267545`.

### S&P 500 gradient boosting

The preprocessing notebook converts `Date` to an ordinal and uses that single column as `X`. The target is `Close`. The split keeps the first 95 percent for training. On the current file that cut is index 22,156: training ends 2016-03-18 and the test block is 2016-03-21 through 2020-11-04 (1,167 rows). The arrays are saved to `preprocessed_data.npz`, which is not tracked.

The model notebook fits three `GradientBoostingRegressor` configurations and selects the lowest test RMSE. Stored results:

- Model 1 (`n_estimators=50`, `learning_rate=0.1`): RMSE `806.03`
- Model 2: RMSE `1328.05`
- Model 3: RMSE `1733.01`

The selected model is Model 1. Stored mean actual test price is `$2686.69`. Stored mean predicted test price is `$1971.47`. No `random_state` is set. The README's description of a linear regression and a random forest for the S&P 500 does not match this notebook. The implemented model is gradient boosting on the date ordinal.

`notebooks/analysis/ModelAnalysis.ipynb` loads `S&P500_gbrModel.joblib`, `S&P500_lastDate.joblib`, and `preprocessed_data.npz`. None of those files is tracked. The notebook cannot be rerun from the repository. Its stored print says the last training ordinal is `736041` and then, in a string literal, "(April 11, 2015)". `datetime.fromordinal(736041)` is 2016-03-18. April 11, 2015 is ordinal `735699`. The next printed ordinal, `736042`, is 2016-03-19, a Saturday. The next session in `SPX.csv` is 2016-03-21. The forecast step is calendar-ordinal plus one, not the next trading day.

## 5. Existing forecasting targets

| Experiment | Target | Alignment in the saved code |
| --- | --- | --- |
| NASDAQ linear regression and random forest | Next-session log return, `log_return.shift(-1)` | Label at row t is the return from close t to close t+1. Features at t use closes through t. |
| NASDAQ weekly random forest and simple LSTM | `log(close.shift(-5) / close)` | Five trading rows ahead, not five calendar days. Consecutive labels overlap. |
| NASDAQ direction label | Next-session return greater than zero | Not used as a model input in Notebook 2. |
| NASDAQ deep LSTM | `y[i + seq_len]` with a 60-row window | One row later than the classical next-day label. See section 4. |
| NASDAQ price notebook | Next close | Features include moving averages through the current close. |
| S&P notebooks | Close level | The only input is the date ordinal. |

`target_direction` is built with `(target > 0).astype(int)` before `dropna()`. A missing target becomes 0 rather than staying missing. The following `dropna()` removes those tail rows because the return targets are missing, so the coerced zeros are not in the saved modeling table. The coercion would matter if the direction column were used without dropping the tail.

## 6. Current evaluation methodology

NASDAQ return models use one chronological 80/20 cut and MSE on that test block. Plots compare actual and predicted returns. There is no validation window, no walk-forward loop, no embargo for overlapping weekly labels, and no baseline. Scaling is fit on the training block only, which avoids one important leakage path.

The price notebook uses MSE in price units and an actual-versus-predicted price chart. The S&P notebooks use RMSE in price units, including RMSE on the same test block that is used to pick the boosting configuration. The analysis notebook then plots fitted training values and one extrapolated ordinal.

LSTM training loss is not a test metric. Deep-LSTM and simple-LSTM test MSEs are not stored, so those models cannot be ranked from the saved numeric outputs.

## 7. Confirmed methodological problems

1. **The published model ranking is not supported by comparable test scores.** Linear regression and random forest have stored test MSEs. The LSTMs do not. Notebook 2's own RQ2 summary says the simple LSTM is similar to or slightly worse than random forest, while the closing section and the README call the deep LSTM or LSTM the best model. Those statements rely on plots and on training loss, not on a common test metric.

2. **A zero-return forecast is a stronger reference than the stored narrative uses.** On the current processed file and the notebook's `int(len * 0.8)` split, test MSE of predicting 0 is `0.0002717194453570787` for the next-day target and `0.0010856965169073383` for the five-session target. Test MSE of predicting the training-period mean is `0.0002715023916977835` and `0.0010804349633244874`. Relative to the zero forecast, the stored linear-regression skill is about `-0.0059`, the stored next-day random-forest skill is about `+0.0316`, and the stored weekly random-forest skill is about `-0.0474`. Skill here means `1 - MSE_model / MSE_zero`. This uses stored model errors and recomputed baseline errors. It is not a refit.

3. **Raw MSE does not make the daily and weekly tasks comparable.** The zero-forecast weekly MSE is about 4.00 times the zero-forecast daily MSE. Weekly returns have a larger scale. A higher weekly MSE, by itself, does not show that the weekly task is less predictable.

4. **The deep LSTM label does not match the classical next-day label.** Section 4 gives the index arithmetic. The model can use history inside its 60-day window. The saved experiment does not score the same target the notebook title states.

5. **The length-1 LSTM does not learn a historical sequence.** The author records this limitation in the notebook. The README still describes the LSTM as learning short-term sequential patterns and as the strongest next-day model.

6. **Price-level moving averages are on a different scale in the test period.** On the return split, `ma20` has training mean about 3,023 and maximum about 8,517, against a test mean about 14,023 and minimum about 7,561. A scaler fit on 2000–2019 maps later price levels far outside the training range. `log_return`, `volatility`, and `rsi` do not show that level shift.

7. **The price-chart experiment has no persistence baseline.** Reconstructing Notebook 3's `dropna()` and 80/20 cut on the current processed file gives 6,249 supervised rows and a test block starting 2020-01-03. Predicting the next close with today's close has test MSE about 42,358. The stored linear-regression test MSE is 57,225. The stored training MSE, 3,090, is also higher than persistence on that reconstructed training block, about 2,382. A chart of two price series near 14,000 can look close at an RMSE near 239 without beating a no-change forecast. Gradient boosting is named in the README and is not fit in the notebook.

8. **The S&P model is a date curve, and the test forecast is a constant.** Predicting 1,971.47 for every row of the current 5 percent test block has RMSE 806.026, which matches the stored Model 1 RMSE of 806.03 to the printed precision. The stored mean prediction is that same constant. Persistence on the same test rows has RMSE about 33.56. The model selection uses that test block. The printed calendar date April 11, 2015 does not match ordinal 736041. Adding one to the ordinal lands on a Saturday.

9. **The cross-index conclusion compares different tasks.** The README says the S&P 500 is easier to predict, that its best model is random forest, and that NASDAQ's best model is LSTM. The S&P code forecasts a price level from a date and selects gradient boosting. The NASDAQ return code forecasts log returns from technical features. The samples, splits, and metrics are not the same. Feature-importance claims such as "RSI, lag_1" and "MA14, volatility" are not computed in the notebooks. `lag_1` is not a column.

10. **Weekly labels overlap, and the test block is also the comparison block.** A five-day label at t uses closes through t+5. The next row's label uses four of those same future closes. MSE treats the rows as separate. Hyperparameters and the verbal model ranking are chosen without a validation window, using the same period that is reported as the result.

## 8. Potential methodological risks

These are risks, not additional confirmed bugs.

- Notebook 3 and the S&P notebooks import `train_test_split`, but the executed splits are chronological `iloc` cuts. A later edit could shuffle the series. The committed runs did not.
- Full-batch LSTM training without a seed can change the unstored test predictions between runs. The random forest sets `random_state=42`. The gradient-boosting models do not.
- RSI in Notebook 1 divides by a rolling mean of losses. A window of only gains produces an infinite relative strength and an RSI of 100. The processed file contains RSI values of 100. That is a property of the formula, not evidence that a price was invalid.
- `target_direction` coerces a missing future return to 0 before rows are dropped. The saved table does not retain those tail rows.
- The sector pie chart uses hardcoded weights that sum to 99.78, not 100. It is illustrative and is not an input to the models. The notebook cites a NASDAQ fact sheet and Investopedia, without a date for the weights.

## 9. Engineering and reproducibility weaknesses

- `requirements.txt` and `.gitignore` are empty. The notebooks import pandas, NumPy, scikit-learn, PyTorch, matplotlib, seaborn, yfinance, and joblib, with no recorded versions.
- There is no package, no tests, and no CI.
- `.DS_Store` files are tracked.
- Notebook 1 downloads data again instead of reading `data/nasdaq100.csv`. Notebook 2 reads `nasdaq_processed.csv` from the process working directory. The S&P notebook reads `SPX.csv` from the working directory, while the file lives in `data/`.
- The S&P analysis notebook depends on joblib and npz artifacts that are not in the repository.
- Random seeds are incomplete, so the LSTM outputs are not pinned.
- Empty Java files are committed beside the notebooks.
- The README, Notebook 2, and Notebook 3 disagree about which model was best and about which models were trained.

## 10. Existing reusable components

- Both raw price files are real historical series with unique, increasing session dates and valid OHLC relationships. They should remain the raw inputs. They should not be overwritten.
- Notebook 1's decision to model log returns, rather than raw price levels, is the right primary target for the research extension.
- The chronological NASDAQ split and the training-only scaler are practices to keep.
- The indicator ideas that are actually implemented, and that can be made causal and scale-free, are lagged returns, rolling volatility, and RSI. Raw price moving averages need a scale-free transform before they are used again.
- The saved notebooks should stay in `notebooks/` as the historical record of the course project.
- The processed CSV is useful as a checked derived artifact. It should not be treated as a second raw source, and it should not silently replace `data/nasdaq100.csv`.

## 11. Recommended improvements

1. Load the committed raw files through one parser that understands the yfinance header and reports quality problems without deleting rows.
2. Define the target as `log(C(t+h) / C(t))` at origin t, for horizons of 1, 5, and 20 trading sessions.
3. Compare every model with a zero-return forecast and a training-mean forecast before interpreting MSE.
4. Keep preprocessing inside each training window. Drop training origins whose label reaches into the evaluation window.
5. Give the LSTM a real sequence that ends at the forecast origin, and score the same rows and the same horizon as the other models.
6. Select hyperparameters on a validation window. Do not choose the reported model on the final test window.
7. Run the same protocol on each index separately. Compare indexes only on the overlapping dates.
8. Add an installable package, tests, a formatter, and CI that does not train models.
9. Write conclusions only after those experiments have been run.

## 12. Proposed research architecture

The foundation revision adds a small `marketpulse` package. Data loading and validation are operational. Target construction, point-forecast metrics, the training embargo rule, and the initial walk-forward settings are implemented as tested functions and as `configs/forecasting.toml`. Model training, walk-forward execution, regime analysis, and interval calibration are not implemented yet.

Later experiment code should call the loader, build targets with `forward_log_return`, apply `exclusive_training_end` at each fold boundary, and score forecasts with MAE, RMSE, bias, directional accuracy, and skill against the baselines. Notebooks remain the historical analysis. New runs should live in the package so they can be tested and repeated.

## 13. Research integrity considerations

- The original notebooks and README are a course project by Yug More and Srihan Cheemangunta. The foundation revision does not present that work as a new single-author study.
- Stored notebook numbers are evidence about those saved runs. They are not a substitute for a seeded, leakage-safe rerun.
- Negative skill and baseline wins are admissible results. They are not something to be explained away by switching metrics after the fact.
- No new market observations were downloaded or synthesized for this audit.
- The sector chart, the hardcoded April 2015 label, and the README ranking should not be cited as findings of the research extension.

## 14. Prioritized implementation roadmap

1. **Phase 1, this revision.** Package layout, environment, raw-data validation, tests, lint, CI, and a written experiment design. No model training.
2. **Phase 2.** Causal features, baselines, linear regression, random forest, and a sequence LSTM, evaluated with walk-forward folds and an embargo. Record seed, fold dates, and metrics. Do not claim a result from a single fit.
3. **Phase 3.** Volatility regimes defined from information available at the forecast origin, stability of model rankings across folds, and chronological prediction intervals with empirical coverage and width. State the dependence limitation instead of claiming an exchangeable conformal guarantee.
4. **Reporting.** Update the README and write the research report only after Phases 2 and 3 have produced saved outputs from the committed code.

Phase 1 of this roadmap is the work contained in the research-foundation revision. Phases 2 and 3 have not been started.
