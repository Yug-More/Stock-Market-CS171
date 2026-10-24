# MarketPulse

MarketPulse measures how reliably machine learning models forecast financial time series when the horizon, the market period, and the data distribution change.

The forecasting target is a forward log return, `log(Close[t + h] / Close[t])`, at horizons of 1, 5, and 20 trading sessions. Models are compared with a zero-return forecast and a historical-mean forecast on the same origins. This repository reports both the original terminal benchmark and a completed 282-fold expanding-window study with historical-stress and causal volatility-regime analysis. It does not claim a trading strategy, and it is not a peer-reviewed study.

## Research objective

The question is whether a model forecast is more accurate than simple baselines when the horizon and the evaluation window change, under a protocol that keeps future prices out of training. The hypotheses and the later walk-forward, regime, and interval designs are in `docs/RESEARCH_DESIGN.md`. The original terminal-split evidence is in `docs/BENCHMARK_RESULTS.md`; repeated-window evidence is in [the Phase 4/5 results](docs/WALKFORWARD_RESULTS.md), with [methodology](docs/WALKFORWARD_METHODOLOGY.md) and [recovery records](docs/RECOVERY.md).

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

The benchmark uses the window lengths in `configs/forecasting.toml`: 2,520 minimum training origins, then 252 validation origins and 252 test origins. The test block is the latest eligible window. Labels that would read a price inside a later block are purged. Scaling is fit on the training origins only. The original benchmark is one split per series and horizon. The completed walk-forward study repeats this split with expanding training history, advancing 252 origins per fold: 13 folds per NASDAQ-100 horizon and 81 per S&P 500 horizon. All five forecasts share each fold’s test origins. No test origin is repeated within an index/horizon. Full fold boundaries are persisted.

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

The `research` extra installs PyTorch and matplotlib. scikit-learn is a core dependency, so the classical tests run without PyTorch. CI has a fast development-check job and a CPU research job that runs LSTM tests, an all-model smoke experiment with figures, and an independent integrity audit of the completed study. CI does not retrain the full research matrix.

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

## Original terminal-benchmark results

The full run completed for both indexes and all three horizons. Every test cell has 252 origins. The NASDAQ-100 test windows fall in late 2023 through 2024. The S&P 500 test windows fall in late 2019 through 2020 and include the early-2020 decline. The tables below are test RMSE. Skill against the zero forecast is in parentheses. Unrounded metrics, biases, and sample dates are in `docs/BENCHMARK_RESULTS.md`.

| Model | NASDAQ-100 h=1 | h=5 | h=20 | S&P 500 h=1 | h=5 | h=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero return | 0.011510 (0.000) | 0.024820 (0.000) | 0.044044 (0.000) | 0.021723 (0.000) | 0.040743 (0.000) | 0.089691 (0.000) |
| Historical mean | 0.011498 (0.002) | 0.024669 (0.012) | 0.042562 (0.066) | 0.021719 (0.000) | 0.040710 (0.002) | 0.089295 (0.009) |
| Linear regression | 0.011541 (-0.005) | 0.024657 (0.013) | 0.041564 (0.109) | 0.021757 (-0.003) | 0.040496 (0.012) | 0.089865 (-0.004) |
| Random forest | 0.011508 (0.000) | 0.024260 (0.045) | 0.040842 (0.140) | 0.020852 (0.079) | 0.041317 (-0.028) | 0.089839 (-0.003) |
| LSTM | 0.011554 (-0.008) | 0.024508 (0.025) | 0.042515 (0.068) | 0.021727 (0.000) | 0.041460 (-0.036) | 0.097422 (-0.180) |

On the NASDAQ-100 one-session window, the RMSE values sit within 0.011498 and 0.011554. The RMSE spread alone does not establish economic usefulness, which was not tested. At the longer NASDAQ-100 horizons, the random forest has the lowest RMSE. Part of the gap versus a zero forecast is the positive average of the overlapping test returns; the historical mean also beats zero there. On the S&P 500 window, the one-session random forest has skill 0.079 against zero, while at five sessions linear regression has lower RMSE than the historical mean; at twenty sessions none of the learned models beats that baseline. The twenty-session LSTM has larger error than the zero forecast. Validation rankings and test rankings disagree in several cells. One window does not show that a ranking will hold on another window.

## Reproducibility

The saved `environment.json` records the package versions and `git rev-parse HEAD` for the code that produced the files. The protocol seed is 171. Forest and LSTM settings are in `configs/benchmark.toml`. PyTorch seeding is set for Python, NumPy, and PyTorch. A later PyTorch build can still change the LSTM numbers. The classical fits use `n_jobs=1` and the protocol seed.

To reproduce the tables, install the `research` extra and run the full command above. Compare `metrics.csv` with the committed file.

## Current limitations

The original Phase 4/5 study supplies descriptive rank stability. Later phases add retrospective dependence-aware inference, selected-fold multi-seed sensitivity, and calibrated interval diagnostics; see the linked reports below. The 210 recovered folds lack original execution revision/environment metadata; their predictions, targets, settings, boundaries, thresholds, metrics, and original file hashes were validated without inventing the missing provenance. The remaining 72 folds record execution metadata. Model weights are not serialized. NASDAQ-100 has no eligible dot-com or financial-crisis test origins under the ten-year training minimum. The fixed financial-crisis slice covers September 2008 through March 2009, not the entire 2007–2009 episode. Source retrieval metadata, early index history, and aggregate constituent selection limit generalization. Horizon 5 and horizon 20 labels overlap, so the test errors in those cells are dependent. The two indexes are scored on different calendars. Directional accuracy of the zero forecast is 0 under the metric definition, and directional accuracy of the positive historical mean equals the share of positive outcomes in that window. Later phases evaluate fixed and adaptive prediction intervals, including undercoverage and nesting failures. These do not establish guaranteed future coverage. No result here is a statement about investment returns.

## Walk-forward and market-regime study

The full study contains 282 completed folds, 1,410 model-fold evaluations, and 355,320 forecast rows. It recovers 210 validated existing folds and completes 72 interrupted S&P 500 twenty-session folds at the original training budgets. The [research report](docs/WALKFORWARD_RESULTS.md) presents all model/horizon comparisons, annual rank stability, historical stress coverage, regime counts, and actual results. The [artifact guide](results/walkforward/README.md) documents schemas and provenance limitations.

Historical slices use inclusive **forecast-origin dates**: dot-com (2000-03-01–2002-12-31), financial crisis (2008-09-01–2009-03-31), and COVID (2020-02-15–2020-04-30). Unavailable episodes are reported explicitly. Low/medium/high volatility uses trailing 20-session return volatility and training-only terciles frozen per fold. Error and baseline skill are pooled on comparable samples; mean fold RMSE is separately labeled.

Across the complete NASDAQ-100 history tested, random forest has modest positive pooled skill against zero at horizons 1 (0.0222) and 5 (0.0185), while zero return wins at horizon 20. By pooled RMSE, LSTM trails both baselines at all three NASDAQ horizons. This differs from some terminal-window rankings and shows why repeated chronological evaluation matters. For S&P 500, random forest wins at one session (skill 0.0131 versus zero), while the historical mean wins at five (0.0023) and twenty sessions (0.0094). LSTM has the highest pooled RMSE in all six index/horizon cells. Stress results and full regime comparisons are in the report; no cross-index ranking is inferred from different native calendars.

```bash
# Check model/plot wiring before a full run.
marketpulse-walkforward --mode smoke --output /tmp/marketpulse-smoke
# Use a new directory for a fresh training run.
marketpulse-walkforward --mode full --output /tmp/marketpulse-full
# Audit the committed study without retraining.
tar -xzf results/walkforward/full/fold_predictions.tar.gz -C results/walkforward/full
python scripts/verify_walkforward.py
python scripts/report_walkforward.py
```

The run saves exact fold plans, predictions, selected hyperparameters, training-only volatility thresholds, pooled/per-fold/regime/stress metrics, ranks, dependency versions, raw/source/config hashes, and per-new-fold execution revision/runtime. Figures come from actual predictions. Resume rejects incompatible inputs or corrupted artifacts. The original terminal benchmark and raw datasets remain unchanged.

## Repository structure

- `src/marketpulse/data/`: canonical loading, catalog, and validation.
- `src/marketpulse/features.py`, `targets.py`, `sequences.py`: causal supervised inputs.
- `src/marketpulse/evaluation/`: purged splits, preprocessing, metrics, and regimes.
- `src/marketpulse/models/`: baselines, classical models, and sequence LSTM.
- `src/marketpulse/experiments/`: benchmark, walk-forward runner, recovery validation, aggregation, and figures.
- `configs/`: predeclared datasets, horizons, windows, seeds, and model budgets.
- `results/benchmarks/full/`, `results/walkforward/full/`: distinct empirical studies.
- `scripts/`: independent study verification and report generation.
- `tests/`, `.github/workflows/`: regression, boundary, artifact, and CPU smoke checks.
- `docs/`: audit, original research design, methods, results, and recovery record.
- `notebooks/`: preserved original course work.

## Robustness, statistical reliability, and forecast uncertainty

Phases 6–8 extend the preserved benchmark with an independent audit, shared-origin-and-endpoint evaluation, five-seed sensitivity, dependence-aware forecast comparisons, and chronological prediction intervals. See the [measured results](docs/ROBUSTNESS_RESULTS.md), [predeclared design](docs/ROBUSTNESS_METHODOLOGY.md), [statistical methods](docs/STATISTICAL_RELIABILITY.md), [calibration](docs/FORECAST_CALIBRATION.md), and [limitations](docs/RESEARCH_LIMITATIONS.md).

- Matching retains 2,232 origins per index/horizon. Equal evaluation dates do not equalize training histories.
- Five seeds retrain RF/LSTM on earliest/middle/latest folds in every cell: 90 full-budget tasks, 180 stochastic model-fold evaluations. This is an 18-fold sensitivity subset, not a five-seed rerun of all 282 folds. Fresh seed-171 outputs match all 36 selected original model/fold forecasts numerically.
- An 18-test retrospective primary family uses paired squared losses, Bartlett HAC and Holm correction, with 2,000-replicate moving-block intervals and 20/40/80-session sensitivity. No RF improvement over a simple baseline survives the primary Holm correction. Nonsignificance does not establish equivalence or future accuracy.
- LSTM seed sensitivity is material on selected NASDAQ twenty-session folds: pooled ranks range from first to fifth. This does not reverse its original full-history ranking.
- All five procedures have prequential 80%/90% intervals using strictly matured OOS residuals: trailing 504, expanding, and volatility-scaled alternatives. The independent audit verifies 1,065,960 wide interval records and 2,085,000 evaluated quantiles/bounds, including every calibration population and all coverage/width/score summaries. Wider intervals are not automatically better; no exchangeability-based guarantee is claimed.

```bash
# Audit the saved extension study; no retraining required.
tar -xzf results/walkforward/full/fold_predictions.tar.gz -C results/walkforward/full
python scripts/research_cycle.py verify
# Separate reproducible steps: audit, matched, seeds, statistics, calibration, figures, report.
```

[Artifact schemas and complete commands](results/robustness/README.md) distinguish historical evidence, newly executed training, interval warmup exclusions, primary versus exploratory inference, and unresolved legacy provenance. Ten new figures derive from actual saved artifacts. The older Phase 4/5 report remains a snapshot of that original one-seed study.

## Research completion and manuscript readiness

Phases 9–12 preserve the previous study and add a frozen prospective protocol,
full-budget temporal/seed expansion, delayed adaptive uncertainty with documented
failure cases, and an executable evidence-to-claim synthesis. Independent validation
remains pending genuinely unexposed, provenance-complete future data.

Start with [final findings](docs/FINAL_RESEARCH_RESULTS.md),
[contributions and boundaries](docs/RESEARCH_CONTRIBUTIONS.md),
[Phase 9 status](docs/PHASE9_INDEPENDENT_VALIDATION.md),
[expanded robustness](docs/PHASE10_ROBUSTNESS.md),
[adaptive uncertainty](docs/PHASE11_ADAPTIVE_UNCERTAINTY.md), and
[prospective intake](docs/PROSPECTIVE_INTAKE.md).
The paper can report a qualified retrospective empirical study; it cannot claim
independently confirmed generalization, universal ML inferiority or trading profit.

```bash
python scripts/final_synthesis.py verify
# Fresh full audit logs, without replacing published logs:
python scripts/audit_all_research.py --output /tmp/marketpulse-final-audit
python scripts/phase10.py verify
python scripts/phase11.py verify
python scripts/phase11_diagnostics.py --verify
```

The logged cross-phase audit and manuscript tables/figure are in `results/phase12`.
The original reports remain snapshots of their respective evidence populations.
Future work requires independent acquisition, separately declared method changes,
model weight preservation and economic evaluation before stronger claims.

## Contributors

- Yug More, yugmore20@gmail.com. NASDAQ-100 course notebooks and data. Research-extension design and engineering.
- Srihan Cheemangunta. S&P 500 course notebooks and data.
