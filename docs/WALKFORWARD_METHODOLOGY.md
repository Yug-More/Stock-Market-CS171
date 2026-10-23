# Walk-forward and regime methodology

## Task and data

At the close of session t, predict `y_t(h) = log(Close[t+h]/Close[t])`
for h in {1, 5, 20}. Sessions are price-file rows, not calendar days.
The existing unmodified index-level files and ten causal features are
specified in [data provenance](../data/README.md) and
[the forecasting pipeline](FORECASTING_PIPELINE.md). This is a price-index
return task, not a dividend-reinvested return or a trading evaluation.
The original one-terminal-window study remains separate and unchanged.

## Expanding windows

The grid is anchored at the latest eligible origin for each index/horizon.
Each test block contains 252 eligible origins; moving backward 252 origins
constructs earlier blocks. Returning them chronologically produces expanding
training histories with no repeated test origin. Each fold uses 252
validation origins and at least 2,520 training origins after purging.
The saved `fold_plan.json` lists all train, validation, test, and purge counts
and exact start/end dates. `fold_metrics.csv` repeats these bounds by model.

Eligibility requires a finite label and a complete 20-session feature window.
On these files eligible origins are consecutive trading-session rows, so
252 origins is 252 trading sessions. If future missing features create
internal holes, the existing generator still advances in eligible origins;
that would require an explicitly revised session-grid protocol. No missing
observation is interpolated. Calendar closures are preserved.

For a later block beginning at row s, an earlier origin t is admitted only
if `t+h < s`. Origins whose labels touch the later block are purged. This
strict rule is used at both train/validation and validation/test boundaries.
It is conservative relative to fitting after the close at s: labels maturing
at s itself are excluded. Training remains fixed throughout the test block;
new feature observations are available at each origin, but no within-block
model refit occurs. This is yearly block walk-forward evaluation, not a
claim of daily retraining. Validation labels all mature before testing.

## Fitting and selection

Each fold refits all five canonical forecasts on the same training origins:
zero return, training-target mean, standardized linear regression, random
forest, and one-layer sequence LSTM. The original notebook Deep LSTM is
historical background, not part of the Phase 2/3 canonical return benchmark.

Scalers see training-origin features only. The LSTM scaler transforms each
causal sequence with those training statistics; a sequence ends at t and
predicts the label stored at t. Labels never enter feature construction.
Feature construction can run once on the entire price frame because all
rolling/lag operations are causal; this is equivalent to constructing each
prefix separately. Existing future-perturbation tests check this property.

Random forest: 200 trees, leaf minimum 5, depth in {4, 8}; choose lowest
validation RMSE, smaller depth on ties. LSTM: lookback 20, hidden size in
{16, 32}, Adam 0.001, batches 256, at most 25 epochs, patience 5; restore
best validation epoch and choose lowest validation RMSE, smaller size on
ties. Test rows never select a depth, hidden size, epoch, or scaler.
Seed 171 is fixed for Python/NumPy/PyTorch and scikit-learn. CPU execution
and forest `n_jobs=1` bound resources. One seed is used: the proposed
multi-seed robustness study has not been performed. LSTM checkpoint metadata
is stored (selected size, best epoch, history), but fitted weights are not
serialized; reproduction requires retraining. No claim of bitwise equality
across hardware, dependency versions, or thread settings is made.

## Metrics and ranking

RMSE is `sqrt(sum((prediction-outcome)^2)/n)`. MAE is the mean absolute
residual. Skill against baseline b is `1-MSE_model/MSE_b`, using predictions
on the exact same dates. The historical-mean baseline is the frozen mean of
each row's own fold, not a mean refitted to the pooled test sample.
Undefined skill at zero baseline MSE is missing. Metric functions reject
empty, misaligned, and nonfinite vectors. Directional accuracy excludes
zero outcomes and does not count a zero prediction as a matching sign.

Pooled metrics concatenate forecasts before calculating errors; they are not
arithmetic means of fold RMSE. `fold_summary.csv` labels mean and median
fold RMSE separately. Fold ranks use lowest test RMSE as rank 1 and minimum
rank for ties. The summary reports rank mean, population standard deviation,
number of rank changes between consecutive folds, fold wins, and counts
beating each baseline. Slice-level ranks are calculated within each
index/horizon/regime or stress slice. These are descriptive rankings with
no significance claim. Overlapping multi-session labels, shared expanding
training sets, and volatility clustering create dependence.

## Historical stress

Membership uses the **forecast-origin date**, with inclusive calendar bounds
fixed in `RESEARCH_DESIGN.md` before the run:

- Dot-com drawdown: 2000-03-01 through 2002-12-31.
- Financial-crisis slice: 2008-09-01 through 2009-03-31.
- COVID shock: 2020-02-15 through 2020-04-30.

The financial-crisis slice is narrower than the full 2007–2009 episode.
Multi-session labels may realize beyond a slice's end; no target-date or
overlap membership is silently substituted. All eligible origins in a slice
are counted, then compared with actually tested origins. Coverage is complete,
partial, or unavailable. Unavailable slices have zero observations and missing
errors, not invented forecasts. NASDAQ-100 begins in 2000; the training
minimum precludes its dot-com and financial-crisis tests. S&P 500's longer
history provides those slices. Outside-stress rows supply a descriptive
comparison, not a matched causal counterfactual. Boundary sensitivity was
not optimized or tested.

## Causal volatility regimes

Volatility at t is trailing 20-session standard deviation of daily log
returns through t with `ddof=1`. Each fold freezes linear NumPy quantiles
1/3 and 2/3 of volatility at its **embargo-clean training origins**. These
thresholds label later test origins: low if v <= q1, medium if
q1 < v <= q2, high if v > q2. Equal thresholds leave medium empty; no
forced equal-size test bins are used. Missing/nonfinite values are rejected.
Thresholds are stored per fold and repeated on forecast rows.

Slice metrics use the same observations for every model and both baselines.
Absent regime groups have no fabricated metrics. `rmse_ratio_to_low` is a
descriptive RMSE ratio to that model's low-volatility slice, missing if the
reference slice is absent or zero. High-error regimes need not imply worse
baseline-relative forecasting skill: the underlying target scale changes.
Small stress samples, especially COVID, limit interpretation.

## Artifacts and recovery

Run `marketpulse-walkforward --mode smoke --output <new-directory>` before
`marketpulse-walkforward --mode full --output results/walkforward/full`.
The full runner plans all folds, preserves valid recovered predictions, and
stops on incompatible or corrupt artifacts rather than silently excluding
them. `manifest.json` fingerprints source, configuration, and raw input bytes
with SHA-256 and records the recovery environment and recovered file hashes.
Changing these inputs requires a fresh output directory. The exact planned
file set must be complete before pooled results are written.

Each new fold records settings, runtime seconds, generating Git revision,
Python/platform/dependency versions, and a prediction checksum. CSV and JSON
files use temporary-file replacement; interruption between those two writes
can leave a detectable orphan, which is preserved for inspection. Completed
folds are independently checked against fresh splits, targets, thresholds,
baselines, and recomputed metrics on resume. No model is reused across folds.
A failed model prevents that fold from being declared complete.

Recovered legacy folds lack execution revision/environment/runtime. Their
internal consistency was validated; those missing provenance facts remain
unknown and are disclosed in `RECOVERY.md` and the manifest. The recovery
revision must not be mistaken for the generating revision of old folds.

## Scope and threats to validity

The indexes use different native historical calendars and training lengths.
They are separate case studies; no raw-error cross-index ranking or matched
calendar generalization claim follows. Price-source versions and retrieval
dates are incomplete, early S&P history predates the modern index, and a
historical index is not a point-in-time constituent panel. Constituent
selection/survivorship effects cannot be measured with these aggregate files.
Market nonstationarity, one-seed neural variability, limited tuning grids,
validation overfitting, overlapping horizons, and few stress episodes limit
conclusions. No confidence intervals, formal significance tests, calibrated
prediction intervals, transaction costs, strategy returns, or evidence of
profitable trading are supplied by these phases.
