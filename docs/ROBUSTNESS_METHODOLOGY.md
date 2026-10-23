# Phases 6–8 preregistered execution design

Declared on 2026-10-23 before new multi-seed, inference, or interval results.
This is a prospective execution registry for a retrospective follow-up on
already examined market history, **not** an independent confirmatory study.
Baseline: main `63d044a3fb9fb603b21c762e2ac752791af27148`; the independent
Phase 4/5 audit passed 282 folds / 355,320 predictions before extensions.
No prior Phase 6–8 work or active experiment was recovered. Original artifacts,
raw datasets, and scientific definitions remain intact.

## Calendar population

Map each saved forecast origin back to its raw price row and actual target
endpoint date. Primary matching requires both shared origin and shared
endpoint, per horizon; origin-only matching is a sensitivity analysis.
Require all five models and valid labels on each date in both indexes.
Record membership and attrition from each native-calendar population.
Matching does not equalize training histories, fold calendars, or information
sets, and does not establish independent markets or transfer learning.

## Stochastic replication and cost

Random forest randomness includes tree resampling/feature selection. LSTM
randomness includes initialization and shuffled optimization. Fix seeds
171, 271, 371, 471, 571 before execution. Select folds at floor(q*(F-1)) for
q=0, 0.5, 1 independently within every index/horizon cell: earliest, middle,
latest, selected without consulting scores. This yields 18 distinct folds,
90 fold-seed tasks and 180 stochastic model-fold evaluations. Retrain seed
171 too, giving new, separate provenance to a targeted replication of 18
original folds. Reuse deterministic forecasts for comparison, not retraining.

The recorded 72-fold continuation used 1,452.56 seconds on Apple M2 Pro CPU,
16 GiB RAM. A full five-seed 282-fold design would require roughly eight
CPU hours at that S&P h=20 average; the balanced subset is estimated at about
30 minutes, with actual runtime recorded. These are planning estimates, not
promised runtimes. Keep the original 200-tree forest, both depth candidates,
20-step LSTM, both hidden sizes, 25-epoch maximum and patience 5. Use the
existing score_split interface, chronological splits, scalers and validation
selection unchanged. Seeds are optimization sensitivity on the same history,
not independent historical replications. This subset cannot establish full
81-fold stochastic stability or represent all crisis episodes.

## Statistical estimands and registry

Primary loss is squared log-return forecast error. The paired estimand is
mean(loss_A-loss_B) over native-calendar test origins in one index/horizon.
Negative favors A. Fixed pairs: RF versus zero, RF versus historical mean,
and RF versus LSTM, across both indexes and all three horizons (18 tests).
Apply Holm correction jointly to these 18 primary two-sided HAC tests at
block length 20 (bandwidth 19). They are primary follow-up comparisons,
not selection-free confirmatory discoveries: RF and these histories have
already been examined. Do not select only favorable comparisons.

Use moving-block bootstrap percentile 95% intervals for paired loss
differences and relative MSE skill, 2,000 replicates with seed 6171.
Block lengths 20/40/80 cover the longest overlapping horizon and test longer
clustering; none is optimized to observed significance. Bartlett HAC variance
uses bandwidth L-1 on the full original session timeline. Never join across
structural origin gaps. Do not treat folds as independent. A block-equivalent
count n/L is a heuristic diagnostic, **not** estimated effective sample size.
Below 10 block equivalents, formal inference is withheld as descriptive.
Report native and endpoint-matched populations separately; only native,
L=20 primary-family tests receive the primary Holm adjustment. Other
bandwidths/populations and MAE intervals are exploratory sensitivity checks.

For regime subsets, retain the entire chronological series and resample
losses together with membership indicators; never compress disjoint high-
volatility days into artificial consecutive sessions. Conditional means use
resampled membership counts. HAC uses the membership-weighted influence
series on that original timeline. Fixed stress periods and training-defined
terciles are retained. Short slices such as COVID and the financial crisis
may be descriptive; unavailable NASDAQ episodes stay unavailable.
Inference is conditional on the fitted forecasting procedure and historical
population. Weak-dependence/stationarity approximations may fail under long
market shifts; HAC/bootstrapping do not fix post-selection or create new data.
Seeds are reported separately, not pooled into a larger historical sample.

## Chronological interval registry

Use genuinely out-of-sample residuals from the original saved procedure,
not fitted training residuals. For a forecast at date t, admit a residual
only when its **actual raw-file target endpoint date is strictly before t**.
Use at least 252 mature residuals. Primary method is the latest 504 eligible
residuals; expanding history is a fixed sensitivity alternative, not chosen
by test coverage. Evaluate both at nominal levels 80% and 90%.

Use symmetric absolute-residual intervals with order statistic
k=ceil((n+1)*coverage), one-based, with ties retained; if k>n, interval is
infinite/unscorable rather than silently clipping. Volatility-scaled
sensitivity uses scores |error|/origin_volatility and rescales by current
causal origin volatility; no future volatility, fitted test thresholds, or
regime outcome enters its construction. Its window is also 504.

This is prequential calibration of an annually refitted forecasting
**procedure**, not held-out split conformal for a fixed model. Past OOS
residuals can later enter a refit's training history. Training and model
selection are completed before each original test block; calibration never
fits or selects a model or uses the current origin's outcome. Within-block
interval updates use only already matured prior forecast outcomes, a declared
online policy. There is no exchangeability or distribution-free coverage
claim, and no choice among these fixed alternatives using final test results.

Record bounds, eligibility/warmup exclusions, calibration n, first/last score
origin, maximum actual residual target date, finite-sample quantile, method,
and nominal level. Report coverage, gap, mean/median width, proper central
interval score, and regime/stress diagnostics on both each method's eligible
population and common comparable eligible origins. Do not call wider
intervals better merely for covering more outcomes. Conditional coverage
is descriptive, not guaranteed. Endpoint-date and future-outcome perturbation
checks must pass before research execution.

## Method sources

Bartlett HAC follows [Newey and West](https://www.nber.org/papers/t0055).
The interval quantile is conformal-style; the distinction between this
empirical baseline and methods with explicit non-exchangeable theory is
motivated by [Barber et al.](https://arxiv.org/abs/2202.13415) and
[Gibbs and Candès](https://arxiv.org/abs/2106.00170). No guarantee from those
methods is attributed to the simpler procedure implemented here.
