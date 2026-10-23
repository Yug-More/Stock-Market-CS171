# Statistical reliability

The exact execution registry is in `configs/robustness.toml` and the pre-execution
[methodology](ROBUSTNESS_METHODOLOGY.md). Results and measured tables are in
[ROBUSTNESS_RESULTS.md](ROBUSTNESS_RESULTS.md); this document specifies interpretation.

## Estimand and time-series inference

For two procedures forecasting identical origins and actual target endpoints,
let d_t = loss_A,t - loss_B,t. The primary estimand is its sample-population
mean over native-calendar historical test origins, using squared log-return
errors. Negative differences favor A. MAE differences and endpoint-matched
populations are exploratory. Original seed-171 forecasts define the full-study
inference population. Optimization-seed sensitivity is reported on 18 selected
folds separately; repeated seeds are not added to historical sample sizes.

A 2,000-replicate moving-block bootstrap samples chronological paired blocks,
never shuffled individual days, with lengths 20/40/80 and random seed 6171.
Every model pair shares a draw. Blocks stop at structural gaps in raw session
positions; they can cross yearly model refit boundaries when actual sessions
are consecutive, reflecting dependence in the forecasting procedure. Prefix
sums implement the same resampling as explicitly materialized row blocks;
a separate synthetic unit test verifies their equality, including the truncated
final block. Percentile intervals use linear quantiles at .025/.975.

Bartlett HAC estimates the variance of the loss-differential sample mean:
(gamma_0 + 2 sum_{k=1}^{L-1}(1-k/L) gamma_k)/N. Autocovariances exclude pairs
crossing structural gaps. A two-sided normal-reference statistic compares the
mean differential with zero. This is a DM-style predictive-accuracy comparison
conditional on the selected expanding-window procedure, not a claim that a
textbook fixed-model DM test exactly applies to repeated estimation here.
No small-sample t or iid calibration is asserted. The original procedures,
shared expanding training sets, long distribution shifts, and overlapping
multi-session labels limit the weak-dependence/asymptotic approximation.

All 18 native-calendar squared-error tests at L=20 form one declared Holm
family. CIs remain **marginal**, not simultaneous familywise intervals.
Changing L or population does not create an opportunity to choose the most
favorable p-value. Other rows are explicitly sensitivity/exploratory analyses.
Because RF and these histories were already examined in Phase 4/5, even this
fixed follow-up family is retrospective and not independent confirmation.

## Regime diagnostics

Resampling carries membership indicators on the full original timeline.
Conditional loss means divide resampled masked losses by resampled counts;
noncontiguous volatility observations are never made adjacent by filtering.
HAC uses influence I_t(d_t-mean_selected)*N/n_selected on that same timeline.

At least ten nominal block equivalents n/L are required. This is a stability
screen, not estimated effective sample size. Empty slices are unavailable;
short slices are descriptive with no p-value/CI. COVID (52 origins) and the
financial-crisis slice (146 origins) fail this screen at L=20. Earlier NASDAQ
crises remain unavailable. Regime p-values are exploratory and not part of
the primary Holm family. No independent-episode replication is inferred.

## Meaning of outcomes

A descriptive RMSE winner can have inconclusive adjusted evidence versus a
baseline. A nonsignificant difference does not establish equivalence. Small
p-values describe approximate historical loss evidence under the assumptions;
they do not establish material economic value, future stability, crash
prediction, or profitable trading. Effect sizes in squared-log-return units,
relative MSE skill, bandwidth sensitivity and uncertainty bounds must be read
together. No comparison is selected solely because it is favorable.

Method reference: [Newey and West's original HAC paper](https://www.nber.org/papers/t0055).
