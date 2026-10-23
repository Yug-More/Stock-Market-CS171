# Chronological forecast calibration

The fixed interval design and limitations were declared in
[ROBUSTNESS_METHODOLOGY.md](ROBUSTNESS_METHODOLOGY.md) before interval outcomes.
All five original forecast procedures, both indexes and horizons 1/5/20 are
supported. Original point predictions and Phase 4/5 artifacts remain intact.

A forecast at origin date t may use a prior out-of-sample residual only if its
**actual target endpoint date is strictly before t**. Endpoints are reconstructed
from each raw index's own ordered sessions. A shared origin is not assumed to
imply a shared endpoint. No residual calibrates its own interval, and endpoint
ordering, positive horizon and finite values are checked. The current origin's
close is available for its causal volatility; future realized volatility is not.

The primary window retains the latest 504 mature absolute residuals, requiring
252. Two fixed alternatives use all mature residuals or the latest 504 scores
|error| / trailing_origin_volatility, rescaled by the current causal volatility.
They are sensitivities, not alternatives selected by final coverage. Windows
count residual observations, not elapsed calendar days. The widths update
online as old outcomes mature even within an annual fixed-model test block.

For nominal coverage c, the radius is the k-th score, k=ceil((n+1)c). Ties
remain; there is no interpolation. If k>n, the public quantile is infinite and
that origin is unscorable rather than clipping to a falsely finite quantile.
The configured n>=252 and c in {.8,.9} make those quantiles finite. Warmup
origins are retained as explicitly excluded rows. Every interval records the
calibration sample size, first/last score origin and maximum target-date cutoff.
Both levels are stored as wide lower_80/upper_80/lower_90/upper_90 columns.

This is prequential residual calibration of an annually refitted procedure.
Past genuinely OOS residuals can later enter refit training histories. It is
not held-out split conformal for the current fixed fitted model. Training and
model selection are not changed by calibration; there is no exchangeability,
finite-sample distribution-free or guaranteed conditional-coverage claim.
[Conformal theory beyond exchangeability](https://arxiv.org/abs/2202.13415)
requires additional methods/assumptions that are not conferred by using an
order statistic alone.

## Evaluation

Coverage counts inclusive bound hits; gap = empirical minus nominal coverage.
Report mean/median width and the proper central interval score:
width + 2/(1-c)*(lower-y)I(y<lower) + 2/(1-c)*(y-upper)I(y>upper).
Lower score favors sharper intervals that avoid costly misses. A wider interval
is not automatically better. Global, training-defined volatility, and fixed
stress diagnostics are reported on the same eligible origins for all methods.
The runner explicitly checks equality of method eligibility populations.
Unavailable historical slices remain missing; small slices are descriptive.

The independent audit reconstructs every mature residual population using
actual endpoint-date binary searches. A separate count-tree order-statistics
algorithm verifies **every evaluated quantile and bound**, not just examples.
It then recomputes every saved coverage, width and interval-score aggregate
from raw interval records with separate arithmetic. Future-outcome
perturbations, manipulated endpoints and changed-bound adversaries are tested.
