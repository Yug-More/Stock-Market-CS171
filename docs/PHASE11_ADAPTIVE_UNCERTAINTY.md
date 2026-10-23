# Phase 11 — Delayed adaptive forecast uncertainty

## Acceptance and frozen design

Use the original saved out-of-sample point forecasts without refitting or selecting
new models. Materialize every interval and adaptation state for all five models,
two indexes, horizons 1/5/20 and nominal levels 80/90%. Require actual endpoint
maturity for both calibration residuals and coverage feedback; independently audit
every bound and feedback update. Compare the same eligible origins and retain
all four fixed variants, including failures. `configs/phase11.json` is frozen
before evaluation. Estimated cost is several CPU minutes for calibration/auditing,
with no neural retraining and no reduced training-budget substitution.

## Method and relationship to prior work

[Gibbs and Candès (2021)](https://arxiv.org/abs/2106.00170v3) motivate adapting a
miscoverage parameter using realized coverage errors. This implementation is a
**bounded, delayed-feedback ACI-inspired variant**, not an implementation of their
coverage theorem. At an origin, process earlier predictions in chronological
endpoint order only when `target_date < origin`. A mature, previously eligible
interval contributes its own miss indicator; warmup forecasts provide residuals
but no fictitious interval feedback. Update each level's state by
`alpha <- clip(alpha + gamma * (target_miscoverage - past_miss), .02, .5)`.
Then take rank `ceil((n+1)*(1-alpha))` of the latest mature absolute residuals.
The finite bounds follow from the declared alpha bounds and minimum history 252;
clipping events are recorded. Interval scores always use the fixed nominal level,
not the changing internal alpha. Point models and the interval center do not change.

The factorial sensitivity is windows 252/504 × gamma .01/.05, minimum 252. The
reference is existing trailing-504; expanding and volatility-scaled methods are
included in coverage/width/score comparisons. The designated primary sensitivity
variant is w504/gamma .01, not the best retrospective performer. Its 60 overall
interval-score comparisons versus trailing (30 point-model cells × two levels)
form one Holm family at L20. L40/L80 and other variants are exploratory. Coverage
bootstrap intervals are marginal diagnostics, not simultaneous coverage tests.
Session-preserving moving blocks and Bartlett HAC use the existing audited methods.

[Barber et al.](https://arxiv.org/abs/2202.13415v5) discuss conformal inference when
exchangeability fails; their guarantees must not be transferred to this different
procedure. Delayed, overlapping financial targets, bounded adaptation, repeatedly
refitted point models and market shifts prevent an arbitrary distribution-free,
finite-sample or conditional-regime guarantee here. Earlier OOS outcomes can enter
later point-model training. This is prequential retrospective procedure calibration,
not split conformal calibration of one permanently fixed fitted model.

## Reproduction and independent validation

`python scripts/phase11.py run` creates a fresh, immutable result bundle under
`results/phase11`. `python scripts/phase11.py verify` independently reconstructs
mature populations using endpoint searches, quantiles using a count tree rather
than sorted lists, feedback using saved past bounds, and metrics using separate
arithmetic. Tests cover irregular sessions, future-outcome perturbations, equal-date
exclusion, clipped states, invalid endpoints and the zero-adaptation reference.
Compressed records omit duplicated outcomes/predictions/regimes: join their unique
dates to the SHA256-pinned original point forecasts. This is a compact lossless
evidence representation, not a change in evaluation population.
