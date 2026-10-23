# Phase 10 — Expanded temporal and seed robustness

## Acceptance and predeclared expansion

Preserve the original 282 folds and all 90 Phase 6 stochastic run artifacts.
Add exactly 60 full-budget runs: five seeds × 12 intervening index/horizon folds.
Retain both RF depth candidates, both LSTM hidden-size candidates, 200 trees,
25 maximum epochs, patience five and all original sample lengths. No shortened
training substitutes. The existing 18-fold sample showed NASDAQ h20 LSTM ranks
1–5 and opposite RF behavior at short NASDAQ versus long S&P horizons. Additional
seeds alone cannot address that temporal sensitivity.

`configs/phase10.json` fixes added fold quantiles .25/.75, seeds 171/271/371/471/571,
pairs, loss aggregation, bootstrap parameters and multiplicity before new runs.
Combined coverage becomes five folds/cell (30 folds), not a five-seed replication
of all 282. Each new run trains both stochastic models at the original budget.
Old deterministic forecasts and previous stochastic runs are reused byte-for-byte.
Cost estimate: 15–25 minutes based on the original 90-run, 21-minute study;
actual timings are recorded per task, never inferred from the estimate.

Acceptance requires identical origin/endpoint/actual vectors across models/seeds,
independent metric and HAC audits, all 60 task checksums, no cross-gap bootstrap
blocks, full settings/environment/revision records and passing regression checks.

## Statistical estimand

Average the five **losses** at each origin, not the five predictions. This estimates
sensitivity averaged over the fixed seeds; it is not an ensemble evaluation and
five seeds do not multiply the temporal sample size. Thirty primary overall MSE
comparisons (five pairs × six cells) use block length 20 and Holm correction.
Lengths 40/80 and volatility-conditional analyses are exploratory. Bootstrap CIs
are marginal 95%, 2,000 replicates. Bartlett HAC approximations retain chronological
session positions. Blocks never cross the separated selected-year gaps. Small
regime samples remain descriptive when fewer than ten nominal block equivalents.

Models within each index/horizon use exactly the same origins and actual target
endpoints. Different indexes' selected folds have different calendars and training
histories: do not interpret their pooled seed results as a common-calendar contest.
The separately saved full-history endpoint-matched Phase 6 comparison remains the
appropriate cross-index descriptive control. Calendar selection is retrospective
and covers only five separated years per cell; neither Holm nor more seeds repairs
historical researcher selection or creates independent market replications.
