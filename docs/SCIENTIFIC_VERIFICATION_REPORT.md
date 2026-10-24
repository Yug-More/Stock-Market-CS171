# Phase 15: adversarial scientific verification

## Scope, dependencies and acceptance

This is a computational scientific review of the existing MarketPulse evidence,
with independently implemented metric, HAC, bootstrap and Holm arithmetic. It is
not independent dataset validation, a blinded human review, or a fresh training
replication. The same raw CSV format loader is reused explicitly. Acceptance
requires reconstructing major reported quantities from saved predictions, checking
chronology and declared comparison products, preserving unfavorable results and
qualifying any conclusion that exceeds the available evidence.

The review code is `src/marketpulse/scientific_review.py`; run
`python scripts/scientific_verification.py --output NEW_REPORT.json`.
`results/phase15/scientific_review.json` records the actually executed review and
all read inputs with SHA256 values. `comparison_families.json` records the separate
structural family check. The six complete earlier audits were executed again in
`results/phase15/cross_phase_audit`; their logs and command return codes distinguish
execution from claims of success. Original experiments were not altered.

## Checks and attempts to falsify the conclusions

1. **Wrong targets or unequal models' populations.** Recomputed log-close targets
   against raw histories for 355,320 native forecasts and 75,600 stochastic seed
   forecasts (430,920 records, including repeated outcomes). Checked uniqueness and
   identical origin, target and fold sequences for all five native models in each
   of six cells. Reconstructed actual endpoint matching: 2,232 per cell. No mismatch
   survived the checks. This validates arithmetic against these raw files, not the
   files' correspondence to an independently acquired market history.
2. **Purging, baseline or chronology errors.** For all 282 folds, mapped recorded
   dates back to raw rows and checked train-end+h < validation-start and
   validation-end+h < test-start; checked test boundaries/counts. Recomputed every
   frozen training-target mean from the actual training rows and checked zero
   predictions exactly. Every condition passed. Scaling, causal feature and
   within-run validation/early-stopping protections were inspected in the initial
   audit and exercised by the passing test suite and original artifact audits.
   This cannot establish that researchers had never seen these periods.
3. **Misleading pooled metrics or seed aggregation.** Recomputed RMSE, MAE, signed
   prediction bias, directional accuracy, both baseline MSE skills and ranks for
   210 metric rows: 30 native, 30 endpoint-matched and 150 selected-fold seed rows.
   Read all 150 seed task records/forecast files, checked their raw-price targets
   and 1,260 origins per cell/seed. Reconstructed seed-averaged squared losses and
   compared them with the saved loss table. No discrepancy was found. These are
   averages of losses, not errors of averaged predictions or 6,300 independent dates.
4. **Incorrect significance or selectively shortened families.** Separately checked
   the exact Cartesian products and uniqueness of the original 18, expanded 30 and
   adaptive 60 primary families against their actual configuration files, including
   block length and variant/level selection. Recomputed all 108 effects, Bartlett
   HAC standard errors and normal-reference p-values with separate arithmetic;
   recalculated marginal 95% block intervals using convolution block sums rather
   than the production two-column prefix-sum implementation. The three actual
   bootstrap seeds were retained. Recomputed Holm with an independent sorted loop.
   All matched within stated numerical tolerances. This establishes reproducibility
   of approximate inference, not validity under arbitrary dependence or correction
   for all historical project choices.
5. **Coverage gains hiding width or method failures.** Recomputed coverage, mean
   width and proper interval score for all 240 overall adaptive model/variant/level
   rows, checking 2,780,000 eligible level/bound pairs, finite ordered bounds,
   matching trailing eligibility and strictly past residual cutoffs. The complete
   fixed-method/regime metrics, residual quantile order statistics and delayed
   adaptive state histories were independently replayed by the six existing audits.
   New checks confirm the saved overall metrics and all 120 diagnostic rows.
6. **Unreported incoherence between nominal levels.** Recomputed crossings from
   actual stored bounds for every variant: 1,368 (252/.01), 25,819 (252/.05),
   1,494 (504/.01) and 26,355 (504/.05). Checked cumulative clipping counters are
   nondecreasing and their terminal values match the diagnostic counts. Violations
   remain in both evidence and manuscript. No post-hoc projection or retuning was
   performed. These repeated procedure-origin pairs are not independent observations.
7. **Manuscript or presentation overstatement.** Regenerated its 37 cited C-claims,
   tables and methodological mappings; verified the 40-claim prior synthesis and
   new paper/figure hashes. The manuscript includes all native cells and original
   primary contrasts, seed ranges, long-horizon undercoverage, separate nominal/Holm
   columns and all four nesting variants. Complete CSV supplements retain other
   models/levels/regimes and null findings. References have authentic recorded
   verification sources. Neither the abstract nor conclusion claims profitability,
   untouched independent data, universal model inferiority or distribution-free
   coverage. The historical Phase 9 rehearsal is explicitly infrastructure evidence.

## Verified findings and their limits

The native RF RMSE wins remain three of six cells, and native LSTM remains worst
in all six. No RF improvement over either simple baseline survives the original
18-test family; RF has corrected evidence of worse S&P h20 loss against both
simple baselines. The expanded 30-test family supports no ML-baseline gain; its
NASDAQ h20 LSTM rank range is 1–5 across the fixed seeds. Rankings change with the
matched population, including S&P h5's linear-regression winner. These are
conditional retrospective findings, not equivalence, universal inferiority or
future performance guarantees.

The designated adaptive method has 20 corrected score improvements, confined to
S&P h1/h5, across five point procedures and both nominal levels. No primary NASDAQ
or h20 improvement survives adjustment. Lower proper scores coexist with reduced
h20 coverage: zero-return S&P 90% coverage is 0.876663 versus trailing 0.885353
(display-rounded). The 1,494 designated-method nesting violations persist. A
particularly instructive nominal-only comparison is NASDAQ h1 historical-mean
80% interval score: p=0.00685712, Holm p=0.246856 (display-rounded). It does not
support a corrected finding. The full primary tables and raw review are authoritative.

## Evidence classification

- **Verified from available predictions and raw-file labels:** the reconstructed
  point metrics, paired populations, boundary conditions, baseline means, primary
  statistical arithmetic, saved overall adaptive metrics and nesting counts above.
- **Supported by processed artifacts plus existing replay audits:** regime/stress
  summaries, full fixed/calibration quantiles and delayed-state trajectories; all
  six audits were actually rerun, but new separate arithmetic in this phase is
  concentrated on the manuscript's primary comparisons and overall metrics.
- **Partially reproducible:** canonical model training. Existing configured fits,
  execution records and targeted seed reruns support comparisons; this phase did
  not refit all models. Weights are absent and 210 recovered folds lack original
  revision/environment/runtime context. Recovery metadata cannot repair that gap.
- **Unverified:** complete original acquisition timestamps/versions, data rights,
  historical exchange-closure completeness, original unseen-data exposure, exact
  legacy generating environment and platform-independent neural replay.
- **Requires new independent evidence:** future chronological performance, other
  markets, practical/economic utility, a coherent revised multilevel interval method
  and any stronger generalization. No such experiment is claimed here.

## Discrepancies and corrective actions

The initial mandatory audit found three README issues (stale absence-of-inference
and intervals, an incorrect terminal S&P h5 narrative, and an unsupported economic
interpretation); Phase 13 corrected them transparently. No stored experimental
value required alteration. During development of the new review, cumulative
clipping counters were initially summed as if they were increments; the failed
assertion exposed this review-code error. The check now uses terminal counter
values and monotonicity, consistent with the state implementation. Per-study
bootstrap seeds are read from their actual configurations. Failed development
runs did not produce a passing report. The committed report was executed after
these corrections. The review does not erase these development failures.

All 103 local tests passed, including hand-calculated Holm and contiguous/gapped
HAC examples for the new arithmetic. Ruff lint/format, manuscript bindings, PDF
hashes, exported-table verification and the prior synthesis passed. All six
cross-phase audits passed; see the saved return codes rather than inferring
success from the presence of a directory.

## Remaining scientific uncertainty

There is no contradictory primary arithmetic remaining in the tested evidence.
The principal threats are inferential assumptions, nonstationarity, overlapping
labels, correlated US indexes, unequal histories, limited seed/fold/architecture
coverage, repeated researcher exposure and incomplete provenance. Lack of corrected
significance is not evidence of no meaningful effect; no equivalence or power
study was run. No held-out independent result or trading experiment exists.
External human scientific review and author/data-rights decisions remain necessary.
