# Phase 9 — Independent chronological validation

## Acceptance and frozen design

Accept only four ordered, nonempty blocks with every earlier label endpoint strictly
before the next block's first origin; reject reuse of the historical period as
independent; preserve all prior artifacts. Execute the same five canonical models
at full original budgets on a historical wiring rehearsal, separately from any
confirmation. The machine-readable freeze is `configs/independent_validation.json`.
It fixes hypotheses, model grids, seed, metrics, interval rule, inference family,
252-origin test and stopping rule before prospective outcomes exist.

The first permitted prospective origin is 2026-10-26 (after this protocol's
2026-10-23 freeze). Data must eventually contain 252 test origins and all actual
20-session endpoints. This is a future requirement, not an acquired dataset or
completed independent experiment. Earlier dates must not be substituted.

## Acquisition and independence audit

Both repository raw files were already used in Phases 1–8 and in historical
notebooks. NASDAQ-100 covers 2000-01-03–2024-12-31; S&P 500 covers
1927-12-30–2020-11-04. Their acquisition timestamps, exact vendor versions and
licenses were not recorded. The NASDAQ notebook's exclusive-end call conflicts
with the last saved date. S&P early history predates the modern index. Price
closes do not establish dividend-adjusted total returns. See `configs/datasets.toml`,
`docs/RECOVERY.md`, `docs/RESEARCH_LIMITATIONS.md` and the saved Phase 6 historical
audit for the decision/provenance record. The raw hashes and limitations remain
unchanged. Later historical observations are not independent simply by date.

No additional acquired, provenance-complete, unexposed dataset is present. We do
not assert that a public historical download establishes lack of prior exposure.
Independent validation is **blocked by unavailable prospective data**, not passed.
No new vendor source or license is claimed. Future intake requires source URL,
version, UTC retrieval, adjustment convention, license, SHA256 and explicit exposure
declaration. A declaration is necessary but cannot prove human independence.

## Implementation and execution

`four_blocks` reserves 252 test origins, then 252 calibration origins, then 252
validation origins with horizon-specific purges, leaving at least 2,520 training
origins. Candidate selection, scaling and early stopping use train/validation only.
The existing full-budget trainer predicts calibration and test in a single call;
its combined diagnostic metrics are not used for selection. Calibration residuals
alone determine fixed symmetric 80/90% radii. They are all mature before test.
The model is not refit after selection. Test residuals never update these radii.

`python scripts/independent_validation.py` executes six terminal historical
rehearsals (two indexes × three horizons), five models each. Output is exclusively
`results/phase9/rehearsal/`, labeled retrospective. The manifest records actual
block dates, raw hashes, execution environment and the frozen protocol hash;
compressed forecasts, selected settings and interval metrics are saved separately.
Rehearsal scores are engineering evidence, not tests of generalization, and are
excluded from prospective inference. To preserve evidence, rerunning into an
existing output is rejected. The future acquisition gate is intentionally fail-closed.

The frozen analysis specifies 24 primary comparisons (two stochastic models vs two
baselines × six cells), paired MSE, primary block length 20, sensitivity 40/80,
2,000 bootstrap replicates and Holm correction. Marginal CIs are not simultaneous.
Actual future inference must be reported only after provenance/exposure review and
all target outcomes mature. Temporal dependence and shifts preclude an arbitrary
distribution-free coverage claim. No independent effect size or ranking exists yet.

## Completed verification

All six full-budget rehearsals completed: 30 model fits, 15,120 stored calibration
and test forecasts, and 60 interval metric rows. `python scripts/audit_phase9.py`
independently checks raw-close targets, all actual endpoint boundaries, calibration
order statistics, coverage, widths, scores and RMSE. The retrospective metrics do
not enter the confirmatory family. Unit tests include horizons 1/5/20, internal
session gaps, insufficient history and rejection of historical holdouts.
