# Phase 16: final research quality assurance

## Scope, evidence, and acceptance

This phase checks the repository state after the Phase 15 computational review.
The authoritative quantitative sources remain the frozen Phase 12 claim registry,
walk-forward predictions and fold plan, Phase 10 seed runs, Phase 11 interval
records, their declared configurations, and the Phase 15 separate-arithmetic
reconstruction. The phase does not train new models or acquire market data.

Acceptance required the complete local test/lint/format suite, configuration and
cross-phase audit success, regeneration and source comparison of manuscript
tables/figures, manuscript claim/reference and PDF checks, and a clean isolated
environment reproduction attempt. The principal risks are unchanged: source-data
acquisition metadata and rights are incomplete, 210 recovered folds lack original
runtime/revision context, fitted weights are unavailable, and no independent future
outcomes exist.

## Checks actually executed

- In the project Python 3.13.2 environment, `python -m pytest` passed all 103
  tests. `ruff check .` passed, and `ruff format --check .` found all 112 Python
  files formatted.
- `scripts/phase10.py verify` passed for 150 full-budget tasks, 150 metric rows,
  330 paired rows, the separate 30-test family, and 1,260 temporal origins per
  cell. Phase 11 verification passed for 1,421,280 interval records, 2,780,000
  independently checked bounds, 1,680 metric rows, 360 score comparisons, and
  its 60-test primary family. The diagnostic verifier retained 55,036 crossings
  across the four variants.
- `scripts/final_synthesis.py verify` passed all 40 registered claims and six
  saved-study audits; its status explicitly says independent confirmation is
  false. The fresh comprehensive audit in
  `results/phase16/cross_phase_audit/manifest.json` records return code zero for
  each of its six commands and `historical_tree_unchanged: true`.
- The separate Phase 15 arithmetic reconstruction was rerun and saved as
  `results/phase16/scientific_review.json`. It rechecked 430,920 raw-label forecast
  records, 282 fold boundaries and frozen means, 210 metric rows, 108 primary
  inference/bootstrap rows across the distinct 18/30/60 comparison families,
  240 adaptive overall metric rows, and 2,780,000 interval bounds. The separate
  family-structure check passed in `results/phase16/comparison_families.json`.
- `scripts/publication_assets.py build` regenerated 12 tables and three figures;
  its source/output verifier passed afterward. Generated publication assets
  matched their recorded sources. Matplotlib emitted Fontconfig cache-location
  warnings on this host, but the build completed and asset verification passed.
- `scripts/manuscript.py build` and `verify` passed for 37 cited quantitative
  claim IDs and nine verified references. Tectonic 0.17.0 compiled the paper to
  14 pages, and the PDF hash verifier passed. All pages were rendered with
  pypdfium2 and inspected as contact sheets; figure pages 7, 9, and 11 and the
  references/reproducibility pages 13–14 were inspected at full page scale. No
  clipping, overlap, unresolved citation, or unreadable figure was found. The
  compiler retains a cosmetic underfull-box warning in one paragraph.
- A new Python 3.12.12 virtual environment installed from
  `requirements-publication.txt` and the local package. In this environment,
  synthesis, publication assets, manuscript, PDF verification, Ruff lint and
  formatting all passed. Its test run passed 101 tests and skipped two
  PyTorch-dependent tests because the publication dependency set intentionally
  omits the optional research extra. The full local environment's 103-test run
  includes those tests.

## Reproduction corrections and failures

The requested audit command using `/tmp/marketpulse-audit-new` initially refused
to run because that directory already existed. The audit script correctly
preserves prior logs rather than replacing them. The same audits passed into a
fresh dated directory and again into the tracked Phase 16 audit directory. The
manuscript appendix and reproduction guide now form a timestamped new path for
each run.

The first PDF build attempt in the project `.venv` could not import `pypdf`, which
is part of the publication dependency set rather than the research environment.
Compilation then succeeded in the fresh environment installed from the documented
publication requirements. Poppler was unavailable; pypdfium2, already included in
those requirements, rendered the PDF for visual inspection. These are environment
limits, not successful Poppler or `.venv` PDF-build checks.

## Outcome and remaining work

No new quantitative discrepancy or high-severity scientific inconsistency was
found in the checked artifacts. The generated paper remains a retrospective
empirical manuscript, not a claim of independent dataset validation, economic
utility, universal model performance, or guaranteed interval coverage. The full
repository README consistency and final release-candidate scan are Phase 17–18
work. This report records local checks; CI status for the Phase 16 pull request
must be reported from the actual GitHub run.
