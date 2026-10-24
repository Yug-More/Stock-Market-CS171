# Phase 18: final research release candidate and handoff

## Scope and release status

This report closes the requested research-completion sequence using only the
existing MarketPulse evidence. It is a release-candidate handoff for technical
and academic review. It is not a journal-ready or peer-reviewed publication, and
no GitHub release or version tag is created by this phase. No new market data or
forecast experiment was added.

## Work completed by phase

- **Phase 13 — manuscript:** added the structured retrospective research paper in
  Markdown, TeX, and PDF, a verified bibliography, 37 quantitative claim bindings,
  and the claim-to-evidence audit. The bibliography verification covers nine
  references. [PR #9](https://github.com/Yug-More/MarketPulse/pull/9) merged.
- **Phase 14 — publication assets:** regenerated three labeled figures and twelve
  evidence tables from saved results, with source manifests and reproduction
  instructions. [PR #10](https://github.com/Yug-More/MarketPulse/pull/10) merged.
- **Phase 15 — scientific verification:** independently reconstructed primary
  forecast metrics, fold boundaries, inference arithmetic, interval metrics and
  nesting diagnostics. This is a computational review of existing evidence, not
  independent dataset validation. [PR #11](https://github.com/Yug-More/MarketPulse/pull/11)
  merged.
- **Phase 16 — final quality assurance:** ran the full local suite, clean-environment
  reproduction, figure/table rebuild, manuscript/PDF checks, and six cross-phase
  audits; corrected the appendix's reused audit-output path. [PR #12](https://github.com/Yug-More/MarketPulse/pull/12)
  merged.
- **Phase 17 — repository presentation:** rewrote the README as a concise research
  overview and checked its links and headline claims. [PR #13](https://github.com/Yug-More/MarketPulse/pull/13)
  merged.
- **Phase 18 — release-candidate audit:** reran synthesis and manuscript checks,
  separate primary arithmetic, comparison-family checks, the full saved-study audit,
  and a tracked-file secret scan. Logs and reconstruction outputs are under
  `results/phase18/`.

The required CI jobs passed on each merged Phase 13–17 pull request. The archived
PR checks are available from the links above.

## Verified scientific findings

The retrospective core contains 282 expanding-window folds and 355,320 stored
forecasts for zero return, a frozen training-target mean, linear regression, random
forest and one sequence LSTM. It covers two stored US price-index histories at
1-, 5- and 20-session horizons.

Random forest has the lowest native pooled RMSE in three of six cells, with a
simple baseline winning the other three. No random-forest improvement over a simple
baseline survives the original 18-comparison Holm family, and no machine-learning
improvement survives the distinct expanded 30-comparison family. The original
family does support corrected evidence of higher random-forest loss than both
simple baselines for S&P 500 at horizon 20. LSTM has the highest native pooled
RMSE in all six cells. These results do not establish equivalence, universal model
inferiority, or future performance.

In the separate 60-test adaptive interval family, the designated bounded,
delayed-feedback variant has 20 Holm-adjusted proper-score improvements, limited to
S&P 500 horizons 1 and 5 against trailing calibration. Its long-horizon intervals
undercover at the nominal 90% level, and 1,494 of 347,500 eligible interval pairs
violate nesting between the nominal 80% and 90% bounds. The procedure has no
distribution-free coverage guarantee. Score improvements do not establish economic
utility.

The Phase 9 historical rehearsal remains infrastructure evidence. Independent
chronological validation is unavailable; there are no prospective outcomes and no
provenance-complete unexposed dataset in the repository. The frozen intake protocol
permits its first new test origin on 2026-10-26, subject to actual data acquisition,
provenance review, and endpoint maturity.

## Verification performed for this handoff

- `scripts/final_synthesis.py verify` passed 40 registered claims, six prior audits,
  and reports `independent_confirmation: false`.
- `scripts/audit_all_research.py --output results/phase18/cross_phase_audit` passed
  all six audits; the manifest records `historical_tree_unchanged: true`.
- The separate-arithmetic review passed after checking 430,920 forecast labels,
  282 fold boundaries and frozen means, 210 metric rows, 108 primary inference and
  bootstrap rows across separate 18/30/60 families, 240 adaptive metric rows, and
  2,780,000 interval bounds. The nesting counts by variant are in
  `results/phase18/scientific_review.json`.
- The family-structure audit passed for the declared 18-, 30-, and 60-comparison
  products in `results/phase18/comparison_families.json`.
- Publication assets verified: twelve tables and three figures. Manuscript
  verification passed for 37 claim bindings and nine references. The compiled
  manuscript PDF has 14 pages and passed its recorded source-hash check; its
  rendered pages were visually checked in Phase 16.
- The full Python 3.13.2 local environment passed all 103 tests, Ruff lint and
  format checks. The isolated Python 3.12.12 publication environment passed 101
  tests and skipped the two tests requiring optional PyTorch; its synthesis,
  publication, manuscript, PDF and Ruff checks passed. Both Phase 16 and Phase 17
  pull requests passed their two GitHub CI jobs; Phases 13–15 did as well.
- A pattern scan covered 752 tracked text files and found no matching credential
  patterns or sensitive key filenames. It skipped 407 binary files. No tracked
  Python cache, bytecode, or editor metadata files were found. The scan is a
  repository check, not a guarantee about secrets in every historical Git object.
- The Phase 13–17 commit records use the configured human Git email
  `yugmore20@gmail.com`; no `Co-authored-by` trailers were present.

The initially requested audit directory `/tmp/marketpulse-audit-new` already
contained evidence and was correctly refused. The audits passed in the fresh,
tracked Phase 18 output directory; the previous audit logs were preserved.

## Reproducibility and evidence classification

- **Recomputed from saved predictions and repository price labels:** forecast
  targets, paired native populations, point metrics and ranks, selected-seed
  summaries, declared primary inference calculations, adaptive overall metrics,
  and interval crossings.
- **Supported by processed artifacts and the saved phase verifiers:** full regime
  and stress summaries, fixed-calibration metrics, quantile histories, and delayed
  feedback state trajectories.
- **Partially reproducible:** canonical historical model training. Predictions and
  configurations are present, but no canonical model weights are saved; 210
  recovered folds have no original execution revision or environment record.
- **Unverified from available repository evidence:** complete acquisition timestamps,
  exact source versions, data rights, full historical exchange-closure completeness,
  and whether researchers had previously seen particular periods.
- **Requires new evidence:** unseen future performance, other-market generalization,
  trading or portfolio utility, and any revised interval method that guarantees
  nesting or coverage under stronger assumptions.

The saved numerical evidence is reproducible to the extent described in
[`docs/REPRODUCIBILITY_GUIDE.md`](REPRODUCIBILITY_GUIDE.md). It is not a full clean-room
replay of every legacy training run or a fresh independent acquisition.

## Readiness and remaining work

The manuscript is a carefully qualified retrospective draft suitable for external
scientific review. It should not be described as publication-ready. Before journal
or conference submission, the research team still needs to resolve source-data
provenance and rights, confirm author identities/affiliations and submission
consent, obtain external review, follow venue formatting and disclosure rules, and
decide whether stronger claims require the frozen prospective study. Any economic
claim would require a separate execution, cost, turnover and portfolio evaluation.

No unresolved contradiction in the recomputed primary arithmetic was found. The
main remaining uncertainties concern data provenance and exposure, temporal
dependence, nonstationarity, overlapping targets, correlated index histories,
limited model/seed coverage, missing legacy runtime context, and the absence of
independent outcomes. These limits remain part of the research result and should
travel with any copy of the paper.

## Post-handoff manuscript revision note (2026-10-23)

This handoff records the Phase 18 snapshot before the manuscript revision merged
as PR #16 (`60e7722`). That later revision changed the figure set: the former
combined `interval_tradeoffs` figure was replaced by `interval_coverage` and
`interval_width`. The current manuscript asset verifier reports 12 tables and 4
figures, and the current PDF has 18 pages; manuscript verification reports 12
verified references. The Phase 18 counts above (3 figures, 14 pages, nine
references) describe that earlier snapshot and should not be read as counts for
the current manuscript. The final evidence-authenticity audit reran the current
asset, manuscript, and PDF verifiers.
