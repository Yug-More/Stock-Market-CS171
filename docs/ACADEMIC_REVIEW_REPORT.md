# Academic review of the MarketPulse manuscript

## Scope and recommendation

This is an internal, skeptical review of the repository at the Phase 18 release
candidate, with targeted rechecks of the evidence bindings, publication assets,
cross-phase audits, tests and formatting. It is not external peer review or
independent dataset validation. The main forecast, inference, calibration and
provenance conclusions are supported by saved records and prior reconstruction
audits, subject to the specific qualifications below. The paper is suitable to
send to a professor as a carefully qualified retrospective draft for feedback,
once the project team confirms how the original contributors should be named.
It is not ready for journal or conference submission while data rights,
acquisition provenance, authorship consent and external review remain unresolved.

## Strengths

- The core comparison shares targets and test origins across five procedures,
  uses chronological expanding folds, and purges training and validation labels
  that would reach into later blocks. The implementation asserts chronology,
  embargo and non-overlap properties.
- The paper reports all six native index/horizon cells and separates pooled
  descriptive rankings from paired inferential comparisons. It distinguishes
  nominal from Holm-adjusted evidence and keeps the 18-, 30- and 60-test families
  separate.
- Sensitivity results are bounded accurately: the five-seed analysis is limited
  to selected folds; the two indexes have unequal histories and are not treated
  as independent market replications.
- The uncertainty section reports score, coverage, width and failures together.
  It identifies long-horizon undercoverage and nonnested adaptive intervals and
  does not transfer theoretical conformal guarantees to the tested procedure.
- Data provenance, researcher exposure, missing historical weights and runtime
  metadata, and the lack of economic evaluation are visible in the paper and
  companion documentation.
- Claim bindings, generated tables and figures, and six cross-phase verifiers
  make the reported artifacts unusually traceable. Those checks support
  reproducibility of saved evidence, not automated proof of narrative meaning.

## Findings by severity

| Severity | Finding and evidence | Disposition |
| --- | --- | --- |
| Major | **Acquisition provenance and rights are incomplete.** The NASDAQ export has no retrieval timestamp or library version; the S&P file lacks a retained source URL, exact version and retrieval time. Neither file establishes licensing rights. See `configs/datasets.toml` and `data/README.md`. | Cannot be resolved from current files. Obtain source records and rights review before submission or redistribution beyond the repository's permitted scope. |
| Major | **Historical exposure limits confirmatory interpretation.** The histories were previously available to the project, follow-up analyses were informed by earlier results, and Phase 9 is an infrastructure rehearsal. The declared family-wise corrections do not account for every project-wide research choice. | Correctly qualified in the manuscript. New prospective or genuinely unexposed evidence is required for confirmation; do not relabel the rehearsal. |
| Major | **Dependence and inference remain approximate.** Multi-session forward targets overlap; fitted procedures share expanding histories; structural change and long-range dependence may not be represented by tested block lengths or the normal-reference HAC approximation. | The paper discloses these limits and reports multiple block lengths. They constrain significance claims and generalization; no available audit can establish validity under arbitrary dependence. |
| Major | **The target is not an investable total-return series.** The inputs are index closes, not confirmed dividend-reinvested total returns; early S&P-labelled observations predate the modern index. | Correctly disclosed. Interpret as price-index changes on these files; avoid direct portfolio or modern-index claims. |
| Major | **Full historical fitting replay is incomplete.** Canonical weights are absent and 210 recovered legacy folds lack original generating revisions and environments. Saved predictions permit metric reconstruction but not exact training replay. | Correctly disclosed. Preserve prediction-level reproducibility language and archive weights/configuration/runtime records in any future study. |
| Major | **Seed evidence is conditional and selected.** Five fixed seeds are evaluated on 30 selected folds, not every native fold. The selected results show instability, including NASDAQ-100 horizon-20 LSTM ranks from first to fifth. | Correctly described. This supports a sensitivity warning, not a population-level distribution over market histories or a full-history seed claim. |
| Minor | **The first related-work draft was method-focused and did not place the historical-mean benchmark in financial return-prediction literature.** | Added a verified Campbell–Thompson reference and explicitly distinguished their equity-premium setting from this study's price-index target. The section remains intentionally selective, not a comprehensive survey. |
| Minor | **The rationale for the chosen 1/5/20 session horizons and the interpretation of the two simple baselines were implicit.** | Clarified these as design choices representing daily, approximate weekly and approximate monthly spans, and as reference procedures rather than trading rules. No claim of prespecification before historical exposure is made. |
| Editorial | **README lacked a compact view of the implemented evidence flow.** The code separates data validation, target/features, purged splits, forecasting, evaluation and publication. | Added a Mermaid diagram limited to those implemented stages and a note that later folds train on earlier outcomes. |
| Editorial | **Automated claim checks can be mistaken for semantic or scientific validation.** | The paper and audit documentation explicitly limit their scope; this report also separates machine-checkable bindings from reviewer judgments. |

## Reference audit

The nine pre-existing bibliography entries were checked against the committed
`paper/evidence/reference_verification.json` records and the matching BibTeX
keys, titles, authors, years, and DOI or source URL fields. The recorded source
routes are Springer for Breiman; the MIT Press journal page for Hochreiter and
Schmidhuber; Crossref DOI records for Diebold and Mariano and for Künsch; NBER
for Newey and West; JSTOR for Holm (no DOI is listed); Taylor & Francis for
Gneiting and Raftery; and version-specific arXiv pages for Gibbs and Candès and
for Barber and colleagues. The repository's earlier verification report says
these metadata were checked against original papers, primary publisher records,
arXiv, or publisher-deposited Crossref records. This refinement re-opened the
Springer and arXiv pages successfully; direct requests to some publisher pages
were blocked, so those checks rely on the committed verification records rather
than a new page fetch. No existing metadata was changed. The added Campbell and
Thompson working-paper metadata was checked against the NBER record, which also
identifies its later published version.

## Checks performed in this review

- The claim-bound manuscript verifier passed for 37 quantitative claim IDs and
  10 references; the 12 publication tables and 3 figures passed asset checks.
- The complete test suite passed, and Ruff lint and format checks passed.
- Six saved-study audits completed with return code 0 in
  `/tmp/marketpulse-academic-review-final-1792816146N/`; the audit manifest
  records that preserved historical result trees were unchanged.
- A separate arithmetic review checked 430,920 raw forecast labels, 282 fold
  boundaries and frozen means, 210 metric rows, 150 seed tasks, 108 inference
  and bootstrap rows across the separate 18/30/60 families, 240 adaptive metric
  rows and 2,780,000 interval bounds. It reproduced designated and sensitivity
  nesting counts. The review explicitly reuses the shared CSV loader and prior
  audits for fixed/regime interval metrics and calibration state histories; it
  does not establish validity under arbitrary dependence or external
  independence.
- The comparison-family checker verified the declared Cartesian products and
  block lengths for all three primary families (18, 30 and 60 tests).
- The PDF compiled with Tectonic 0.17.0 to 14 pages. The saved PDF hashes
  verified, and all 14 rendered pages were visually inspected. Tables and plots
  were legible at normal page scale; no clipped content, overlap, broken glyphs,
  or orphan page remained after shortening the final appendix paragraph.
- All 22 local README links resolved. The three manuscript figures were
  visually inspected in the rendered paper. All 12 CSV table outputs were
  checked for row counts and column schemas; displayed values remain guarded
  by the publication and manuscript verifiers.

These checks verify the available repository artifacts and presentation. They
do not close the data-rights, exposure, dependence, exact-training-replay or
external-review limitations above.

## Likely reviewer questions

1. Can the exact source files, retrieval dates, source revisions and licenses be
   documented, and can the data be redistributed or independently reacquired?
2. Why should the reader interpret close-to-close price-index changes as a
   relevant financial forecasting target, given dividends, index composition
   changes and the early S&P-labelled history?
3. How sensitive are inference results to the overlap in 5- and 20-session
   targets, repeated estimation, plausible long-memory dependence and the
   chosen block-length range?
4. How were the comparison families fixed relative to the full sequence of
   exploratory analyses, and what conclusions survive this project-wide
   selection history beyond the declared within-family Holm adjustments?
5. Would retraining with preserved weights, explicit library defaults and
   original runtime metadata reproduce the stored historical predictions?
6. Do conclusions change under a larger, contemporaneously sourced market set,
   point-in-time constituent data, total returns or a truly prospective period?
7. What decision loss, execution timing, costs, turnover and portfolio
   constraints would be required before forecast differences had economic
   meaning?
8. Can adaptive intervals be made nested while retaining acceptable scores and
   coverage, and what assumptions would justify any coverage guarantee?

The repository cannot answer these questions with the present artifacts. They
are concrete next steps or limitations, not evidence that the reported saved
sample arithmetic is incorrect.

## Presentation and reproducibility assessment

The manuscript follows a coherent empirical-paper structure and retains complete
primary comparison tables, negative results, clear interval diagnostics and a
reproducibility appendix. Its main remaining presentation risk is density: much
of the methods detail is necessary to define the audited procedure, while the
paper's 14-page length is not itself evidence of adequacy. The regenerated PDF
was inspected page by page. The three figures and twelve tables are generated
from committed source result files; the figure set is useful but does not
replace inspection of the numerical tables.

The repository supports reconstruction of numerical results from saved
predictions and source labels to the documented extent. It does not provide a
clean-room reacquisition, exact replay of every historical model fit, unseen
chronological outcomes, or proof of economic value. Passing CI and audit tools
must not be represented as resolving those gaps.
