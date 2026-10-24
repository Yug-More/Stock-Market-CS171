# Phase 14: publication assets and reproducibility

## Deliverables and acceptance

Regenerate figures and supplementary tables from saved outputs; retain baseline,
negative and null comparisons; label units, populations and uncertainty; record
source/code/output hashes and executable reproduction commands. No new fits are
justified. Original source files must remain unchanged. Graphics are presentation
artifacts and may be revised transparently in Git.

Three figures are provided as vector PDFs and 300-dpi PNGs. Twelve tables include
complete metrics/statistics and the three distinct primary families (18/30/60).
The manuscript incorporates the new figures; its Markdown/TeX/PDF are rebuilt.
The evidence manifest and reproducibility guide define selections, units, marginal
intervals, nominal/Holm distinctions, warmup exclusions and incomplete replication.

## Verification and corrections

During authoring, an incorrect short regime selector (`high` rather than the saved
`volatility_high`) caused a build failure; it was corrected without modifying any
source data. A table verification initially compared retained pandas row indexes
rather than CSV row positions; the generated primary tables now explicitly reset
those indexes. These were presentation-code failures, not experimental revisions.
Visual inspection found a legend covering a point; legends were moved outside the
plot areas. All three figures were then regenerated from unchanged source CSVs.

The numerical verifier checks all twelve exported tables against their sources
with rtol 1e-13 / atol 1e-15 and validates committed input/output hashes. Exact
binary figure equality across platforms is not claimed. All original experimental
artifacts remain unchanged. No plot omits a failed method to improve appearances.
The figures are explicitly descriptive except for labelled marginal effect bars;
all adaptive coverage uncertainty intervals remain in the supplement.

Final checks executed: all 100 local tests passed; Ruff lint and format passed;
all 12 table reconstructions and 3 figure hashes passed; manuscript and PDF source
hash verification passed. PDF compilation succeeded with 14 pages; all fourteen
rendered pages were inspected in four contact sheets, including all three vector
figures and their captions. The remaining underfull paragraph warning is cosmetic;
no unresolved citations or overfull text boxes were reported. Standalone vector
figures are provided for venue-specific resizing; current paper formatting is a
draft. Clean-environment reproduction remains Phase 16 work.
