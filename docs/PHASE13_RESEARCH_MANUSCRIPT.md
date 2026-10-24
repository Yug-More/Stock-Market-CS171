# Phase 13: manuscript scope and audit corrections

## Acceptance

Produce complete Markdown and LaTeX manuscripts, authentic references with recorded
verification sources, a compiled and visually inspected PDF if tooling permits,
and an executable manuscript claim-to-evidence audit. All primary model cells,
null/negative findings, seed instability and uncertainty failures must remain
accessible. Do not fit new models merely for presentation. Preserve raw files,
notebooks and original result trees. Treat later authoring revisions as document
revisions, not new experiments or independent confirmation.

## Mandatory initial audit

Both requested commands were executed successfully before tracked edits:
`final_synthesis.py verify` (40 evidence claims) and `audit_all_research.py --output
/tmp/marketpulse-audit-new` (six saved-study audits). Their actual logs are copied
unchanged to `results/phase13/initial_audit`. The tracked-file inventory records
sizes and SHA256 values at the starting main revision
`3f52a79cb2fb4bbf79848d35d77eeaa2c2fe3929`.

Source inspection covered data loading/quality validation, causal features and
labels, purged partitions, preprocessing, model selection/training, metric pooling,
matched populations, seed aggregation, HAC/bootstrap/Holm and calibration feedback;
configuration, test and artifact inventories identify the remaining supporting
modules. Earlier notebook outputs are historical records and cannot substitute for
canonical shared-target evaluations. Missing notebook model files and unsaved
canonical weights limit retraining replay. Existing result verification does not
prove unrecorded legacy execution context or independent data exposure.

## Discrepancies corrected before manuscript drafting

- D01: README said no statistical/multi-seed analysis and no calibrated intervals
  existed. Those sentences described an earlier state. Replaced them with explicit
  Phase 4/5 scope and links to the later analyses. No experimental values changed.
- D02: Terminal-window narrative incorrectly said all learned S&P h5 models fail
  to beat historical mean. The saved test CSV gives linear regression lower RMSE
  than historical mean. Corrected the narrative; the existing table was correct.
- D03: A small RMSE spread was characterized as not useful. That conclusion requires
  an economic decision loss or trading experiment. Replaced it with an explicit
  statement that economic usefulness was not tested.
- D04: Older phase reports are dated methodological snapshots, including statements
  that later work was deferred. Preserve those snapshots; the manuscript and new
  navigation explicitly distinguish them from current project status.

No change to an experimental result was justified by these discrepancies. Remaining
risks include researcher selection, nonstationarity, overlapping labels, correlated
indexes, unequal training histories, incomplete acquisition/license metadata,
210 legacy folds without original runtime context, unrecorded library defaults,
unsaved weights and unavailable prospective outcomes. Small nominal p-values and
marginal bootstrap intervals must not be presented as Holm-adjusted evidence.

The scientific review is a computational review within this workflow, not a blinded
review by an independent human and not independent dataset validation.

## Completed verification

The evidence-bound Markdown and LaTeX compile to a 12-page PDF using Tectonic
0.17.0. All twelve rendered pages were inspected; a caret escaping error, duplicate
table anchor and table breaks were corrected during layout review. A sandboxed
font download failed and was retried with authorized network access. The final
successful compiler console and file hashes are retained in `paper/evidence`.
One underfull paragraph warning remains; no final overfull boxes or unresolved
citations were reported. Renderer/text extraction succeeded.

Ruff lint and formatting passed. The full local suite passed: 100 tests. Manuscript
verification passed for 37 cited quantitative claims and nine references; the
40-claim earlier synthesis also passed. The six-study comprehensive audit passed
before edits. PDF verification checks the recorded source/figure/bibliography/PDF
hashes; it does not claim platform-independent binary PDF regeneration. The
manuscript is a draft requiring external review and submission decisions.
