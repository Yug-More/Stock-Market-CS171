# Reproducing the MarketPulse manuscript

## Scope and prerequisites

This guide reconstructs saved-result tables, figures, claims and the manuscript.
It does not claim a complete independent reacquisition or exact replay of every
historical fit. Use a full Git clone: old audits resolve historical source blobs.
Git LFS is not used. Python 3.13 was used locally; CI uses 3.12. Both environments
are recorded with actual checks. Run commands from the repository root.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-publication.txt
.venv/bin/python -m pip install -e . --no-deps
```

The requirements pin the local analysis/presentation versions; they are not a
claim that these were the original versions for the 210 recovered legacy folds.
PyTorch is separate. For neural smoke tests/refitting, install the appropriate CPU
PyTorch build and `pip install -e '.[research,dev]'`. Hardware/library differences
can change fitted results. No saved canonical weights exist.

## Verify preserved evidence

```bash
.venv/bin/python scripts/final_synthesis.py verify
.venv/bin/python scripts/audit_all_research.py --output /tmp/marketpulse-reproduction-audit
.venv/bin/python scripts/publication_assets.py verify
.venv/bin/python scripts/manuscript.py verify
.venv/bin/python scripts/verify_manuscript_pdf.py
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest
```

Choose a new comprehensive-audit output directory each time. The command restores
ignored per-fold CSVs from `results/walkforward/full/fold_predictions.tar.gz` and
runs all six earlier study audits. It does not overwrite published results. The
standalone original verifier needs those CSVs restored first if run independently:

```bash
tar -xzf results/walkforward/full/fold_predictions.tar.gz -C results/walkforward/full
```

## Regenerate presentation artifacts

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/marketpulse-mpl \
  .venv/bin/python scripts/publication_assets.py build
.venv/bin/python scripts/manuscript.py build
```

Authoring builds explicitly update current paper outputs/manifests. Original
experiments are untouched. Review the resulting diff. Twelve supplementary tables
retain source columns and available precision; CSV parsing/serialization can
change harmless final floating digits, so verification uses tight numerical
agreement (rtol 1e-13, atol 1e-15). Binary PNG/PDF output can vary across plotting
libraries/platforms. Checksums verify the committed instance; numerical table
regeneration checks agreement with the sources. The original phase reports and
prior draft revisions remain in Git.

`paper/evidence/assets_manifest.json` maps each figure to source CSVs and records
input/output hashes, row counts and the generating code. `manuscript_manifest.json`
records C-claims, methodological E-groups, citation keys, source and output hashes.
`reference_verification.json` records primary-source bibliography verification.
`paper/CLAIM_EVIDENCE_AUDIT.md` explains the limits of these checks.

## Compile and inspect the PDF

Install Tectonic from its official project release and put it on PATH. Version
0.17.0 was tested. The official macOS arm64 archive used locally had SHA256
`a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667`;
other architectures require their own release asset. Compiler binaries are not
committed. The first build downloads standard TeX resources, requiring network.

```bash
XDG_CACHE_HOME=/tmp/marketpulse-tex-cache \
  .venv/bin/python scripts/build_manuscript_pdf.py
.venv/bin/python scripts/verify_manuscript_pdf.py
```

The compiler runs BibTeX and writes auxiliary files in a temporary directory.
Only a successful build replaces the current PDF and compilation record. Render
every page with a PDF renderer (pypdfium2 is pinned), inspect tables, captions,
legends, math, references, page boundaries and any compiler warnings. Compilation
and hash success do not assert visual approval. TeX uses the vector PDF figures;
Markdown uses their 300-dpi PNG equivalents. Page count and compiler console are
recorded. PDF timestamps/fonts can prevent byte-identical binary regeneration.

## Table and figure definitions

- `native_metrics`: all 30 native model/cell rows; RMSE and MAE are log-return units,
  bias is predicted minus actual return, skill is relative MSE reduction.
- `matched_metrics`: all saved population rows, explicitly labelled; endpoint-matched
  cells have 2,232 shared origins/endpoints, without equalized training histories.
- `seed_metrics`: 150 rows, five fixed seeds, five selected folds and 1,260 origins
  per cell. Seeds repeat deterministic baselines; they are not independent markets.
- `original_comparisons`, `expanded_comparisons`, `adaptive_comparisons`: complete
  216/330/360-row outputs, including exploratory sensitivity; `_primary` tables
  contain their separate 18/30/60-test families. MSE differences are squared
  log-return units; interval-score differences are log-return units. Negative
  first-minus-second differences favor the first procedure. CIs are marginal 95%
  moving-block intervals, not Holm-inverted simultaneous intervals. Nominal and
  Holm p-values are retained in separate original source columns.
- `interval_metrics`: all 2,940 model/method/level/slice rows, including unavailable
  slices and excluded warmup counts. Coverage is a fraction, widths/scores use
  log-return units; 80% and 90% are nominal targets, not guarantees.
- `coverage_intervals`: all 240 marginal adaptive coverage CIs, length 20; no
  family-wise coverage claim. `nesting_diagnostics`: all 120 variant/model/cell
  rows, with repeated procedure-origin counts and separate clipping counts.
- `point_skill`: all five models, all six cells, native versus matched populations,
  common horizontal scale and visible zero. Descriptive points have no error bars.
- `baseline_effects`: all twelve original RF-versus-simple-baseline contrasts, with
  marginal 95% block intervals and nominal/Holm p-values from the full 18-test
  family. The six RF-versus-LSTM contrasts remain in Table 2 and the full CSV.
- `interval_tradeoffs`: all seven methods, zero-return 90% intervals in every cell,
  overall and high-volatility slices; coverage axis spans 0–1, widths start at zero.
  Points are descriptive; uncertainty intervals are in the coverage supplement.

## Refitting and limits

See `docs/WALKFORWARD_METHODOLOGY.md`, `ROBUSTNESS_METHODOLOGY.md`, `PHASE10_ROBUSTNESS.md`
and `PHASE11_ADAPTIVE_UNCERTAINTY.md` for the actual budgets and commands. Always
use fresh output directories; do not bypass source-identity guards to resume an
old run under new code. Preserve weights and full environment metadata in any new
study. The historical four-block rehearsal is not independent dataset validation.
Prospective execution requires genuine sourced, unexposed, mature observations
and the frozen intake checks; none are supplied or simulated by this guide.

Unresolved acquisition timestamps/versions/licenses, 210 legacy execution contexts,
unsaved weights and past researcher exposure prevent complete end-to-end independent
replication. Dependence assumptions and selected-fold seed coverage further limit
scientific generalization even when arithmetic is reproducible.
