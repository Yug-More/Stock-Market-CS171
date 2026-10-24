# Manuscript artifacts

`MarketPulse_Research_Paper.md` and `.tex` are generated from
`manuscript_source.md`. Edit that source, not the generated versions. Numerical
bindings are implemented in `src/marketpulse/publication.py` and audited in
`CLAIM_EVIDENCE_AUDIT.md` and `evidence/manuscript_manifest.json`.

Claim and evidence-group tags appear as HTML comments in the source. The renderer
validates those tags against the claim registry and evidence manifest, then omits
them from the reader-facing paper. The manifest continues to list every bound
claim and supporting file. Tables and figures are regenerated from saved CSV
artifacts; model fitting is not part of manuscript reconstruction.

From the repository root:

```bash
.venv/bin/python scripts/manuscript.py build
.venv/bin/python scripts/manuscript.py verify
.venv/bin/python scripts/publication_assets.py build
.venv/bin/python scripts/publication_assets.py verify
```

The assets workflow exports all manuscript table CSVs and regenerates the
point-skill, paired-baseline, interval-coverage, and interval-width figures.
`evidence/assets_manifest.json` records source hashes and figure/table provenance.
The interval coverage and width views are separate to keep their scales and
interpretations legible. Empirical points are descriptive; inference remains in
the comparison tables and saved analysis files.

Compile in `paper/` with Tectonic (tested version 0.17.0):

```bash
tectonic --outdir /tmp/marketpulse-paper MarketPulse_Research_Paper.tex
```

Create the temporary output directory first. Tectonic downloads standard TeX
resources on first use; this requires network access. It runs BibTeX automatically.
Copy the successful PDF into this directory only after inspecting the rendered
pages. Build evidence records the exact source, bibliography, figure, and PDF
hashes. Render every final page and inspect it at normal reading size.
Compiler binaries, caches and auxiliary TeX files are not research artifacts and
are not committed. Rendering used pypdfium2 and text inspection used pypdf in the
available document runtime; these are optional review dependencies.

This is a research draft. Submission still requires authorship/affiliation consent,
source-data rights/provenance clarification, venue formatting, external review and
an honest decision on whether new chronological evidence is required. A locally
compiled PDF does not establish publication readiness.
