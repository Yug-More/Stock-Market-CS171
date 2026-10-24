# Manuscript artifacts

`MarketPulse_Research_Paper.md` and `.tex` are generated from
`manuscript_source.md`. Edit that source, not the generated versions. Numerical
bindings are implemented in `src/marketpulse/publication.py` and audited in
`CLAIM_EVIDENCE_AUDIT.md` and `evidence/manuscript_manifest.json`.

From the repository root:

```bash
.venv/bin/python scripts/manuscript.py build
.venv/bin/python scripts/manuscript.py verify
```

Compile in `paper/` with Tectonic (tested version 0.17.0):

```bash
tectonic --outdir /tmp/marketpulse-paper MarketPulse_Research_Paper.tex
```

Create the temporary output directory first. Tectonic downloads standard TeX
resources on first use; this requires network access. It runs BibTeX automatically.
Copy the successful PDF into this directory only after inspecting the rendered
pages. Build evidence records the exact source, bibliography and PDF hashes.
Compiler binaries, caches and auxiliary TeX files are not research artifacts and
are not committed. Rendering used pypdfium2 and text inspection used pypdf in the
available document runtime; these are optional review dependencies.

This is a research draft. Submission still requires authorship/affiliation consent,
source-data rights/provenance clarification, venue formatting, external review and
an honest decision on whether new chronological evidence is required. A locally
compiled PDF does not establish publication readiness.
