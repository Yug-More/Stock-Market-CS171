# Manuscript claim-to-evidence audit

Run `.venv/bin/python scripts/manuscript.py verify` from the repository root.
The authoritative source stores claim and evidence-group bindings in HTML comments.
The generated Markdown and TeX omit those internal identifiers, while
`evidence/manuscript_manifest.json` retains all 37 distinct quantitative claim IDs,
the bound evidence groups, and SHA256 checksums for source artifacts. Twelve
references are cited and checked against the bibliography and the records in
`evidence/reference_verification.json`. Numerical tables and comparisons are
generated from CSVs, not transcribed from earlier reports.

This is traceability, not a machine proof of prose semantics. The initial audit
compared the narrative with code and raw/processed evidence; the later Phase 15
review will independently challenge calculations. Original execution context for
210 recovered folds, original acquisition metadata and prospective outcomes cannot
be established by checksum verification. Related-work citations motivate methods;
they do not confer their theoretical guarantees on the bounded delayed procedure.

Evidence groups include data, features, models, metrics, folds, regimes, statistics,
calibration, adaptive calibration, diagnostics, protocols, prospective intake,
historical recovery and tests. Their complete file-level mapping is in the manifest.
Marginal intervals, nominal p-values and family-wise Holm p-values appear separately.
The original 18, expanded 30 and adaptive 60 comparison families are separate studies.
All six native cells, all original primary comparisons, both stochastic models'
seed ranges, zero-return interval examples and all four nesting variants are shown.
Full model/level/regime results remain available in the cited source artifacts.

References were checked against original papers, primary publisher/arXiv records,
or deposited Crossref metadata. `evidence/reference_verification.json` records the
actual verification route, including cases where publisher full text was inaccessible.
The Bergmeir et al. publisher page could not be opened directly in the review tool;
its bibliographic metadata was checked in the official journal listing and DOI record.
No external human peer review or independent dataset validation is claimed.
