# Manuscript claim-to-evidence audit

Run `.venv/bin/python scripts/manuscript.py verify` from the repository root.
The generated manifest in `evidence/manuscript_manifest.json` binds every cited
C-number to the Phase 12 registry and every E-tag to the listed source files,
configurations or methodological documentation. All bound files have SHA256
checksums. Thirty-seven distinct quantitative claim IDs and ten authentic
references are cited. Numerical tables and illustrative comparisons are generated
from the CSVs, not transcribed from earlier reports. The manuscript source contains
the explicit bindings; the generated Markdown and TeX contain display-rounded values.

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
No external human peer review or independent dataset validation is claimed.
