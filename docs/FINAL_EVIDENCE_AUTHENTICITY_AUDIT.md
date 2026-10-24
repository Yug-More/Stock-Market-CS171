# Final Evidence Authenticity and Scientific Integrity Audit

**Audit date:** 2026-10-23  
**Audited repository revision:** `60e77222a467e54d3fa28cbfda54970a5252079f`  
**Scope:** computational review of existing MarketPulse evidence and a limited current-code rerun. This is not independent dataset validation, prospective validation, or external human peer review.

## Overall assessment

The repository contains internally consistent historical market-price records, saved model forecasts, configurations, and analysis artifacts. Independent recalculation reproduces the primary quantitative results. A fresh run of the current code also reproduced exactly the saved prediction values for the final fold in each of the six index-by-horizon cells. The review found no indication that synthetic test fixtures or placeholder values were mixed into the research result files.

That evidence does not establish the complete acquisition history, source version, licensing, or prior exposure of the datasets. In particular, the S&P 500 CSV has no exact source URL or version record and begins one session before the S&P DJI page's stated first value date. Also, 210 recovered walk-forward folds lack original execution revision and environment metadata; their records are inspectable, but their original training runs cannot be independently replayed. The evidence supports a retrospective computational study with the stated scope, not independently validated forecasting performance.

## Scope, source evidence, and method

The audit examined the repository tree and Git state; the Phase 9–12 reports and outputs; `docs/FINAL_RESEARCH_RESULTS.md`, `docs/RESEARCH_CONTRIBUTIONS.md`, `results/phase12/claims.json`; dataset and model configurations; raw index CSVs; saved forecast, robustness, calibration, and adaptive-interval records; figures and tables; test fixtures; manuscript sources and PDF; and relevant analysis and verification code.

The independent arithmetic is implemented in [`scripts/audit_evidence_authenticity.py`](../scripts/audit_evidence_authenticity.py). It parses raw CSV and JSON files and uses direct pandas/NumPy calculations rather than the publication and scientific-review helpers for its core checks. Its machine-readable output is [`independent_recalculation.json`](../results/final_evidence_authenticity_audit/independent_recalculation.json). The complete Phase 12 claim registry is cross-referenced in [`claim_inventory.csv`](../results/final_evidence_authenticity_audit/claim_inventory.csv). Existing, separately implemented verification is recorded in [`scientific_verification.json`](../results/final_evidence_authenticity_audit/scientific_verification.json). These checks share repository artifacts and are computationally independent in parts; they are not independent acquisition or independent data validation.

## Data authenticity, lineage, and provenance

The NASDAQ-100 file contains 6,289 daily rows from 2000-01-03 through 2024-12-31. The S&P 500 file contains 23,323 daily rows from 1927-12-30 through 2020-11-04. Both files passed structural checks for missing/non-finite values, duplicate or disordered dates, nonpositive OHLC values, and OHLC relationship violations. The NASDAQ file has four consecutive unchanged closes; the S&P file has 313. These observations are retained and do not by themselves establish an error. The S&P file has 5,496 zero-volume rows, and its `Adj Close` equals `Close` throughout; volume semantics and adjustment provenance are not established.

The file hashes are recorded in `independent_recalculation.json`. Repository history shows when each file entered Git, not when or how it was acquired. The NASDAQ source notebook records a yfinance download call, but it does not provide a complete capture/version record; the stored endpoint also needs to be reconciled with the downloader's end-date semantics. The S&P notebook describes a Kaggle download without a dataset identifier, URL, version, or retrieval date. Consequently, full-file lineage, acquisition date, source revision, license, and prior exposure remain unverified.

External spot checks support the sampled values, not the complete files. NASDAQ-100 closes at 2000-01-03 (3,790.55), 2000-01-10 (3,717.41), 2008-09-15 (1,705.46), 2020-03-16 (7,020.38), and 2024-12-31 (21,012.17) agree with the Nasdaq-sourced FRED observations ([FRED NASDAQ100 observations](https://fred.stlouisfed.org/data/NASDAQ100)). S&P 500 closes at 2018-12-24 (2,351.10), 2020-03-16 (2,386.13), and 2020-11-04 (3,443.44) agree with the S&P DJI-sourced FRED series ([FRED SP500 observations](https://fred.stlouisfed.org/data/SP500)). The index provider reports a first value date of 1928-01-03 and an index launch date of 1957-03-04 ([S&P 500 index profile](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)). The repository's 1927-12-30 record therefore precedes the provider's first stated value date. This discrepancy is unresolved; it is not evidence by itself that the row is fabricated. FRED also identifies licensing restrictions for the S&P series, so publication and redistribution rights require review.

The S&P history spans periods before the modern index launch and should be treated as provider-labelled historical index data whose exact construction and source lineage have not been established in this repository. No claim of independent acquisition, an unseen dataset, or complete historical source authentication is warranted.

## Experiment provenance and artifact inventory

The primary walk-forward manifest reports 282 folds, 13 per NASDAQ-100 horizon and 81 per S&P 500 horizon. The saved native prediction file contains 355,320 rows for zero return, frozen historical-target mean, linear regression, random forest, and LSTM. There are 72 markers with execution metadata and 210 recovered historical markers without original execution revision/environment details. Current recovery metadata cannot fill those historical gaps. Canonical model weights are absent, so the legacy training process cannot be replayed end to end.

The robustness and Phase 10 sensitivity artifacts include 90 and 60 task files respectively, with seeds 171, 271, 371, 471, and 571. The 150 files contain 75,600 prediction rows checked here against raw-price-derived targets. They cover 30 selected fold schedules, not full-history reruns for every seed. Phase 11 interval and adaptive-calibration results are supported by saved per-origin records, metrics, and diagnostics; the independent checks below recalculate the designated interval score family and nesting counts from these records. Some additional interval histories and feedback-state summaries remain supported primarily by the repository's existing processing and verifier rather than a new independent reconstruction of every state transition.

Searches found synthetic data only in clearly identified test fixtures. No placeholder or mock output was found in the scientific result paths. This search cannot prove that every historical input is authentic; the missing provenance described above remains material.

## Independent recalculation results

All 355,320 native forecast labels were reconstructed as `log(close[t+h] / close[t])` from the committed input CSVs. The maximum absolute discrepancy was zero. There were no duplicate model/fold/origin keys, no endpoint ordering errors, and no mismatched common samples across forecast procedures within a fold. All 282 test windows matched their recorded chronological fold boundaries and test dates. These are checks of stored records and present code/data, not proof of original data acquisition or research blinding.

All 30 dataset-by-horizon-by-model pooled metric rows were recomputed. The maximum absolute differences from saved metrics were at floating-point noise (below `5e-16` for reported metrics and skills); model ranks matched exactly. The 18-comparison original native-MSE family, 30-comparison expanded five-seed sensitivity family, and 60-comparison adaptive interval-score family were reconstructed with the recorded HAC, Holm, and moving-block bootstrap procedures. Effects, standard errors, nominal p-values, adjusted p-values, and bootstrap endpoints matched the saved analysis to numerical tolerance. The comparison-family audit independently confirmed the declared family sizes and unique contrast keys.

The original results support no Holm-adjusted random-forest improvement over either simple baseline; they do support higher random-forest loss than both simple baselines in the S&P 500 20-session cell. The expanded five-seed family supports no adjusted machine-learning improvement over a simple baseline. In the 60-test adaptive proper-score family, the designated method has 20 adjusted improvements, all for the S&P 500 at one- and five-session horizons across the five forecast procedures and two nominal levels. Significance in these specified families does not establish practical value, equivalence, or future performance.

The interval-bound audit recalculated all 120 saved variant diagnostics. For the designated `aci_w504_g001` variant, 1,494 of 347,500 eligible 80%-versus-90% interval pairs violate nesting. Coverage and score records also retain the documented long-horizon undercoverage. The method does not have a distribution-free coverage guarantee; nominal significance in interval scores does not establish calibrated coverage or economic utility.

## Current-code rerun

Using the committed full-mode configuration and the repository's current walk-forward runner, the audit reran the final chronological fold for each of the two indexes at horizons 1, 5, and 20. The new outputs are isolated in [`reruns/`](../results/final_evidence_authenticity_audit/reruns/). This produced six fold tasks and 7,560 model/baseline prediction records (252 origins × five procedures per fold). Every prediction value matched the corresponding saved terminal-fold value exactly. The run used Python 3.13.2, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, PyTorch 2.14.1, and commit `60e7722` on macOS ARM. A temporary matplotlib cache warning did not prevent completion.

This targeted replay demonstrates reproducibility for these six current-code fold tasks. It does not recreate the other 276 folds, recover missing historical weights, establish the 210 legacy executions' original provenance, or authenticate the input files. It was not an independent validation experiment.

## Claim-by-claim disposition

The 40 entries in `results/phase12/claims.json` each have an inventory row with the registered artifact, its saved hash, operation, registered value and classification, verification performed, evidence status, and limitation. The headline claims were directly recalculated as described above. Several claims are counts or status assertions (for example, historical rehearsal and absence of independent validation); they are verified only within the repository's documented scope. Detailed regime summaries, some calibration trajectories, and certain seed-rank summaries remain supported by processed artifacts and existing audit code rather than a fresh reconstruction of every model fit or interval state.

No numerical contradiction was found in the independently recalculated primary results. The source provenance mismatch at the start of the S&P 500 history is the clearest unresolved data-quality question and is explicitly classified as unresolved, not silently repaired. A documentation mismatch was found: the Phase 18 handoff's three-figure, 14-page, nine-reference counts describe its pre-PR #16 snapshot, while the current verifier reports four figures, 18 pages, and 12 verified references. A dated amendment was added to that handoff, and the current reproducibility guide now distinguishes the historical snapshot from the current assets. No historical result file or original dataset was rewritten.

## Reproducibility, limitations, and classification

- **Recomputed from raw repository data and prediction records:** target labels, common sample alignment, fold chronology, pooled metrics/ranks, primary comparison-family statistics, adaptive score-family statistics, designated nesting count, and selected-seed target reconstruction.
- **Reproduced from current code on selected folds:** all 7,560 prediction values in six final fold tasks.
- **Supported by processed artifacts and existing verifiers:** full regime/stress summaries, some calibration trajectories and coverage summaries, and some seed-rank summaries.
- **Partially reproducible:** canonical historical training. Configurations and predictions exist, but weights are absent and 210 recovered folds lack original execution metadata.
- **Unverified:** complete raw-data acquisition and version history, S&P prehistory provenance, usage rights, full historical source completeness, and whether the researchers had prior exposure to particular periods.
- **Requires new evidence:** future or independent-market performance, trading/portfolio utility, and any claim of generalized uncertainty guarantees.

No prospective or independent dataset was added. The Phase 9 historical rehearsal remains infrastructure evidence. No new experiment beyond the six-fold reproduction check was run. The research remains a retrospective empirical study with limited provenance and temporal generalization. It should not be described as journal-ready or independently validated.

## Verification commands and outcomes

The following checks were executed on this branch:

- `.venv/bin/python scripts/final_synthesis.py verify` — passed 40 registered claims and six prior audits; reports `independent_confirmation: false`.
- `.venv/bin/python scripts/audit_all_research.py --output /tmp/marketpulse-final-evidence-audit-new` — passed all six cross-phase audits; reported `historical_tree_unchanged: true`.
- `.venv/bin/python scripts/scientific_verification.py --output results/final_evidence_authenticity_audit/scientific_verification.json` — passed its computational review of 430,920 forecast rows, 282 fold boundaries, 210 metric rows, the three inference families, 240 adaptive metric rows, and 2,780,000 saved interval bounds. It explicitly reports no independent dataset validation or external human peer review.
- `.venv/bin/python scripts/verify_comparison_families.py --output results/final_evidence_authenticity_audit/comparison_families.json` — passed the 18-, 30-, and 60-comparison family checks.
- `.venv/bin/python scripts/manuscript.py verify` — passed 37 quantitative claim bindings and 12 verified references.
- `.venv/bin/python scripts/publication_assets.py verify` — passed 12 tables and four figures.
- `.venv/bin/python scripts/verify_manuscript_pdf.py` — passed the saved PDF hash/build check; 18 pages.
- `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` — passed after formatting the new audit script.
- `.venv/bin/pytest -q` — passed; 105 tests were collected and completed successfully.
- A tracked-file pattern scan examined 763 text files, skipped 409 binary files, and found no credential-pattern matches. Dedicated credential scanners were unavailable; the scan is not a guarantee about every historical Git object.

The repository does not provide a fully clean replay of all original fits because original weights and some execution provenance are missing. Reproduction of saved numerical analyses is strong for the checks listed above but remains conditional on the committed data files and the repository's described pipeline.

## Git and release status

The work is on the requested feature branch, `research/final-evidence-authenticity-audit`, based on the fetched `origin/main` commit listed above. The checkout's configured origin URL still uses the former repository name, `https://github.com/Yug-More/Stock-Market-CS171.git`; GitHub confirmed the move to `https://github.com/Yug-More/MarketPulse.git` and accepted the branch push at the new location. No release or tag was created.

## Final disposition

The available records provide defensible, reproducible evidence for the limited retrospective results reported in the manuscript. No indication of fabricated market rows, placeholder experiment output, or contradictory primary arithmetic was found. The evidence does not establish full dataset lineage, historical blinding, licensing, original execution provenance for all folds, independent validation, or deployable performance. Those points remain explicit limitations for public technical review and any future submission.
