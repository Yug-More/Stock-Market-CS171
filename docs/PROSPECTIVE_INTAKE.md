# Future prospective intake contract

Independent confirmation is not currently completed. Available raw files are
historically exposed, and the frozen future test and its actual endpoints have
not matured. This entry point integrates the Phase 9 infrastructure; it does not
supply new observations or repair old licensing/acquisition metadata.

Run `python scripts/prospective_validation.py --manifest PATH --output NEW_DIRECTORY`
only after a complete new acquisition with reviewed exposure history. The JSON
manifest requires `protocol_sha256` equal to the original Phase 9 freeze and a
`datasets` list containing exactly NASDAQ-100 and S&P 500. Each entry requires
`name`, `index_id`, `path`, `source_url`, `version`, `retrieved_utc` (timezone aware),
`license`, `adjustments`, `sha256` and `prior_outcome_exposure: "none declared"`.
Record the actual provider's terms and price-index conventions, not a guessed
license or a dividend-return interpretation. Source files must pass raw validation.
An exposure declaration is a research attestation, not proof from timestamps.

All source SHA256s, the original frozen protocol and original training dependencies
are checked before fitting. The runner uses the first frozen test window beginning
on 2026-10-26, rejects a missing first origin, and waits for all 252 origins and each
horizon's actual target endpoints. A later download cannot slide the window to
another favorable period. Train, validation, calibration and test have target-mature
purges. Calibration alone sets the fixed 80/90% radii. Canonical models run at the
original budgets; only validation RMSE selects candidates. No test-driven refit,
calibration update or optional stopping occurs.

Outputs preserve calibration/test forecasts, bounds, selected settings, point and
interval metrics, dependence-aware comparisons, the frozen 24-comparison primary
family and acquisition/environment checksums. L20 is primary; longer blocks may
be descriptive when the fixed sample lacks the declared minimum block support.
A completed run is contingent on truthful source/exposure review and still carries
no arbitrary-dependence coverage guarantee. Independent future calculation audits
must accompany reporting when real acquired data becomes available.

Unit fixtures test fixed-window selection and failure gates; they are synthetic
engineering tests, never validation data. Full-budget historical four-block
rehearsals remain separately labeled in the Phase 9 evidence. No independent point
ranking, effect or calibration outcome is reported until the prospective experiment
can actually be executed.
