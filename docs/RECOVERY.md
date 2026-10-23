# Phase 4/5 recovery checkpoint

Recovery began on 2026-10-23 on `research/marketpulse-walkforward-regimes`,
HEAD `3db447a` (following `d30c40a` and `4457e6e`). Local `main` and remote
`main` were at `5f7b9099ad0ac0cc52adece8deb50e192c4fae80`.
There were no modified or staged tracked files. The only untracked work was
`results/walkforward/`; ignored fold prediction CSVs were also present.
No active experiment process was found. No existing PR or remote feature
branch was present. Existing history and raw files have been preserved.

## Recovered implementation

- Expanding, terminal-anchored folds, 252-origin step and test blocks,
  252-origin validation, minimum 2,520 embargo-clean training origins.
- Strict label maturity at train/validation and validation/test boundaries.
- Five forecasts, including the sequence LSTM; training-only preprocessing
  and validation-only forest/LSTM selection.
- Training-only volatility terciles and predeclared origin-date stress slices.
- Pooled prediction metrics, fold summaries, stress coverage, and figures.
- 56 existing tests passed at recovery; original Ruff checks passed.

## Recovered experiments

The full plan has 282 folds: 13 for each NASDAQ-100 horizon and 81 for each
S&P 500 horizon. At recovery 210 complete JSON/CSV pairs were present:
all 39 NASDAQ folds, all 162 S&P h=1/h=5 folds, and 9 S&P h=20 folds.
72 S&P h=20 folds remained. Existing smoke artifacts were preserved.

Every recovered fold passed comparison with a fresh split/data plan:
model set, seed, lookback, search grids and training budgets, fold boundaries,
all test dates and targets, finite predictions, zero/mean forecasts,
training-only volatility thresholds, regimes, and recomputed test metrics.
These checks establish internal consistency, not independent proof of the
original training execution. Legacy files did **not** record execution
revision, runtime, hardware, dependency versions, data hashes, or full
configuration provenance. These absent facts cannot be reconstructed.
The recovery manifest records this limitation rather than assigning a new
execution revision to old predictions. Recovered file hashes preserve their
identity. Their model metadata includes selected LSTM checkpoint epoch and
training history; weights were not saved by the original runner.

## Completion checklist

- [x] Inspect and preserve prior code, history, data, and artifacts.
- [x] Verify all 210 recovered folds against the actual plan and data.
- [x] Add guarded resume, input fingerprints, prediction checksums, atomic
  per-file writes, exact fold plans, and per-new-fold environment/runtime.
- [x] Add corruption/configuration and 252-session boundary tests.
- [ ] Execute a fresh all-model smoke experiment with figures.
- [ ] Execute remaining 72 full-budget folds and aggregate all 282 folds.
- [ ] Verify complete artifacts, figures, and real findings.
- [ ] Finish methodology/results/README and reproducibility records.
- [ ] Run quality checks, commit with mandated identity, push, open PR,
  verify CI, and merge/synchronize when permitted.

Deep LSTM remains in the historical notebooks. The canonical Phase 2/3
benchmark explicitly requires one-layer LSTM and defers new architectures;
recovery retains that five-forecast design rather than adding a new model.
The narrower financial-crisis slice (2008-09 through 2009-03) is predeclared;
it must not be described as a complete evaluation of 2007–2009.
