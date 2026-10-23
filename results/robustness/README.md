# Phases 6–8 artifacts and reproduction

Original datasets, terminal benchmarks and `results/walkforward/full/` remain
unchanged. New files reference that saved seed-171 forecasting procedure.
The shared configuration is `configs/robustness.toml`; its preregistered design
is `docs/ROBUSTNESS_METHODOLOGY.md`. New training uses the original architecture,
hyperparameter grids, fitting/validation policy and full training budgets.

## Run order

Install `python -m pip install -e ".[research,dev]"` from repository root.
For a fresh clone restore original fold CSVs before the historical audit:

```bash
tar -xzf results/walkforward/full/fold_predictions.tar.gz -C results/walkforward/full
python scripts/research_cycle.py audit
python scripts/research_cycle.py matched
python scripts/research_cycle.py seeds --smoke --output /tmp/marketpulse-seed-smoke
python scripts/research_cycle.py seeds
python scripts/research_cycle.py statistics
python scripts/research_cycle.py calibration
python scripts/research_cycle.py figures
python scripts/research_cycle.py report
python scripts/research_cycle.py verify
```

The seed command resumes checksummed completed tasks and summarizes all 90
when complete. It fingerprints only actual immutable training dependencies,
allowing unrelated analysis modules to develop without changing training.
A mismatched seed manifest or corrupt/partial run stops with an error.
To recreate research outputs after changing generating inputs, use a new
output directory; do not present overwritten old training as an original run.
The saved research audit requires the original code revisions in Git history;
CI fetches that history. Most analyses reuse saved forecasts and need no fitting.
Full seed reproduction is a separate bounded CPU workload; smoke is not evidence.

For auditing the committed study, only archive restoration and `verify` are
needed; no training or recalibration is required. `verify` independently checks
mature score populations and **every evaluated bound**, all coverage/width/score
summaries, paired statistical effects/HAC arithmetic/Holm adjustment, matching,
seed forecast populations/metrics and file hashes. It does not prove the missing
execution history of 210 legacy folds.

## Layout and schemas

- `audit/`: original-study machine report, prose provenance findings, execution
  context and hashes. All raw target endpoints are independently reconstructed.
- `matched/`: compressed outer-origin membership (`origin_match`, `endpoint_match`,
  both actual target dates), per-model counts/attrition, native versus matched
  RMSE/MAE/skills/ranks, configuration references and input hash.
- `multiseed/manifest.json`: fixed seeds, 18 selected folds, training-input hashes,
  expected 90 tasks, initial execution context and declaration timestamp.
- `multiseed/runs/`: 90 JSON reports and gzip prediction files. Reports retain
  original split boundaries, validation-selected hyperparameters, candidate
  scores, LSTM checkpoint epoch/history, seed, actual runtime/date, execution
  environment and prediction checksum. No fitted weights are serialized.
- `seed_metrics.csv`, `seed_fold_metrics.csv`, `seed_stability.csv`: pooled
  and fold metrics/ranks by seed; each cell/seed contains 756 origins, not the
  original full-calendar population. Deterministic forecasts are reused.
- `seed171_replication.csv`: fresh versus original selected-fold numerical
  differences, never retroactively assigned to the old provenance records.
- `../statistical_reliability/`: 18-comparison registry, all 216 native/matched,
  MSE/MAE, three-block-length comparisons, 108 regime diagnostics, manifests.
  Negative paired loss difference favors `first`. Only `primary_family=true`
  rows receive the one-family `holm_p_value`; other p-values are exploratory.
  `ci_lower/upper` are marginal percentile bootstrap bounds; `hac_ci_*` are
  unadjusted normal-reference HAC bounds. `nominal_block_equivalents` is n/L,
  not estimated effective sample size. Empty/short regimes are not assigned
  unsupported inference.
- `../calibration/records/`: 90 gzip files, one index/horizon/model/method each.
  Wide rows contain both levels (`lower_80`, `upper_80`, `lower_90`, `upper_90`),
  actual origin/target dates, actual outcome/point prediction, causal volatility,
  existing regime/stress, sample n, eligible/status and calibration first/last
  origin and maximum target-date cutoff. Warmup rows are retained with empty bounds.
- `../calibration/metrics.csv`: all 180 overall model/method/level cells and
  corresponding volatility/stress diagnostics: sample/exclusion counts, coverage,
  gap, mean/median width and proper interval score. The same eligible dates are
  explicitly verified for all methods.
- Per-area `execution.json` and `checksums.json`: actual execution revision,
  dates, runtime, versions, source fingerprints and file inventories. Historical
  source changes are checked against recorded Git blobs, never falsely relabeled
  as current execution. Later figure/report sources can be separate commits.
- `final_audit.json`: independent machine-readable completion evidence.
- `figures/`: ten 220-DPI measured figures and exact captions; regenerate using
  the `figures` command. No synthetic market data enter research results.

Compression is used to retain auditable evidence without committing large
plain CSVs; no external storage restoration or account access is required.
Values remain float64 precision. Dates are ISO session dates. Losses/intervals
use the existing log-return target, and squared losses have squared-log-return
units. Numerical reproducibility is distinct from gzip bytes or neural equality
across versions/hardware. No inference or coverage guarantee follows merely
from repeating a seed or applying a conformal-style quantile.

`python scripts/package_research.py` inventories the completed bundle after
its audit. `study_environment.json` records measured training totals, software
versions and CPU context; its workstation reference is explicitly identified.
`audited_inputs.json` is a post-execution audit reference, not retroactive
historical metadata. `study_checksums.json` covers all three result directories,
including figures and interval records; `verify` checks it when present.

In statistical tables, `skill_vs_comparator` is one minus the ratio of the
selected mean loss: relative MSE for `loss=squared_error`, and relative MAE
for explicitly exploratory `loss=absolute_error`. It is not silently treated
as MSE skill on MAE rows. Primary comparisons always use squared error.
