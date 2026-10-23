# Walk-forward artifacts

The `full/` directory is the native-calendar Phase 4/5 study. It covers both
indexes, horizons 1/5/20, and five forecasts. The original terminal benchmark
is separately preserved in `../benchmarks/full/`.

## Files

- `manifest.json`: schema version, exact source/config/data SHA-256 hashes,
  protocol and model budgets, expected fold count, recovery environment,
  hashes of unchanged recovered artifacts, and explicit legacy provenance gap.
- `fold_plan.json`: exact boundaries and counts for every planned fold.
- `folds/<index>/h<horizon>/fold_<number>.json`: fold report, all model test and
  validation metrics, selected hyperparameters and validation search scores,
  LSTM selected-epoch metadata/history, and training-only volatility thresholds.
  New reports also contain execution environment, revision, runtime, full model
  settings, and prediction checksum. Original reports remain byte-identical.
- `fold_predictions.tar.gz`: byte-preserving archive of individual fold CSVs;
  these are otherwise gitignored to avoid hundreds of large text diffs.
- `predictions.csv.gz`: pooled forecasts for analysis; the uncompressed local
  file is ignored. gzip uses a fixed timestamp for deterministic packaging.
- `fold_metrics.csv`: separate validation/test rows with boundaries and counts.
- `pooled_metrics.csv`: sample-pooled errors, skills against both baselines,
  and ranks by dataset/horizon.
- `fold_summary.csv`: explicitly labeled mean/median fold RMSE, mean rank,
  population rank standard deviation, consecutive rank changes, fold winners,
  and counts beating each baseline.
- `stress_metrics.csv`: inclusive origin-date slices, eligibility/test counts,
  coverage status, and metrics; unavailable episodes have no fabricated errors.
- `regime_metrics.csv`: low/medium/high training-tercile test slices with counts,
  errors, baseline skills, ranks, and RMSE ratio to each model's low-volatility
  slice (undefined without a positive low-volatility reference).
- `volatility_thresholds.csv`: per-fold training quantiles and validation/test
  volatility means. Test volatility means are descriptive, never fit inputs.
- `protocol.json`: short protocol summary; the manifest contains full settings.
- `environment.json`: **aggregation** environment and revision, not the
  undocumented generating environment of legacy folds.
- `recovery_hardware.json`, `requirements-recovery.txt`: recovery CPU/software
  context; dependency installation is platform-specific, and legacy context
  was not recorded.
- `verification.json`: independent full-plan, data, metric, and preservation audit.
- `artifact_checksums.json`: SHA-256 inventory of published artifacts, including figures
  and compressed forecasts.
- `execution-recovery.log`: actual recovery execution log, showing reused versus new folds.
- `figures/`: 220-DPI PNG comparisons from actual saved predictions/metrics.

Prediction columns are `dataset,index_id,horizon,model,date,y_true,y_pred,fold,
volatility,regime,vol_low_max,vol_high_min`. `date` is the decision date,
`horizon` is sessions ahead, and targets/predictions are dimensionless log
returns. Baseline skill is based on MSE, not RMSE. `rank=1` is lowest RMSE;
ties take minimum rank. `fold` is zero-based within index/horizon. Rows with
empty metrics are genuinely unavailable slices, not zero errors.

## Reproduce or audit

From the repository root after installing `.[research,dev]`:

```bash
# Restore exact fold forecast bytes from a fresh clone.
tar -xzf results/walkforward/full/fold_predictions.tar.gz -C results/walkforward/full
python scripts/verify_walkforward.py

# Fresh experiment: a new directory is essential if source/config/data changed.
marketpulse-walkforward --mode smoke --output /tmp/marketpulse-smoke
marketpulse-walkforward --mode full --output /tmp/marketpulse-full

# Resume the published run only with its exact fingerprinted inputs.
marketpulse-walkforward --mode full --output results/walkforward/full
```

Use code revision recorded in the manifest's recovery environment to recreate
new-fold training. The earlier 210 folds have no generating revision record;
internal validation cannot independently prove their execution provenance.
Raw CSVs are versioned and hashed; do not redownload replacement data.
Pooled metrics are recomputed independently by the audit script without
training. Training is a separate potentially lengthy CPU task.

No smoke forecast contributes to the research results. Existing ignored
`smoke/` outputs are preserved locally.

To regenerate compressed deliverables after a complete fresh run has been
verified, run `python scripts/package_walkforward.py`, then rerun the audit.
Packaging normalizes gzip/tar timestamps and tar owner fields and checks
that archived per-fold CSV bytes are identical to their source files.
