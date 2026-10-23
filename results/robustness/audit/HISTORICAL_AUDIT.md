# Independent historical audit

282 folds and 355,320 predictions passed the original audit and fresh actual target-date reconstruction. Pooled metrics use individual losses, not mean fold RMSE. All recovered-file hashes remain unchanged.

Predictions, date/label alignment, configured selection records, thresholds, regimes, and errors can be verified. Actual historical execution of 210 legacy training runs and their absent environment/revision/runtime cannot be independently proven. The 72 newer folds have execution metadata, though weights were not saved. A new seed-171 rerun of 18 time-stratified folds supplies separate targeted provenance; it does not repair the old records or reproduce all 282 original fits.
