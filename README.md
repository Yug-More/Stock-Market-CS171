# MarketPulse

MarketPulse is a retrospective empirical study of machine-learning forecasts for
financial time series. It asks whether added model complexity produces reliable
improvements over simple return forecasts when the horizon and historical period
change. The repository preserves the original course project and adds an audited,
chronological research pipeline. The results do not establish trading profitability
or performance on unseen data.

The study asks whether learned models improve on simple baselines after repeated
chronological evaluation and multiple-testing correction; how rankings change with
market period, matched origins, and stochastic seeds; and whether prequential
prediction intervals retain useful score, coverage, and nesting behavior.

## Data and scope

The experiments use two stored US price-index histories:

- NASDAQ-100 (`^NDX`): 6,289 sessions from 2000-01-03 through 2024-12-31.
- S&P 500 (`^GSPC`): 23,323 sessions from 1927-12-30 through 2020-11-04.

The files do not have complete acquisition timestamps, exact source versions, or
license records. The NASDAQ preprocessing notebook records a yfinance export but
not when it was retrieved; the S&P file is described as a Kaggle download without
a retained source URL. The inputs are closing-price indexes, not dividend-adjusted
total-return series. The early S&P history predates the modern index. Both files
were used during the project, so chronology alone cannot establish that any period
was previously unseen. See [data provenance](data/README.md) and [research
limitations](docs/RESEARCH_LIMITATIONS.md).

## Method

The target is the forward log change in closing price over 1, 5, or 20 trading
sessions. A 282-fold expanding-window study stores 355,320 out-of-sample forecasts.
Each fold uses purged training, validation, and test windows; all five procedures
share the same test origins. The walk-forward protocol and boundaries are described
in [the methodology](docs/WALKFORWARD_METHODOLOGY.md).

The procedures are zero return, the embargoed training-target mean, standardized
linear regression, random forest, and a one-layer sequence LSTM. The LSTM and
random forest are stochastic; additional seed sensitivity covers selected folds,
not the full history. Statistical comparisons use paired losses, dependence-aware
HAC or moving-block calculations, and separate Holm families for the original,
expanded, and interval studies. See the [statistical methods](docs/STATISTICAL_RELIABILITY.md)
and [robustness reports](docs/PHASE10_ROBUSTNESS.md).

## Verified findings

In native pooled RMSE, random forest ranks first in three of six index/horizon
cells; a simple baseline ranks first in the other three. No random-forest gain over
either simple baseline survives the original 18-comparison Holm correction, and no
machine-learning gain over a simple baseline survives the separate expanded
30-comparison family. The original comparisons do show corrected evidence of
higher random-forest loss than both simple baselines for S&P 500 at horizon 20.
LSTM has the highest native pooled RMSE in all six cells. These are conditional
retrospective results; nonsignificance does not establish equivalence or universal
model inferiority. Full findings, null results, and sample definitions are in
[FINAL_RESEARCH_RESULTS.md](docs/FINAL_RESEARCH_RESULTS.md).

The uncertainty study evaluates fixed and four bounded, delayed-feedback adaptive
interval variants. The designated adaptive variant has 20 Holm-adjusted interval
score improvements among 60 primary comparisons, limited to S&P 500 horizons 1 and
5 against trailing calibration. This score result coexists with 90% interval
undercoverage at long horizons and 1,494 crossings between nominal 80% and 90%
intervals among 347,500 eligible designated-method pairs. Intervals are empirical
diagnostics: no distribution-free coverage guarantee is claimed. See the
[adaptive uncertainty report](docs/PHASE11_ADAPTIVE_UNCERTAINTY.md).

The Phase 9 historical rehearsal tested infrastructure only. Independent
chronological validation remains unavailable; no new unexposed, provenance-complete
market data or prospective outcomes are in the repository. The [scientific
verification report](docs/SCIENTIFIC_VERIFICATION_REPORT.md) is an adversarial
computational review of existing evidence, not independent dataset validation or
external human peer review.

## Reproduce the saved evidence

Python 3.11 or newer is required. For the full research and development checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[research,dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python scripts/final_synthesis.py verify
.venv/bin/python scripts/publication_assets.py verify
.venv/bin/python scripts/manuscript.py verify
```

To rerun the six saved-study audits, use a fresh output directory because prior
logs are preserved:

```bash
audit_dir="/tmp/marketpulse-audit-$(date +%Y%m%d%H%M%S)"
.venv/bin/python scripts/audit_all_research.py --output "$audit_dir"
```

The figures and tables can be regenerated from the committed result files. Full
model refitting is a separate, more expensive operation; weights for the historical
fits are not saved. The 210 recovered folds lack original execution revision and
environment records. The [reproducibility guide](docs/REPRODUCIBILITY_GUIDE.md)
explains what can be reconstructed, required tools, and the remaining limits.

## Research artifacts

- [Research manuscript (Markdown)](paper/MarketPulse_Research_Paper.md) and
  [compiled PDF](paper/MarketPulse_Research_Paper.pdf).
- [Claim-to-evidence audit](paper/CLAIM_EVIDENCE_AUDIT.md), references, source
  tables, and publication figures under [`paper/`](paper/).
- [Final findings](docs/FINAL_RESEARCH_RESULTS.md) and
  [contribution boundaries](docs/RESEARCH_CONTRIBUTIONS.md).
- [Methods](docs/WALKFORWARD_METHODOLOGY.md), [results](docs/WALKFORWARD_RESULTS.md),
  [prospective intake protocol](docs/PROSPECTIVE_INTAKE.md), and
  [quality assurance](docs/PHASE16_FINAL_QUALITY_ASSURANCE.md) and
  [final research handoff](docs/PHASE18_FINAL_HANDOFF.md).
- Original notebooks and contributor history are documented in
  [the repository audit](docs/REPOSITORY_AUDIT.md). Historical rehearsal results
  are not independent confirmation.

The manuscript is a qualified retrospective draft. External scientific review,
authorship decisions, source-data provenance and rights review, and genuinely
independent chronological evidence remain necessary before stronger claims or
submission to a venue.
