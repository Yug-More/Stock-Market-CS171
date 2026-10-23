# Research limitations after Phases 6–8

- The 210 legacy folds still lack original execution revision, runtime and
  environment. Checksums and reconstructed errors establish consistency, not
  historical execution. New seed-171 reruns are separate targeted evidence.
- Five seeds cover three selected folds per cell, not all 282 folds. They
  measure optimization variation on repeated data, not independent market
  replications. Original architectures and full budgets are preserved.
- Endpoint matching leaves 2,232 origins per index/horizon but does not
  equalize training histories, fold calendars or all information sets. Only
  two dependent US indexes are available, with no genuinely unseen market.
- Inference reuses history and model rankings already examined. The primary
  family is retrospective. Holm correction does not repair adaptive historical
  selection. Bootstrap/HAC rely on approximate weak dependence and can be
  unreliable under long nonstationary shifts. CIs are marginal, not simultaneous.
- Overlapping multi-session labels and shared expanding training histories
  create dependence. Nominal n/L block equivalents are not effective sample size.
  Long-memory and structural-break robustness are not established by three Ls.
- Short crisis windows provide few observations; unavailable NASDAQ stress
  episodes remain unavailable. Origin-based slices do not require target
  realizations to fall inside the same episode. Boundaries are fixed, not optimized.
- Online intervals calibrate a changing procedure, with old OOS residuals
  potentially reused in later training histories. No exchangeability, conditional
  coverage or finite-sample distribution-free guarantee is claimed. Historical
  marginal coverage cannot establish future coverage after a new regime shift.
- Raw acquisition metadata remain incomplete, early S&P history predates the
  modern index, and aggregate indexes do not expose point-in-time constituents
  or quantify survivorship/selection effects. Neither file establishes a
  dividend-reinvested total-return series.
- Small fixed model searches, one neural architecture and finite training
  budgets restrict conclusions about broad model classes. Model weights are
  not serialized; numerical replay requires retraining. Hardware/software changes
  can alter neural results despite controlled seeds.
- No transaction costs, portfolio execution, strategy returns, investment
  recommendation, profitability evidence or crash-prediction claim is supplied.
