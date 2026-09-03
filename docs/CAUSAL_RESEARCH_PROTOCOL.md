# Causal Research Protocol

## Scope

Price-Call is a paper-only forecasting and statistical evaluation project. Outputs
are research artifacts, not instructions to buy, sell, or hold a security. Part 10
must remain in dry-run mode until all explicit evidence and health gates pass.

## Temporal contract

For a prediction stamped at time *t*:

1. Training labels and feature values must have timestamps strictly earlier than *t*.
2. A macro observation may be carried forward only after it was published.
3. Backward filling is prohibited.
4. Regime labels are generated walk-forward. Every emitted label records the final
   training timestamp and an out-of-sample flag.
5. Event prevalence is estimated from the row's historical training and validation
   windows, never from the full realized tape.
6. A signal produced for row *t* is paired with returns no earlier than row *t+1*.
7. FRED values use their earliest ALFRED release and become visible only on the
   corresponding real-time availability date. A revised-history fallback is labeled
   non-point-in-time and cannot clear governance.

The secret-free `point_in_time_macro.py` adapter runs after Part 0 and rebuilds both
`macro_data.parquet` and `features_full.parquet` before the regime engine starts.
This placement prevents the regime model from reading the revised-history feature file.

## Calibration contract

Platt parameters are fit on the older 75% of a chronological calibration sample and
accepted only if they improve Brier loss on the newer 25%. Inverted, near-constant,
insignificant, or holdout-worsening calibrators fall back to passthrough behavior.

## Data freshness contract

Part 1 measures freshness before forward fill or proxy substitution. Core market
series may be at most one business day behind the expected completed session;
secondary series may be at most five. Breaching either limit records diagnostics
and forces downstream governance to fail closed.

The execution layer publishes zero instructions while data are stale, point-in-time
lineage is incomplete, or governance is uncleared. It must not publish a schedule
that merely looks current while relying on an older decision.

## Statistical contract

Raw accuracy is descriptive only. Because tail events are imbalanced, inference uses
balanced accuracy, Matthews correlation, AUC, Brier skill, calibration error, and a
deterministic label-permutation null that preserves event prevalence. Historical
evidence is not considered positive unless full-period AUC is at least 0.50 and
Brier skill is non-negative versus the causal prevalence forecast.

Every score uses the causal base-rate value attached to that observation. AUC and
balanced-accuracy inference requires at least five positive and five negative outcomes.
Until that class-count guard clears, point estimates are descriptive and every
directional significance flag remains false.

## Evidence cohort contract

`causal-integrity-v2` is a new live-evidence cohort. Earlier prediction-log rows are
retained as `legacy-pre-causal-integrity-v2` for reproducibility, with
`evidence_eligible=0`. Promotion counts only current-protocol rows that were produced
with fresh, point-in-time inputs. Code SHA and workflow run ID are stored on each new row.

## Operational contract

Scheduled production jobs are serialized and use an Eastern-date completion marker.
A delayed GitHub scheduler run is allowed to execute; later duplicate runs skip only
after the date marker has been committed. Pull or merge failures are fatal and may
not be hidden with `|| true`.

Backfill may run only on weekdays at or after 16:20 Eastern. Manual dispatch does not
bypass this settlement gate. Failure to regenerate Part 9 is fatal. Both production
and backfill refresh the dashboard snapshot and SHA-256 artifact manifest before commit.

## Dependency contract

Core and development dependencies exclude PyTorch. The experimental BNN sleeve
requires both `requirements-bnn.txt` and `PRICECALL_ENABLE_BNN=1`.
Core dependencies have bounded major versions; CI runs on `main`, pull requests, and
hardening branches and validates production modules as well as tests.

## Deferred issue

The public FRED credential finding is intentionally **not changed in this release**
at the repository owner's request. It remains unresolved and should be rotated and
removed from source in a separate security change.
