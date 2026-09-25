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
7. FRED values use the earliest retrieved ALFRED release and become visible on
   the first exchange session after that release date. ALFRED gives dates but not
   intraday publication times, so same-day forecast use is disallowed. If the
   archive begins after the model calendar, earlier dates remain missing. A failed
   or unavailable series
   remains missing and cannot clear governance; revised-history values are never
   substituted into training features.

The secret-free `point_in_time_macro.py` adapter runs after Part 0 and rebuilds
`macro_data.parquet`, `features_full.parquet`, and any Part 0 DuckDB copies before
the regime engine starts. The regime loader checks the feature-file hash and
refuses a DuckDB fallback when the point-in-time parquet is missing.

Core VOO/IEF closes can be replayed from a manifest-verified older production
snapshot with per-ticker run provenance. The newest settled session may use a
separately verified backfill only if it records the exact target date and both
prices agree with the preceding verified anchor. A missing target in a backfill
remains missing, including for legacy forecast rows; it never moves to the next
available provider date. If either source cannot be verified, Part 0 stops.

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
evidence is not considered positive unless full-period AUC exceeds 0.50, its
one-sided DeLong p-value is at most 0.10, and Brier skill is at least 0.005 versus
the rowwise causal prevalence forecast.

Every score uses the causal base-rate value attached to that observation. AUC and
balanced-accuracy inference requires at least five positive and five negative outcomes.
Until that class-count guard clears, point estimates are descriptive and every
directional significance flag remains false.

## Evidence cohort contract

`causal-integrity-v4` is the new live-evidence cohort after removing revised macro
history. All v3 rows, including the eleven realized v3 rows previously excluded for
incomplete macro provenance, remain in the ledger and never count toward v4.
Promotion counts only current-protocol rows produced on main with fresh, dated
point-in-time inputs, an exact row-level tail threshold, and a recorded issuance
time before the target session's close. Code SHA and workflow run ID are retained.

The 2020–2026 historical holdout has already informed debugging and is descriptive
for v4. The v4 method is frozen for a new prospective evaluation starting no earlier
than 2026-09-24. Historical AUC p <= 0.10 and causal Brier skill >= 0.005 retain
their numerical thresholds, but cannot promote v4 without an independently reviewed
prospective holdout. The `independent_validation_ok` publication gate remains false
until a separate review assesses the prospective series. Eligible paper forecasts
can accumulate while the publication and allocation gates stay closed; 60 is only
the minimum for live inference, not an automatic release rule.

`python evaluate_prospective_cohort.py --root .` checks issuance, the next
exchange session, realized price provenance, and the first 60 eligible v4
outcomes. It uses the already specified `p_final_cal` probability (the Part 2
base probability), each row's causal baseline, and the same one-sided DeLong
AUC p <= 0.10 and Brier skill >= 0.005 thresholds. The first 60 are fixed
before scoring; later outcomes cannot change that confirmatory result. Part 9's
blended/recalibrated live metrics remain descriptive. Even a passing report
requires a separate review of integrity and the historical gate before any
allocation permission changes. An already eligible forecast cannot be replaced
by a manual rerun of its decision session.

Only a main-branch production forecast issued before its target close can qualify.
Research replays on feature branches retain their diagnostic artifacts in GitHub
Actions, but cannot commit artifacts or deploy Pages.

The event label, distributional overlay, prediction log, and live attribution all use
the same backward-looking 63-observation 20th-percentile threshold shifted by one row.
The fixed `-0.015` value is only a cold-start fallback inside Part 1 and is never silently
substituted for a missing threshold in current-protocol evidence.

## Operational contract

Scheduled production jobs are serialized and use an Eastern-date completion marker.
A delayed GitHub scheduler run is allowed to execute; later duplicate runs skip only
after the date marker has been committed. Pull or merge failures are fatal and may
not be hidden with `|| true`.

Scheduled backfill triggers run on weekdays after the settlement boundary. Manual
dispatch uses the same latest-completed-XNYS-session gate and cannot admit an
unsettled close. Failure to regenerate Part 9 is fatal. Both workflows
synchronize before computation, abort if the branch changes during computation, build
the SHA-256 manifest after all outputs, push without conflict-merging, and explicitly
dispatch the verified Pages deployment. Pages has no independent push trigger, so a code
merge cannot publish an old or incomplete artifact snapshot before production succeeds.

## Dependency contract

Core and development dependencies exclude PyTorch. The experimental BNN sleeve
requires both `requirements-bnn.txt` and `PRICECALL_ENABLE_BNN=1`.
Core dependencies have bounded major versions; CI runs on `main`, pull requests, and
hardening branches and validates production modules as well as tests.

## Credential posture

Current source contains no embedded FRED credential. `point_in_time_macro.py` reads
`FRED_API_KEY` only from the runtime environment, and GitHub Actions supplies it from
the repository secret. Any credential exposed by earlier history should remain
revoked; source cleanup does not invalidate a previously disclosed key.
