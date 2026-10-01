# Production input audit — October 1, 2026

## Scope and evidence

Audited main commit `1e4f11757b4d9bd084e4d632c7f68a48c40e65e8`, its published
manifest/ledgers, and the September 30 production and backfill job logs. Times
below are September 30 in America/New_York (EDT, UTC−4).

| Run | Created | Duration | Finding |
| --- | --- | --- | --- |
| [Production #368](https://github.com/salemmorelli1/Price-Call/actions/runs/36794780849) | 8:10 p.m. | 2m16s | Part 0 failed before training: September 30 VOO/IEF closes were missing after individual/paired retries and archive recovery. |
| [Production #369](https://github.com/salemmorelli1/Price-Call/actions/runs/36798174757) | 8:50 p.m. | 1m28s | Same current-session input failure. |
| [Backfill #413](https://github.com/salemmorelli1/Price-Call/actions/runs/36804023719) | 10:03 p.m. | about 1m20s | Exact closes became available; due outcomes were reconciled. Earlier backfill attempts correctly reported DATA_PENDING. |
| [Production #370](https://github.com/salemmorelli1/Price-Call/actions/runs/36804826262) | 10:13 p.m. | 25m34s | Market inputs recovered and training ran, but a required macro retrieval failed. The older code still marked production complete. |

In #370 the final `T10Y2Y` real-time window, `2024-09-09..2026-09-30`, raised
`ValueError(None)` through fredapi. Earlier windows succeeded. The adapter then
made `curve_2s10s` entirely unavailable, rebuilt/trained with the missing input,
and returned success. Published Part 0 has `historical_point_in_time_complete=false`,
and Part 2 has `macro_point_in_time_ok=false`. A green Actions conclusion did not
mean the required input contract or allocation gate had passed.

## Corrections

1. Classify a remaining gap exclusively on the latest completed XNYS session as
   DATA_PENDING. Older core gaps remain fatal. Do not synthesize closes.
2. Retry transient ALFRED request failures up to three times per four-year window.
   Keep successful chunks when a retry succeeds. Do not retry schema/credential
   errors or replace failed windows with revised history.
3. Reject incomplete required macro retrieval before rebuilding the PIT feature
   file or starting model fitting. Transport failures that exhaust retries defer
   the attempt; malformed/permanent failures terminate it.
4. Propagate exit 75 through the direct runner. The production workflow records
   DATA_PENDING explicitly, retains attempt diagnostics, and guards all outcome
   reconciliation, completion markers, dashboard synchronization, artifact
   publication, and deployment on `steps.pipeline.outputs.ready == 'true'`.
5. Permit a new attempt for an older completed/incomplete-macro session only when
   publication hashes verify, the decision is ineligible, and its next-session
   target remains open. This issues a new forecast; it does not relabel the
   earlier diagnostic. Eligible issued forecasts are never replaced.
6. Add an automatic read-only feature-branch replay using the production lock
   and real providers. It tests the full suite with production dependencies,
   runs the direct pipeline, checks all 14 PIT macro series, and retains evidence.
   It has no commit/deploy permissions or steps.

## Statistical status at the audited commit

The ledger contains 45 forecasts: 28 pre-v3 legacy, 12 v3, four eligible v4
forecasts with realized outcomes, and one ineligible September 30 v4 diagnostic.
That diagnostic is excluded because the macro input contract failed. All older
rows are retained. Its exclusion must not be reversed after seeing its outcome.

The published overall historical AUC is 0.5153209 and causal Brier skill is
−0.0030100, below the prespecified 0.005 minimum. Independent validation is
pending. Fixing provider readiness does not establish forecasting skill.
The current protocol, target definition, AUC p-value threshold (0.10), Brier
threshold (0.005), minimum 60 eligible realized outcomes, and paper-only neutral
60/40 allocation remain in force.

## Timing and verification

The intended production triggers remain `45 20`, `45 21`, and `22 23` UTC on
weekdays. In EDT these are 4:45, 5:45, and 7:22 p.m. GitHub documents that scheduled
events can be delayed or dropped under load; the logs establish the actual
creation times above but do not expose the scheduler's internal cause.
There is no minimum correct runtime: a full replay can take about 25 minutes,
an already-completed session skips computation, and a pending input stops early.

Primary scheduling reference:
https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule

Regression coverage includes the observed ValueError(None), exhausted transport
retries, permanent/schema failures, current versus historical close gaps,
no downstream fitting after pending input, actual workflow shell exit handling,
publication readiness guards, target-close recovery boundaries, and immutable
eligible forecasts. The local CI lock passes the suite and main-ref simulation;
the optional DuckDB test requires the production dependencies. GitHub CI and the
real-provider replay are reported separately in the reviewable PR.
