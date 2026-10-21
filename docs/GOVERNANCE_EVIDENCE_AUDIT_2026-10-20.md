# Governance and publication evidence audit — 2026-10-20

## Observed state

The audit began from main `c9d3888f678bc47c33524b6bb98a6802b6f93a71`
(tree `b8040958e5703d15bcd332d64acf159bbe01ea1f`). Its last completed production
decision was October 19. October 20's backfill had completed, and the October 20
production run was still computing when the screenshots were captured/read.
The screenshots' stale-session warning is consistent with those separate dates.

The last completed full production run inspected was
[39344540151](https://github.com/salemmorelli1/Price-Call/actions/runs/39344540151),
job 118373855002. It completed its pipeline, preserved the ledgers, scored the
prospective cohort, committed its outputs, and dispatched Pages successfully.
The newer production run is
[39514706324](https://github.com/salemmorelli1/Price-Call/actions/runs/39514706324),
job 118943964338. A running job is not evidence of either a crash or successful
completion. GitHub's creation times establish delivery delay, not its cause.

| Observed control at the audited snapshot | Value | Required conclusion |
| --- | --- | --- |
| Historical AUC | 0.5139108455 | Above 0.50 alone is insufficient |
| Stored one-sided AUC p-value | 0.212965 | Fails p <= 0.10 |
| Model Brier loss | 0.1610254505 | Worse than causal baseline |
| Causal baseline Brier loss | 0.1605762547 | Rowwise, prior-only prevalence |
| Causal Brier skill | -0.0027973992 | Fails skill >= 0.005 |
| Current v4 realized outcomes | 18 | Fewer than the fixed 60 |
| Current positive / negative outcomes | 4 / 14 | Class-count inference guard also remains closed |
| Legacy realized rows excluded | 40 | Must not be counted as v4 evidence |
| Independent validation | prospective_holdout_pending | Separate review remains required |
| Core publication | FAIL_CLOSED_NEUTRAL | Neutral 60/40 paper allocation |
| Auxiliary alpha | FLAT_VETOED | Does not override core clearance |

These values justify keeping the gate closed. A successful process or a fresher
publication cannot create evidence of forecasting skill. `NEUTRAL` is an
allocation state, not a reference to a neural-model failure. October 20 is a
completed XNYS session in the repository's exchange calendar; the evidence blocks
are not explained by a market holiday.

## Reproduced defects and corrections

1. **Outcome-dependent holdout membership.** The old evaluator removed unresolved
   forecasts before selecting its first 60. With 61 issued forecasts and the
   first outcome missing, it scored forecasts 2–61 and even reported a passing
   threshold result on the regression fixture. The corrected evaluator fixes
   forecasts 1–60 by decision date first, reports the unresolved target, and
   waits. Its issuance-only SHA-256 fingerprint does not change when an outcome
   arrives or later forecasts are appended. Existing forecast-writing immutability
   controls remain in force; the fingerprint adds an independent comparison key.
2. **Fractional evidence flags.** `astype(int)` admitted `evidence_eligible=1.9`
   as eligible and truncated fractional legacy-horizon flags. Exact numeric
   equality now admits only 1 for eligibility and 0 for the horizon exclusion.
   Regression tests for both the missing-outcome and fractional-flag cases failed
   before the patch and pass afterward.
3. **Uncertainty estimator in the review.** The frozen forecast helper's Q1/Q2
   expression is not the DeLong structural-component sample-variance estimator.
   The new read-only review and prospective evaluator use
   `var(V10, ddof=1)/n_positive + var(V01, ddof=1)/n_negative`, with half credit
   for ties, class-count guards, unrounded calculations, and no normal-inference
   claim for zero estimated variance. The audited historical tape yields
   SE 0.0174672541 and p 0.2129010473 under this reference calculation, versus
   stored SE 0.017472 and p 0.212965. Both fail the same threshold.
   **The frozen Part 2/Part 3 forecast implementation is not changed or silently
   relabeled.** Changing its regime/Platt selection would alter the forecasting
   method and requires a separately versioned prospective protocol. The corrected
   reference result cannot approve allocation. DeLong's normal approximation also
   does not account for serial dependence; this remains part of independent review.
4. **Static freshness could remain green indefinitely.** The browser previously
   used a stored boolean with no expiry. A page from a successful day could remain
   apparently fresh if both writers stopped. The snapshot now records the next
   relevant exchange settlement timestamp. The actual dashboard JavaScript closes
   expired/missing-deadline publications on initial load, every minute, and on
   focus/visibility changes, while preserving the original source snapshot.
5. **Summary flags could contradict the displayed evidence.** The dashboard
   previously trusted `final_pass` without independently requiring its displayed
   historical metrics, macro coverage, prospective result, and separate review.
   It now requires all controls. Its maturity counter uses outcomes within the
   fixed issuance window; later resolved forecasts cannot fill an earlier hole.
6. **Pages checked paths but not source authenticity.** `prepare_pages.py` formerly
   copied artifacts without checking their manifest or execution identity and
   reused existing output directories. It now checks source hashes, status
   markers, and execution lineage; builds in temporary staging; refreshes only
   staged JSON/HTML; hashes every deployed file; rejects modified/unlisted files;
   and atomically publishes into a new directory. It also detects a changed
   source manifest during the build. Pages dispatches are restricted to main;
   feature-branch checks build previews without deployment permission.

## Preserved contracts

No raw market observations, frozen forecasts, realized outcomes, adjusted-price
anchors, existing attribution report, model artifacts, or tracked publication
bytes are rewritten by this audit. All changes are source, tests, workflow
verification, or documentation. In particular:

- `causal-integrity-v4` and the September 24 earliest decision date remain fixed;
- the target is next-XNYS-session VOO-minus-IEF log return below the row's
  trailing-63-observation 20th-percentile threshold shifted by one observation;
- confirmatory scoring uses the issued `p_final_cal` and rowwise causal baseline;
- the 60-outcome minimum, five observations per class, AUC direction/p <= 0.10,
  Brier skill >= 0.005, and separate independent review remain required;
- corrected review statistics do not change the frozen forecast generator;
- 60/40, DRY_RUN, the alpha veto, and no-action execution remain in force while
  governance is uncleared; no automatic promotion is introduced;
- the scheduled providers, training order, retries, idempotency, writer
  synchronization, and main publication schedule are unchanged.

## Verification

The baseline CI lock produced **152 passed, 1 skipped** locally; the skipped
check needed optional DuckDB. Adding the exact production DuckDB 1.5.5 and
PyArrow 23.0.1 pins enables that integration check. The corrected suite has
**180 passing checks** locally, including a main-ref simulation. Verification
also covers pinned dependency consistency, compilation/lint, artifact hashes,
execution lineage, read-only audit behavior, Pages staging/source preservation,
tampering, output-path guards, the published Stata tied-rating AUC/SE example,
an independent dense imbalanced-sample variance calculation, and execution of
the actual dashboard binding in Node through an expiry boundary. CI runs this
suite on Python 3.12 in isolated Windows and Linux environments and verifies
that checks did not rewrite tracked artifacts. Both run the delivered
`bash verify_audit.sh` entry point, including Git Bash activation and fresh
environment/preview directories. Exact GitHub runs are recorded
in the PR after completion.

The read-only audit's `integrity_status=PASS` is compatible with
`governance_status=BLOCKED`. Missing/corrupt source evidence produces a nonzero
process exit; insufficient statistical evidence is an explicit report result.

## Primary references

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
  Comparing the areas under two or more correlated receiver operating
  characteristic curves: A nonparametric approach. *Biometrics, 44*, 837–845.
  https://doi.org/10.2307/2531595
- StataCorp, *roccomp — Tests of equality of ROC areas*, methods/formulas p. 12
  and the tied-rating reference example pp. 6, 9–10:
  https://www.stata.com/manuals/rroccomp.pdf
- GitHub scheduling reference (scheduled events may be delayed):
  https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
