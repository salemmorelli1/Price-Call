# Governance contract audit — October 27, 2026

Audited main: `768057d37e57ae0dbab97c451ae4299e66a9afe1`.
Audited tree: `5180fb9dafb31fbb0f1d1b534c580533b799711f`.

## Why the published gates are closed

The [last full production run inspected](https://github.com/salemmorelli1/Price-Call/actions/runs/40423957083)
successfully computed the October 26 decision, reconciled outcomes, verified
the ledgers, committed artifacts and dispatched Pages. The later short
[catch-up run](https://github.com/salemmorelli1/Price-Call/actions/runs/40436856940)
correctly skipped a completed session. Its green status does not represent
another forecast. This is not evidence of a model crash or a market holiday.

| Control | Published evidence | Frozen requirement |
| --- | --- | --- |
| Historical AUC | 0.5173419863 | Above 0.50 is necessary, not sufficient |
| One-sided historical AUC p-value | 0.156917 | At most 0.10 |
| Causal Brier skill | 0.00234928635 | At least 0.005 |
| Fixed prospective cohort | 21 realized, 22 issued | 60 locked outcomes before confirmatory scoring |
| Prospective class counts | 5 positive, 16 negative | Class counts alone do not satisfy maturity |
| Legacy outcomes excluded | 40 | Retained separately from current v4 evidence |
| Independent review | prospective_holdout_pending | Separate prospective review required |
| Auxiliary alpha reason | fail_closed_live_veto | Core closure continues to veto the sleeve |

`FAIL_CLOSED_NEUTRAL` means the neutral **60/40 paper allocation**. It is not a
neural-network exception. `NOT_VALIDATED` and `FLAT_VETOED` agree with these
requirements. Part 9's descriptive live AUC/Brier statistics cannot replace the
fixed, issued-`p_final_cal` prospective evaluation or the historical controls.

At the audit's read time, October 27 had settled but the last publication still
covered October 26. The read-only audit correctly added a one-session freshness
block. Its expiry was October 27 at 20:20 UTC (4:20 p.m. Eastern). A fresh
production publication can clear that operational block while statistical and
review blocks remain. Scheduled workflow delivery and outcome availability must
be inspected separately from publication approval.

## Confirmed defects and corrections

### Dashboard approval could contradict the frozen policy

The prior dashboard used threshold values from the Part 2 artifact as its
decision rules. In a copied fixture with the other controls satisfied, changing
the artifact to p-max 0.50 and skill-min -0.01 caused p=0.15 and skill=0.002 to
display approval. The metadata had redefined the policy it was supposed to
describe. The dashboard now uses the frozen `Part2Gen53Config` constants and
blocks any recorded-threshold mismatch.

The prior dashboard also relied on the Part 3 `final_pass` flag without requiring
the Part 2 flag and both publication modes to agree. Passing-source fixtures
could display an open gate while the core flag was false or a mode was
`FAIL_CLOSED_NEUTRAL`/unknown. JSON booleans used as numeric metrics, an AUC above
one, a negative p-value, or Brier skill above one were accepted by the earlier
comparison-only checks. These cases now fail closed. A fully consistent passing
fixture remains able to display validation; the correction does not force every
possible publication to stay blocked.

Before correction, **11 fault-injection cases reproduced false dashboard
approval**, two additional diagnostic tests failed, and the passing control
succeeded. These are separate validation defects; they do not explain or
invalidate the correctly closed gate in the real October 26 snapshot.

The snapshot and read-only `audit_governance.py` report now expose `gate_checks`:
each control has an identifier, observed value, requirement and pass/fail result.
The alpha veto reason is also explicit. The original issuance and forecasting
code does not change, and an audit report cannot grant allocation permission.

### The uploaded Pages artifact differed from its verified manifest

The [current successful Pages run](https://github.com/salemmorelli1/Price-Call/actions/runs/40428236461)
uploaded artifact `12701046696`. Inspection of its actual ZIP/tar found:

- the publication manifest expected **13 files**;
- the archive contained **12**, excluding the manifest itself;
- `.nojekyll` was missing; all common file hashes and byte counts matched.

The pinned Pages uploader v5 excludes hidden files by default, after the local
builder has already verified its directory. The workflow now sets
`include-hidden-files: true`, preserving the complete verified file set. This
manifest mismatch did not cause the statistical rejection.

PRs now build and upload a preview with read permissions; deployment stays
restricted to main. Preview runs have a separate concurrency key and do not
cancel active production publication. Production Pages remains manually
dispatched by the existing writers; no independent push deployment is added.

## Verification and preserved method

The baseline suite passed **180 tests**. The corrected suite has **202 tests**,
including the passing-control fixture, threshold and metric-domain checks,
core/governance consistency, separate review/freshness/macro controls, maturity,
read-only audit behavior, and the Pages packaging/permissions contract. Existing
causal data, real model, prospective cohort, execution lineage, browser expiry,
artifact tamper detection and source-preservation tests remain active.

Windows and Ubuntu CI run the delivered `bash verify_audit.sh`: fresh pinned
Python 3.12 environments, dependency checks, compile/lint, the normal suite and
main-ref simulation, artifact verification, read-only governance audit, and
offline Pages build/verification. Exact heads, trees and final CI links are
recorded in the PR and delivery response. The PR preview artifact is separately
inspected against its manifest after upload.

No issued probability, forecast, realized outcome, price anchor, eligibility
flag, model file or tracked publication bytes are rewritten in this patch.
`causal-integrity-v4`, its cohort start, fixed first-60 selection, frozen event
definition, p <= 0.10, Brier skill >= 0.005, independent review, DRY_RUN, and the
neutral allocation remain in force. This strengthens reporting validation; it
does not establish new predictive skill or improve accuracy by changing a gate.

## Activation

After review and merge, a main-branch **Deploy GitHub Pages** run can publish
the corrected diagnostics from verified source artifacts. Subsequent successful
production/backfill runs continue their existing dispatches. A forced forecast
rerun, forced retraining or retrospective cohort relabeling is not required.

Primary implementation references:

- [Frozen project protocol](CAUSAL_RESEARCH_PROTOCOL.md).
- [Pinned uploader v5](https://github.com/actions/upload-pages-artifact/blob/fc324d3547104276b827a68afc52ff2a11cc49c9/action.yml).
- [Pinned Pages settings action](https://github.com/actions/configure-pages/blob/45bfe0192ca1faeb007ade9deae92b16b8254a0d/action.yml).
