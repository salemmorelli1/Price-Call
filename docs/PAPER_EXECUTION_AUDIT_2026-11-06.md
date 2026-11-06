# Price-Call paper execution and DQN assessment — November 6, 2026

Audited main: `fea434aa2aae8fa4036f2a80648b960f03b65a86`.
Audited tree: `2e4c1f6d76b1be47e33420c8ddf594e1a9dabd7f`.

## Why the gate is closed

The [full production run 42156817952](https://github.com/salemmorelli1/Price-Call/actions/runs/42156817952)
successfully generated the November 5 decision, reconciled outcomes, verified
the ledgers, committed the artifacts and dispatched the
[successful Pages run 42159521034](https://github.com/salemmorelli1/Price-Call/actions/runs/42159521034).
The earlier [run 42146284626](https://github.com/salemmorelli1/Price-Call/actions/runs/42146284626)
deferred input processing and retained diagnostics without committing a new
forecast. Successful Actions and a closed scientific gate can coexist.

| Prespecified control | November 5 decision snapshot | Required |
| --- | --- | --- |
| Historical AUC | 0.5112345754 (1,720 realized historical rows) | Above 0.50 |
| One-sided historical AUC p-value | 0.259431 stored; 0.258090 read-only DeLong reference | At most 0.10 |
| Causal historical Brier skill | −0.0030597119 | At least +0.005 |
| Fixed prospective paper cohort | 30 issued; 29 realized; November 6 target pending | 60 locked realized outcomes |
| Independent review | `prospective_holdout_pending` | Separate review after the fixed holdout |

The DeLong reference is not adjusted for serial dependence. It is a read-only
cross-check, not a new approval test. The negative Brier skill says the model's
mean squared probability error exceeded that of the causal baseline on these
historical observations. Neither historical test clears its threshold. The
pipeline's `FAIL_CLOSED_NEUTRAL`, `NOT_VALIDATED` and `FLAT_VETOED` are therefore
supported. A 60/40 target describes the policy; the recorded paper book is
$1,000 cash, zero VOO/IEF holdings, zero trades and `DRY_RUN`.

The snapshot's input freshness, point-in-time macro, and same-run execution
lineage checks passed at publication. Its data extend through November 5 and
the publication expired after the November 6 XNYS settlement boundary (4:20
p.m. Eastern). At the audit read time, the November 6 decision was not yet
published. That operational freshness block is separate from the statistical
and review blocks. A fresh run cannot supply the missing independent evidence.

## Why a DQN is the wrong production change now

The core in `part2_predictor.py` is supervised logistic regression, random
forest and histogram gradient boosting with optional XGBoost. `part2c_bnn_sleeve.py`
is an optional PyTorch uncertainty experiment. The runner requires both
explicit opt-in and a Part 2B recommendation to activate it. It is not the
current allocation policy or a Q-value network.

Price-Call estimates the chance that the next-session VOO-minus-IEF spread
falls below a trailing, shifted tail threshold. This event probability alone
does not give the expected magnitude of gains and losses or net portfolio
utility. A useful forecast and a useful trading policy must be tested
separately, with their corresponding outcomes.

At the scale assumed here, treat the market price path as exogenous to the
paper allocation. Under a fixed price path and a specified cost model, the
two realized asset returns let a researcher compute the hypothetical one-step
payoff for each candidate weight. There is no need for the paper bot to
explore the market to reveal which historical price path follows an action.
Holdings, turnover costs, settlement and drawdown constraints do carry state
forward; they can create a genuine sequential problem. Test whether this
dependence warrants a dynamic policy before fitting a deep Q-function.

Replaying the same historical dates with different actions, seeds or episode
cuts does not create independent market paths. A DQN adds an estimated future
value and extra choices about state, reward and exploration to a short,
changing sample. Its policy can exploit the assumed fill/cost model, rather
than learn a repeatable allocation advantage. In fixed-batch settings where
some actions lack evidence, Q-learning can also extrapolate poorly. This
limitation is demonstrated by [Fujimoto, Meger and Precup (2019)](https://arxiv.org/abs/1812.02900);
[Levine et al. (2020)](https://arxiv.org/abs/2005.01643) explain the underlying
offline distribution shift. The full-information historical-price assumption
helps estimate hypothetical one-step payoffs; it does not validate fills or
remove regime and sample-selection uncertainty.

In this repository the decision is made after the prior close settles, so
that already observed close is a mark and a forecast-label anchor. It cannot
also be assumed to be an available fill after issuance. The production paper
book has no recorded fills. A sparse trade log cannot measure daily NAV,
true drawdown, or a daily-return Sharpe ratio. An apparent reward can even be
an accounting artifact: the funding bug below turned $1,000 into $1,999.94
before any market movement in a controlled test.

The appropriate next research test is a separate, read-only walk-forward
comparison of a specified 60/40 benchmark and a bounded, cost-aware tilt with
a no-trade band. Use actual post-issuance executable reference prices, drifted
holdings, two-sided turnover, spread/slippage and fees, and daily net NAV.
Compare paired net returns and drawdown under predeclared cost stresses and
chronological, untouched evaluation periods. If holding-dependent decisions
show incremental value, compare a small state/action dynamic program before
adding a deep learner. A future learner gets a new protocol and an untouched
prospective test; it cannot train on and then claim confirmation from the
locked v4 cohort. This PR changes execution controls and accounting only.

## Reproduced execution defects and corrections

Eight original-code replay cases failed using isolated artifacts with real
same-run IDs, exact-session observed price pairs and synthetic gate inputs.
They are latent defects; no paper trade was observed on audited main.

1. **Part 8 could generate orders with stale data.** The override considered
   publication flags and modes but omitted freshness. `bool("false")` and
   `bool("0")` also turned textual rejections into approval. Instruction
   generation now requires fresh input, an explicit Part 2 boolean, an
   explicit Part 3 boolean or its integer `1`, and both NORMAL modes. The
   output carries the complete governed target for downstream verification.
2. **Part 10 could ignore a valid no-action plan.** It checked the run/session
   lineage but chose `DRY_RUN` solely from Part 9. With a favorable Part 9
   fixture (or its optional check disabled), it executed the Part 7 proposal
   even when the authoritative Part 8 artifact had zero orders and a veto.
   Now Part 10 consumes the same verified plan it read for lineage. It requires
   explicit approval, valid nonempty orders and matching governed weights;
   an inconsistent signal, absent plan or missing statistical inputs cannot
   authorize a paper trade. The Part 9 gate retains its numeric limits and
   requires finite, current-cohort evidence. The Part 7 proposal cannot
   substitute for the authorized target.
3. **DRY_RUN could liquidate a previously held paper position.** A passive
   stop-loss check ran before the dry-run safeguard. The blocked path now
   updates marks and diagnostics without liquidation, trade fees or changes
   to existing cash, shares or trade count.
4. **Paper purchases did not debit their principal.** A cash-only portfolio
   bought $1,000 of holdings and retained its $1,000 cash less the fee. The
   old fixture reported $1,999.94 NAV and used the ungoverned 85% VOO proposal.
   A positive authorized fixture now uses the governed 75/25 target. It solves
   `invested_notional + fees(invested_notional) = pre_trade_NAV`, including
   both assets' absolute traded notionals. At 0.60 bps on the initial buy it
   records $999.9400 NAV and spends the cash. Rebalances after price drift
   are also tested. These remain synthetic valuations, not executable fills.
5. **Sparse marks produced a Sharpe estimate.** Averaging separately annualized
   returns over unequal trade intervals does not yield daily-return Sharpe.
   Part 10 now withholds Sharpe and identifies its trade-mark valuation basis.

The publication validator now rejects an active signal, an active paper state,
or a same-session trade in a Part 8 no-action publication. The read-only
governance audit distinguishes the plan and target from the paper book. The
new checks do not upgrade evidence, change the historical thresholds, alter the
prediction generator, retrain models, rewrite ledgers, or activate the BNN.

## Verification and activation

Baseline: 202 tests passed in both normal and main-ref simulation. The corrected
suite executes the actual Part 8 writer and Part 10 paper bot on isolated files,
with observed Parquet price inputs and the network disabled. It tests failing
and passing plans, forged or missing approval flags, stale data, disabled
optional accuracy checks, incomplete health evidence, preservation during
DRY_RUN, the self-financing cash equation, and publication contradictions.

Use `bash verify_audit.sh` on the exact clean checkout with Python 3.12. It
installs the locked environment, runs the full suite in both modes, verifies
the source manifest, prints the read-only audit, and builds and checks a local
Pages preview. It needs no GitHub CLI or credentials. The current snapshot
should report integrity PASS and governance BLOCKED. Main retains its existing
production schedule; after review and merge the next eligible production run
uses the corrected execution path. The PR preview does not deploy main.
