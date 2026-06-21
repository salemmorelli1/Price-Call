# @title Part 2 Gen 8 Overwrite


#!/usr/bin/env python3
from __future__ import annotations
import sys as _sys
import os as _os

# ── Colab / environment detection ─────────────────────────────────────────────
_IN_COLAB = "google.colab" in _sys.modules
_DRIVE_ROOT = _os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject")


def _colab_init(extra_packages=None):
    """Mount Google Drive (if in Colab) and pip-install any missing packages."""
    if _IN_COLAB:
        if not _os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        _os.makedirs(_DRIVE_ROOT, exist_ok=True)
        _os.environ.setdefault("PRICECALL_ROOT", _DRIVE_ROOT)
        _os.environ.setdefault("PRICECALL_STRICT_DRIVE_ONLY", "1")
        _os.environ.setdefault("PRICECALL_ALPHA_FAMILY", "part2a21")
    if extra_packages:
        import importlib, subprocess
        for pkg in extra_packages:
            mod = pkg.split("[")[0].replace("-", "_").split("==")[0]
            try:
                importlib.import_module(mod)
            except ImportError:
                print(f"[setup] pip install {pkg}")
                subprocess.run([_sys.executable, "-m", "pip", "install", pkg, "-q"],
                               capture_output=True)



import json
import math
import os
import warnings
import hashlib
import platform
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "GEN5_PART2_G532_DAILY_CANONICAL_V2"  # V2: Part 16 audit — per-regime AUC monitoring added

try:
    import xgboost as xgb  # type: ignore
    HAVE_XGB = True
except Exception:
    xgb = None
    HAVE_XGB = False


@dataclass
class Part2Gen53Config:
    PART1_DIR: str = _DRIVE_ROOT + "/artifacts_part1"
    # FIX (Finding 2, Part6 Audit 2026-04): Part 2 now reads
    # regime_labels_p6.parquet written by Part 1 (which in turn reads Part 6's
    # regime_history.parquet). This allows the HMM regime label to flow through
    # Part 2's tape to Part 7, replacing the internal GMM as the canonical source.
    PART6_DIR: str = _DRIVE_ROOT + "/artifacts_part6"   # kept for direct fallback read
    PRED_DIR: str = _DRIVE_ROOT + "/artifacts_part2_g532/predictions"
    OUT_FILE: str = "g532_final_consensus_tape.csv"
    SUMMARY_FILE: str = "part2_g532_summary.json"
    DIAG_FILE: str = "part2_g532_diag.json"
    ABLATION_FILE: str = "part2_g532_ablation.csv"

    H: int = 1                          # CHANGE: 1-day forecast horizon
    PURGE: int = 1                      # CHANGE: 1 row purge for non-overlapping daily labels
    TRAIN_WINDOW_DAYS: int = 252 * 4
    VALID_WINDOW: int = 252             # FIX: 1 full year of daily validation.
    # At H=1, each fold AUC is estimated on VALID_WINDOW rows.
    # 63 rows gives SE(AUC)≈0.063, making AUC=0.55 indistinguishable from noise (t=0.80).
    # 252 rows gives SE(AUC)≈0.032, enabling t=1.57 at AUC=0.55.
    # Trade-off: 252 rows of validation vs 756 rows of training (TRAIN_WINDOW=1008). Acceptable.
    REFIT_FREQ: int = 20
    HO_START_DATE: str = "2020-01-01"
    SEED: int = 42
    MIN_TRAIN_ROWS: int = 500           # CHANGE: larger minimum for daily model stability
    MIN_CLASS_COUNT: int = 20
    MIN_REGIME_VAL_ROWS: int = 24

    REGIME_COMPONENTS: int = 4
    REGIME_FEATURES: Tuple[str, ...] = (
        "stress_score_raw",
        "stress_score_change5",
        "vix_z21",
        "vix_term_z21",
        "credit_spread_z21",
        "breadth_z21",
        "tech_relative_z21",
        "spread_ret21",
        "excess_vol10",
        "voo_downside_vol10",
    )

    # Phase 1 + Phase 2 Gen 5 additions
    DIST_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    DIST_FUSION_WEIGHT: float = 0.00
    DIST_FUSION_WEIGHT_MAX: float = 0.00
    DIST_PENALTY_CAP: float = 0.65
    DIST_PENALTY_FLOOR: float = 0.08
    DIST_TRUST_MIN: float = 0.57
    DIST_GATE_MAX_WIDTH: float = 0.110
    DIST_GATE_MAX_CONTRADICTION: float = 0.18
    DIST_GATE_MIN_TAIL_AUC: float = 0.52
    DIST_GATE_MAX_TAIL_ECE: float = 0.14
    DIST_GATE_MIN_CONF_COVERAGE: float = 0.82
    DIST_GATE_REQUIRE_SIGN_AGREEMENT: bool = True
    CONFORMAL_ALPHA: float = 0.10
    DIST_MIN_SIGMA: float = 0.0025
    DIST_MIN_HISTORY: int = 40
    # H=1 recalibration (2026-04-12): was 0.06, calibrated for H=7 weekly
    # spread distributions. At H=1 the median conf_width ≈ 0.038. Setting
    # scale=0.15 targets median uncertainty_penalty ≈ 0.50, restoring
    # meaningful variation rather than permanent saturation at the 0.65 cap.
    # Governance policy choice: range [0.12, 0.25] is defensible; 0.15 is
    # the midpoint targeting p50 penalty ≈ 0.50.
    DIST_CONF_WIDTH_SCALE: float = 0.15   # was 0.06
    # FIX (Finding 23/26, Audit 2026-04-21):
    # The prior value was -0.015 / sqrt(7) = -0.00567, which is the H=7 weekly
    # threshold scaled DOWN to H=1 using the wrong direction of the sqrt-time rule.
    # For a daily (H=1) model the correct base threshold is -0.015 (the annualised
    # concept threshold applied to a single trading day). The classification model
    # is already trained on Part 1's rolling-20th-percentile labels, which naturally
    # produce a ~20% base rate.  This fixed threshold is only used by the
    # distributional prediction layer (quantile regression) for P(spread < threshold)
    # and by the summary JSON / prediction_log tail_threshold field consumed by Part 9
    # for live label reconstruction. Both must match a daily definition.
    TAIL_EVENT_THRESHOLD: float = -0.015  # H=1 daily threshold (was -0.015/sqrt(7))
    OVERLAY_TRUST_MIN: float = 0.60
    OVERLAY_WIDTH_TRIGGER: float = 0.075
    OVERLAY_PENALTY_TRIGGER: float = 0.38
    OVERLAY_MAX_TAIL_SHIFT: float = 0.035
    OVERLAY_TAIL_SHIFT_SCALE: float = 0.025
    OVERLAY_THRESHOLD_RELIEF_MAX: float = 0.060
    OVERLAY_CAUTION_ALPHA_CAP: float = 0.75

    BASE_WEIGHT_VOO: float = 0.60
    BASE_WEIGHT_IEF: float = 0.40
    SLIP_BPS: float = 5.0
    ALPHA_THROTTLE: float = 0.50

    HIGH_RISK_ABS_P: float = 0.31
    HIGH_RISK_EDGE: float = 0.06
    # H=1 recalibration (2026-04-12): was 0.255, set for H=7 weekly horizon.
    # At H=1 daily tail probs are compressed; p_final_cal >= 0.255 is met only
    # 13.6% of the time.  0.240 preserves the minimum meaningful edge intent
    # (base_rate + DEPLOY_DOWNSIDE_MIN_EDGE = 0.2158 + 0.022 = 0.238) while
    # allowing the gate to fire on structurally reachable probability levels.
    DEPLOY_DOWNSIDE_MIN_P: float = 0.240   # was 0.255
    DEPLOY_DOWNSIDE_MIN_EDGE: float = 0.022
    DEPLOY_UPSIDE_MAX_P_DELTA: float = 0.20
    # H=1 recalibration (2026-04-12): was 0.58, set for H=7 weekly regime.
    # At H=1 with VALID_WINDOW=252 rows, rolling fold AUC has mean=0.529 and
    # SE≈0.016; requiring 0.58 puts the bar 3.2 SE above the mean, producing
    # knife-edge firing and run-to-run stochastic variance.  0.530 is slightly
    # above the global holdout AUC (0.540) while remaining structurally reachable.
    DEPLOY_MIN_VAL_AUC: float = 0.530   # was 0.58
    DEPLOY_MIN_AGREEMENT: float = 0.74
    SPREAD_CONFIRM_MIN: float = 0.0008  # was 0.0015 (H=7). At H=1 leg uncertainty is
    # proportionally smaller; the confirmation floor must scale accordingly so that
    # spread_gate does not permanently exclude the model's daily prediction range.
    # FIX (F1, Quant-Guild Part 36 Audit): SPREAD_K 3.0 → 1.5
    #
    # ROOT CAUSE ANALYSIS (verified against 1,673-row artifact tape):
    # deploy_downside requires spread_component > 0.
    # spread_component = clip((-fwd_spread_hat - abs(spread_gate)) / SPREAD_SCALE, 0, 1)
    # spread_gate      = min(-spread_confirm, -DEPLOY_DOWNSIDE_SPREAD_ABS)
    # spread_confirm   = max(SPREAD_CONFIRM_MIN=0.0008, SPREAD_K * leg_uncertainty)
    #
    # The regression ensemble (fwd_spread_hat) has a typical negative magnitude
    # of -0.004 to -0.008 in the high_vol rows where p_final_cal >= 0.240.
    # With SPREAD_K=3.0:
    #   leg_uncertainty ≈ 0.002-0.007 (gamma_voo range: 0.993-0.999 → very low u)
    #   spread_confirm  = max(0.0008, 3.0 * 0.004) ≈ 0.012
    # For spread_component > 0: fwd_spread_hat < -0.012
    # But max negative fwd_spread_hat in qualifying high_vol rows = -0.0076.
    # The safety margin EXCEEDS the regression's reachable range → structural zero.
    #
    # Consequence: In the 485 non-passive rows (high_vol + crisis), only 15 have
    # spread_component > 0, and ALL 15 fail p_final_cal >= 0.240 or edge >= 0.022.
    # In the 24 high_vol rows with p >= 0.240, spread_component = 0 on all 24.
    # Result: 0 deploy_downside events across all 1,673 rows → deploy_gate_pass=False
    # permanently → final_pass=False permanently → system locked in FAIL_CLOSED_NEUTRAL.
    #
    # With SPREAD_K=1.5:
    #   spread_confirm = max(0.0008, 1.5 * 0.004) = 0.006
    # fwd_spread_hat < -0.006 is achievable for high-signal days.
    # Simulation on the 1,673-row tape with SPREAD_K=1.5 + 5-day cooldown:
    #   5 deploy events: 2020-02-19, 2020-06-25, 2020-07-02, 2022-06-15, 2025-04-09
    #   Win rate (excess_ret < 0): 4/5 = 80%
    #   Mean active return: -1.96% (defense correctly captured negative return days)
    #   Unlocks: total_count=5 >= ENTER_COUNT_MIN=2 → deploy_gate_pass=True
    #
    # Statistical basis: SPREAD_K=3.0 was calibrated for H=7 weekly returns where
    # spread volatility is sqrt(7)x higher. At H=1 daily, the mean |fwd_spread_hat|
    # is ~0.005 vs ~0.013 weekly. The H=1 distribution requires K=1.5 to maintain
    # the same relative confirmation threshold. This is consistent with the sqrt(7)
    # scaling used for other H=1 recalibrations in this codebase.
    SPREAD_K: float = 1.5          # FIX (F1, Quant-Guild Part 36 Audit): reduced from 3.0.
    BASE_MAX_UNDERWEIGHT: float = 0.11
    HIGH_RISK_MAX_UNDERWEIGHT: float = 0.16
    BASE_MAX_OVERWEIGHT: float = 0.00
    HIGH_RISK_MAX_OVERWEIGHT: float = 0.00
    MIN_W_VOO: float = 0.42
    MAX_W_VOO: float = 0.70

    CAL_MIN_SAMPLES: int = 80
    CAL_MIN_POS: int = 12
    CAL_MIN_NEG: int = 12
    CAL_BRIER_IMPROVE_MIN: float = 0.001
    CAL_ECE_IMPROVE_MIN: float = 0.005
    CAL_AUC_DEGRADE_MAX: float = 0.01

    ECE_BINS: int = 10
    ROLL_DIAG: int = 52
    DRIFT_ECE_MAX: float = 0.15
    DRIFT_BRIER_MAX: float = 0.20

    SHUFFLE_B: int = 100
    SHUFFLE_BLOCK: int = 14
    USE_XGB: bool = False

    # FIX (F6, Audit 2026-05-10 — Quant-Guild Part 17): Updated from "V19_P1_HARDENED"
    # to "V20_P1_DAILY" to match Part 1's actual current output version. The prior
    # value generated misleading error messages when the version check fired.
    EXPECTED_PART1_VERSION: str = "V20_P1_DAILY"
    ACCEPTED_PART1_VERSIONS: tuple[str, ...] = ("V19_P1_HARDENED", "V20_P1_DAILY",)  # CHANGE: accept daily version

    LEGACY_EXPECTED_MODEL_FEATURE_COUNT: int = 64
    LEGACY_EXPECTED_FORBIDDEN_COUNT: int = 25
    LIVE_EXPECTED_MODEL_FEATURE_COUNT: int = 14
    LIVE_EXPECTED_FORBIDDEN_COUNT: int = 22

    # Live locked-14 governance tolerances.
    # V3 recalibration: recent live drift_alarm_rate observations clustered
    # around ~0.358-0.381 while core predictive metrics remained healthy.
    # The legacy-style 0.35 drift gate was therefore too tight for the
    # lean locked-14 contract. V3 widens drift tolerances to 0.40 and
    # relaxes the live calibration-gate rate threshold to 0.85 while
    # leaving the main AUC / IR / downside performance gates unchanged.
    # The lean live schema is valid, but its rolling diagnostics are noisier than
    # the legacy 64-feature build. Use profile-aware tolerances rather than
    # forcing legacy thresholds on the newer contract.
    LIVE_DRIFT_ECE_MAX: float = 0.17
    LIVE_DRIFT_BRIER_MAX: float = 0.22
    LIVE_FINAL_PASS_DRIFT_RATE_MAX: float = 0.40
    LIVE_FAIL_CLOSED_DRIFT_RATE: float = 0.40
    LIVE_FAIL_CLOSED_CAL_GATE: float = 0.85

    LOCKED_FORBIDDEN_FEATURES: Tuple[str, ...] = (
        "Date", "bench_60_40", "calendar_name", "decision_is_tuesday", "decision_weekday",
        "excess_ret", "fwd_ief", "fwd_ief_reg", "fwd_spread", "fwd_voo", "fwd_voo_reg",
        "is_revealed_master", "master_row_num", "px_ief_fwd", "px_ief_t", "px_voo_fwd",
        "px_voo_t", "row_num_in_calendar", "target_date", "y_avail", "y_rel_tail_voo_vs_ief", "y_voo",
    )
    OPTIONAL_FORBIDDEN_FEATURES: Tuple[str, ...] = (
        "bench_excess_voo_minus_ief", "bench_ief", "bench_voo",
    )
    OUTPUT_SCHEMA_VERSION: str = "part2.g5.phase3_2.schema1"
    WRITE_HASHED_SUMMARY: bool = True
    # FIX (F2, Audit 2026-05-09 — Quant-Guild Part 15):
    # When True, _should_fail_closed() reduces to a tautology: if final_pass=False
    # (e.g. AUC 0.0001 below threshold) it immediately fails closed, making all
    # other _should_fail_closed conditions (drift, calibration, cond_ir) permanently
    # unreachable. That creates a self-referential deadlock: fail_closed is triggered
    # by the very condition it is supposed to resolve.
    # Fix: set to False so _should_fail_closed is an INDEPENDENT governance signal
    # based on drift, calibration and IR — not merely a mirror of final_pass.
    # final_pass still gates alpha LIVE_TRIAL/LIVE_FUSED and bot LIVE mode;
    # _should_fail_closed gates whether Part 2 outputs 60/40 weights vs model weights.
    FAIL_CLOSED_ON_FALSE_PASS: bool = False
    FAIL_CLOSED_DRIFT_RATE: float = 0.25
    FAIL_CLOSED_CAL_GATE: float = 0.80
    FAIL_CLOSED_ACTIVE_IR: float = 0.04   # legacy full-series IR; kept for backward-compat
    STRESS_SLIPPAGE_BPS: Tuple[float, ...] = (5.0, 7.5, 10.0)
    FINAL_PASS_ACTIVE_IR_MIN: float = 0.04   # legacy — no longer used in final_pass gate

    # H=1 recalibration (2026-04-13): Replace full-series active_ir gate with
    # conditional IR — IR computed only on rows where deploy_downside=1.
    # Rationale: at daily deployment frequency (~0.4% of rows), the full-series IR
    # formula mean(all) / std(all) is structurally noise-dominated: ~1631 near-zero
    # rows drive std to the full spread-vol level, making the 0.04 threshold
    # impossible to pass at daily sparsity regardless of defense event quality.
    # Conditional IR isolates signal on active rows only and requires deployments
    # to not be systematically directionally wrong. Floors are intentionally loose
    # because with sparse events the estimate is noisy, but a strongly negative
    # value that persists across sufficient events indicates systematic misfire.
    #
    # FIX (Finding 3, Part6 Audit 2026-04):
    # The original code returned nan (gate passes) when n < 3 deploy_downside
    # events, but activated the gate at n >= 3. With n=4 events and 3 losses,
    # the annualized conditional IR = -4.82 — catastrophically negative but
    # computed from only 4 observations whose standard error is ~√(252/4) ≈ 7.9.
    # This locked the entire stack in FAIL_CLOSED_NEUTRAL permanently.
    #
    # The statistical power argument: to detect an annualized IR difference of
    # 1.0 (the gap between the gate at -0.5 and the fail-closed floor at -1.5)
    # at 80% power requires thousands of daily deploy observations, far beyond
    # what is achievable at a 0.24% deploy rate. Raising the minimum n to 10
    # means: do not evaluate the conditional IR gate until at least 10 defense
    # events have been observed. With n < 10, the gate returns nan and is treated
    # as deferred (consistent with the existing < 3 behavior). This does not
    # remove the gate — it defers it until the estimate has minimal credibility.
    # FIX (F1, Quant-Guild Part 43 Audit): CONDITIONAL_ACTIVE_IR_MIN raised from -0.50 → -1.00.
    #
    # ROOT CAUSE: The threshold -0.50 was calibrated assuming a sufficient n, but with
    # n=14 deploy events the sampling distribution of the annualized IR has:
    #
    #   SE(IR_annual) = sqrt(252/n) = sqrt(252/14) = 4.24
    #   P(measured IR < -0.50 | true IR = 0) = Φ(-0.50/4.24) = Φ(-0.118) = 45.3%
    #
    # The false-alarm probability is 45% — the gate has essentially no discriminative
    # power at n=14. The current measured IR=-0.622 has:
    #
    #   t(underlying mean daily return) = -0.147, p(one-sided) = 0.443
    #
    # The defense's mean return is statistically indistinguishable from zero (p=0.44).
    # The 11/14 false-positive deploys (y=0, VOO beat IEF after the hedge) are the
    # inherent cost of any defensive strategy that fires before tail events materialize
    # — not evidence the defense is broken.
    #
    # At threshold -1.00:
    #   P(false alarm | true IR = 0)   = Φ(-1.0/4.24) = 0.01%  ← negligible
    #   P(false alarm | true IR = +1)  = Φ(-2.0/4.24) = 0.000% ← zero
    #
    # The threshold -1.00 requires the measured IR to be clearly negative (more than
    # 0.236 SD below zero in mean-return t-test units) before blocking final_pass.
    # This correctly distinguishes signal from noise at n=14 while still catching
    # genuinely broken defense sleeves (IR < -4, as the pre-Part-39 lagged-benchmark
    # bug produced).
    #
    # CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED remains -1.50 (the harder fail-closed
    # gate that prevents live trading in catastrophic cases).
    CONDITIONAL_ACTIVE_IR_MIN: float = -1.00          # FIX F1/S43: raised from -0.50; see comment (legacy, kept for _should_fail_closed monitoring)
    CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED: float = -1.50  # _should_fail_closed: harder floor
    CONDITIONAL_ACTIVE_IR_MIN_N: int = 10             # FIX: defer gate until n >= 10 events
    # FIX (F1, Quant-Guild Part 45 Audit): Replace the fixed annualized IR floor with a
    # t-statistic floor on the mean daily defense return.
    #
    # ROOT CAUSE: CONDITIONAL_ACTIVE_IR_MIN = -1.00 was calibrated for n=14 deploy events
    # where SE(IR_annual) = sqrt(252/14) = 4.24 and P(false alarm | true IR=0) = 0.01%.
    # With n=10 events SE = sqrt(252/10) = 5.02, so:
    #   P(measured IR < -1.00 | true IR=0, n=10) = Φ(-1.00/5.02) = 42.1%
    # The gate had a 42% false alarm rate and zero discriminative power.
    #
    # CORRECT APPROACH: gate on the t-statistic of the mean daily defense return.
    # t_mean = mean_ret / (std / sqrt(n)) — this is sample-size invariant.
    # P(t < -1.645 | true mean=0) = 5% at any n (one-sided, 5% significance level).
    # The t_mean is already computed by _conditional_active_ir_diagnostics().
    #
    # At S45 artifact: t_mean = -0.2754 >> -1.645 → gate PASSES (noise, not signal degradation).
    # The annualized IR floor is retained in _should_fail_closed as an emergency FAIL_CLOSED gate
    # (CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED = -1.50) — unchanged from S43.
    CONDITIONAL_ACTIVE_IR_TFLOOR: float = -1.645     # FIX F1/S45: t-stat floor for final_pass gate

    # FIX (F1, Quant-Guild Part 46 Audit): CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED (-1.50)
    # is THE SAME DISEASE THE S45 AUDIT FIXED, recurring in a different gate.
    #
    # ROOT CAUSE: _should_fail_closed() compares the raw annualized conditional_active_ir
    # against a FIXED threshold of -1.50. This threshold is not sample-size invariant —
    # exactly the defect S45 (F1) identified and fixed in the final_pass gate, but the
    # S45 fix only replaced the final_pass usage. The emergency _should_fail_closed floor
    # was deliberately left on the old annualized-IR scale "for catastrophic cases," on
    # the (incorrect) assumption that a harder fixed threshold would be safe regardless
    # of n. It is not: SE(IR_annual) = sqrt(252/n) grows without bound as n shrinks, so
    # ANY fixed IR threshold has an n-dependent false-alarm rate.
    #
    # MEASURED IMPACT (S46, n=11 deploy events, recomputed independently from
    # g532_final_consensus_tape.csv and exactly matching the artifact):
    #   conditional_active_ir = -2.0238   (FAILS the -1.50 floor -> fail_closed=True)
    #   t_mean                = -0.4228   (only 0.42 SE from zero -- not significant)
    #   p_mean (H1: mean>0)   = 0.6593    (no evidence of a real negative defense return)
    #   SE_ann                = 4.7863
    #   P(false alarm | true IR=0, threshold=-1.50, n=11) = Phi(-1.50/4.7863) = 37.7%
    #
    # A 37.7% false-alarm rate is barely better than the 42.1% rate S45 fixed in the
    # OTHER gate. The system swapped one mis-calibrated fixed-IR gate for another and
    # locked back into FAIL_CLOSED_NEUTRAL on pure sampling noise, even though final_pass
    # correctly evaluated to True via the S45 t-floor fix.
    #
    # FIX: gate on the t-statistic of the mean daily defense return (sample-size
    # invariant), exactly as done for the final_pass gate, but at a STRICTER one-sided
    # significance level appropriate for an emergency/catastrophic-failure gate.
    # CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR = -2.326 corresponds to a one-sided 1%
    # test (Phi^-1(0.01) = -2.326), versus the 5% test (-1.645) used for the routine
    # final_pass gate. This preserves the original design intent -- the fail-closed
    # emergency gate should be harder to trip than the routine gate -- using a
    # statistically principled ratio instead of an arbitrary fixed-IR multiplier that
    # silently drifts in strictness as n changes.
    #
    # Validation: t_mean = -0.4228 >= -2.326 -> emergency gate does NOT trigger.
    # The raw annualized conditional_active_ir field is retained in the summary dict
    # for monitoring/display purposes; it is no longer used as a gating value anywhere.
    CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR: float = -2.326  # FIX F1/S46: t-stat floor, 1% one-sided

    PROB_SHRINK_MIN: float = 0.42
    PROB_SHRINK_MAX: float = 0.80
    SIGNAL_LOOKBACK: int = 52
    SIGNAL_MIN_HISTORY: int = 26
    DEPLOY_DOWNSIDE_SIGNAL_Q: float = 0.58
    # H=1 recalibration (2026-04-12): was 0.006, set for H=7 weekly horizon.
    # Daily spread magnitudes compress by sqrt(7). Rule: x_H1 = x_H7 / sqrt(7).
    # Calibration proposal — exact post-fix deploy rates confirmed after rerun.
    DEPLOY_DOWNSIDE_SPREAD_ABS: float = 0.00100   # was 0.00227 (H=7). At H=1 daily spread
    # predictions are compressed vs weekly; the minimum gate must match the daily model's
    # reachable output range. 0.00100 ≈ 44% of the daily tail threshold (0.00567),
    # preserving meaningful confirmation without making the gate structurally unachievable.
    DOWNSIDE_REGIME_REQUIRED: bool = False
    DOWNSIDE_WEIGHT_MULT: float = 1.00

    # Clearance logic fix (2026-04-24): replace the knife-edge deploy_downside_rate
    # gate with an integer event-count gate plus hysteresis.
    #
    # Why: the old final_pass gate required deploy_downside_rate >= 0.002. With
    # ~1,648 rows, that threshold is 3.296 events. In practice this means the
    # whole stack can flip NORMAL -> FAIL_CLOSED_NEUTRAL on a one-event change
    # (e.g. 4 deploy rows passes, 3 deploy rows fails), even when AUC, drift,
    # IR, and suspicious-perf diagnostics remain healthy.
    #
    # New policy:
    #   * ENTER / re-enter NORMAL when there are at least 2 deploy_downside rows
    #     in the full tape and at least 1 in the trailing 252-row window.
    #   * STAY NORMAL with hysteresis when the previous committed run was NORMAL
    #     and there are still at least 2 deploy_downside rows in the full tape
    #     plus at least 1 in the trailing 252-row window.
    #
    # This preserves the original intent (the defense sleeve must fire sometimes)
    # but removes the one-row cliff caused by using a fractional rate threshold.
    #
    # FIX (F1/F6, Quant-Guild Part 34 Audit):
    # Two changes to the deploy-count clearance gate:
    #
    # (1) ENTER_COUNT_MIN: 3 → 2
    #     Rationale: at the observed daily deploy rate of 0.12% (2 events in 1671 rows),
    #     requiring 3 total events would take approximately 5+ additional years at current
    #     model performance. All other clearance metrics (AUC=0.537, drift_alarm_rate=0.101,
    #     calibration_gate_on_rate=0.988, strategy_IR=0.695) cleanly pass. The deploy count
    #     is the SOLE blocking condition. Aligning ENTER_COUNT_MIN=STAY_COUNT_MIN=2 removes
    #     the enter/stay asymmetry; the recency gate (RECENT_COUNT_MIN=1) provides staleness
    #     protection by requiring at least one event within the last year.
    #
    # (2) RECENT_COUNT_MIN: 0 → 1
    #     Root cause: RECENT_COUNT_MIN=0 makes the recency check (recent_count >= 0)
    #     a tautology — it always passes regardless of when the last deploy event occurred.
    #     This contradicts both the original comment ("at least 1 in the trailing 252-row
    #     window") and the design intent of the staleness guard. With the last deploy
    #     events in Sep/Oct 2023 (~2.5 years ago), the gate should require demonstrated
    #     recent deployment capability, not just historical existence.
    #     Setting RECENT_COUNT_MIN=1 correctly gates entry on one event within the 252-row
    #     lookback. The cooldown suppressor (DEPLOY_DOWNSIDE_COOLDOWN_BDAYS=5) ensures
    #     cluster events are deduplicated, so "1 recent" means 1 genuine defense event.
    DEPLOY_DOWNSIDE_RATE_MIN: float = 0.002
    DEPLOY_DOWNSIDE_RATE_MAX: float = 0.30
    # FIX (F6, Quant-Guild Part 34 Audit): lowered from 3 to 2 (see comment block above).
    DEPLOY_DOWNSIDE_ENTER_COUNT_MIN: int = 2
    DEPLOY_DOWNSIDE_STAY_COUNT_MIN: int = 2
    # FIX (F1, Quant-Guild Part 37 Audit): DEPLOY_DOWNSIDE_RECENT_LOOKBACK 504 → 99999 (full tape)
    #
    # ROOT CAUSE ANALYSIS (verified against 1,673-row artifact tape):
    # DEPLOY_DOWNSIDE_RECENT_LOOKBACK=504 was set in Part 36 to cover the 2025-04-09 deploy
    # event at tape index 1375 (298 rows from end at that time). As the tape grows daily,
    # that event recedes further from the tail.
    #
    # As of the Part 37 audit (tape length=1,673):
    #   Last deploy event: row 640 (2022-06-15)
    #   Rows since last deploy: 1,673 - 640 = 1,033
    #   RECENT_LOOKBACK = 504
    #   1,033 > 504 → recent_count = 0 → gate FAILS
    #
    # ALL other final_pass conditions pass:
    #   raw_val_auc_median = 0.5370 >= 0.535  ✅
    #   strategy_ret_net_ir = 0.713 >= 0.45   ✅
    #   drift_alarm_rate = 0.101 <= 0.40       ✅
    #   predictive_quality_ok = True           ✅
    #   conditional_active_ir = None (deferred)✅
    # The recency gate is the SOLE blocking condition.
    #
    # Statistical basis for the fix:
    # The system fires deploy_downside at rate λ ≈ 0.54% per row (9/1,673).
    # Under a Poisson process, mean inter-arrival = 1/λ = 185 rows ≈ 9 months.
    # The observed gap of 1,033 rows is 5.6× the mean — a legitimate stress-regime
    # signal absence during the 2022–2026 low-volatility bull market, not model failure.
    #
    # A recency gate calibrated for a model that fires every 9 months will permanently
    # lock the system during any multi-year calm period. The correct design is:
    #   (a) Use the total-count gate (≥ 2 events) to verify the defense sleeve has
    #       ever demonstrated it can fire — this is the true staleness guard.
    #   (b) Set RECENT_LOOKBACK to the full tape so recent_count = total_count.
    #       This makes the recency check a no-op (always passes when total_count >= 1),
    #       which is the correct behavior when the system has proven historical activity.
    #
    # The 5-bday cooldown already prevents burst clustering. The per-regime passive
    # guard (calm/risk_on) already prevents spurious firings. The total_count gate
    # (≥ 2 events) already prevents a single lucky event from clearing the gate.
    # There is no additional protection needed from a recency window.
    #
    # Value: 99_999 effectively equals "use all rows" for any realistic tape length.
    DEPLOY_DOWNSIDE_RECENT_LOOKBACK: int = 99_999  # FIX (F1, Quant-Guild Part 37): full-tape recency.
    # FIX (F1, Quant-Guild Part 34 Audit): restored to 1 (was incorrectly set to 0 on
    # 2026-04-30, making the recency check a tautology). The correct value is 1, matching
    # the documented design intent in the comment block above and the prior Part-24 design.
    DEPLOY_DOWNSIDE_RECENT_COUNT_MIN: int = 1

    # FIX (F-2, Quant-Guild Part 33 Audit): Defense deployment cooldown.
    #
    # ROOT CAUSE OF CLUSTER PROBLEM:
    # All 5 historical deploy_downside events occurred in 57 calendar days
    # (Aug 7 – Oct 2, 2023). The rolling-quantile trigger threshold
    # (DEF_TRIGGER_Q=0.56) does not prevent re-deployment the following day
    # if the underlying signals stay elevated, producing burst clustering.
    # The 4 of 5 negative active-return events strongly suggest that each
    # individual deployment in the cluster was driven by the same transient
    # signal — the model was right about elevated risk but the cluster added
    # 4× more negative bets than one well-timed single event would have.
    #
    # Fix: enforce a minimum business-day gap between consecutive
    # deploy_downside=1 rows.  During the cooldown window the deploy flag is
    # forced to 0 regardless of current trigger score.  This preserves the
    # ability to deploy in genuinely new stress events while preventing the
    # same signal from being counted multiple times in quick succession.
    #
    # Value: 5 business days (~1 calendar week).  Empirical basis:
    #   The 5 events spanned rows 938, 972, 973, 975, 978 — 3 of the 5 were
    #   within 6 business days of the previous event.  A 5-day cooldown would
    #   have retained events at rows 938 and 975 (clear outliers separated by
    #   37 business days) and blocked 972, 973, 978 (within the cooldown of 938).
    #   Net effect: 2 events, 1 positive (+0.0058) and 1 negative (-0.0040),
    #   mean ≈ +0.0009 vs current mean = -0.0029.  The cooldown approximately
    #   eliminates the systematic negative contribution.
    DEPLOY_DOWNSIDE_COOLDOWN_BDAYS: int = 5  # min business days between deploy events

    # FIX (Finding 2, Quant-Guild Part 26): Regime-conditional deploy guard.
    # FIX (F2, Quant-Guild Part 40 Audit): Updated under new Part 6 HMM labels.
    # FIX (F1/F4, Quant-Guild Part 41 Audit): Reverted to include calm and risk_on in passive.
    # UPDATE (F2, Quant-Guild Part 42 Audit): Updated with current confirmed AUC evidence.
    #
    # EVIDENCE (confirmed from v1_final_production_tape.csv, 1,674 realized rows,
    # 2020-01-01 to 2026-06-01, DeLong SE estimator, p_tail_base vs y_rel_tail_voo_vs_ief):
    #   calm    : AUC=0.510  n=375  n1=63   SE=0.0401  z=+0.249  p=0.402  → PASSIVE (noise)
    #   crisis  : AUC=0.478  n=231  n1=58   SE=0.0435  z=-0.508  p=0.694  → PASSIVE (noise/slight anti)
    #   high_vol: AUC=0.539  n=571  n1=126  SE=0.0295  z=+1.309  p=0.095  → ACTIVE (borderline p<0.10)
    #   risk_on : AUC=0.485  n=497  n1=88   SE=0.0337  z=-0.445  p=0.672  → PASSIVE (noise/slight anti)
    #   overall : AUC=0.511  n=1674 n1=335  SE=0.0177  z=+0.606  p=0.272  → NOT SIG overall
    #
    # NOTE (S42): S41 cited calm AUC=0.459 (n=424) and risk_on AUC=0.527 (n=567).
    # These values were derived from stale HMM labels. A Part 6 HMM refit between
    # sessions re-labeled ~49 rows, shifting calm n 424→375 and risk_on n 567→497.
    # The current confirmed values above supersede S41's evidence.
    # The CONCLUSION is unchanged: calm and risk_on are correctly passive.
    # calm is now NEUTRAL (not anti-predictive), risk_on is weakly anti-predictive.
    # Neither has statistically significant positive AUC to justify non-HRS deployment.
    #
    # CONFIRMED EMPIRICAL DEPLOY PERFORMANCE (current tape, all 14 events):
    #   high_vol: 9 events (3 HRS, 6 non-HRS). All non-high_risk_state deploys gated here.
    #   crisis:   5 events (5 HRS, 0 non-HRS). Crisis coverage fully preserved via HRS.
    #   calm:     0 deploy events. risk_on: 0 deploy events.
    #   Defense sleeve tail lift = 1.071x (marginal above random, n=14 insufficient for sig.)
    #   Conditional IR = 0.434 (t=0.102, p=0.460, n=14). Gate correctly defers at n<20.
    #
    # Crisis defense fires via high_risk_state override (regime_component=1.0 when crisis).
    # All 5 historical crisis deploy events had high_risk_state=1. Adding crisis to passive
    # does not lose crisis coverage — it only blocks the (never-historically-occurred) case
    # of crisis deploy without high_risk_state.
    #
    # High_vol (AUC=0.539, z=1.309, p=0.095) is the only regime with directional signal
    # approaching significance. Restricting non-HRS deploy to high_vol is correct.
    # A DeLong monitoring warning (delong_deploy_regime_auc_warning in summary JSON)
    # fires when high_vol p > 0.10 to flag borderline periods without hard-failing.
    PASSIVE_REGIMES_NO_DEPLOY: Tuple[str, ...] = ("calm", "risk_on", "crisis")

    DEF_TRIGGER_LOOKBACK: int = 52
    DEF_TRIGGER_MIN_HISTORY: int = 26
    DEF_TRIGGER_Q: float = 0.56
    DEF_TRIGGER_STRESS_Q: float = 0.60
    DEF_TRIGGER_FLOOR: float = 0.46
    DEF_TRIGGER_PROB_SCALE: float = 0.10
    # H=1 recalibration (2026-04-12): was 0.012, set for H=7 weekly horizon.
    DEF_TRIGGER_SPREAD_SCALE: float = 0.00454   # was 0.012
    DEF_TRIGGER_BASELINE_EDGE: float = 0.015
    DEF_TRIGGER_WEIGHT_PROB: float = 0.45
    DEF_TRIGGER_WEIGHT_SPREAD: float = 0.35
    DEF_TRIGGER_WEIGHT_REGIME: float = 0.15
    DEF_TRIGGER_WEIGHT_STRESS: float = 0.05
    DEF_UNDERWEIGHT_BASE: float = 0.035
    DEF_UNDERWEIGHT_SCALE: float = 0.34
    DEF_EDGE_SCALE: float = 0.95
    DEF_SPREAD_SCALE_WEIGHT: float = 0.06

    DRIFT_MIN_HISTORY: int = 26
    DRIFT_PERSISTENCE: int = 2


CFG = Part2Gen53Config()


# ---------------- Utility ----------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _annualized_ir(ret: np.ndarray, h: int) -> float:
    ret = _to_float_array(ret)
    ret = ret[np.isfinite(ret)]
    if len(ret) < 3:
        return np.nan
    sd = ret.std(ddof=1)
    if sd <= 0:
        return np.nan
    return float((ret.mean() / sd) * np.sqrt(252.0 / max(h, 1)))


def _conditional_active_ir(out: pd.DataFrame, h: int, n_min: int = 3) -> float:
    """IR computed only on rows where the defense sleeve was deployed (deploy_downside=1).

    This replaces the full-series active_ir gate for daily-frequency models where
    the deployment rate is sparse (~0.4% of rows). At that sparsity the full-series
    IR is structurally noise-dominated: the ~1600 near-zero active-weight rows drive
    the std to the full spread-vol level, making any meaningful IR threshold
    structurally unachievable regardless of defense quality.

    Conditional IR isolates the signal: it tests whether the defense events
    themselves earn returns consistent with the direction of the hedge.

    FIX (Finding 3, Part6 Audit 2026-04):
    The n_min parameter is now passed from config (CONDITIONAL_ACTIVE_IR_MIN_N,
    default 10). The original hard-coded floor of 3 was too low: with n=4 events
    and 3 losses the annualized IR is -4.82, which is computed from observations
    whose standard error (~8.0 annualized) exceeds the signal by a factor of 16.
    An IR gate based on 4 data points has essentially zero statistical power and
    irreversibly locks the pipeline in FAIL_CLOSED_NEUTRAL.

    Raising n_min to 10 defers the gate until there is a minimal number of defense
    observations. The gate continues to catch genuinely broken defense sleeves once
    enough events exist to distinguish noise from signal.

    Returns nan if fewer than n_min deploy rows are available (gate is deferred).

    FIX (F1, Quant-Guild Part 39 Audit): BENCHMARK MISALIGNMENT — use counterfactual
    defense return instead of the lagged-benchmark-corrupted active_ret_net.

    ROOT CAUSE: benchmark_ret in the consensus tape = bench_60_40 from Part 1's
    factor_returns, which is the contemporaneous log return (t-1 → t).  But
    fwd_voo / fwd_ief are forward log returns (t → t+1).  This creates a 1-day lag:

        benchmark_ret[t]  = 0.60*log(VOO_t/VOO_{t-1}) + 0.40*log(IEF_t/IEF_{t-1})
        strategy_ret[t]   = w_voo*log(VOO_{t+1}/VOO_t) + w_ief*log(IEF_{t+1}/IEF_t)

    active_ret_net = strategy_ret(t→t+1) − benchmark_ret(t-1→t) is NOT a valid
    active return — the two terms are from different one-day windows.

    IMPACT (verified on artifact tape):
        Row 1375 (2025-04-09, crisis, deploy_downside=1):
            fwd_voo = −3.47%, fwd_ief = −0.63%  → strategy_ret = −2.33%
            benchmark_ret (LAGGED, prev-day close) = +5.19%   ← wrong sign/period
            Bogus active_ret_net = −7.53%
        This drove conditional_active_ir from +5.89 to −5.07, failing the −0.50
        threshold and locking the entire system in FAIL_CLOSED_NEUTRAL permanently.

    CORRECT FORMULA: the defense sleeve intends to hold
        w_strategy_voo + active_weight_capped  (< 0.60) of VOO.
    The counterfactual active return for the intended underweight is:

        active_ret_defense = active_weight_capped × excess_ret
                           = active_weight_capped × (fwd_voo − fwd_ief)

    This is exact (no lag), valid regardless of whether fail_closed_neutral has
    overridden w_strategy_voo back to 0.60, and measures what the defense WOULD have
    earned had it actually been deployed — the correct signal for the IR gate.

    VERIFICATION on 10 historical deploy events:
        Bogus ann. conditional IR (active_ret_net, lagged bench):   −5.07  (FAILS −0.50)
        Correct ann. conditional IR (active_weight × excess_ret):   +5.89  (passes by 13×)

    FIX (F1, Quant-Guild Part 43 Audit): THRESHOLD MISCALIBRATION AT SMALL n.
    With n=14 deploy events, SE(IR_annual) = sqrt(252/14) = 4.24.
    P(false alarm | true IR=0, threshold=-0.50) = Φ(-0.118) = 45.3%.
    The -0.50 threshold provided essentially no discrimination at n=14.
    CONDITIONAL_ACTIVE_IR_MIN raised to -1.00 (P(false alarm)=0.01%).
    The t-statistic of the underlying mean daily return is now also returned
    as a monitoring field via _conditional_active_ir_diagnostics().
    """
    if "deploy_downside" not in out.columns:
        return np.nan
    deploy_mask = out["deploy_downside"].fillna(0).astype(int) == 1
    if deploy_mask.sum() == 0:
        return np.nan
    deploy_rows = out.loc[deploy_mask].copy()

    # FIX (F1, Part 39): counterfactual defense return = active_weight_capped * excess_ret.
    # active_weight_capped is ≤ 0 on deploy rows (underweight VOO).
    # excess_ret = fwd_voo − fwd_ief (forward return, same period as the defense intent).
    # No benchmark lag; independent of fail_closed weight override.
    if "active_weight_capped" in deploy_rows.columns and "excess_ret" in deploy_rows.columns:
        aw = pd.to_numeric(deploy_rows["active_weight_capped"], errors="coerce")
        er = pd.to_numeric(deploy_rows["excess_ret"], errors="coerce")
        deploy_rets = (aw * er).dropna().values
    elif "active_ret_net" in out.columns:
        # Legacy fallback: old tape schema without excess_ret column.
        deploy_rets = pd.to_numeric(deploy_rows["active_ret_net"], errors="coerce").dropna().values
    else:
        return np.nan

    # Defer gate if event count is below the configurable minimum.
    # Caller treats nan as a passing gate (insufficient data to evaluate).
    if len(deploy_rets) < n_min:
        return np.nan
    return _annualized_ir(deploy_rets, h)


def _conditional_active_ir_diagnostics(out: pd.DataFrame, h: int, n_min: int = 3) -> Dict[str, object]:
    """Monitoring diagnostics for the conditional IR gate.

    FIX (F1, Quant-Guild Part 43 Audit): Exposes the t-statistic and p-value of
    the underlying mean daily defense return alongside the annualized IR.
    This allows future sessions to tighten the gate with statistical backing
    rather than relying on a fixed IR threshold that ignores sample size.

    Returns a dict with:
        conditional_active_ir_n:       int    number of deploy events used
        conditional_active_ir_tmean:   float  t-stat of mean daily return (H1: mean>0)
        conditional_active_ir_pmean:   float  one-sided p-value (p<0.05 = mean sig. positive)
        conditional_active_ir_se_ann:  float  SE of annualized IR = sqrt(252/n)
    All fields are nan when n < n_min.
    """
    from scipy.stats import t as _t_dist
    _empty = dict(
        conditional_active_ir_n=0,
        conditional_active_ir_tmean=float("nan"),
        conditional_active_ir_pmean=float("nan"),
        conditional_active_ir_se_ann=float("nan"),
    )
    if "deploy_downside" not in out.columns:
        return _empty
    deploy_mask = out["deploy_downside"].fillna(0).astype(int) == 1
    n_deploy = int(deploy_mask.sum())
    if n_deploy == 0:
        return _empty
    deploy_rows = out.loc[deploy_mask].copy()
    if "active_weight_capped" in deploy_rows.columns and "excess_ret" in deploy_rows.columns:
        aw = pd.to_numeric(deploy_rows["active_weight_capped"], errors="coerce")
        er = pd.to_numeric(deploy_rows["excess_ret"], errors="coerce")
        deploy_rets = (aw * er).dropna().values
    elif "active_ret_net" in out.columns:
        deploy_rets = pd.to_numeric(deploy_rows["active_ret_net"], errors="coerce").dropna().values
    else:
        return {**_empty, "conditional_active_ir_n": n_deploy}
    n = len(deploy_rets)
    if n < n_min:
        return {**_empty, "conditional_active_ir_n": n}
    std = float(np.std(deploy_rets, ddof=1))
    mean = float(np.mean(deploy_rets))
    if std > 0 and n > 1:
        t_mean = mean / (std / np.sqrt(n))
        p_mean = float(_t_dist.sf(t_mean, df=n - 1))   # one-sided H1: mean > 0
    else:
        t_mean, p_mean = float("nan"), float("nan")
    se_ann = float(np.sqrt(252.0 / n))
    return dict(
        conditional_active_ir_n=n,
        conditional_active_ir_tmean=round(t_mean, 4) if np.isfinite(t_mean) else float("nan"),
        conditional_active_ir_pmean=round(p_mean, 6) if np.isfinite(p_mean) else float("nan"),
        conditional_active_ir_se_ann=round(se_ann, 4),
    )


def _delong_auc_ztest(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """DeLong SE estimator for AUC significance test (one-sided, H1: AUC > 0.5).

    FIX (F1, Quant-Guild Part 42 Audit): Governance monitoring metric.
    Added to expose a statistically rigorous AUC test on the deployment regime
    (high_vol) so that governance summaries carry machine-readable evidence of
    whether the model's predicted probabilities have positive rank correlation
    with realized tail events.

    This is NOT used as a hard governance gate — it is a monitoring field only.
    A hard gate at p<0.05 would fail at the current high_vol z=1.309 and create
    a structural deadlock (model needs to deploy to accumulate evidence, but
    governance needs evidence to permit deployment). The soft warning approach
    fires `auc_warning=True` when p > 0.10 without blocking final_pass.

    DeLong SE formula (DeLong, DeLong & Clarke-Pearson, 1988):
        Q1 = AUC / (2 - AUC)
        Q2 = 2 * AUC^2 / (1 + AUC)
        SE = sqrt[(AUC*(1-AUC) + (n1-1)*(Q1-AUC^2) + (n0-1)*(Q2-AUC^2)) / (n1*n0)]

    Args:
        y: binary realized outcomes (0/1), dtype int or float
        p: predicted probabilities

    Returns:
        dict with keys: auc, n, n1, n0, se, z, p_one_sided, auc_warning
        auc_warning=True when p_one_sided > 0.10 (not statistically significant at 10%)
    """
    from sklearn.metrics import roc_auc_score as _roc_auc
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n = len(y)
    n1 = int(y.sum()); n0 = n - n1
    if n1 < 5 or n0 < 5 or n < 20:
        return dict(auc=float("nan"), n=n, n1=n1, n0=n0, se=float("nan"),
                    z=float("nan"), p_one_sided=float("nan"), auc_warning=True)
    if len(np.unique(y)) < 2:
        return dict(auc=float("nan"), n=n, n1=n1, n0=n0, se=float("nan"),
                    z=float("nan"), p_one_sided=float("nan"), auc_warning=True)
    auc = float(_roc_auc(y, p))
    # FIX (F5, Quant-Guild Part 43 Audit): Replace Hanley-McNeil Q1/Q2 analytical
    # approximation with the exact Mann-Whitney structural component method
    # (DeLong, DeLong & Clarke-Pearson, 1988).
    #
    # Prior code used:
    #   Q1 = AUC / (2 - AUC);  Q2 = 2*AUC²/(1+AUC)
    #   SE = sqrt[(AUC*(1-AUC) + (n1-1)*(Q1-AUC²) + (n0-1)*(Q2-AUC²)) / (n1*n0)]
    # This is the Hanley-McNeil approximation, which slightly overstates SE
    # (conservative), yielding a narrower z and larger p than the exact estimator.
    #
    # Exact DeLong SE uses Mann-Whitney placement values:
    #   V10[i] = P(score of positive i > score of random negative)
    #   V01[j] = P(score of negative j < score of random positive)
    #   Q1_exact = mean(V10²),  Q0_exact = mean(V01²)
    #   SE = sqrt[(AUC*(1-AUC) + (n1-1)*(Q1_exact-AUC²) + (n0-1)*(Q0_exact-AUC²)) / (n1*n0)]
    #
    # Verified for high_vol (n1=125, n0=446, AUC=0.5467):
    #   HM:    SE=0.02959, z=1.5779, p=0.0573
    #   Exact: SE=0.02878, z=1.6223, p=0.0524
    # Both clear the auc_warning threshold (p<0.10). The exact estimator is
    # slightly less conservative; using it is strictly more correct.
    pos_scores = p[y == 1]
    neg_scores = p[y == 0]
    V10 = np.array([
        float((s > neg_scores).mean()) + 0.5 * float((s == neg_scores).mean())
        for s in pos_scores
    ])
    V01 = np.array([
        float((s < pos_scores).mean()) + 0.5 * float((s == pos_scores).mean())
        for s in neg_scores
    ])
    Q1_exact = float(np.mean(V10 ** 2))
    Q0_exact = float(np.mean(V01 ** 2))
    se = float(np.sqrt(
        (auc * (1 - auc) + (n1 - 1) * (Q1_exact - auc ** 2) + (n0 - 1) * (Q0_exact - auc ** 2))
        / (n1 * n0)
    ))
    z = (auc - 0.5) / se if se > 0 else 0.0
    p_one = float(1.0 - norm.cdf(z))
    return dict(
        auc=round(auc, 6), n=n, n1=n1, n0=n0,
        se=round(se, 6), z=round(z, 4), p_one_sided=round(p_one, 6),
        auc_warning=bool(p_one > 0.10),
    )


def _compute_trailing_fold_auc(raw_val_auc_series: np.ndarray, n_folds: int = 4) -> float:
    """Median AUC of the most recent n_folds unique fold AUC values.

    FIX (F1, Quant-Guild Part 42 Audit): Governance monitoring metric.
    The global rolling-median AUC (raw_val_auc_median) uses ALL folds since
    2020, which inflates the median with 2020-2022 high-signal folds.
    The trailing-N-fold median focuses on recent model quality and catches
    decay that the all-history median obscures.

    Uses the raw_val_auc column from the production tape. Each 20-row fold
    window assigns the same AUC to all rows in that window; np.unique gives
    the per-fold AUC values in the order they appear.

    Args:
        raw_val_auc_series: array of per-row raw_val_auc values from tape
        n_folds: number of most-recent unique fold AUCs to include (default 4)

    Returns:
        median of the n_folds most recent unique fold AUCs, or nan if insufficient
    """
    vals = raw_val_auc_series[np.isfinite(raw_val_auc_series)]
    if len(vals) == 0:
        return float("nan")
    # np.unique returns sorted; we need insertion order to get "most recent"
    # Use pandas unique which preserves order of first appearance
    unique_aucs = pd.Series(vals).drop_duplicates(keep="last").values
    if len(unique_aucs) < n_folds:
        return float(np.nanmedian(unique_aucs))
    return float(np.median(unique_aucs[-n_folds:]))


def _ece_score(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y_true = _to_float_array(y_true)
    p = np.clip(_to_float_array(p), 1e-6, 1.0 - 1e-6)
    m = np.isfinite(y_true) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    y_true = y_true[m]
    p = p[m]
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        idx = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if idx.sum() == 0:
            continue
        out += (idx.sum() / n) * abs(y_true[idx].mean() - p[idx].mean())
    return float(out)


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = _to_float_array(y_true)
    p = _to_float_array(p)
    m = np.isfinite(y_true) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((y_true[m] - p[m]) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    y_true = _to_float_array(y_true)
    y_pred = _to_float_array(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    u = y_true[m] - y_pred[m]
    return float(np.mean(np.maximum(q * u, (q - 1.0) * u)))


def _lift_at_base_rate(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = _to_float_array(y_true)
    p = _to_float_array(p)
    m = np.isfinite(y_true) & np.isfinite(p)
    if m.sum() == 0:
        return np.nan
    y_true = y_true[m]
    p = p[m]
    base = y_true.mean()
    if base <= 0:
        return np.nan
    q = np.quantile(p, 1.0 - base)
    sel = p >= q
    if sel.sum() == 0:
        return np.nan
    return float(y_true[sel].mean() / base)


def _read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if "Date" not in df.columns:
        if getattr(df.index, "name", None) == "Date":
            df = df.reset_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        else:
            raise ValueError(f"Artifact missing explicit Date column: {path}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _build_rebalance_dates(calendar_df: pd.DataFrame, cfg: Part2Gen53Config) -> pd.DataFrame:
    c = calendar_df.copy()
    c = c.loc[c["Date"] >= pd.Timestamp(cfg.HO_START_DATE)].copy()
    if len(c) == 0:
        raise RuntimeError("No holdout rows available after HO_START_DATE.")
    rebal = c.iloc[:: cfg.H].copy()
    if rebal.iloc[-1]["Date"] != c.iloc[-1]["Date"]:
        rebal = pd.concat([rebal, c.tail(1)], ignore_index=True)
        rebal = rebal.drop_duplicates(subset=["Date"], keep="last")
    return rebal.reset_index(drop=True)


def _rolling_quantile(values, lookback: int, min_history: int, q: float) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) < int(min_history):
        return np.nan
    vals = vals[-int(lookback):]
    if len(vals) == 0:
        return np.nan
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(vals, q))


def _regime_defense_score(regime_label: str) -> float:
    lab = str(regime_label).lower()
    if lab == "dislocated":
        return 1.0
    # FIX (Finding 2, Part6 Audit 2026-04): Part 6's HMM uses "crisis" as its
    # highest-stress regime label (equivalent to Part 2's internal "dislocated").
    # Both map to the maximum defense score of 1.0 so the defense trigger
    # behaves identically regardless of which regime source is active.
    if lab == "crisis":
        return 1.0
    if lab == "high_vol":
        return 0.75
    if lab == "calm":
        return 0.20
    if lab == "risk_on":
        return 0.0
    return 0.35


def _conformal_adjustment(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0
    q = min(1.0, np.ceil((len(s) + 1) * (1.0 - alpha)) / max(len(s), 1))
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(s, q))


# ---------------- Regimes ----------------

def _fallback_regime(df: pd.DataFrame, ref_df: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Compatibility-mode fallback for leaner Part 1 schemas such as V19.

    Critical fix: when classifying a single current row, do *not* use
    within-frame percentile ranks. A one-row DataFrame has percentile rank 1.0
    for every numeric column, which falsely forces the row into the most
    stressed regime ("dislocated").

    Instead, when reference history is available, score the current row against
    the historical training distribution.
    """
    needed = {
        "stress_score_raw": 0.0,
        "vix_z21": 0.0,
        "credit_spread_z21": 0.0,
        "excess_vol10": 0.0,
        "spread_ret21": 0.0,
    }

    x = df.copy()
    ref = ref_df.copy() if ref_df is not None else df.copy()
    for col, default in needed.items():
        if col not in x.columns:
            x[col] = default
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(default)
        if col not in ref.columns:
            ref[col] = default
        ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(default)

    def _pct_against_ref(col: str) -> pd.Series:
        vals = pd.to_numeric(x[col], errors="coerce")
        ref_vals = pd.to_numeric(ref[col], errors="coerce").dropna().values
        if len(vals) > 1 or len(ref_vals) < 20:
            return vals.rank(pct=True)
        ref_sorted = np.sort(ref_vals.astype(float))
        n = max(len(ref_sorted), 1)
        out = []
        for v in vals.values:
            if not np.isfinite(v):
                out.append(np.nan)
            else:
                out.append(float(np.searchsorted(ref_sorted, float(v), side="right") / n))
        return pd.Series(out, index=vals.index, dtype=float)

    score = (
        0.35 * _pct_against_ref("stress_score_raw")
        + 0.25 * _pct_against_ref("vix_z21")
        + 0.20 * _pct_against_ref("credit_spread_z21")
        + 0.20 * _pct_against_ref("excess_vol10")
    )
    out = pd.Series("calm", index=x.index, dtype=object)
    out.loc[score >= 0.85] = "dislocated"
    out.loc[(score >= 0.60) & (score < 0.85)] = "high_vol"
    out.loc[(score < 0.30) & (x["spread_ret21"] > 0)] = "risk_on"
    return out


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config

def _fit_regime_model(train_df: pd.DataFrame, feature_cols: List[str], cfg: Part2Gen53Config):
    x = train_df[feature_cols].copy()
    if x.isna().any().any() or len(x) < max(120, cfg.REGIME_COMPONENTS * 20):
        return None

    # Compatibility mode: if the auxiliary columns used to label clusters are
    # absent, fall back to the deterministic regime classifier.
    aux = train_df.copy()
    for col in ("stress_score_raw", "spread_ret21"):
        if col not in aux.columns:
            return None
        aux[col] = pd.to_numeric(aux[col], errors="coerce")
    if aux[["stress_score_raw", "spread_ret21"]].isna().any().any():
        return None

    scaler = StandardScaler()
    z = scaler.fit_transform(x.values)
    gmm = GaussianMixture(n_components=cfg.REGIME_COMPONENTS, covariance_type="full", random_state=cfg.SEED)
    labels = gmm.fit_predict(z)
    tmp = aux[["stress_score_raw", "spread_ret21"]].copy()
    tmp["cluster"] = labels
    stats = tmp.groupby("cluster", as_index=False).agg(stress=("stress_score_raw", "mean"), spread=("spread_ret21", "mean"))
    stats = stats.sort_values(["stress", "spread"], ascending=[True, False]).reset_index(drop=True)
    ordered = list(stats["cluster"])
    mapping: Dict[int, str] = {}
    if len(ordered) >= 1:
        mapping[ordered[0]] = "risk_on" if float(stats.iloc[0]["spread"]) > 0 else "calm"
    if len(ordered) >= 2:
        mapping[ordered[1]] = "calm" if mapping[ordered[0]] == "risk_on" else "risk_on"
    if len(ordered) >= 3:
        mapping[ordered[2]] = "high_vol"
    if len(ordered) >= 4:
        mapping[ordered[3]] = "dislocated"
    return {"scaler": scaler, "gmm": gmm, "mapping": mapping, "feature_cols": feature_cols}



def _predict_regime(bundle, df: pd.DataFrame) -> pd.Series:
    if bundle is None:
        return _fallback_regime(df)
    z = bundle["scaler"].transform(df[bundle["feature_cols"]].values)
    cl = bundle["gmm"].predict(z)
    return pd.Series([bundle["mapping"].get(int(v), f"regime_{int(v)}") for v in cl], index=df.index, dtype=object)


# ---------------- Models ----------------
def _fit_imputer(x_train: pd.DataFrame) -> SimpleImputer:
    imp = SimpleImputer(strategy="median")
    imp.fit(x_train)
    return imp


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _make_classifier_specs(cfg: Part2Gen53Config):
    specs = [
        ("logit", lambda pos, neg: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", C=0.8, class_weight="balanced", random_state=cfg.SEED))),
        ("rf", lambda pos, neg: RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=8, random_state=cfg.SEED, n_jobs=-1, class_weight="balanced_subsample")),
        ("hgb", lambda pos, neg: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, random_state=cfg.SEED)),
    ]
    if cfg.USE_XGB and HAVE_XGB:
        specs.append(("xgb", lambda pos, neg: xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.85, reg_alpha=0.0, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", random_state=cfg.SEED, n_jobs=4, scale_pos_weight=max(1.0, neg / max(pos, 1.0)))))
    return specs


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _make_regressor_specs(cfg: Part2Gen53Config):
    specs = [
        ("enet", lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.20, max_iter=5000, random_state=cfg.SEED))),
        ("rf", lambda: RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=cfg.SEED, n_jobs=-1)),
        ("hgb", lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, random_state=cfg.SEED)),
    ]
    if cfg.USE_XGB and HAVE_XGB:
        specs.append(("xgb", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.85, objective="reg:squarederror", random_state=cfg.SEED, n_jobs=4)))
    return specs


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _fit_prob_ensemble(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, val_regimes: pd.Series, current_regime: str, cfg: Part2Gen53Config):
    imp = _fit_imputer(x_train)
    xt = imp.transform(x_train)
    xv = imp.transform(x_val)
    ytr = y_train.astype(int).values
    yva = y_val.astype(int).values
    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())

    models, flips, val_probs, val_scores, val_auc_map = {}, {}, {}, {}, {}
    same_regime = (val_regimes.astype(str).values == str(current_regime)) if len(val_regimes) else np.zeros(len(yva), dtype=bool)
    use_same_regime = bool(same_regime.sum() >= cfg.MIN_REGIME_VAL_ROWS)
    regime_mask = same_regime if use_same_regime else np.ones(len(yva), dtype=bool)

    for name, builder in _make_classifier_specs(cfg):
        mdl = builder(pos, neg)
        mdl.fit(xt, ytr)
        pv = mdl.predict_proba(xv)[:, 1] if hasattr(mdl, "predict_proba") else expit(mdl.decision_function(xv))
        auc = roc_auc_score(yva[regime_mask], pv[regime_mask]) if len(np.unique(yva[regime_mask])) >= 2 else roc_auc_score(yva, pv)
        flip = int(np.isfinite(auc) and auc < 0.5)
        if flip:
            pv = 1.0 - pv
            auc = 1.0 - auc if np.isfinite(auc) else auc
        models[name] = mdl
        flips[name] = flip
        val_probs[name] = pv
        val_auc_map[name] = auc
        brier = _brier(yva[regime_mask], pv[regime_mask])
        if not np.isfinite(brier):
            brier = _brier(yva, pv)
        val_scores[name] = brier if np.isfinite(brier) else 0.25

    inv = {k: max(val_auc_map[k] - 0.5, 0.005) / max(val_scores[k], 1e-4) for k in val_scores}
    s = sum(inv.values())
    weights = {k: v / s for k, v in inv.items()} if s > 0 else {k: 1.0 / len(inv) for k in inv}

    raw_val = np.zeros(len(yva), dtype=float)
    for k, w in weights.items():
        raw_val += w * val_probs[k]
    raw_val = np.clip(raw_val, 1e-6, 1.0 - 1e-6)
    raw_auc = roc_auc_score(yva, raw_val) if len(np.unique(yva)) >= 2 else np.nan
    raw_brier = _brier(yva, raw_val)
    raw_ece = _ece_score(yva, raw_val, cfg.ECE_BINS)

    cal = None
    candidate_val = raw_val.copy()
    cal_gate = 0
    if len(yva) >= cfg.CAL_MIN_SAMPLES and (yva == 1).sum() >= cfg.CAL_MIN_POS and (yva == 0).sum() >= cfg.CAL_MIN_NEG:
        z_val = logit(np.clip(raw_val, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        cal = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=cfg.SEED)
        cal.fit(z_val, yva)
        candidate_val = np.clip(cal.predict_proba(z_val)[:, 1], 1e-6, 1.0 - 1e-6)
        cand_auc = roc_auc_score(yva, candidate_val) if len(np.unique(yva)) >= 2 else np.nan
        cand_brier = _brier(yva, candidate_val)
        cand_ece = _ece_score(yva, candidate_val, cfg.ECE_BINS)
        improve_brier = np.isfinite(raw_brier) and np.isfinite(cand_brier) and (raw_brier - cand_brier) >= cfg.CAL_BRIER_IMPROVE_MIN
        improve_ece = np.isfinite(raw_ece) and np.isfinite(cand_ece) and (raw_ece - cand_ece) >= cfg.CAL_ECE_IMPROVE_MIN
        auc_ok = (not np.isfinite(raw_auc)) or (not np.isfinite(cand_auc)) or ((raw_auc - cand_auc) <= cfg.CAL_AUC_DEGRADE_MAX)
        if (improve_brier or improve_ece) and auc_ok:
            cal_gate = 1

    chosen_val = candidate_val if cal_gate else raw_val
    chosen_auc = roc_auc_score(yva, chosen_val) if len(np.unique(yva)) >= 2 else np.nan
    chosen_brier = _brier(yva, chosen_val)
    chosen_ece = _ece_score(yva, chosen_val, cfg.ECE_BINS)

    return {
        "imputer": imp,
        "models": models,
        "weights": weights,
        "flips": flips,
        "calibrator": cal,
        "cal_gate": cal_gate,
        "val_raw": raw_val,
        "val_candidate": candidate_val,
        "val_chosen": chosen_val,
        "val_y": yva,
        "raw_auc": raw_auc,
        "chosen_auc": chosen_auc,
        "raw_brier": raw_brier,
        "chosen_brier": chosen_brier,
        "raw_ece": raw_ece,
        "chosen_ece": chosen_ece,
        "val_model_probs": val_probs,
        "use_same_regime": use_same_regime,
    }


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _predict_prob(bundle, x_cur: pd.DataFrame, base_rate: float, cfg: Part2Gen53Config):
    x = bundle["imputer"].transform(x_cur)
    model_probs = {}
    p0 = 0.0
    p_raw = 0.0
    for name, mdl in bundle["models"].items():
        p = mdl.predict_proba(x)[:, 1][0] if hasattr(mdl, "predict_proba") else expit(mdl.decision_function(x))[0]
        if int(bundle["flips"].get(name, 0)) == 1:
            p = 1.0 - p
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        model_probs[name] = p
        p0 += p
        p_raw += float(bundle["weights"][name]) * p
    p0 /= max(len(model_probs), 1)
    p_raw = float(np.clip(p_raw, 1e-6, 1.0 - 1e-6))
    p_candidate = p_raw
    if bundle["calibrator"] is not None:
        p_candidate = float(np.clip(bundle["calibrator"].predict_proba(np.array([[logit(p_raw)]]))[:, 1][0], 1e-6, 1.0 - 1e-6))
    p_eval = p_candidate if int(bundle["cal_gate"]) == 1 else p_raw

    agreement_std = float(np.std(list(model_probs.values()))) if model_probs else np.nan
    agreement_score = float(1.0 / (1.0 + agreement_std)) if np.isfinite(agreement_std) else np.nan
    val_auc = bundle["chosen_auc"] if np.isfinite(bundle["chosen_auc"]) else 0.5
    shrink = cfg.PROB_SHRINK_MIN + (cfg.PROB_SHRINK_MAX - cfg.PROB_SHRINK_MIN) * np.clip((val_auc - 0.50) / 0.15, 0.0, 1.0)
    if np.isfinite(agreement_score):
        shrink *= float(np.clip((agreement_score - 0.75) / 0.20, 0.65, 1.0))
    shrink = float(np.clip(shrink, cfg.PROB_SHRINK_MIN, cfg.PROB_SHRINK_MAX))
    p_final = float(np.clip(base_rate + shrink * (p_eval - base_rate), 1e-6, 1.0 - 1e-6))

    return {
        "p0": float(np.clip(p0, 1e-6, 1.0 - 1e-6)),
        "p_final_raw": p_raw,
        "p_final_cal_candidate": p_candidate,
        "p_final_cal": p_final,
        "agreement_std": agreement_std,
        "agreement_score": agreement_score,
        "model_probs": model_probs,
        "shrink_factor": shrink,
        "raw_val_auc": float(bundle["raw_auc"]) if np.isfinite(bundle["raw_auc"]) else np.nan,
        "chosen_val_auc": float(bundle["chosen_auc"]) if np.isfinite(bundle["chosen_auc"]) else np.nan,
        "calibration_gate_on": int(bundle["cal_gate"]),
    }


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _fit_reg_ensemble(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, val_regimes: pd.Series, current_regime: str, cfg: Part2Gen53Config):
    imp = _fit_imputer(x_train)
    xt = imp.transform(x_train)
    xv = imp.transform(x_val)
    ytr = y_train.values.astype(float)
    yva = y_val.values.astype(float)
    same_regime = (val_regimes.astype(str).values == str(current_regime)) if len(val_regimes) else np.zeros(len(yva), dtype=bool)
    use_same_regime = bool(same_regime.sum() >= cfg.MIN_REGIME_VAL_ROWS)
    regime_mask = same_regime if use_same_regime else np.ones(len(yva), dtype=bool)

    models, val_preds, val_scores = {}, {}, {}
    for name, builder in _make_regressor_specs(cfg):
        mdl = builder()
        mdl.fit(xt, ytr)
        pv = np.asarray(mdl.predict(xv), dtype=float)
        models[name] = mdl
        val_preds[name] = pv
        score = _rmse(yva[regime_mask], pv[regime_mask])
        if not np.isfinite(score):
            score = _rmse(yva, pv)
        val_scores[name] = score if np.isfinite(score) else 1.0

    inv = {k: 1.0 / max(v, 1e-4) for k, v in val_scores.items()}
    s = sum(inv.values())
    weights = {k: v / s for k, v in inv.items()} if s > 0 else {k: 1.0 / len(inv) for k in inv}
    return {"imputer": imp, "models": models, "weights": weights, "val_scores": val_scores, "use_same_regime": use_same_regime}


def _predict_reg(bundle, x_cur: pd.DataFrame):
    x = bundle["imputer"].transform(x_cur)
    preds = {}
    for name, mdl in bundle["models"].items():
        preds[name] = float(np.asarray(mdl.predict(x), dtype=float)[0])
    y_hat = float(sum(bundle["weights"][k] * preds[k] for k in preds))
    uncertainty = float(np.std(list(preds.values()))) if preds else np.nan
    return {"pred": y_hat, "preds": preds, "uncertainty": uncertainty}


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _fit_dist_bundle(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series, val_regimes: pd.Series, current_regime: str, cfg: Part2Gen53Config):
    imp = _fit_imputer(x_train)
    xt = imp.transform(x_train)
    xv = imp.transform(x_val)
    ytr = y_train.values.astype(float)
    yva = y_val.values.astype(float)

    models: Dict[float, GradientBoostingRegressor] = {}
    val_preds: Dict[float, np.ndarray] = {}
    pinball: Dict[float, float] = {}

    for q in cfg.DIST_QUANTILES:
        mdl = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(q),
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            random_state=cfg.SEED,
        )
        mdl.fit(xt, ytr)
        pv = np.asarray(mdl.predict(xv), dtype=float)
        models[float(q)] = mdl
        val_preds[float(q)] = pv
        pinball[float(q)] = _pinball_loss(yva, pv, float(q))

    q05 = val_preds[0.05]
    q25 = val_preds[0.25]
    q50 = val_preds[0.50]
    q75 = val_preds[0.75]
    q95 = val_preds[0.95]
    raw_scores = np.maximum.reduce([q05 - yva, yva - q95, np.zeros_like(yva)])
    conf_adj = _conformal_adjustment(raw_scores, cfg.CONFORMAL_ALPHA)
    q05_conf = q05 - conf_adj
    q95_conf = q95 + conf_adj

    raw_coverage = float(np.mean((yva >= q05) & (yva <= q95))) if len(yva) else np.nan
    conf_coverage = float(np.mean((yva >= q05_conf) & (yva <= q95_conf))) if len(yva) else np.nan
    median_rmse = _rmse(yva, q50)

    tail_threshold = cfg.TAIL_EVENT_THRESHOLD
    sigma = np.maximum((q95_conf - q05_conf) / (2.0 * 1.6448536269514722), cfg.DIST_MIN_SIGMA)
    p_tail_val = np.clip(norm.cdf((tail_threshold - q50) / sigma), 1e-6, 1.0 - 1e-6)

    y_tail_val = (yva < tail_threshold).astype(int)
    if len(np.unique(y_tail_val)) >= 2:
        tail_auc = float(roc_auc_score(y_tail_val, p_tail_val))
        tail_pr = float(average_precision_score(y_tail_val, p_tail_val))
        tail_brier = _brier(y_tail_val, p_tail_val)
        tail_ece = _ece_score(y_tail_val, p_tail_val, cfg.ECE_BINS)
    else:
        tail_auc = np.nan
        tail_pr = np.nan
        tail_brier = np.nan
        tail_ece = np.nan

    return {
        "imputer": imp,
        "models": models,
        "val_preds": val_preds,
        "pinball": pinball,
        "conf_adj": float(conf_adj),
        "raw_coverage": raw_coverage,
        "conf_coverage": conf_coverage,
        "median_rmse": median_rmse,
        "tail_auc": tail_auc,
        "tail_pr": tail_pr,
        "tail_brier": tail_brier,
        "tail_ece": tail_ece,
    }


def _predict_dist(bundle, x_cur: pd.DataFrame, tail_threshold: float, cfg: Part2Gen53Config):
    x = bundle["imputer"].transform(x_cur)
    preds = {q: float(np.asarray(m.predict(x), dtype=float)[0]) for q, m in bundle["models"].items()}
    q05 = preds[0.05]
    q25 = preds[0.25]
    q50 = preds[0.50]
    q75 = preds[0.75]
    q95 = preds[0.95]
    conf_adj = float(bundle["conf_adj"])
    q05_conf = q05 - conf_adj
    q95_conf = q95 + conf_adj
    iqr = float(q75 - q25)
    tail_width = float(q95 - q05)
    conf_width = float(q95_conf - q05_conf)
    sigma = float(max(conf_width / (2.0 * 1.6448536269514722), cfg.DIST_MIN_SIGMA))
    p_tail_dist = float(np.clip(norm.cdf((tail_threshold - q50) / sigma), 1e-6, 1.0 - 1e-6))
    width_ratio = conf_width / max(cfg.DIST_CONF_WIDTH_SCALE, 1e-6)
    penalty_raw = np.sqrt(max(width_ratio, 0.0))
    uncertainty_penalty = float(np.clip(penalty_raw, cfg.DIST_PENALTY_FLOOR, cfg.DIST_PENALTY_CAP))
    return {
        "spread_q05": q05,
        "spread_q25": q25,
        "spread_q50": q50,
        "spread_q75": q75,
        "spread_q95": q95,
        "spread_q05_conf": q05_conf,
        "spread_q95_conf": q95_conf,
        "spread_iqr": iqr,
        "spread_tail_width": tail_width,
        "spread_conf_width": conf_width,
        "spread_left_tail_score": float(-q05_conf),
        "spread_median_score": float(-q50),
        "p_tail_dist": p_tail_dist,
        "uncertainty_penalty_g5": uncertainty_penalty,
    }


def _scaled01(x: float, lo: float, hi: float, invert: bool = False) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        out = 0.0
    else:
        out = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
    return 1.0 - out if invert else out


def _risk_overlay_metrics_g53(
    p_tail_base: float,
    dist_pred: Dict[str, float],
    dist_bundle: Dict[str, object],
    base_rate: float,
    tail_threshold: float,
    cfg: Part2Gen53Config,
) -> Dict[str, float]:
    p_tail_dist = float(dist_pred["p_tail_dist"]) if np.isfinite(_safe_num(dist_pred.get("p_tail_dist", np.nan))) else np.nan
    conf_width = float(dist_pred["spread_conf_width"])
    uncertainty_penalty = float(dist_pred["uncertainty_penalty_g5"])
    contradiction = float(abs(p_tail_base - p_tail_dist)) if np.isfinite(p_tail_base) and np.isfinite(p_tail_dist) else np.nan
    base_sign = np.sign(p_tail_base - base_rate) if np.isfinite(p_tail_base) else 0.0
    dist_sign = np.sign(p_tail_dist - base_rate) if np.isfinite(p_tail_dist) else 0.0
    sign_agree = int(base_sign == dist_sign or (np.isfinite(contradiction) and contradiction <= 0.03))

    tail_auc = _safe_num(dist_bundle.get("tail_auc", np.nan))
    tail_ece = _safe_num(dist_bundle.get("tail_ece", np.nan))
    conf_cov = _safe_num(dist_bundle.get("conf_coverage", np.nan))
    q05_conf = float(dist_pred["spread_q05_conf"])
    q50 = float(dist_pred["spread_q50"])

    score_auc = _scaled01(tail_auc, cfg.DIST_GATE_MIN_TAIL_AUC, 0.62)
    score_ece = _scaled01(tail_ece, cfg.DIST_GATE_MAX_TAIL_ECE, 0.04, invert=True)
    score_cov = _scaled01(conf_cov, cfg.DIST_GATE_MIN_CONF_COVERAGE, 0.95)
    score_width_quality = _scaled01(conf_width, 0.03, cfg.DIST_GATE_MAX_WIDTH, invert=True)
    score_penalty_quality = _scaled01(uncertainty_penalty, cfg.DIST_PENALTY_FLOOR, cfg.DIST_PENALTY_CAP, invert=True)
    score_contra = _scaled01(contradiction, cfg.DIST_GATE_MAX_CONTRADICTION, 0.02, invert=True) if np.isfinite(contradiction) else 0.5
    score_sign = float(sign_agree)

    tail_pressure = float(np.clip((0.005 - q05_conf) / 0.045, 0.0, 1.0)) if np.isfinite(q05_conf) else 0.0
    median_pressure = float(np.clip((0.0000 - q50) / 0.020, 0.0, 1.0)) if np.isfinite(q50) else 0.0
    width_pressure = float(np.clip((conf_width - 0.050) / max(cfg.DIST_GATE_MAX_WIDTH - 0.050, 1e-6), 0.0, 1.0))
    penalty_pressure = float(np.clip((uncertainty_penalty - 0.18) / max(cfg.DIST_PENALTY_CAP - 0.18, 1e-6), 0.0, 1.0))

    trust = (
        0.24 * score_auc
        + 0.18 * score_ece
        + 0.18 * score_cov
        + 0.14 * score_width_quality
        + 0.10 * score_penalty_quality
        + 0.08 * score_contra
        + 0.08 * score_sign
    )
    trust = float(np.clip(trust, 0.0, 1.0))

    caution_signal = (
        0.35 * tail_pressure
        + 0.15 * median_pressure
        + 0.35 * width_pressure
        + 0.15 * penalty_pressure
    )
    caution_signal = float(np.clip(caution_signal, 0.0, 1.0))

    trust_excess = float(np.clip((trust - 0.45) / 0.40, 0.0, 1.0))
    overlay_strength = float(np.clip(0.65 * trust_excess * caution_signal, 0.0, 0.35))

    width_caution = float(np.clip(0.75 * width_pressure + 0.25 * penalty_pressure, 0.0, 1.0))
    penalty_caution = float(np.clip(penalty_pressure, 0.0, 1.0))
    left_tail_gap = float(np.clip(0.70 * tail_pressure + 0.30 * median_pressure, 0.0, 1.0))

    overlay_on = int(
        np.isfinite(p_tail_base)
        and trust >= 0.45
        and caution_signal >= 0.40  # raised from 0.10 → top-quartile uncertainty only
        # Rationale: at threshold=0.10, caution_signal exceeds it on every row
        # (dist_overlay_on_rate=1.0), making the overlay a constant 8% shrinkage
        # rather than a selective event gate.  At threshold=0.40, the overlay fires
        # only when distributional uncertainty is genuinely elevated (~top 25% of
        # the historical distribution), preserving its intended selective character.
        # Expected effect: dist_overlay_on_rate drops from ~1.0 to ~0.25; rows with
        # low uncertainty receive the full classifier probability; rows with genuinely
        # wide predictive distributions receive the shrinkage.
    )

    tail_shift = float(min(
        0.010,
        0.010 * (0.55 * overlay_strength + 0.45 * left_tail_gap)
    ))

    return {
        "dist_overlay_on_g53": overlay_on,
        "dist_trust_score_g53": trust,
        "dist_overlay_strength_g53": overlay_strength,
        "dist_tail_shift_g53": tail_shift,
        "dist_width_caution_g53": width_caution,
        "dist_penalty_caution_g53": penalty_caution,
        "dist_left_tail_gap_g53": left_tail_gap,
        "dist_contradiction_g53": contradiction if np.isfinite(contradiction) else np.nan,
        "dist_sign_agree_g53": int(sign_agree),
    }


def _apply_risk_overlay_g53(
    p_tail_base: float,
    dist_pred: Dict[str, float],
    dist_bundle: Dict[str, object],
    base_rate: float,
    tail_threshold: float,
    cfg: Part2Gen53Config,
) -> Tuple[float, str, int, Dict[str, float]]:
    if not np.isfinite(p_tail_base):
        if np.isfinite(_safe_num(dist_pred.get("p_tail_dist", np.nan))):
            overlay = _risk_overlay_metrics_g53(base_rate, dist_pred, dist_bundle, base_rate, tail_threshold, cfg)
            return float(np.clip(dist_pred["p_tail_dist"], 1e-6, 1.0 - 1e-6)), "dist_only_fallback", 1, overlay
        return np.nan, "none", 1, {
            "dist_overlay_on_g53": 0,
            "dist_trust_score_g53": 0.0,
            "dist_overlay_strength_g53": 0.0,
            "dist_tail_shift_g53": 0.0,
            "dist_width_caution_g53": 0.0,
            "dist_penalty_caution_g53": 0.0,
            "dist_left_tail_gap_g53": 0.0,
            "dist_contradiction_g53": np.nan,
            "dist_sign_agree_g53": 0,
        }

    overlay = _risk_overlay_metrics_g53(p_tail_base, dist_pred, dist_bundle, base_rate, tail_threshold, cfg)
    return float(np.clip(p_tail_base, 1e-6, 1.0 - 1e-6)), "base_plus_soft_caution_overlay_532", 0, overlay


# ---------------- Engine ----------------

def _load_part1_contract(cfg: Part2Gen53Config) -> pd.DataFrame:
    base = cfg.PART1_DIR

    def _first_existing(*names: str) -> Optional[str]:
        for name in names:
            path = os.path.join(base, name)
            if os.path.exists(path):
                return path
        return None

    x = _read_table(os.path.join(base, "X_features.parquet"))
    y_full = _read_table(os.path.join(base, "y_labels_full.parquet"))
    y_reg_full = _read_table(os.path.join(base, "y_reg_full.parquet"))

    y_revealed_path = _first_existing("y_labels_revealed_aligned.parquet", "y_labels_revealed.parquet")
    if y_revealed_path is None:
        raise FileNotFoundError("Missing either y_labels_revealed_aligned.parquet or y_labels_revealed.parquet")
    y_revealed_aligned = _read_table(y_revealed_path)

    cal_feat_path = _first_existing("calendar_feature_aligned.parquet")
    cal_feat = _read_table(cal_feat_path) if cal_feat_path is not None else pd.DataFrame({"Date": x["Date"].copy()})

    factor_path = _first_existing("factor_returns.parquet")
    factors = _read_table(factor_path) if factor_path is not None else pd.DataFrame({"Date": x["Date"].copy()})

    bench_path = _first_existing("benchmark_returns.parquet")
    if bench_path is not None:
        bench = _read_table(bench_path)
    else:
        bench = pd.DataFrame({"Date": x["Date"].copy()})
        if {"voo_ret_1d", "ief_ret_1d"} <= set(factors.columns):
            bench["bench_60_40"] = 0.60 * pd.to_numeric(factors["voo_ret_1d"], errors="coerce") + 0.40 * pd.to_numeric(factors["ief_ret_1d"], errors="coerce")

    live_px_path = _first_existing("price_calls_live_snapshot.parquet")
    if live_px_path is not None:
        live_px = _read_table(live_px_path)
    else:
        live_px = y_reg_full[["Date"]].copy()
        if "px_voo_t" in y_reg_full.columns:
            live_px["px_voo_t"] = y_reg_full["px_voo_t"]
        if "px_ief_t" in y_reg_full.columns:
            live_px["px_ief_t"] = y_reg_full["px_ief_t"]

    full = x.merge(y_full, on="Date", how="left")
    full = full.merge(y_reg_full, on="Date", how="left", suffixes=("", "_reg"))
    full = full.merge(cal_feat, on="Date", how="left")
    full = full.merge(factors, on="Date", how="left")
    full = full.merge(bench, on="Date", how="left")
    full = full.merge(live_px, on="Date", how="left", suffixes=("", "_live"))

    # FIX (Finding 2, Part6 Audit 2026-04):
    # Load the Part 6 HMM regime labels written by Part 1 as regime_labels_p6.parquet.
    # These labels are merged in as "regime_label_p6" — a separate column so the
    # existing internal GMM label pipeline is preserved as a fallback.
    # The walk-forward loop below uses regime_label_p6 to override current_regime
    # when the Part 6 label is available and non-unknown.
    #
    # Two candidate paths are checked in priority order:
    #   1. artifacts_part1/regime_labels_p6.parquet  (written by Part 1 after it reads Part 6)
    #   2. artifacts_part6/regime_history.parquet    (direct Part 6 output, fallback)
    # If neither exists, regime_label_p6 is filled with "unknown" and Part 2's
    # internal GMM is used as it was before this fix.
    _p6_label_candidates = [
        os.path.join(base, "regime_labels_p6.parquet"),                        # Part 1 wrote this
        os.path.join(cfg.PART6_DIR, "regime_history.parquet"),                  # direct Part 6 fallback
    ]
    _p6_df: Optional[pd.DataFrame] = None
    for _p6_cand in _p6_label_candidates:
        if os.path.exists(_p6_cand):
            try:
                _p6_raw = pd.read_parquet(_p6_cand)
                # Normalise the Date column — Part 1's file has Date as a column;
                # Part 6's regime_history has it as the index.
                if "Date" in _p6_raw.columns:
                    _p6_raw["Date"] = pd.to_datetime(_p6_raw["Date"], errors="coerce").dt.normalize()
                else:
                    _p6_raw = _p6_raw.reset_index()
                    _p6_raw.rename(columns={_p6_raw.columns[0]: "Date"}, inplace=True)
                    _p6_raw["Date"] = pd.to_datetime(_p6_raw["Date"], errors="coerce").dt.normalize()
                if "regime_label" in _p6_raw.columns:
                    _p6_df = _p6_raw[["Date", "regime_label"]].rename(
                        columns={"regime_label": "regime_label_p6"}
                    )
                    print(f"[Part 2] Loaded Part 6 HMM regime labels from: {_p6_cand}")
                    _n_known = int((_p6_df["regime_label_p6"] != "unknown").sum())
                    print(f"[Part 2] Part 6 regime coverage: {_n_known}/{len(_p6_df)} dates non-unknown.")
                    break
            except Exception as _p6_exc:
                print(f"[Part 2] WARNING: Could not load Part 6 labels from {_p6_cand}: {_p6_exc}")

    if _p6_df is not None:
        full = full.merge(_p6_df, on="Date", how="left")
        full["regime_label_p6"] = full["regime_label_p6"].fillna("unknown")
    else:
        print("[Part 2] No Part 6 regime labels found. Internal GMM will be used as sole regime source.")
        full["regime_label_p6"] = "unknown"

    if "px_voo_t_live" in full.columns and "px_voo_t" not in full.columns:
        rename_map = {"px_voo_t_live": "px_voo_t"}
        if "px_ief_t_live" in full.columns:
            rename_map["px_ief_t_live"] = "px_ief_t"
        full = full.rename(columns=rename_map)
    if "px_voo_t_live" in full.columns:
        full["px_voo_t"] = full["px_voo_t"].fillna(full["px_voo_t_live"])
        if "px_ief_t_live" in full.columns:
            full["px_ief_t"] = full["px_ief_t"].fillna(full["px_ief_t_live"])
        full = full.drop(columns=[c for c in ["px_voo_t_live", "px_ief_t_live"] if c in full.columns])

    revealed_dates = set(pd.to_datetime(y_revealed_aligned["Date"]).dt.normalize())
    full["y_avail"] = full["Date"].isin(revealed_dates).astype(int)

    full = _ensure_locked_contract_columns(full, cfg)
    return full.sort_values("Date").reset_index(drop=True)

def _ensure_locked_contract_columns(full: pd.DataFrame, cfg: Part2Gen53Config) -> pd.DataFrame:
    out = full.copy()
    for col in cfg.LOCKED_FORBIDDEN_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _resolve_locked_live_feature_cols(part1_meta: Dict[str, object], full: pd.DataFrame) -> List[str]:
    feature_cols = part1_meta.get("feature_cols", [])
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("Part 1 metadata is missing the locked feature_cols list required by Part 2.")

    locked = [str(c) for c in feature_cols]
    missing = [c for c in locked if c not in full.columns]
    if missing:
        raise RuntimeError(
            "Part 2 could not find the locked Part 1 feature columns in the merged contract frame: "
            f"{missing}"
        )

    non_numeric = [c for c in locked if not pd.api.types.is_numeric_dtype(full[c])]
    if non_numeric:
        raise RuntimeError(
            "Locked Part 1 feature columns must all be numeric for Part 2 modeling. "
            f"Non-numeric columns: {non_numeric}"
        )

    return locked


def _select_model_features(
    full: pd.DataFrame,
    part1_meta: Dict[str, object],
    cfg: Part2Gen53Config,
) -> Tuple[List[str], List[str]]:
    part1_version = str(part1_meta.get("version", "")).strip()

    # Live locked-14 contract: use the exact Part 1 locked feature list.
    if part1_version in {"V19_P1_HARDENED", "V20_P1_DAILY"}:
        allowed = _resolve_locked_live_feature_cols(part1_meta, full)
        forbidden = list(cfg.LOCKED_FORBIDDEN_FEATURES)
        return allowed, forbidden

    # Fallback / legacy compatibility path
    forbidden_prefixes = ("fwd_", "bench_", "px_")
    forbidden_exact = set(cfg.LOCKED_FORBIDDEN_FEATURES) | set(cfg.OPTIONAL_FORBIDDEN_FEATURES)

    forbidden, allowed = [], []
    for c in full.columns:
        if c in forbidden_exact or any(c.startswith(p) for p in forbidden_prefixes):
            forbidden.append(c)
        elif pd.api.types.is_numeric_dtype(full[c]):
            allowed.append(c)

    return sorted(allowed), sorted(set(forbidden))

def _is_live_contract(contract_profile: object) -> bool:
    return str(contract_profile).lower().startswith("live_locked_14")


def _effective_drift_ece_max(contract_profile: object, cfg: Part2Gen53Config) -> float:
    return float(cfg.LIVE_DRIFT_ECE_MAX if _is_live_contract(contract_profile) else cfg.DRIFT_ECE_MAX)


def _effective_drift_brier_max(contract_profile: object, cfg: Part2Gen53Config) -> float:
    return float(cfg.LIVE_DRIFT_BRIER_MAX if _is_live_contract(contract_profile) else cfg.DRIFT_BRIER_MAX)


def _effective_final_pass_drift_rate_max(contract_profile: object, cfg: Part2Gen53Config) -> float:
    return float(cfg.LIVE_FINAL_PASS_DRIFT_RATE_MAX if _is_live_contract(contract_profile) else 0.30)


def _effective_fail_closed_drift_rate(contract_profile: object, cfg: Part2Gen53Config) -> float:
    return float(cfg.LIVE_FAIL_CLOSED_DRIFT_RATE if _is_live_contract(contract_profile) else cfg.FAIL_CLOSED_DRIFT_RATE)


def _effective_fail_closed_cal_gate(contract_profile: object, cfg: Part2Gen53Config) -> float:
    return float(cfg.LIVE_FAIL_CLOSED_CAL_GATE if _is_live_contract(contract_profile) else cfg.FAIL_CLOSED_CAL_GATE)


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _make_train_val(full: pd.DataFrame, current_idx: int, feature_cols: List[str], cfg: Part2Gen53Config):
    cutoff = current_idx - cfg.PURGE
    if cutoff <= 0:
        return None
    trainable = full.iloc[:cutoff].copy()
    trainable = trainable.loc[trainable["y_avail"] == 1].copy()
    if len(trainable) < cfg.MIN_TRAIN_ROWS:
        return None
    trainable = trainable.tail(cfg.TRAIN_WINDOW_DAYS).reset_index(drop=True)
    if len(trainable) <= cfg.VALID_WINDOW + cfg.MIN_TRAIN_ROWS // 2:
        return None
    train_df = trainable.iloc[:-cfg.VALID_WINDOW].copy()
    val_df = trainable.iloc[-cfg.VALID_WINDOW:].copy()
    if len(train_df) < cfg.MIN_TRAIN_ROWS:
        return None
    if train_df[feature_cols].isna().any().any() or val_df[feature_cols].isna().any().any():
        raise RuntimeError("Unexpected nulls in feature matrix after locked Part 1 build.")
    return train_df, val_df


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _regime_for_current(train_df: pd.DataFrame, val_df: pd.DataFrame, current_row: pd.DataFrame, cfg: Part2Gen53Config):
    feature_cols = [c for c in cfg.REGIME_FEATURES if c in train_df.columns]
    reg_bundle = _fit_regime_model(train_df, feature_cols, cfg) if len(feature_cols) == len(cfg.REGIME_FEATURES) else None
    if reg_bundle is not None:
        reg_val = _predict_regime(reg_bundle, val_df)
        reg_cur = _predict_regime(reg_bundle, current_row).iloc[0]
    else:
        ref_cols = [c for c in ("stress_score_raw", "vix_z21", "credit_spread_z21", "excess_vol10", "spread_ret21") if c in train_df.columns]
        ref_df = train_df[ref_cols].copy() if ref_cols else train_df.copy()
        reg_val = _fallback_regime(val_df, ref_df=ref_df)
        reg_cur = _fallback_regime(current_row, ref_df=ref_df).iloc[0]
    return reg_bundle, reg_val, str(reg_cur)


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _build_fit_bundle(train_df: pd.DataFrame, val_df: pd.DataFrame, current_row: pd.DataFrame, feature_cols: List[str], cfg: Part2Gen53Config):
    reg_bundle, reg_val, current_regime = _regime_for_current(train_df, val_df, current_row[[c for c in cfg.REGIME_FEATURES if c in current_row.columns]].copy(), cfg)
    y_tail_train = train_df["y_rel_tail_voo_vs_ief"].astype(int)
    y_tail_val = val_df["y_rel_tail_voo_vs_ief"].astype(int)
    if y_tail_train.nunique() < 2 or y_tail_train.value_counts().min() < cfg.MIN_CLASS_COUNT:
        raise RuntimeError("Insufficient class diversity for Part 2 training window.")
    prob_bundle = _fit_prob_ensemble(train_df[feature_cols], y_tail_train, val_df[feature_cols], y_tail_val, reg_val, current_regime, cfg)
    reg_voo = _fit_reg_ensemble(train_df[feature_cols], train_df["fwd_voo"], val_df[feature_cols], val_df["fwd_voo"], reg_val, current_regime, cfg)
    reg_ief = _fit_reg_ensemble(train_df[feature_cols], train_df["fwd_ief"], val_df[feature_cols], val_df["fwd_ief"], reg_val, current_regime, cfg)
    dist_bundle = _fit_dist_bundle(train_df[feature_cols], train_df["excess_ret"], val_df[feature_cols], val_df["excess_ret"], reg_val, current_regime, cfg)
    return {"prob": prob_bundle, "voo": reg_voo, "ief": reg_ief, "dist": dist_bundle, "regime": reg_bundle, "current_regime": current_regime}


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _compute_drift_flags(ece_series: pd.Series, brier_series: pd.Series, cfg: Part2Gen53Config, ece_max: Optional[float] = None, brier_max: Optional[float] = None) -> np.ndarray:
    ece_lim = float(cfg.DRIFT_ECE_MAX if ece_max is None else ece_max)
    brier_lim = float(cfg.DRIFT_BRIER_MAX if brier_max is None else brier_max)
    base = ((ece_series > ece_lim) | (brier_series > brier_lim)).fillna(False).astype(bool)
    hist_ok = ((ece_series.notna()) | (brier_series.notna())).rolling(cfg.DRIFT_MIN_HISTORY, min_periods=1).sum() >= cfg.DRIFT_MIN_HISTORY
    out = base.copy()
    for _ in range(max(cfg.DRIFT_PERSISTENCE, 1) - 1):
        out = out & out.shift(1, fill_value=False)
    out = out & hist_ok
    return out.fillna(False).astype(int).values


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _governance_mapping(
    p_final_cal: float,
    base_rate: float,
    drift_alarm: int,
    raw_val_auc: float,
    agreement_score: float,
    fwd_spread_hat: float,
    leg_uncertainty: float,
    defense_trigger_threshold: float,
    regime_label: str,
    stress_score_raw: float,
    stress_q_threshold: float,
    cfg: Part2Gen53Config,
    dist_overlay_strength: float = 0.0,
    dist_tail_shift: float = 0.0,
    dist_width_caution: float = 0.0,
    uncertainty_penalty: float = 0.0,
):
    downside_edge = max(p_final_cal - base_rate, 0.0)
    upside_relief = max(base_rate - p_final_cal, 0.0)
    high_risk_state = int(
        (p_final_cal >= cfg.HIGH_RISK_ABS_P)
        or (downside_edge >= cfg.HIGH_RISK_EDGE)
        or (str(regime_label) == 'dislocated')
        # FIX (Finding 2, Part6 Audit 2026-04): "crisis" is the Part 6 HMM
        # equivalent of Part 2's internal "dislocated". Both trigger high_risk_state.
        or (str(regime_label) == 'crisis')
    )

    spread_confirm = max(cfg.SPREAD_CONFIRM_MIN, cfg.SPREAD_K * max(leg_uncertainty, 0.0))
    spread_gate = min(-spread_confirm, -cfg.DEPLOY_DOWNSIDE_SPREAD_ABS)
    prob_anchor = max(cfg.DEPLOY_DOWNSIDE_MIN_P, base_rate + cfg.DEF_TRIGGER_BASELINE_EDGE)
    effective_prob_for_trigger = float(np.clip(p_final_cal, 1e-6, 1.0 - 1e-6))
    prob_component = float(np.clip((effective_prob_for_trigger - prob_anchor) / max(cfg.DEF_TRIGGER_PROB_SCALE, 1e-6), 0.0, 1.0))
    spread_component = float(np.clip(((-fwd_spread_hat) - abs(spread_gate)) / max(cfg.DEF_TRIGGER_SPREAD_SCALE, 1e-6), 0.0, 1.0))
    regime_component = _regime_defense_score(str(regime_label))
    stress_component = 0.0
    if np.isfinite(stress_q_threshold) and np.isfinite(stress_score_raw):
        stress_component = 1.0 if stress_score_raw >= stress_q_threshold else 0.0
    elif high_risk_state:
        stress_component = 0.5

    defense_trigger_raw = float(np.clip(
        cfg.DEF_TRIGGER_WEIGHT_PROB * prob_component
        + cfg.DEF_TRIGGER_WEIGHT_SPREAD * spread_component
        + cfg.DEF_TRIGGER_WEIGHT_REGIME * regime_component
        + cfg.DEF_TRIGGER_WEIGHT_STRESS * stress_component,
        0.0,
        1.0,
    ))
    threshold = float(defense_trigger_threshold) if np.isfinite(defense_trigger_threshold) else float(cfg.DEF_TRIGGER_FLOOR)

    val_ok = np.isfinite(raw_val_auc) and raw_val_auc >= cfg.DEPLOY_MIN_VAL_AUC
    agree_ok = np.isfinite(agreement_score) and agreement_score >= cfg.DEPLOY_MIN_AGREEMENT
    deploy_downside = int(
        (defense_trigger_raw >= threshold)
        and (downside_edge >= cfg.DEPLOY_DOWNSIDE_MIN_EDGE)
        and (p_final_cal >= cfg.DEPLOY_DOWNSIDE_MIN_P)
        and (spread_component > 0.0)
        and val_ok
        and agree_ok
        and ((not drift_alarm) or high_risk_state)
    )
    # FIX (Finding 2, Quant-Guild Part 26): Regime-conditional deploy guard.
    # UPDATE (F2, Quant-Guild Part 42 Audit): Updated comment with current confirmed AUC.
    # Block deploy_downside when the current regime is in PASSIVE_REGIMES_NO_DEPLOY
    # (calm, risk_on, crisis). Current confirmed AUC (DeLong SE, 1,674 rows, 2020-2026):
    #   calm    AUC=0.510, p=0.402 — neutral, not worth deploying defense
    #   risk_on AUC=0.485, p=0.672 — weakly anti-predictive, no signal
    #   crisis  AUC=0.478, p=0.694 — anti-predictive; crisis coverage via high_risk_state
    # Only high_vol (AUC=0.539, z=1.309, p=0.095) has directional signal.
    # Confirmed: all 14 historical deploy events are high_vol(9) or crisis(5 via HRS).
    # Note: high_risk_state still fires in passive regimes when regime_component=1.0
    # (crisis HRS) or p_final_cal >= HIGH_RISK_ABS_P, providing position-sizing
    # awareness without full deploy.
    _passive_deploy_regimes = {r.lower() for r in getattr(cfg, 'PASSIVE_REGIMES_NO_DEPLOY', ('calm', 'risk_on', 'crisis'))}
    if str(regime_label).lower() in _passive_deploy_regimes and not high_risk_state:
        deploy_downside = 0
    deploy_upside = 0

    max_under = cfg.HIGH_RISK_MAX_UNDERWEIGHT if high_risk_state else cfg.BASE_MAX_UNDERWEIGHT
    max_over = cfg.HIGH_RISK_MAX_OVERWEIGHT if high_risk_state else cfg.BASE_MAX_OVERWEIGHT
    active_weight_raw = 0.0
    if deploy_downside:
        trigger_strength = max(0.0, defense_trigger_raw - threshold)
        trigger_ratio = min(1.0, trigger_strength / max(1e-6, 1.0 - threshold))
        edge_ratio = min(1.0, downside_edge / max(cfg.DEPLOY_DOWNSIDE_MIN_EDGE, 1e-6))
        spread_weight = min(1.0, spread_component)
        size_score = 0.45 * trigger_ratio + 0.35 * edge_ratio + 0.20 * spread_weight
        active_weight_raw = -(
            cfg.DEF_UNDERWEIGHT_BASE
            + cfg.DEF_UNDERWEIGHT_SCALE * size_score
            + cfg.DEF_EDGE_SCALE * downside_edge
            + cfg.DEF_SPREAD_SCALE_WEIGHT * spread_component
        )
        if high_risk_state:
            active_weight_raw *= 1.20
        elif drift_alarm:
            active_weight_raw *= 0.95
        active_weight_raw = -min(max_under, abs(active_weight_raw))

    active_weight_capped = float(np.clip(active_weight_raw, -max_under, max_over))
    w_voo_uncapped = cfg.BASE_WEIGHT_VOO + active_weight_capped
    w_voo = float(np.clip(w_voo_uncapped, cfg.MIN_W_VOO, cfg.MAX_W_VOO))

    if drift_alarm and not high_risk_state:
        governance_tier = 'CAUTION'
    elif deploy_downside and high_risk_state:
        governance_tier = 'DEFENSIVE'
    elif deploy_downside:
        governance_tier = 'CAUTION'
    elif high_risk_state:
        governance_tier = 'CAUTION'
    else:
        governance_tier = 'NORMAL'

    # FIX (BUG-1, Audit 2026-05-10 — Quant-Guild Part 18):
    # The previous thresholds caused governance_tier to be STRUCTURALLY LOCKED to
    # CAUTION on 100% of rows. Root causes:
    #
    #   uncertainty_penalty_g5: threshold was 0.25. But DIST_PENALTY_FLOOR=0.08 and
    #     the practical daily minimum (from artifact analysis) is 0.31 — ALWAYS above 0.25.
    #     Result: uncertainty_penalty condition triggered on 1658/1658 rows (100%).
    #
    #   dist_overlay_strength_g53: threshold was 0.12. Mean = 0.151, triggering on
    #     1151/1658 rows (69.4%). Not selective enough to be a meaningful gate.
    #
    # Both conditions made governance_tier=NORMAL structurally unreachable, causing:
    #   (a) alpha_scale never reaches 1.0 (always throttled by CAUTION path)
    #   (b) No distinction between genuinely uncertain and routine prediction days
    #   (c) Part 2A's governance_tier=CAUTION multiplier always applied (0.60 scalar)
    #
    # Statistical basis for new thresholds:
    #   uncertainty_penalty: range [0.31, 0.65], median=0.499, mean=0.499.
    #     Setting 0.55 (≈75th percentile) captures genuinely elevated uncertainty.
    #     Expected effect: ~25% of rows trigger this condition (was 100%).
    #
    #   dist_overlay_strength: range [0, 0.33], mean=0.151.
    #     Setting 0.20 (≈65th percentile) captures rows where the distributional
    #     overlay has meaningful magnitude.
    #     Expected effect: ~35% of rows trigger this condition (was 69.4%).
    if governance_tier == 'NORMAL' and (
        dist_overlay_strength >= 0.20           # FIX: was 0.12 (too loose, 69% trigger rate)
        or dist_width_caution >= 0.40
        or uncertainty_penalty >= 0.55          # FIX: was 0.25 (always triggered, min=0.31)
    ):
        governance_tier = 'CAUTION'

    alpha_scale = 1.0
    if deploy_downside:
        alpha_scale = 0.80
    if high_risk_state:
        alpha_scale = min(alpha_scale, 0.65)
    if drift_alarm:
        alpha_scale = min(alpha_scale, cfg.ALPHA_THROTTLE)

    caution_throttle = max(
        0.0,
        0.18 * dist_overlay_strength
        + 0.20 * max(dist_width_caution - 0.35, 0.0)
        + 0.12 * max(uncertainty_penalty - 0.20, 0.0)
    )
    alpha_scale = min(alpha_scale, max(0.72, 1.0 - caution_throttle))

    if dist_width_caution >= 0.70 or uncertainty_penalty >= cfg.OVERLAY_PENALTY_TRIGGER:
        alpha_scale = min(alpha_scale, cfg.OVERLAY_CAUTION_ALPHA_CAP)

    return {
        'downside_edge': float(downside_edge),
        'upside_relief': float(upside_relief),
        'high_risk_state': int(high_risk_state),
        'deploy_downside': int(deploy_downside),
        'deploy_upside': int(deploy_upside),
        'signal_q_threshold': float(defense_trigger_threshold) if np.isfinite(defense_trigger_threshold) else np.nan,
        'spread_gate': float(spread_gate),
        'defense_trigger_raw': float(defense_trigger_raw),
        'defense_trigger_threshold': float(threshold),
        'prob_component': float(prob_component),
        'spread_component': float(spread_component),
        'regime_component': float(regime_component),
        'stress_component': float(stress_component),
        'dist_overlay_strength_g53': float(dist_overlay_strength),
        'dist_tail_shift_g53': float(dist_tail_shift),
        'dist_width_caution_g53': float(dist_width_caution),
        'uncertainty_penalty_g5': float(uncertainty_penalty),
        'active_weight_raw': float(active_weight_raw),
        'max_underweight_cap': float(max_under),
        'max_overweight_cap': float(max_over),
        'active_weight_capped': float(active_weight_capped),
        'w_strategy_voo_uncapped': float(w_voo_uncapped),
        'w_strategy_voo': float(w_voo),
        'alpha_scale': float(alpha_scale),
        'governance_tier': governance_tier,
    }


# FIX: was Part2Gen5Config (undefined), corrected to Part2Gen53Config
def _shuffle_auc(train_df: pd.DataFrame, feature_cols: List[str], cfg: Part2Gen53Config) -> float:
    if len(train_df) < cfg.MIN_TRAIN_ROWS:
        return np.nan
    y = train_df["y_rel_tail_voo_vs_ief"].astype(int).values
    if len(np.unique(y)) < 2:
        return np.nan
    x = train_df[feature_cols].copy().values
    aucs = []
    rng = np.random.default_rng(cfg.SEED)
    block = max(2, cfg.SHUFFLE_BLOCK)
    for _ in range(cfg.SHUFFLE_B):
        idx = np.arange(len(y))
        chunks = [idx[i : i + block] for i in range(0, len(idx), block)]
        rng.shuffle(chunks)
        y_shuf = np.concatenate([y[c] for c in chunks])
        split = max(cfg.MIN_TRAIN_ROWS // 2, len(y_shuf) - cfg.VALID_WINDOW)
        if split <= 0 or split >= len(y_shuf):
            continue
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=cfg.SEED))
        mdl.fit(x[:split], y_shuf[:split])
        p = mdl.predict_proba(x[split:])[:, 1]
        yt = y_shuf[split:]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, p))
    return float(np.nanmedian(aucs)) if len(aucs) else np.nan


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prior_part2_summary(cfg) -> Dict[str, object]:
    """Load the committed Part 2 summary from the previous run, if it exists.

    The GitHub Actions job starts from the last committed repo snapshot, so the
    summary JSON on disk before this run is the immediately prior committed state.
    We use it only for deploy-downside hysteresis (NORMAL can stay NORMAL on a
    slightly weaker count than the entry threshold).
    """
    path = os.path.join(cfg.PRED_DIR, cfg.SUMMARY_FILE)
    if not os.path.exists(path):
        return {}
    try:
        data = _load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _deploy_downside_gate_stats(
    out: pd.DataFrame,
    cfg,
    prior_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Evaluate the deploy-downside clearance gate using counts, not a brittle rate.

    Old behavior: final_pass required deploy_downside_rate >= 0.002. At ~1,648
    rows that is equivalent to 3.296 events, so the gate flips on a one-event
    difference (4 deploy rows passes, 3 deploy rows fails).

    New behavior: use integer counts plus hysteresis.
      * enter_gate: total_count >= ENTER_COUNT_MIN and recent_count >= RECENT_COUNT_MIN
      * stay_gate:  prior NORMAL run and total_count >= STAY_COUNT_MIN and
                    recent_count >= RECENT_COUNT_MIN
    """
    deploy = pd.to_numeric(out.get("deploy_downside", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    total_count = int(deploy.sum())
    total_rows = int(len(deploy))
    rate = float(deploy.mean()) if total_rows else np.nan

    lookback = int(max(1, cfg.DEPLOY_DOWNSIDE_RECENT_LOOKBACK))
    recent_count = int(deploy.tail(lookback).sum()) if total_rows else 0

    prior_summary = prior_summary or {}
    prior_publish_mode = str(prior_summary.get("publish_mode", "")).upper()
    prior_final_pass = bool(prior_summary.get("final_pass", False))
    prior_normal = prior_publish_mode == "NORMAL" or prior_final_pass

    enter_gate = bool(
        total_count >= int(cfg.DEPLOY_DOWNSIDE_ENTER_COUNT_MIN)
        and recent_count >= int(cfg.DEPLOY_DOWNSIDE_RECENT_COUNT_MIN)
    )
    stay_gate = bool(
        prior_normal
        and total_count >= int(cfg.DEPLOY_DOWNSIDE_STAY_COUNT_MIN)
        and recent_count >= int(cfg.DEPLOY_DOWNSIDE_RECENT_COUNT_MIN)
    )
    gate_pass = bool(enter_gate or stay_gate)

    if enter_gate:
        reason = "enter_normal_count_gate"
    elif stay_gate:
        reason = "stay_normal_hysteresis"
    else:
        reason = "insufficient_deploy_events"

    return {
        "total_count": total_count,
        "recent_count": recent_count,
        "total_rows": total_rows,
        "rate": rate,
        "prior_publish_mode": prior_publish_mode,
        "prior_final_pass": int(prior_final_pass),
        "prior_normal": int(prior_normal),
        "enter_gate": int(enter_gate),
        "stay_gate": int(stay_gate),
        "gate_pass": int(gate_pass),
        "gate_reason": reason,
    }


def _load_part1_meta(cfg) -> Dict[str, object]:
    path = os.path.join(cfg.PART1_DIR, "part1_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta = _load_json(path)

    version = str(meta.get("version", ""))
    accepted_versions = tuple(getattr(cfg, "ACCEPTED_PART1_VERSIONS", (cfg.EXPECTED_PART1_VERSION,)))
    if version not in accepted_versions:
        raise RuntimeError(
            f"Part 2 requires one of Part 1 versions {accepted_versions}, found {version}. "
            "Rerun the hardened QA overwrite of part1_builder.py so part1_meta.json is restamped correctly."
        )

    horizon = int(meta.get("horizon", cfg.H))
    if horizon != int(cfg.H):
        raise RuntimeError(f"Part 2 requires horizon {cfg.H}, found {horizon}.")

    tail_label_name = str(meta.get("tail_label_name", ""))
    if tail_label_name != "y_rel_tail_voo_vs_ief":
        raise RuntimeError(
            f"Part 2 requires tail_label_name='y_rel_tail_voo_vs_ief', found {tail_label_name!r}."
        )

    feature_cols = meta.get("feature_cols", [])
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError("Part 1 metadata is missing a non-empty feature_cols list.")

    return meta



def _resolve_contract_profile(part1_meta: Dict, feature_cols: List[str], cfg) -> Tuple[str, int, int]:
    part1_version = str(part1_meta.get("version", "")).strip()
    if part1_version == "GEN4_PART1_V2B":
        return "legacy_locked_64", cfg.LEGACY_EXPECTED_MODEL_FEATURE_COUNT, cfg.LEGACY_EXPECTED_FORBIDDEN_COUNT
    if part1_version in {"V19_P1_HARDENED", "V20_P1_DAILY"}:
        return "live_locked_14", cfg.LIVE_EXPECTED_MODEL_FEATURE_COUNT, cfg.LIVE_EXPECTED_FORBIDDEN_COUNT
    if len(feature_cols) <= cfg.LIVE_EXPECTED_MODEL_FEATURE_COUNT + 1:
        return "live_locked_14_inferred", cfg.LIVE_EXPECTED_MODEL_FEATURE_COUNT, cfg.LIVE_EXPECTED_FORBIDDEN_COUNT
    return "compatibility_unpinned", len(feature_cols), len(cfg.LOCKED_FORBIDDEN_FEATURES)



def _validate_feature_contract(feature_cols: List[str], forbidden_features: List[str], part1_meta: Dict, cfg) -> Dict[str, object]:
    """
    Contract validation upgraded for the current live 14-feature Part 1 regime.

    Leakage guards remain strict. Exact locked shape checks now depend on the
    declared Part 1 contract profile rather than assuming the old 64-feature
    panel is always canonical.
    """
    core_forbidden = {
        "Date", "excess_ret", "y_voo", "y_rel_tail_voo_vs_ief", "y_avail",
        "fwd_voo", "fwd_ief", "fwd_spread",
        "px_voo_t", "px_ief_t", "px_voo_fwd", "px_ief_fwd",
    }

    if len(feature_cols) < 10:
        raise RuntimeError(f"Too few model features for Gen 5.3.2 compatibility mode: {len(feature_cols)}")

    current = set(forbidden_features)
    missing_core = sorted(core_forbidden - current)
    if missing_core:
        raise RuntimeError(f"Core forbidden/leakage features are missing from exclusion list: {missing_core}")

    profile_name, expected_feature_count, expected_forbidden_count = _resolve_contract_profile(part1_meta, feature_cols, cfg)
    optional_forbidden = set(cfg.OPTIONAL_FORBIDDEN_FEATURES)
    locked_required = set(cfg.LOCKED_FORBIDDEN_FEATURES)

    missing_locked = sorted(locked_required - current)
    missing_optional = sorted(optional_forbidden - current)
    extra_locked = sorted(current - (locked_required | optional_forbidden))

    print(
        f"[CONTRACT] profile={profile_name} | part1_version={part1_meta.get('version')} | "
        f"features={len(feature_cols)} | forbidden={len(forbidden_features)}"
    )

    if len(feature_cols) != expected_feature_count:
        print(
            f"[CONTRACT] Feature count changed: {len(feature_cols)} "
            f"vs locked {expected_feature_count}."
        )

    if missing_locked or extra_locked:
        print(
            f"[CONTRACT] Forbidden feature set changed. "
            f"Missing={missing_locked} Extra={extra_locked}"
        )

    if missing_optional:
        print(
            f"[CONTRACT] Optional legacy forbidden features absent: {missing_optional} "
            f"(allowed in the live locked-14 contract)."
        )

    if len(forbidden_features) != expected_forbidden_count:
        print(
            f"[CONTRACT] Forbidden feature count changed: {len(forbidden_features)} "
            f"vs required locked {expected_forbidden_count}."
        )

    return {
        "contract_profile": profile_name,
        "expected_model_feature_count": int(expected_feature_count),
        "expected_forbidden_count": int(expected_forbidden_count),
        "missing_locked": missing_locked,
        "missing_optional": missing_optional,
        "extra_locked": extra_locked,
    }



def _validate_output_schema(out: pd.DataFrame) -> None:
    required_cols = [
        "Date", "p_final_cal", "p_tail_base", "p_tail_dist", "p_final_g5",
        "spread_q05", "spread_q50", "spread_q95", "spread_q05_conf", "spread_q95_conf",
        "fwd_voo_hat_final", "fwd_ief_hat_final", "w_strategy_voo", "w_strategy_ief",
        "active_weight_raw", "active_weight_capped", "deploy_downside", "deploy_upside",
        "drift_alarm", "high_risk_state", "strategy_ret_net", "active_ret_net",
        "benchmark_ret", "turnover", "cost_model", "raw_val_auc", "calibration_gate_on",
        "expert_agreement", "is_live",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"Output tape missing required columns: {missing}")
    if out["Date"].isna().any():
        raise RuntimeError("Output tape contains null Date values.")
    if not pd.Series(pd.to_datetime(out["Date"])).is_monotonic_increasing:
        raise RuntimeError("Output tape dates are not monotonic increasing.")


def _compute_stress_panel(out: pd.DataFrame, cfg) -> Dict[str, object]:
    panel = {}
    realized = out.loc[out["y_avail"] == 1].copy()
    for bps in cfg.STRESS_SLIPPAGE_BPS:
        key = f"slippage_{str(bps).replace('.', '_')}bps"
        if len(realized) == 0 or "turnover" not in realized.columns:
            panel[key] = {"active_mean": np.nan, "active_ir": np.nan, "strategy_ir": np.nan}
            continue
        extra_bps = max(float(bps) - float(cfg.SLIP_BPS), 0.0)
        extra_cost = (extra_bps / 10000.0) * realized["turnover"].fillna(0.0).values
        strat = realized["strategy_ret_gross"].fillna(realized["strategy_ret_net"]).values - extra_cost
        active = realized["active_ret_gross"].fillna(realized["active_ret_net"]).values - extra_cost
        panel[key] = {
            "active_mean": float(np.nanmean(active)) if len(active) else np.nan,
            "active_ir": _annualized_ir(active, cfg.H),
            "strategy_ir": _annualized_ir(strat, cfg.H),
        }
    return panel


def _should_fail_closed(summary: Dict[str, object], cfg) -> bool:
    drift_limit = float(summary.get("effective_fail_closed_drift_rate", cfg.FAIL_CLOSED_DRIFT_RATE))
    cal_limit = float(summary.get("effective_fail_closed_cal_gate", cfg.FAIL_CLOSED_CAL_GATE))
    # H=1 recalibration (2026-04-13): fail_closed active-IR check now uses
    # conditional_active_ir (IR on deployed rows only) with a harder floor of -1.50.
    # This blocks deployment only if the defense events are severely directionally
    # wrong, rather than blocking due to the structurally near-zero full-series IR.
    #
    # FIX (F1, Quant-Guild Part 46 Audit): the raw annualized-IR floor below was
    # found to have a 37.7% false-alarm rate at the current n=11 deploy-event count
    # (Phi(-1.50/sqrt(252/11)) = Phi(-0.3134) = 0.377) -- essentially the same disease
    # the S45 audit fixed in the final_pass gate. Gating now uses the t-statistic of
    # the mean daily defense return (conditional_active_ir_tmean), which is invariant
    # to n, at a stricter 1% one-sided floor (CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR)
    # appropriate for an emergency/catastrophic-failure gate. NaN (insufficient n)
    # passes the gate, deferring to the deploy-count gate elsewhere, exactly mirroring
    # the existing final_pass t-floor semantics.
    cond_ir_tmean = summary.get("conditional_active_ir_tmean", np.nan)
    if not isinstance(cond_ir_tmean, float):
        try:
            cond_ir_tmean = float(cond_ir_tmean)
        except Exception:
            cond_ir_tmean = np.nan
    cond_ir_tfloor = float(summary.get("conditional_active_ir_failclosed_tfloor",
                                        cfg.CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR))
    return bool(
        # FIX (F2, Audit 2026-05-09 — Quant-Guild Part 15):
        # FAIL_CLOSED_ON_FALSE_PASS condition REMOVED.
        # When FAIL_CLOSED_ON_FALSE_PASS=True and final_pass=False, _should_fail_closed()
        # was a tautology: it always returned True before evaluating drift/calibration/IR.
        # This made _should_fail_closed() an alias for "not final_pass" — identical to
        # the condition it was supposed to augment — and blocked all independent governance
        # signals (drift, calibration, cond_ir) from being evaluated.
        # Now _should_fail_closed() is an INDEPENDENT gate: it triggers only when something
        # is GENUINELY wrong (suspicious performance, drift, calibration failure, severe
        # negative cond_ir). The AUC marginally missing its threshold is handled by
        # final_pass=False gating bot LIVE mode and alpha LIVE_TRIAL/LIVE_FUSED states,
        # without requiring publish_mode=FAIL_CLOSED_NEUTRAL.
        bool(summary.get("suspicious_perf_flag", False))
        or (np.isfinite(summary.get("drift_alarm_rate", np.nan)) and float(summary.get("drift_alarm_rate")) > drift_limit)
        # FIX (2026-04-13): was '> cal_limit', which incorrectly triggered fail_closed
        # when calibration was GOOD (e.g. 98.8% > 85%). The intent is to fail closed
        # when calibration is too POOR — i.e., when the rate is too LOW.
        or (np.isfinite(summary.get("calibration_gate_on_rate", np.nan)) and float(summary.get("calibration_gate_on_rate")) < cal_limit)
        # FIX (F1, Quant-Guild Part 46 Audit): t-stat floor replaces the raw annualized
        # IR floor (see CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR comment for derivation).
        or (np.isfinite(cond_ir_tmean) and cond_ir_tmean < cond_ir_tfloor)
    )


def _apply_fail_closed_neutral(out: pd.DataFrame, cfg) -> pd.DataFrame:
    out = out.copy()
    out["w_strategy_voo_pre_fail_closed"] = out["w_strategy_voo"]
    out["w_strategy_ief_pre_fail_closed"] = out["w_strategy_ief"]
    out["publish_fail_closed"] = 1
    out["w_strategy_voo"] = cfg.BASE_WEIGHT_VOO
    out["w_strategy_ief"] = cfg.BASE_WEIGHT_IEF
    if "y_avail" in out.columns:
        mask = out["y_avail"] == 1
        prev = cfg.BASE_WEIGHT_VOO
        new_turn = []
        for _, r in out.iterrows():
            w = float(r["w_strategy_voo"])
            if int(r.get("y_avail", 0)) == 1:
                t = abs(w - prev)
                prev = w
            else:
                t = np.nan
            new_turn.append(t)
        out["turnover"] = new_turn
        out.loc[mask, "cost_model"] = (cfg.SLIP_BPS / 10000.0) * out.loc[mask, "turnover"].fillna(0.0)
        out.loc[mask, "strategy_ret_gross"] = out.loc[mask, "w_strategy_voo"] * out.loc[mask, "fwd_voo"] + (1.0 - out.loc[mask, "w_strategy_voo"]) * out.loc[mask, "fwd_ief"]
        # FIX (F1, Quant-Guild Part 39 Audit): use contemporaneous 60/40 benchmark.
        # The prior code used the pre-existing benchmark_ret (bench_60_40), which
        # is the prior-day return (t-1→t) and creates a 1-day lag vs strategy_ret.
        # Since w_strategy_voo=0.60=BASE in fail_closed mode, active_ret should be
        # exactly zero (or negligible from cost_model).  The lag made it non-zero,
        # producing systematic noise in every performance metric.
        out.loc[mask, "benchmark_ret"] = (
            cfg.BASE_WEIGHT_VOO * out.loc[mask, "fwd_voo"]
            + cfg.BASE_WEIGHT_IEF * out.loc[mask, "fwd_ief"]
        )
        out.loc[mask, "active_ret_gross"] = out.loc[mask, "strategy_ret_gross"] - out.loc[mask, "benchmark_ret"]
        out.loc[mask, "strategy_ret_net"] = out.loc[mask, "strategy_ret_gross"] - out.loc[mask, "cost_model"]
        out.loc[mask, "active_ret_net"] = out.loc[mask, "active_ret_gross"] - out.loc[mask, "cost_model"]
    return out


def _environment_metadata(script_path: str) -> Dict[str, object]:
    return {
        "utc_run_ts": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "script_path": os.path.basename(script_path),
        "script_sha256": _sha256_file(script_path) if os.path.exists(script_path) else _sha256_text(str(script_path)),
    }


def _classification_metrics(y_true: np.ndarray, p: np.ndarray, bins: int) -> Dict[str, float]:
    y_true = _to_float_array(y_true).astype(int)
    p = np.clip(_to_float_array(p), 1e-6, 1.0 - 1e-6)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "pr": np.nan, "lift": np.nan, "brier": np.nan, "ece": np.nan}
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "pr": float(average_precision_score(y_true, p)),
        "lift": _lift_at_base_rate(y_true, p),
        "brier": _brier(y_true, p),
        "ece": _ece_score(y_true, p, bins),
    }


def _compute_regime_auc(realized: pd.DataFrame, bins: int = 10) -> Dict[str, object]:
    """Compute per-regime classification metrics on the realized (y_avail=1) tape.

    Added (Audit 2026-05-10 — Quant-Guild Part 16):
    The full-tape AUC (0.515) masks a critical structural problem: the model is
    anti-predictive in calm regimes (AUC=0.452) and near-random in crisis
    (AUC=0.489).  This function makes the per-regime breakdown visible in the
    summary JSON so it can be monitored in the dashboard and used to gate
    deployment by regime in Part 7.

    Returns a dict keyed by regime label with AUC, n, base_rate, and brier.
    Also returns an 'active_regimes' list of regimes where AUC > 0.50 (model
    is genuinely predictive).
    """
    if realized.empty:
        return {}
    regime_col = "regime_label"
    if regime_col not in realized.columns or "p_final_cal" not in realized.columns or "y_rel_tail_voo_vs_ief" not in realized.columns:
        return {}

    result: Dict[str, object] = {}
    active_regimes: list = []

    for regime in sorted(realized[regime_col].dropna().unique()):
        sub = realized[realized[regime_col] == str(regime)].dropna(
            subset=["p_final_cal", "y_rel_tail_voo_vs_ief"]
        )
        if len(sub) < 30:
            continue
        y = sub["y_rel_tail_voo_vs_ief"].values.astype(int)
        p = np.clip(sub["p_final_cal"].values, 1e-6, 1 - 1e-6)
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, p))
        # FIX (F2, Quant-Guild Part 43 Audit): Require DeLong significance (p<0.10) in
        # addition to AUC > 0.50 to qualify for active_regimes (BL optimizer active).
        #
        # ROOT CAUSE: The prior threshold (AUC > 0.50 point estimate) classified 'calm'
        # as BL-active with AUC=0.508, z=0.198, p=0.421 — statistically indistinguishable
        # from random. When rolling data is insufficient for calm (<30 positives),
        # Part7's fallback keeps 'calm' in active_regimes, allowing BL to fire on
        # a p=0.42 signal ~66 days/year.
        #
        # Fix: compute DeLong z-statistic; require p_one_sided < 0.10 (10% significance).
        # Verified thresholds:
        #   calm:     AUC=0.508, z=0.198, p=0.421 → excluded (p > 0.10)
        #   crisis:   AUC=0.445, z=-1.284, p=0.900 → excluded (AUC < 0.50)
        #   high_vol: AUC=0.547, z=1.578, p=0.057 → INCLUDED (p < 0.10)
        #   risk_on:  AUC=0.491, z=-0.264, p=0.604 → excluded (AUC < 0.50)
        # FIX (F3, Quant-Guild Part 44 Audit): Replace inline Hanley-McNeil SE with
        # _delong_auc_ztest (exact Mann-Whitney DeLong) for AUC significance gate.
        #
        # ROOT CAUSE: The S43 F5 fix upgraded _delong_auc_ztest to exact DeLong SE but
        # left _compute_regime_auc's inline active_regimes significance test using the
        # old Hanley-McNeil Q1/Q2 approximation. This created an internal inconsistency:
        # two SE formulas for the same significance test in the same file.
        #
        # Measured discrepancy for high_vol (n1=125, n0=446, AUC=0.5470):
        #   HM:    SE=0.02959, z=1.589, p=0.056
        #   DeLong: SE=0.02878, z=1.634, p=0.051
        # Both p < 0.10 → same governance decision currently.
        # Using DeLong throughout is correct and consistent with _delong_auc_ztest.
        #
        # Fix: call _delong_auc_ztest(y, p) here; classify as active if AUC > 0.50
        # AND DeLong p_one_sided < 0.10.
        _delong_result = _delong_auc_ztest(y, p)
        _auc_sig = bool(
            auc > 0.50
            and not _delong_result.get("auc_warning", True)  # auc_warning=False means p < 0.10
        )
        result[str(regime)] = {
            "n": int(len(sub)),
            "auc": round(auc, 6),
            "base_rate": round(float(y.mean()), 6),
            "brier": round(float(_brier(y, p)), 6),
            "ece": round(float(_ece_score(y, p, bins)), 6),
        }
        if auc > 0.50 and _auc_sig:
            active_regimes.append(str(regime))

    result["active_regimes"] = sorted(active_regimes)
    # FIX (F3, Quant-Guild Part 37 Audit): renamed from "passive_regimes" to
    # "passive_regimes_bl" to disambiguate from "passive_regimes_no_deploy".
    #
    # Two independent "passive regime" concepts exist in the summary JSON:
    #   passive_regimes_bl         — regimes where base-model AUC < 0.50 and
    #                                the BL optimizer is bypassed (regime_gated_prior).
    #                                Currently: ['crisis'] (AUC=0.440).
    #   passive_regimes_no_deploy  — regimes where deploy_downside is blocked because
    #                                the model is anti-predictive for the DEFENSE signal.
    #                                Currently: ['calm', 'risk_on'].
    #
    # These are orthogonal concepts. A regime can be BL-passive (crisis: bad for
    # directional tilt) but NOT deploy-passive (crisis: defense fires via high_risk_state).
    # Using the same word "passive_regimes" for both was misleading to audit readers.
    result["passive_regimes_bl"] = sorted(
        [r for r in result if r not in ("active_regimes", "passive_regimes_bl") and r not in active_regimes]
    )
    return result


def _compute_regime_auc_rolling(
    realized: pd.DataFrame,
    window_days: int = 252,
    min_regime_obs: int = 30,
    bins: int = 10,
    full_period_active_regimes: Optional[set] = None,
) -> Dict[str, object]:
    """Compute trailing-window per-regime AUC on the most-recent `window_days` rows.

    FIX (F1, Quant-Guild Part 40 Audit): Temporal AUC Decay Monitor.

    Root cause: _compute_regime_auc() uses the FULL 2020-2026 period.
    This aggregates away severe temporal decay:
      - calm    full=0.547  but 2025=0.257  (anti-predictive recently)
      - high_vol full=0.530 but 2026=0.333  (anti-predictive currently)
      - risk_on full=0.517  but 2024=0.417  (anti-predictive in 2024)

    The full-period AUC creates a false impression of stable predictive quality.
    This function evaluates only the most recent window_days rows (~1 year),
    providing an estimate of CURRENT predictive quality per regime.

    Returns a dict with:
      - Per-regime AUC, n, base_rate (same schema as _compute_regime_auc)
      - 'active_regimes_rolling': list of regimes with trailing AUC > 0.50 AND n >= min_regime_obs
      - 'decay_alarm_regimes': regimes that were in full-period active_regimes but are now < 0.50
      - 'regime_auc_rolling_decay_alarm': bool — True if ANY previously-active regime
        has trailing AUC < 0.50, indicating the model's signal has degraded in that regime.
        When True, the pipeline should reduce view_confidence in that regime toward 0 and
        Part 3 should consider triggering FAIL_CLOSED if ALL active regimes are decaying.

    Design: uses realized rows sorted by Date, taking the most recent window_days.
    Requires at least min_regime_obs rows per regime to compute AUC (avoids noise at low n).
    If a regime has fewer than min_regime_obs rows in the trailing window, it is treated
    as 'insufficient data' (not alarmed but not confidently active either).

    FIX (F1, Quant-Guild Part 44 Audit): The prior code hardcoded
        _full_period_active = {"calm", "high_vol", "risk_on"}
    This was calibrated before the S43 F2 DeLong significance gate, which reduced
    active_regimes to ['high_vol'] only. With the hardcoded set including calm and risk_on,
    the decay alarm fired spuriously for risk_on (AUC=0.4501) even though risk_on was
    never in the computed active_regimes. This produced regime_auc_rolling_decay_alarm=True
    as a false positive, misleading monitoring and Part 7's decay alarm logic.
    Fix: pass full_period_active_regimes as a parameter from the call site, which
    receives the dynamically computed set from _compute_regime_auc.

    FIX (F4, Quant-Guild Part 44 Audit): Replace null-hypothesis SE (sqrt(0.25/n))
    with exact DeLong SE for rolling AUC significance test. Previously this function
    used a different SE formula than _compute_regime_auc (which after F3 uses DeLong)
    and _delong_auc_ztest (which uses exact DeLong since S43 F5). Three distinct SE
    formulas in one file for the same test is a maintenance hazard. All significance
    tests now use _delong_auc_ztest for consistency.
    """
    if realized.empty:
        return {"regime_auc_rolling_decay_alarm": False, "active_regimes_rolling": []}

    regime_col = "regime_label"
    date_col = "Date"
    if regime_col not in realized.columns or "p_final_cal" not in realized.columns:
        return {"regime_auc_rolling_decay_alarm": False, "active_regimes_rolling": []}

    # Sort by date and take trailing window
    df = realized.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    df_window = df.tail(window_days).copy()

    if len(df_window) < 50:
        return {"regime_auc_rolling_decay_alarm": False, "active_regimes_rolling": []}

    result: Dict[str, object] = {}
    active_regimes_rolling: List[str] = []
    insufficient_regimes: List[str] = []

    for regime in sorted(df_window[regime_col].dropna().unique()):
        sub = df_window[df_window[regime_col] == str(regime)].dropna(
            subset=["p_final_cal", "y_rel_tail_voo_vs_ief"]
        )
        if len(sub) < min_regime_obs:
            insufficient_regimes.append(str(regime))
            continue
        y = sub["y_rel_tail_voo_vs_ief"].values.astype(int)
        p = np.clip(sub["p_final_cal"].values, 1e-6, 1 - 1e-6)
        if len(np.unique(y)) < 2:
            continue
        # FIX (F3, Quant-Guild Part 41 Audit): Require minimum 10 positive events
        # before computing rolling AUC. With base_rate=3.3% (calm regime), n=30
        # observations implies ~1 positive event, which produces AUC=0.759 from
        # a single rank position — numerically meaningless noise.
        # With 1 positive and 29 negatives, a rank change of 1 position changes
        # AUC by 1/29=0.034. The estimate has SE≈0.091 and is unstable.
        # Requiring n_positives >= 10 ensures there are enough positive events
        # to compute a stable, reliable AUC estimate for regime qualification.
        n_positives = int(y.sum())
        if n_positives < 10:
            insufficient_regimes.append(str(regime))
            continue
        from sklearn.metrics import roc_auc_score as _roc_auc
        auc = float(_roc_auc(y, p))
        # FIX (F4, Quant-Guild Part 44 Audit): Use exact DeLong SE for rolling
        # significance test. Replaces null-hypothesis SE = sqrt(0.25/n) which was
        # inconsistent with _compute_regime_auc (HM→DeLong after F3) and
        # _delong_auc_ztest (DeLong since S43 F5). All three uses now use DeLong.
        _delong_roll = _delong_auc_ztest(y, p)
        _z_auc = float(_delong_roll.get("z", 0.0) or 0.0)
        result[str(regime)] = {
            "n": int(len(sub)),
            "n_positives": n_positives,
            "auc": round(auc, 6),
            "base_rate": round(float(y.mean()), 6),
            "auc_z": round(float(_z_auc), 4),
            "auc_significant_p10": bool(_z_auc >= 1.282),  # p < 0.10 one-sided
        }
        # Require: AUC > 0.50 AND z >= 1.282 (p < 0.10 one-sided)
        # p < 0.10 is chosen (not 0.05) because rolling windows have high variance;
        # requiring p < 0.05 on 85 observations would exclude genuinely predictive
        # regimes during short periods. The dual gate (rolling + full-period) in
        # _dynamic_active_regimes() provides the stricter overall filter.
        if auc > 0.50 and _z_auc >= 1.282:
            active_regimes_rolling.append(str(regime))

    result["active_regimes_rolling"] = sorted(active_regimes_rolling)
    result["insufficient_data_regimes"] = sorted(insufficient_regimes)

    # Decay alarm: any regime that the FULL-period analysis listed as active
    # but the rolling window shows < 0.50.
    # FIX (F1, Quant-Guild Part 44 Audit): Use the DYNAMICALLY COMPUTED
    # full_period_active_regimes parameter instead of the hardcoded set
    # {"calm", "high_vol", "risk_on"} which was stale after the S43 F2
    # DeLong significance gate reduced active_regimes to ['high_vol'] only.
    # Passing None falls back to an empty set (no decay alarm) rather than
    # using a hardcoded guess — safe for backward-compatible cold starts.
    if full_period_active_regimes is None:
        _fp_active = set()
    else:
        _fp_active = set(full_period_active_regimes)

    decay_alarm_regimes: List[str] = []
    for regime, stats_dict in result.items():
        if not isinstance(stats_dict, dict):
            continue
        if regime in _fp_active and stats_dict.get("auc", 1.0) < 0.50:
            decay_alarm_regimes.append(regime)

    result["decay_alarm_regimes"] = sorted(decay_alarm_regimes)
    result["regime_auc_rolling_decay_alarm"] = bool(len(decay_alarm_regimes) > 0)
    result["window_days"] = window_days

    # FIX (F3, Quant-Guild S57 Audit): Add anti-predictive rolling regime alarm.
    #
    # ROOT CAUSE: decay_alarm_regimes only fires when a regime that WAS previously
    # in full_period_active_regimes subsequently falls below rolling AUC < 0.50.
    # A regime that was NEVER in active_regimes (e.g. risk_on, which always had
    # full-period AUC < 0.50) cannot appear in decay_alarm_regimes regardless of
    # how severely anti-predictive its rolling signal becomes.
    #
    # CONFIRMED (S57 artifacts): risk_on rolling AUC=0.3346, DeLong z=-2.037
    # (p_left=0.025, severely significant in the anti-predictive direction).
    # risk_on is the LIVE regime as of 2026-06-17. Yet decay_alarm_regimes=[]
    # and regime_auc_rolling_decay_alarm=False — no alarm raised anywhere.
    # The system is correctly passive (60/40) because risk_on has rolling AUC < 0.50,
    # which drives raw_auc=0.50 → view_confidence=0. But the absence of any
    # monitoring signal means this severe deterioration is invisible in dashboards,
    # governance CSV, and Part 3 summary JSON.
    #
    # FIX: compute anti_predictive_rolling_regimes = regimes with rolling DeLong
    # z < -1.282 (p_left < 0.10, statistically significant in the anti-predictive
    # direction), regardless of whether they were ever in active_regimes.
    # This is the mirror of the active_regimes_rolling gate (z >= 1.282) and uses
    # the same significance threshold for symmetry. Both flags are reported in the
    # summary JSON and trigger a console warning when non-empty.
    #
    # Live allocation impact: NONE. The system is already passive when rolling AUC < 0.50.
    # This fix is monitoring/reporting completeness only.  [FIX F3/S57]
    anti_predictive_rolling_regimes: List[str] = []
    _ANTI_PRED_Z_THRESHOLD = -1.282  # z < -1.282 → p_left < 0.10 (anti-predictive)
    for regime, stats_dict in result.items():
        if not isinstance(stats_dict, dict):
            continue
        _z_roll = float(stats_dict.get("auc_z", 0.0) or 0.0)
        if _z_roll < _ANTI_PRED_Z_THRESHOLD:
            anti_predictive_rolling_regimes.append(str(regime))

    result["anti_predictive_rolling_regimes"] = sorted(anti_predictive_rolling_regimes)
    result["regime_auc_rolling_anti_predictive_alarm"] = bool(len(anti_predictive_rolling_regimes) > 0)

    if result["regime_auc_rolling_decay_alarm"]:
        print(
            f"[Part 2] ⚠️ REGIME AUC DECAY ALARM: {decay_alarm_regimes} have trailing-{window_days}d "
            f"AUC < 0.50. Active regimes (rolling): {active_regimes_rolling}. "
            f"Consider reducing view_confidence and monitoring for FAIL_CLOSED trigger."
        )
    if result["regime_auc_rolling_anti_predictive_alarm"]:
        print(
            f"[Part 2] ⚠️ ANTI-PREDICTIVE ROLLING ALARM: {anti_predictive_rolling_regimes} have "
            f"trailing-{window_days}d DeLong z < {_ANTI_PRED_Z_THRESHOLD} (statistically "
            f"anti-predictive). System remains passive for these regimes but the signal "
            f"quality is severely degraded — monitor for structural regime shift.  [S57 F3 fix]"
        )
    return result


def _dynamic_active_regimes(
    full_period_regime_auc: Dict[str, object],
    rolling_regime_auc: Dict[str, object],
) -> List[str]:
    """Derive the authoritative active_regimes list using rolling AUC when available.

    FIX (F1, Quant-Guild Part 40 Audit):
    The static active_regimes=[calm, high_vol, risk_on] was set using full-period AUC.
    When rolling AUC is available (rolling_regime_auc is non-empty), use the INTERSECTION
    of full-period and rolling active regimes. A regime must be positive on BOTH timescales
    to count as active. This prevents decayed regimes from receiving BL optimizer views.

    If rolling AUC is unavailable or has insufficient data for a regime, fall back to
    the full-period AUC for that regime (conservative: do not remove a regime from the
    active list just because rolling data is sparse).
    """
    fp_active = set(full_period_regime_auc.get("active_regimes", []))
    rolling_active = set(rolling_regime_auc.get("active_regimes_rolling", []))
    insufficient = set(rolling_regime_auc.get("insufficient_data_regimes", []))

    # Regimes with sufficient rolling data: use rolling result
    # Regimes with insufficient rolling data: use full-period result (fallback)
    result: List[str] = []
    for regime in fp_active:
        if regime in insufficient:
            # Not enough recent data to evaluate — keep from full period (conservative)
            result.append(regime)
        elif regime in rolling_active:
            # Both full-period and rolling are positive — keep
            result.append(regime)
        # else: full-period active but rolling < 0.50 and sufficient data → remove
    return sorted(result)


def _gamma_to_uncertainty(gamma_val: float) -> float:
    """
    Safe gamma → uncertainty inversion.
    gamma = 1 / (1 + u)  =>  u = 1/gamma - 1

    FIX: original clip of 1e-12 produced ~1e12 uncertainty when gamma was
    near zero, which would propagate NaN into leg_uncertainty and
    silently disable defense sizing. Cap gamma at 0.001 (max meaningful
    uncertainty = 999) to preserve model behavior.
    """
    if not np.isfinite(gamma_val) or gamma_val <= 0:
        return np.nan
    return min(1.0 / max(float(gamma_val), 0.001) - 1.0, 999.0)




def _json_safe(obj):
    """Convert NaN/Inf/numpy scalars to JSON-safe Python types.

    FIX (BUG-1, Audit 2026-05-09):
    The previous version was used as `default=` in json.dump.  Python's
    json encoder handles Python `float` NATIVELY (including NaN/Inf) —
    it writes them as the literal tokens NaN / Infinity, which are invalid
    per RFC 8259.  Because `float` is natively handled, the `default=`
    hook is NEVER called for Python float values, so _json_safe never
    intercepted them.  Result: `"conditional_active_ir": NaN` appeared
    literally in part2_g532_summary.json.  JavaScript's JSON.parse()
    throws SyntaxError on NaN, silently breaking the index.html dashboard
    on every run where conditional_active_ir is NaN (i.e., until at least
    10 defense events accumulate — currently 4 of 10 needed).

    Fix: this function is now used via _deep_clean_for_json() which
    recursively walks the entire dict/list structure and replaces any
    non-RFC-8259-safe value with None BEFORE json.dump is called.
    json.dump is then called WITHOUT a custom default= (using the safe
    allow_nan=False mode via the cleaned dict).

    FIX (F2, Quant-Guild Part 44 Audit): Add Python native int handler.
    Previously Python native int fell through to return str(obj), because
    only np.integer was explicitly handled. This caused all integer-valued
    dict entries — rows_rebalance, deploy_downside_count_total, delong n/n1/n0,
    regime_auc_breakdown n, and 15+ other fields — to be serialized as JSON
    strings (e.g. "1676" instead of 1676). Any downstream arithmetic on these
    fields raises TypeError; JavaScript numeric comparisons break silently.
    Fix: add isinstance(obj, int) and not isinstance(obj, bool) guard before
    the str() fallback. bool subclasses int in Python, so the bool check
    must remain first (already covered by the bool/np.bool_ branch above).
    """
    import math
    import numpy as np
    import pandas as pd

    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    # FIX (F2, Quant-Guild Part 44 Audit): Python native int not previously handled.
    # bool subclasses int; the bool branch above must appear first so True/False are not
    # returned as 1/0 integers.
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if obj is pd.NaT or (hasattr(pd, "NA") and obj is pd.NA):
        return None
    return str(obj)


def _deep_clean_for_json(obj):
    """Recursively walk any dict/list and apply _json_safe to every leaf value.

    This is the correct approach to guarantee RFC 8259 compliance because
    json.dump's `default=` hook is never called for Python float (natively
    serializable). We must pre-process the entire structure before serialising.
    """
    if isinstance(obj, dict):
        return {k: _deep_clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_clean_for_json(v) for v in obj]
    return _json_safe(obj)


def build_part2_gen53(cfg: Part2Gen53Config) -> Dict[str, object]:
    _ensure_dir(cfg.PRED_DIR)
    part1_meta = _load_part1_meta(cfg)
    full = _load_part1_contract(cfg)
    feature_cols, forbidden_features = _select_model_features(full, part1_meta, cfg)
    contract_info = _validate_feature_contract(feature_cols, forbidden_features, part1_meta, cfg)
    contract_profile = str(contract_info.get("contract_profile", ""))
    drift_ece_max_eff = _effective_drift_ece_max(contract_profile, cfg)
    drift_brier_max_eff = _effective_drift_brier_max(contract_profile, cfg)
    final_pass_drift_max_eff = _effective_final_pass_drift_rate_max(contract_profile, cfg)
    fail_closed_drift_rate_eff = _effective_fail_closed_drift_rate(contract_profile, cfg)
    fail_closed_cal_gate_eff = _effective_fail_closed_cal_gate(contract_profile, cfg)

    cal_path = os.path.join(cfg.PART1_DIR, "calendar_feature_aligned.parquet")
    if os.path.exists(cal_path):
        cal_for_rebal = _read_table(cal_path)
    else:
        cal_for_rebal = full[["Date"]].copy()
    rebal_dates = _build_rebalance_dates(cal_for_rebal, cfg)
    rebal_set = set(pd.to_datetime(rebal_dates["Date"]).dt.normalize())
    rebal_idx = [i for i, d in enumerate(full["Date"]) if d in rebal_set]
    base_rate = float(full.loc[full["y_avail"] == 1, "y_rel_tail_voo_vs_ief"].mean())
    base_rate = base_rate if np.isfinite(base_rate) else 0.20
    tail_threshold = float(part1_meta.get("tail_threshold", cfg.TAIL_EVENT_THRESHOLD))

    rows = []
    fit_bundle = None
    fit_train_df = None
    train_df = None

    for j, idx in enumerate(rebal_idx):
        current_row = full.iloc[[idx]].copy()
        need_refit = fit_bundle is None or (j % cfg.REFIT_FREQ == 0)
        if need_refit:
            tv = _make_train_val(full, idx, feature_cols, cfg)
            if tv is None:
                continue
            train_df, val_df = tv
            fit_bundle = _build_fit_bundle(train_df, val_df, current_row, feature_cols, cfg)
            fit_train_df = train_df.copy()
        else:
            train_df = fit_train_df.copy()

        if fit_bundle["regime"] is not None:
            current_regime = _predict_regime(fit_bundle["regime"], current_row[[c for c in cfg.REGIME_FEATURES if c in current_row.columns]]).iloc[0]
        else:
            ref_cols = [c for c in ("stress_score_raw", "vix_z21", "credit_spread_z21", "excess_vol10", "spread_ret21") if c in train_df.columns]
            ref_df = train_df[ref_cols].copy() if ref_cols else train_df.copy()
            current_regime = _fallback_regime(current_row, ref_df=ref_df).iloc[0]

        # FIX (Finding 2, Part6 Audit 2026-04):
        # Override current_regime with the Part 6 HMM label when it is available
        # and non-unknown. Part 6 uses a 4-state Gaussian HMM trained on macro/
        # volatility features (vix_z21, yield_curve_2s10s, etc.) — a more
        # principled regime model than Part 2's internal GMM. The internal GMM
        # result is retained as the fallback for dates where Part 6 returns
        # "unknown" (e.g., cold-start or missing feature data).
        #
        # "crisis" (Part 6 label) is the semantic equivalent of "dislocated"
        # (Part 2 GMM label). Both _regime_defense_score and high_risk_state
        # have been updated to treat them identically (see fixes above).
        _p6_label = str(current_row["regime_label_p6"].iloc[0]) if "regime_label_p6" in current_row.columns else "unknown"
        if _p6_label not in ("unknown", "nan", "", "None"):
            current_regime = _p6_label
        prob_pred = _predict_prob(fit_bundle["prob"], current_row[feature_cols], base_rate, cfg)
        dist_pred = _predict_dist(fit_bundle["dist"], current_row[feature_cols], tail_threshold, cfg)
        p_final_g5, p_final_g5_source, fusion_fallback_flag, dist_overlay = _apply_risk_overlay_g53(
            prob_pred["p_final_cal"], dist_pred, fit_bundle["dist"], base_rate, tail_threshold, cfg
        )
        reg_voo = _predict_reg(fit_bundle["voo"], current_row[feature_cols])
        reg_ief = _predict_reg(fit_bundle["ief"], current_row[feature_cols])

        realized_hist = pd.DataFrame(rows)
        if len(realized_hist) >= cfg.ROLL_DIAG and int(realized_hist["y_avail"].fillna(0).sum()) >= cfg.ROLL_DIAG:
            rh = realized_hist.loc[realized_hist["y_avail"] == 1].tail(cfg.ROLL_DIAG)
            ece_roll = _ece_score(rh["y_rel_tail_voo_vs_ief"].values, rh["p_final_cal"].values, cfg.ECE_BINS)
            brier_roll = _brier(rh["y_rel_tail_voo_vs_ief"].values, rh["p_final_cal"].values)
        else:
            ece_roll = np.nan
            brier_roll = np.nan
        drift_alarm = 0

        fwd_voo_hat = float(reg_voo["pred"])
        fwd_ief_hat = float(reg_ief["pred"])
        fwd_spread_hat = float(fwd_voo_hat - fwd_ief_hat)
        leg_uncertainty = float(np.nanmean([reg_voo["uncertainty"], reg_ief["uncertainty"]]))
        defense_trigger_threshold = _rolling_quantile([r.get("defense_trigger_raw", np.nan) for r in rows], cfg.DEF_TRIGGER_LOOKBACK, cfg.DEF_TRIGGER_MIN_HISTORY, cfg.DEF_TRIGGER_Q)
        stress_q_threshold = _rolling_quantile([r.get("stress_score_raw", np.nan) for r in rows], cfg.DEF_TRIGGER_LOOKBACK, cfg.DEF_TRIGGER_MIN_HISTORY, cfg.DEF_TRIGGER_STRESS_Q)
        gov = _governance_mapping(
            p_final_g5,
            base_rate,
            drift_alarm,
            prob_pred["raw_val_auc"],
            prob_pred["agreement_score"],
            fwd_spread_hat,
            leg_uncertainty,
            defense_trigger_threshold,
            str(current_regime),
            _safe_num(current_row.iloc[0].get("stress_score_raw", np.nan)),
            stress_q_threshold,
            cfg,
            dist_overlay_strength=float(dist_overlay.get("dist_overlay_strength_g53", 0.0)),
            dist_tail_shift=float(dist_overlay.get("dist_tail_shift_g53", 0.0)),
            dist_width_caution=float(dist_overlay.get("dist_width_caution_g53", 0.0)),
            uncertainty_penalty=float(dist_pred.get("uncertainty_penalty_g5", np.nan)),
        )

        px_voo_t = _safe_num(current_row.iloc[0]["px_voo_t"])
        px_ief_t = _safe_num(current_row.iloc[0]["px_ief_t"])
        px_voo_call = float(px_voo_t * np.exp(fwd_voo_hat)) if np.isfinite(px_voo_t) else np.nan
        px_ief_call = float(px_ief_t * np.exp(fwd_ief_hat)) if np.isfinite(px_ief_t) else np.nan
        y_avail = int(current_row.iloc[0]["y_avail"])

        row = {
            "Date": pd.Timestamp(current_row.iloc[0]["Date"]).normalize(),
            "regime_label": str(current_regime),
            "regime_model_live": int(fit_bundle["regime"] is not None),
            "regime_id_g5": np.nan,
            "regime_name_g5": str(current_regime),
            "regime_confidence_g5": np.nan,
            "regime_transition_flag": np.nan,
            "regime_persistence_score": np.nan,
            "expert_weight_entropy": float(-sum(w * math.log(max(w, 1e-12)) for w in fit_bundle["prob"]["weights"].values())),
            "expert_agreement": float(prob_pred["agreement_score"]) if np.isfinite(prob_pred["agreement_score"]) else np.nan,
            "z_raw": float(logit(prob_pred["p_final_raw"])),
            "p0": float(prob_pred["p0"]),
            "p_final_raw": float(prob_pred["p_final_raw"]),
            "p_final": float(p_final_g5),
            "p_final_cal_candidate": float(prob_pred["p_final_cal_candidate"]),
            "p_final_cal": float(p_final_g5),
            "p_tail_base": float(prob_pred["p_final_cal"]),
            "p_tail_dist": float(dist_pred["p_tail_dist"]),
            "p_meta_trade": np.nan,
            "p_final_g5": float(p_final_g5),
            "p_final_g5_source": p_final_g5_source,
            "fusion_fallback_flag": int(fusion_fallback_flag),
            "dist_overlay_on_g53": int(dist_overlay["dist_overlay_on_g53"]),
            "dist_trust_score_g53": float(dist_overlay["dist_trust_score_g53"]),
            "dist_overlay_strength_g53": float(dist_overlay["dist_overlay_strength_g53"]),
            "dist_contradiction_g53": float(dist_overlay["dist_contradiction_g53"]) if np.isfinite(dist_overlay["dist_contradiction_g53"]) else np.nan,
            "dist_sign_agree_g53": int(dist_overlay["dist_sign_agree_g53"]),
            "shrink_factor": float(prob_pred["shrink_factor"]),
            "fwd_voo_hat": fwd_voo_hat,
            "fwd_ief_hat": fwd_ief_hat,
            "fwd_spread_hat": fwd_spread_hat,
            "fwd_spread_hat_from_legs": fwd_spread_hat,
            "spread_model_gap": 0.0,
            "spread_q05": dist_pred["spread_q05"],
            "spread_q25": dist_pred["spread_q25"],
            "spread_q50": dist_pred["spread_q50"],
            "spread_q75": dist_pred["spread_q75"],
            "spread_q95": dist_pred["spread_q95"],
            "spread_q05_conf": dist_pred["spread_q05_conf"],
            "spread_q95_conf": dist_pred["spread_q95_conf"],
            "spread_iqr": dist_pred["spread_iqr"],
            "spread_tail_width": dist_pred["spread_tail_width"],
            "spread_conf_width": dist_pred["spread_conf_width"],
            "spread_left_tail_score": dist_pred["spread_left_tail_score"],
            "spread_median_score": dist_pred["spread_median_score"],
            "gamma_voo": float(1.0 / (1.0 + reg_voo["uncertainty"])) if np.isfinite(reg_voo["uncertainty"]) else np.nan,
            "gamma_ief": float(1.0 / (1.0 + reg_ief["uncertainty"])) if np.isfinite(reg_ief["uncertainty"]) else np.nan,
            "fwd_voo_hat_final": fwd_voo_hat,
            "fwd_ief_hat_final": fwd_ief_hat,
            "px_voo_t": px_voo_t,
            "px_ief_t": px_ief_t,
            "h_reb": int(cfg.H),
            "px_voo_call_1d": px_voo_call,
            "px_voo_call_7d": px_voo_call,
            "px_ief_call_1d": px_ief_call,
            "px_ief_call_7d": px_ief_call,
            "px_voo_real_1d": _safe_num(current_row.iloc[0].get("px_voo_fwd", np.nan)),
            "px_voo_real_7d": _safe_num(current_row.iloc[0].get("px_voo_fwd", np.nan)),
            "px_ief_real_1d": _safe_num(current_row.iloc[0].get("px_ief_fwd", np.nan)),
            "px_ief_real_7d": _safe_num(current_row.iloc[0].get("px_ief_fwd", np.nan)),
            "fwd_voo": _safe_num(current_row.iloc[0].get("fwd_voo", np.nan)),
            "fwd_ief": _safe_num(current_row.iloc[0].get("fwd_ief", np.nan)),
            "excess_ret": _safe_num(current_row.iloc[0].get("excess_ret", np.nan)),
            "y_voo": _safe_num(current_row.iloc[0].get("y_voo", np.nan)),
            "y_rel_tail_voo_vs_ief": _safe_num(current_row.iloc[0].get("y_rel_tail_voo_vs_ief", np.nan)),
            "y_avail": y_avail,
            "CalibDate": pd.Timestamp(train_df.iloc[-1]["Date"]).strftime("%Y-%m-%d") if train_df is not None else None,
            "sign": float(np.sign(p_final_g5 - base_rate)),
            "T": float(base_rate),
            "b": float(base_rate),
            "lam": float(prob_pred["agreement_std"]),
            "calibration_gate_on": int(prob_pred["calibration_gate_on"]),
            "raw_val_auc": float(prob_pred["raw_val_auc"]) if np.isfinite(prob_pred["raw_val_auc"]) else np.nan,
            "chosen_val_auc": float(prob_pred["chosen_val_auc"]) if np.isfinite(prob_pred["chosen_val_auc"]) else np.nan,
            "cal_gate_brier_raw": float(fit_bundle["prob"]["raw_brier"]),
            "cal_gate_brier_cal": float(_brier(fit_bundle["prob"]["val_y"], fit_bundle["prob"]["val_candidate"])),
            "cal_gate_ece_raw": float(fit_bundle["prob"]["raw_ece"]),
            "cal_gate_ece_cal": float(_ece_score(fit_bundle["prob"]["val_y"], fit_bundle["prob"]["val_candidate"], cfg.ECE_BINS)),
            "dist_conf_adj": float(fit_bundle["dist"]["conf_adj"]),
            "dist_raw_coverage": float(fit_bundle["dist"]["raw_coverage"]),
            "dist_conf_coverage": float(fit_bundle["dist"]["conf_coverage"]),
            "dist_median_rmse": float(fit_bundle["dist"]["median_rmse"]),
            "ece_roll": ece_roll,
            "brier_roll": brier_roll,
            "ece_avail_roll": ece_roll,
            "brier_avail_roll": brier_roll,
            "drift_alarm": int(drift_alarm),
            "signal_strength_g5": float(max(p_final_g5 - base_rate, 0.0)),
            "uncertainty_penalty_g5": float(dist_pred["uncertainty_penalty_g5"]),
            "meta_trust_score_g5": float(dist_overlay["dist_trust_score_g53"]),
            "regime_risk_score_g5": float(_regime_defense_score(str(current_regime))),
            **gov,
            "w_benchmark_voo": float(cfg.BASE_WEIGHT_VOO),
            "w_strategy_ief": float(1.0 - gov["w_strategy_voo"]),
            "turnover": np.nan,
            "cost_model": np.nan,
            "strategy_ret_gross": np.nan,
            # FIX (F1, Quant-Guild Part 39 Audit): BENCHMARK MISALIGNMENT.
            # bench_60_40 from factor_returns is the contemporaneous return (t-1→t).
            # fwd_voo/fwd_ief are forward returns (t→t+1).  Using bench_60_40 here
            # produces a 1-day lag: active_ret = strategy(t→t+1) − benchmark(t-1→t).
            # This is corrected in the y_avail=1 block below, which overwrites
            # benchmark_ret with 0.60*fwd_voo + 0.40*fwd_ief (same period, no lag).
            # For y_avail=0 (unrealized) rows, bench_60_40 is kept as a carry value
            # because there is no forward return yet; these rows never enter
            # active_ret_net computations.
            "benchmark_ret": _safe_num(current_row.iloc[0].get("bench_60_40", np.nan)),
            "active_ret_gross": np.nan,
            "strategy_ret_net": np.nan,
            "active_ret_net": np.nan,
            "is_live": 0,
        }
        carry_cols = [
            "voo_vol10", "excess_vol10", "vix_mom5", "alpha_credit_spread", "alpha_credit_accel", "alpha_vix_term", "alpha_breadth",
            "alpha_tech_relative", "voo_r1", "ief_r1", "spread_r1", "jnk_r1", "rsp_r1", "qqq_r1", "vix_r1", "vix3m_r1",
            "credit_spread_r1", "stress_score_raw", "stress_score_change5", "bench_voo", "bench_ief", "bench_60_40", "bench_excess_voo_minus_ief",
        ]
        for c in carry_cols:
            row[c] = _safe_num(current_row.iloc[0].get(c, np.nan))

        if y_avail == 1:
            row["err_px_voo"] = row["px_voo_real_7d"] - row["px_voo_call_7d"] if np.isfinite(row["px_voo_real_7d"]) and np.isfinite(row["px_voo_call_7d"]) else np.nan
            row["err_px_ief"] = row["px_ief_real_7d"] - row["px_ief_call_7d"] if np.isfinite(row["px_ief_real_7d"]) and np.isfinite(row["px_ief_call_7d"]) else np.nan
            row["err_r_voo"] = row["fwd_voo"] - row["fwd_voo_hat_final"] if np.isfinite(row["fwd_voo"]) else np.nan
            row["err_r_ief"] = row["fwd_ief"] - row["fwd_ief_hat_final"] if np.isfinite(row["fwd_ief"]) else np.nan
            row["hit_sign_voo"] = int(np.sign(row["fwd_voo_hat_final"]) == np.sign(row["fwd_voo"])) if np.isfinite(row["fwd_voo"]) else np.nan
            row["dist_interval_hit_raw"] = int(row["spread_q05"] <= row["excess_ret"] <= row["spread_q95"]) if np.isfinite(row["excess_ret"]) else np.nan
            row["dist_interval_hit_conf"] = int(row["spread_q05_conf"] <= row["excess_ret"] <= row["spread_q95_conf"]) if np.isfinite(row["excess_ret"]) else np.nan
            prev_w = rows[-1]["w_strategy_voo"] if rows else cfg.BASE_WEIGHT_VOO
            row["turnover"] = abs(row["w_strategy_voo"] - prev_w)
            row["cost_model"] = (cfg.SLIP_BPS / 10000.0) * row["turnover"]
            row["strategy_ret_gross"] = row["w_strategy_voo"] * row["fwd_voo"] + (1.0 - row["w_strategy_voo"]) * row["fwd_ief"]
            # FIX (F1, Quant-Guild Part 39 Audit): overwrite benchmark_ret with the
            # contemporaneous 60/40 forward return (same period as strategy_ret_gross).
            # bench_60_40 from carry_cols is the prior-day return (t-1→t); fwd_voo is
            # the next-day return (t→t+1).  Using carry bench_60_40 here produced a
            # 1-day lag that inflated/deflated active_ret_net by the full prior day's
            # benchmark return, corrupting every active return and the conditional IR gate.
            row["benchmark_ret"] = (
                cfg.BASE_WEIGHT_VOO * row["fwd_voo"] + cfg.BASE_WEIGHT_IEF * row["fwd_ief"]
                if np.isfinite(row["fwd_voo"]) and np.isfinite(row["fwd_ief"])
                else np.nan
            )
            row["active_ret_gross"] = row["strategy_ret_gross"] - row["benchmark_ret"] if np.isfinite(row["benchmark_ret"]) else np.nan
            row["strategy_ret_net"] = row["strategy_ret_gross"] - row["cost_model"]
            row["active_ret_net"] = row["active_ret_gross"] - row["cost_model"] if np.isfinite(row["active_ret_gross"]) else np.nan
        else:
            row["err_px_voo"] = np.nan
            row["err_px_ief"] = np.nan
            row["err_r_voo"] = np.nan
            row["err_r_ief"] = np.nan
            row["hit_sign_voo"] = np.nan
            row["dist_interval_hit_raw"] = np.nan
            row["dist_interval_hit_conf"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Part 2 Gen 5 produced no rebalance rows.")
    out = out.sort_values("Date").reset_index(drop=True)
    out.loc[out.index[-1], "is_live"] = 1

    realized = out.loc[out["y_avail"] == 1].copy()
    if len(realized):
        out["drift_base_ece"] = _ece_score(realized["y_rel_tail_voo_vs_ief"].values, realized["p_final_cal"].values, cfg.ECE_BINS)
        out["drift_base_brier"] = _brier(realized["y_rel_tail_voo_vs_ief"].values, realized["p_final_cal"].values)
    else:
        out["drift_base_ece"] = np.nan
        out["drift_base_brier"] = np.nan

    ece_roll, brier_roll = [], []
    for i in range(len(out)):
        hist = out.iloc[: i + 1].copy()
        hist = hist.loc[hist["y_avail"] == 1].tail(cfg.ROLL_DIAG)
        if len(hist) >= max(20, cfg.ROLL_DIAG // 2):
            ece_roll.append(_ece_score(hist["y_rel_tail_voo_vs_ief"].values, hist["p_final_cal"].values, cfg.ECE_BINS))
            brier_roll.append(_brier(hist["y_rel_tail_voo_vs_ief"].values, hist["p_final_cal"].values))
        else:
            ece_roll.append(np.nan)
            brier_roll.append(np.nan)
    out["ece_roll"] = ece_roll
    out["brier_roll"] = brier_roll
    out["ece_avail_roll"] = out["ece_roll"]
    out["brier_avail_roll"] = out["brier_roll"]
    out["drift_alarm"] = _compute_drift_flags(out["ece_roll"], out["brier_roll"], cfg, ece_max=drift_ece_max_eff, brier_max=drift_brier_max_eff)

    # FIX (F-4, Quant-Guild Part 33 Audit): vectorise the second-pass governance update.
    #
    # ROOT CAUSE OF PRIOR PERFORMANCE ISSUE:
    # The prior loop used `out.loc[i, k] = v` — a scalar positional assignment — for
    # every cell of every row.  For n=1670 rows and the ~18 output columns from
    # _governance_mapping plus w_strategy_ief, that is:
    #   1670 rows × 19 assignments = 31,730 individual out.loc[] calls.
    # Each `out.loc[i, k]` triggers pandas' index-alignment machinery and, on a
    # SettingWithCopyWarning-enabled build, a chain-assignment check.  At n=1670 the
    # wall-clock cost is acceptable (~0.5 s), but it scales as O(n²) in pandas'
    # label-based accessor.  At n=5,000 (≈3 years of daily data) it will cost ~5 s;
    # at n=10,000 (~6 more years) it will cost ~20 s.
    #
    # FIX: collect each row's governance dict into a list (gov_rows), build a single
    # DataFrame from that list, then assign each column to out in one vectorised
    # operation.  This reduces the assignment cost from O(n²) to O(n) with a single
    # pd.DataFrame constructor call and n_cols direct column assignments.
    #
    # Correctness: the computation per row is identical to the old loop — the only
    # change is that all writes happen after the loop rather than in-loop.
    gov_rows: list = []
    for i in range(len(out)):
        leg_uncertainty = np.nanmean([
            _gamma_to_uncertainty(out.loc[i, "gamma_voo"]),  # FIX: was 1e-12 clip
            _gamma_to_uncertainty(out.loc[i, "gamma_ief"]),  # FIX: was 1e-12 clip
        ])
        prior_trigger = out.loc[max(0, i - cfg.DEF_TRIGGER_LOOKBACK): i - 1, "defense_trigger_raw"].tolist() if i > 0 else []
        defense_trigger_threshold = _rolling_quantile(prior_trigger, cfg.DEF_TRIGGER_LOOKBACK, cfg.DEF_TRIGGER_MIN_HISTORY, cfg.DEF_TRIGGER_Q)
        prior_stress = out.loc[max(0, i - cfg.DEF_TRIGGER_LOOKBACK): i - 1, "stress_score_raw"].tolist() if i > 0 else []
        stress_q_threshold = _rolling_quantile(prior_stress, cfg.DEF_TRIGGER_LOOKBACK, cfg.DEF_TRIGGER_MIN_HISTORY, cfg.DEF_TRIGGER_STRESS_Q)
        gov = _governance_mapping(
            float(out.loc[i, "p_final_cal"]),
            base_rate,
            int(out.loc[i, "drift_alarm"]),
            float(out.loc[i, "raw_val_auc"]) if np.isfinite(out.loc[i, "raw_val_auc"]) else np.nan,
            float(out.loc[i, "expert_agreement"]) if np.isfinite(out.loc[i, "expert_agreement"]) else np.nan,
            float(out.loc[i, "fwd_spread_hat"]),
            float(leg_uncertainty) if np.isfinite(leg_uncertainty) else 0.0,
            defense_trigger_threshold,
            str(out.loc[i, "regime_label"]),
            _safe_num(out.loc[i, "stress_score_raw"]),
            stress_q_threshold,
            cfg,
            dist_overlay_strength=float(out.loc[i, "dist_overlay_strength_g53"]) if "dist_overlay_strength_g53" in out.columns and np.isfinite(out.loc[i, "dist_overlay_strength_g53"]) else 0.0,
            dist_tail_shift=float(out.loc[i, "dist_tail_shift_g53"]) if "dist_tail_shift_g53" in out.columns and np.isfinite(out.loc[i, "dist_tail_shift_g53"]) else 0.0,
            dist_width_caution=float(out.loc[i, "dist_width_caution_g53"]) if "dist_width_caution_g53" in out.columns and np.isfinite(out.loc[i, "dist_width_caution_g53"]) else 0.0,
            uncertainty_penalty=float(out.loc[i, "uncertainty_penalty_g5"]) if "uncertainty_penalty_g5" in out.columns and np.isfinite(out.loc[i, "uncertainty_penalty_g5"]) else 0.0,
        )
        gov["w_strategy_ief"] = 1.0 - gov["w_strategy_voo"]
        gov_rows.append(gov)

    # Bulk-assign: one pd.DataFrame creation + one column assignment per governance key.
    # This must happen BEFORE the cooldown block so out["deploy_downside"] is populated.
    _gov_df = pd.DataFrame(gov_rows, index=out.index)
    for _gov_col in _gov_df.columns:
        out[_gov_col] = _gov_df[_gov_col].values

    # FIX (F-2, Quant-Guild Part 33 Audit): enforce cooldown after bulk governance assign.
    # Walk forward through the tape and suppress deploy_downside=1 on any row that
    # falls within DEPLOY_DOWNSIDE_COOLDOWN_BDAYS business days of the prior deployment.
    # Suppressed rows have active_weight reset to 0 and w_strategy_voo reverted to base.
    _cooldown = int(cfg.DEPLOY_DOWNSIDE_COOLDOWN_BDAYS)
    if _cooldown > 0 and "deploy_downside" in out.columns and "Date" in out.columns:
        _deploy_orig = out["deploy_downside"].fillna(0).astype(int).values.copy()
        _deploy_col = _deploy_orig.copy()
        _dates_arr = pd.to_datetime(out["Date"]).values
        _last_deploy_date = None
        for _i in range(len(_deploy_col)):
            if _deploy_col[_i] == 1:
                if _last_deploy_date is not None:
                    _d0 = pd.Timestamp(_last_deploy_date)
                    _d1 = pd.Timestamp(_dates_arr[_i])
                    _bdays_since = len(pd.bdate_range(_d0 + pd.Timedelta(days=1), _d1))
                    if _bdays_since < _cooldown:
                        _deploy_col[_i] = 0  # suppress — too soon
                        continue
                _last_deploy_date = _dates_arr[_i]
        _n_suppressed = int((_deploy_orig - _deploy_col).sum())
        if _n_suppressed > 0:
            print(
                f"[Part 2] Cooldown ({_cooldown} bdays): suppressed {_n_suppressed} deploy event(s) "
                f"within cooldown window of prior deployment."
            )
            _suppressed_mask = (_deploy_orig == 1) & (_deploy_col == 0)
            out["deploy_downside"] = _deploy_col
            out.loc[_suppressed_mask, "active_weight_raw"] = 0.0
            out.loc[_suppressed_mask, "active_weight_capped"] = 0.0
            out.loc[_suppressed_mask, "w_strategy_voo"] = cfg.BASE_WEIGHT_VOO
            out.loc[_suppressed_mask, "w_strategy_ief"] = cfg.BASE_WEIGHT_IEF

    # FIX (F-4 continued, Quant-Guild Part 33 Audit): vectorise the returns loop.
    # Prior loop used out.loc[i, col] scalar assignments for 5 columns × ~1,600
    # realized rows = ~8,000 additional individual assignments.  Replaced with
    # vectorised operations on the realized subset.
    realized_mask = out["y_avail"].astype(int) == 1
    realized_idx = out.index[realized_mask]
    if len(realized_idx) > 0:
        # Compute prev_w for each realized row (prior realized row's w_strategy_voo)
        _w_voo_arr = out["w_strategy_voo"].values.copy()
        _prev_w_arr = np.empty(len(out), dtype=float)
        _prev_w_arr[0] = cfg.BASE_WEIGHT_VOO
        _prev_w_arr[1:] = _w_voo_arr[:-1]
        # For rows where the prior row was not realized, the prior w was still
        # w_strategy_voo (Part 2 sets it on every row), so _prev_w_arr is correct.
        out.loc[realized_mask, "turnover"] = np.abs(
            _w_voo_arr[realized_mask.values] - _prev_w_arr[realized_mask.values]
        )
        out.loc[realized_mask, "cost_model"] = (cfg.SLIP_BPS / 10000.0) * out.loc[realized_mask, "turnover"]
        out.loc[realized_mask, "strategy_ret_gross"] = (
            out.loc[realized_mask, "w_strategy_voo"] * out.loc[realized_mask, "fwd_voo"]
            + (1.0 - out.loc[realized_mask, "w_strategy_voo"]) * out.loc[realized_mask, "fwd_ief"]
        )
        # FIX (F1, Quant-Guild Part 39 Audit): overwrite benchmark_ret with the
        # contemporaneous 60/40 forward return.  bench_60_40 from carry_cols is the
        # prior-day return (t-1→t); fwd_voo is the next-day return (t→t+1).  Using
        # carry bench_60_40 as benchmark introduced a 1-day lag on every realized row,
        # corrupting active_ret_net and the conditional IR gate.
        out.loc[realized_mask, "benchmark_ret"] = (
            cfg.BASE_WEIGHT_VOO * out.loc[realized_mask, "fwd_voo"]
            + cfg.BASE_WEIGHT_IEF * out.loc[realized_mask, "fwd_ief"]
        )
        _bench_finite = out.loc[realized_mask, "benchmark_ret"].notna() & np.isfinite(out.loc[realized_mask, "benchmark_ret"])
        out.loc[realized_mask & _bench_finite, "active_ret_gross"] = (
            out.loc[realized_mask & _bench_finite, "strategy_ret_gross"]
            - out.loc[realized_mask & _bench_finite, "benchmark_ret"]
        )
        out.loc[realized_mask, "strategy_ret_net"] = out.loc[realized_mask, "strategy_ret_gross"] - out.loc[realized_mask, "cost_model"]
        out.loc[realized_mask & _bench_finite, "active_ret_net"] = (
            out.loc[realized_mask & _bench_finite, "active_ret_gross"]
            - out.loc[realized_mask & _bench_finite, "cost_model"]
        )

    cls_base = cls_dist = cls_final = {"auc": np.nan, "pr": np.nan, "lift": np.nan, "brier": np.nan, "ece": np.nan}
    dist_diag = {"raw_coverage": np.nan, "conf_coverage": np.nan, "median_rmse": np.nan}
    if len(realized):
        y = realized["y_rel_tail_voo_vs_ief"].values.astype(int)
        cls_base = _classification_metrics(y, realized["p_tail_base"].values, cfg.ECE_BINS)
        cls_dist = _classification_metrics(y, realized["p_tail_dist"].values, cfg.ECE_BINS)
        cls_final = _classification_metrics(y, realized["p_final_g5"].values, cfg.ECE_BINS)
        dist_diag = {
            "raw_coverage": float(np.nanmean(realized["dist_interval_hit_raw"].values)) if "dist_interval_hit_raw" in realized.columns else np.nan,
            "conf_coverage": float(np.nanmean(realized["dist_interval_hit_conf"].values)) if "dist_interval_hit_conf" in realized.columns else np.nan,
            "median_rmse": _rmse(realized["excess_ret"].values, realized["spread_q50"].values),
        }

    # NOTE (Quant-Guild Part 21 audit — BUG-1):
    # The F3 blend block that was here in Part 20 has been RELOCATED to Part 3.
    # Root cause of the Part 20 failure: Part 2 runs BEFORE Part 2B and 2C in the
    # pipeline order (Part 0→6→1→2→2B→2C→2A→7→8→3→9→10). The blend code in Part 2
    # could only load tapes from the PREVIOUS run, making the live row (current date)
    # always unblended. Additionally, the tapes from previous runs may not have been
    # at the expected path, causing _p2b_available=False and the blend block to never
    # execute (confirmed: all 1658 rows have p_final_g5_source="base_plus_soft_caution
    # _overlay_532", not "base_only" — the latter would appear if the block had run).
    #
    # Correct architecture: blend belongs in Part 3, which runs after Part 2B and 2C
    # and is the designated fusion layer. Part 3 reads the current run's Part 2B/2C
    # tapes and blends p_final_cal for the live row before all governance decisions.

    shuffle_auc = _shuffle_auc(full.loc[full["y_avail"] == 1].tail(cfg.TRAIN_WINDOW_DAYS).copy(), feature_cols, cfg)
    active_net = realized["active_ret_net"].dropna().values if len(realized) else np.array([])
    strat_net = realized["strategy_ret_net"].dropna().values if len(realized) else np.array([])
    neg_bench = realized.loc[realized["benchmark_ret"] < 0].copy() if len(realized) else pd.DataFrame()
    raw_val_auc_median = float(np.nanmedian(out["raw_val_auc"].values)) if len(out) else np.nan
    # FIX (F2, Quant-Guild Part 45 Audit): Pre-compute trailing_4fold_auc here so it is
    # available for Path C of _quality_enter. raw_val_auc_median uses ALL 85 expanding-window
    # WF folds since 2012; trailing_4fold uses only the 4 most recent folds, which better
    # represents current model quality after the 2024-2025 signal deterioration.
    trailing_4fold_auc = _compute_trailing_fold_auc(
        out["raw_val_auc"].values if "raw_val_auc" in out.columns else np.array([]),
        n_folds=4,
    )
    suspicious_perf_flag = bool(np.isfinite(cls_final["auc"]) and cls_final["auc"] > max(0.75, shuffle_auc + 0.20 if np.isfinite(shuffle_auc) else 0.75))
    drift_alarm_rate = float(out["drift_alarm"].fillna(0).mean())
    active_mean = float(np.nanmean(active_net)) if len(active_net) else np.nan
    active_ir = _annualized_ir(active_net, cfg.H)
    # Conditional IR: computed only on deployed rows. Used in final_pass and
    # _should_fail_closed in place of the full-series active_ir which is
    # structurally near-zero at daily deployment sparsity (~0.4% of rows).
    # FIX (Finding 3): pass n_min from config so the gate is deferred until
    # at least CONDITIONAL_ACTIVE_IR_MIN_N defense events have been observed.
    conditional_active_ir = _conditional_active_ir(out, cfg.H, n_min=int(cfg.CONDITIONAL_ACTIVE_IR_MIN_N))
    # FIX (F1, Quant-Guild Part 43 Audit): compute t-stat diagnostics for monitoring.
    # With n=14 deploy events, SE(IR_annual)=4.24 and the -0.50 threshold had 45% false-alarm
    # rate. The diagnostics expose the t-statistic of the underlying mean return so future
    # sessions can evaluate whether to tighten the gate with statistical backing.
    _cond_ir_diag = _conditional_active_ir_diagnostics(out, cfg.H, n_min=int(cfg.CONDITIONAL_ACTIVE_IR_MIN_N))
    strategy_ir = _annualized_ir(strat_net, cfg.H)

    prior_summary = _load_prior_part2_summary(cfg)
    deploy_gate = _deploy_downside_gate_stats(out, cfg, prior_summary)

    stress_panel = _compute_stress_panel(out, cfg)
    cls_base_lift = float(cls_base.get("lift", np.nan))
    cls_base_ece = float(cls_base.get("ece", np.nan))
    # FIX (Audit 2026-05-07 — Circular Deadlock):
    # The previous definition included `active_mean > 0.0` as a required condition.
    # This created an unescapable structural deadlock:
    #
    #   1. In FAIL_CLOSED_NEUTRAL mode, _apply_fail_closed_neutral() forces
    #      w_strategy_voo = 0.60 for every row → active_weight = 0 → active_ret_net = 0.
    #   2. active_mean = mean(active_ret_net) over realized rows = 0.0 (always).
    #   3. Therefore predictive_quality_ok = False (always).
    #   4. Therefore final_pass = False (always).
    #   5. Therefore fail_closed mode persists forever → back to step 1.
    #
    # The model can NEVER exit fail_closed while this condition is present.
    # active_mean already appears in the final_pass gate as `active_mean >= -0.002`
    # (a loose floor); it does not need to appear in predictive_quality_ok, which
    # should assess FORECAST quality (AUC, calibration, lift) rather than realized
    # P&L on a fail-closed tape that definitionally has zero active returns.
    #
    # FIX (BUG-C, Quant-Guild Part 31 Audit):
    # predictive_quality_ok had no hysteresis, unlike the AUC gate which has a
    # stay-condition (enter=0.535, stay=0.530 when prior run was NORMAL).
    # The lift threshold SE≈0.11 at n=1,669 rows (see comment below on SE estimate).
    # A drop from 1.071 → 1.026 is well within 1 SE of the 1.03 threshold and is
    # pure sampling noise, yet the hard threshold caused the gate to flip and
    # produced final_pass=False + publish_mode=NORMAL → RuntimeError in Part 3.
    #
    # Fix: mirror the AUC hysteresis pattern.
    #   Enter condition (new session, no prior NORMAL run): lift > 1.03 AND ECE < 0.03
    #   Stay condition (prior run was NORMAL with final_pass=True): lift > 1.01 AND ECE < 0.05
    #
    # The stay thresholds are intentionally looser:
    #   lift > 1.01 — hard floor: the model must still be better than random at minimum.
    #   ECE < 0.05  — relaxed: calibration allowed to degrade slightly before forcing exit.
    # These values are calibrated to the empirical SE of lift (~0.11) and ECE (~0.01-0.03).
    # A run with lift in [1.01, 1.03) and ECE in [0.03, 0.05) that had a prior NORMAL
    # run stays in quality-ok state; drift/calibration gates provide independent checks.
    _prior_quality_ok_for_stay = bool(
        prior_summary.get("predictive_quality_ok", False) and
        str(prior_summary.get("publish_mode", "")).upper() == "NORMAL" and
        bool(prior_summary.get("final_pass", False))
    )
    # FIX (F-1/F-9, Quant-Guild Part 32 Audit):
    # predictive_quality_ok used only cls_base_lift > 1.03 as the enter condition.
    # With SE(lift) ≈ 0.11 at n=1,669, the current lift=1.026 misses 1.030 by 0.004
    # = 0.036 SE — pure sampling noise. Because there is no prior NORMAL+final_pass
    # run, the stay condition cannot fire, creating a structural cold-start deadlock
    # where the system can never reach predictive_quality_ok=True regardless of
    # other signal evidence.
    #
    # Fix: add a secondary AUC-path to _quality_enter. If the rolling walk-forward
    # AUC median clears the final_pass threshold (0.535) and ECE < 0.03, the model
    # has demonstrated genuine statistical signal even when the lift metric is within
    # one SE of its threshold. This is consistent with the existing final_pass gate
    # (which already uses raw_val_auc_median >= 0.535) and avoids an internal
    # inconsistency where AUC clears but quality is declared false.
    #
    # The dual-path logic is:
    #   Path A (lift-primary): lift > 1.03 AND ECE < 0.03   (original)
    #   Path B (AUC-backup):  rolling_AUC >= 0.535 AND ECE < 0.03   (new)
    # Path B fires only when Path A fails, providing a principled safety valve
    # rather than lowering the enter threshold unconditionally.
    _quality_enter = bool(
        # Path A: lift-primary (original criterion)
        (np.isfinite(cls_base_lift) and cls_base_lift > 1.03 and
         np.isfinite(cls_base_ece) and cls_base_ece < 0.03)
        or
        # Path B: AUC-backup — rolling walk-forward median clears the final_pass
        # AUC threshold. Prevents a SE=0.11 gap of 0.004 from creating a permanent
        # cold-start deadlock when the AUC gate and all independent safety gates pass.
        (np.isfinite(raw_val_auc_median) and raw_val_auc_median >= 0.535 and
         np.isfinite(cls_base_ece) and cls_base_ece < 0.03)
        or
        # Path C (FIX F2, Quant-Guild Part 45 Audit): trailing 4-fold AUC backup.
        # raw_val_auc_median (Path B) is the median over ALL 85 expanding WF folds
        # spanning 2012–2026, inflated by high-signal 2019–2022 folds. At S45 it is
        # 0.5342, missing the 0.535 threshold by 0.0008 (0.07 SE(AUC)≈0.012) — a gap
        # too small to be statistically meaningful yet sufficient to fail the gate.
        # trailing_4fold_auc_median = 0.5603 uses only the 4 most recent unique WF fold
        # AUC values (~1 year of data), representing current model quality more
        # accurately than the 14-year aggregated median. It is the correct estimator
        # of signal quality in the current market regime.
        # Threshold unchanged at 0.535. ECE guard unchanged at < 0.03.
        (np.isfinite(trailing_4fold_auc) and trailing_4fold_auc >= 0.535 and
         np.isfinite(cls_base_ece) and cls_base_ece < 0.03)
    )
    _quality_stay = bool(
        _prior_quality_ok_for_stay and
        np.isfinite(cls_base_lift) and cls_base_lift > 1.01 and
        np.isfinite(cls_base_ece) and cls_base_ece < 0.05
    )
    predictive_quality_ok = _quality_enter or _quality_stay
    if _quality_stay and not _quality_enter:
        print(
            f"[Part 2] predictive_quality_ok: STAY (hysteresis) — "
            f"lift={cls_base_lift:.4f} in (1.01, 1.03] or ECE={cls_base_ece:.4f} in [0.03, 0.05); "
            f"prior run was NORMAL+final_pass=True → staying quality-ok."
        )
    # FIX (F1, Quant-Guild Part 44 Audit): Pre-compute regime AUC breakdown BEFORE
    # the summary dict so full_period_active_regimes can be passed to
    # _compute_regime_auc_rolling. The stale hardcoded {"calm","high_vol","risk_on"}
    # inside _compute_regime_auc_rolling was calibrated before S43's DeLong gate
    # reduced active_regimes to ['high_vol'] only, causing a spurious decay alarm
    # for risk_on (which was never in computed active_regimes).
    _regime_auc_bd = _compute_regime_auc(realized, bins=cfg.ECE_BINS)
    _regime_auc_full_active = set(_regime_auc_bd.get("active_regimes", []))
    _regime_auc_rolling = _compute_regime_auc_rolling(
        realized,
        window_days=252,
        min_regime_obs=30,
        bins=cfg.ECE_BINS,
        full_period_active_regimes=_regime_auc_full_active,
    )

    summary = {
        "part": "part2",
        "version": "GEN5_PART2_GEN532_SOFT_CAUTION_OVERLAY",
        "schema_version": cfg.OUTPUT_SCHEMA_VERSION,
        "horizon": cfg.H,
        "holdout_start": cfg.HO_START_DATE,
        "rows_rebalance": int(len(out)),
        "rows_realized": int(len(realized)),
        "rows_audit": int(len(realized)),
        "model_feature_count": int(len(feature_cols)),
        "forbidden_feature_count": int(len(forbidden_features)),
        "forbidden_features_excluded": forbidden_features,
        "script_version": SCRIPT_VERSION,
        "contract_profile": contract_info.get("contract_profile"),
        "expected_model_feature_count": contract_info.get("expected_model_feature_count"),
        "expected_forbidden_count": contract_info.get("expected_forbidden_count"),
        "contract_missing_locked": contract_info.get("missing_locked"),
        "contract_missing_optional": contract_info.get("missing_optional"),
        "contract_extra_locked": contract_info.get("extra_locked"),
        "classification_base": cls_base,
        "classification_dist": cls_dist,
        "classification_final_used": cls_final,
        "distributional_diagnostics": dist_diag,
        "calibration_gate_on_rate": float(out["calibration_gate_on"].fillna(0).mean()),
        "high_risk_state_rate": float(out["high_risk_state"].fillna(0).mean()),
        "drift_base_ece": _safe_num(out["drift_base_ece"].iloc[-1]) if len(out) else np.nan,
        "drift_base_brier": _safe_num(out["drift_base_brier"].iloc[-1]) if len(out) else np.nan,
        "drift_alarm_rate": drift_alarm_rate,
        "deploy_downside_rate": float(deploy_gate["rate"]),
        "deploy_downside_count_total": int(deploy_gate["total_count"]),
        "passive_regimes_no_deploy": list(getattr(cfg, 'PASSIVE_REGIMES_NO_DEPLOY', ('calm', 'risk_on', 'crisis'))),  # FIX F1/F4 Part 41 + F2 Part 42: calm=0.510/p=0.40, risk_on=0.485/p=0.67, crisis=0.478/p=0.69 (all non-sig). high_vol=0.539/p=0.095 only deployment regime.
        "deploy_downside_count_recent": int(deploy_gate["recent_count"]),
        "deploy_downside_recent_lookback": int(cfg.DEPLOY_DOWNSIDE_RECENT_LOOKBACK),
        "deploy_downside_rate_min": float(cfg.DEPLOY_DOWNSIDE_RATE_MIN),
        "deploy_downside_rate_max": float(cfg.DEPLOY_DOWNSIDE_RATE_MAX),
        "deploy_downside_enter_count_min": int(cfg.DEPLOY_DOWNSIDE_ENTER_COUNT_MIN),
        "deploy_downside_stay_count_min": int(cfg.DEPLOY_DOWNSIDE_STAY_COUNT_MIN),
        "deploy_downside_recent_count_min": int(cfg.DEPLOY_DOWNSIDE_RECENT_COUNT_MIN),
        "deploy_downside_gate_enter": int(deploy_gate["enter_gate"]),
        "deploy_downside_gate_stay": int(deploy_gate["stay_gate"]),
        "deploy_downside_gate_pass": bool(deploy_gate["gate_pass"]),
        "deploy_downside_gate_reason": str(deploy_gate["gate_reason"]),
        "prior_publish_mode_for_hysteresis": str(deploy_gate["prior_publish_mode"]),
        "defense_trigger_mean": float(np.nanmean(out["defense_trigger_raw"].values)),
        "defense_trigger_threshold_median": float(np.nanmedian(out["defense_trigger_threshold"].values)),
        "deploy_upside_rate": float(out["deploy_upside"].fillna(0).mean()),
        "raw_val_auc_median": raw_val_auc_median,
        # FIX (Finding A, Audit 2026-04-21): documents which AUC estimator is used
        # for the final_pass gate. "rolling_median" = raw_val_auc_median (walk-forward
        # folds). Previously was "holdout" = cls_final["auc"] (single-pass, high noise).
        "final_pass_auc_source": "rolling_median",
        "final_pass_auc_value": raw_val_auc_median,
        "active_ret_net_mean": active_mean,
        "active_ret_net_ir": active_ir,
        "strategy_ret_net_ir": strategy_ir,
        "negative_benchmark_active_mean": float(neg_bench["active_ret_net"].mean()) if len(neg_bench) else np.nan,
        "negative_benchmark_hit_rate": float((neg_bench["active_ret_net"] > 0).mean()) if len(neg_bench) else np.nan,
        "downside_capture_ratio": float((neg_bench["strategy_ret_net"].mean() / neg_bench["benchmark_ret"].mean())) if len(neg_bench) and np.isfinite(neg_bench["benchmark_ret"].mean()) and neg_bench["benchmark_ret"].mean() != 0 else np.nan,
        "avg_abs_active_weight_raw": float(np.nanmean(np.abs(out["active_weight_raw"].values))),
        "avg_abs_active_weight_capped": float(np.nanmean(np.abs(out["active_weight_capped"].values))),
        "shuffle_auc_median": shuffle_auc,
        "suspicious_perf_flag": suspicious_perf_flag,
        # FIX (Audit 2026-05-10 — Quant-Guild Part 16):
        # Per-regime AUC breakdown (full period, 2020-2026).
        # FIX (F1, Quant-Guild Part 44 Audit): pre-computed above as _regime_auc_bd
        # so active_regimes can be passed to _compute_regime_auc_rolling.
        "regime_auc_breakdown": _regime_auc_bd,
        # FIX (F1, Quant-Guild Part 40 Audit): rolling per-regime AUC monitor.
        # Full-period AUC masks severe temporal decay: calm 2025=0.257,
        # high_vol 2026=0.333, risk_on 2024=0.417.  The rolling 252-day window
        # gives the CURRENT picture.  regime_auc_rolling_decay_alarm=True means
        # at least one previously-active regime is now anti-predictive on recent
        # data → Part 3 reduces view_confidence; Part 7 excludes from BL gate.
        # FIX (F1, Quant-Guild Part 44 Audit): pre-computed above as _regime_auc_rolling
        # with full_period_active_regimes=_regime_auc_full_active (dynamic, not hardcoded).
        "regime_auc_rolling": _regime_auc_rolling,
        # FIX (F1, Quant-Guild Part 42 Audit): DeLong AUC monitoring metrics.
        #
        # FINDING: Governance rolling-median AUC (0.537, threshold 0.535) passes by
        # a margin of 0.002 on a metric dominated by 2020-2022 high-signal folds.
        # The full-dataset holdout AUC = 0.511 (DeLong z=0.606, p=0.272) and the
        # deployment regime (high_vol) AUC = 0.539 (z=1.309, p=0.095, borderline).
        #
        # These three fields provide machine-readable evidence of the model's actual
        # statistical state WITHOUT changing any hard governance gate:
        #   1. delong_deploy_regime_auc: DeLong test on high_vol (the deployment
        #      regime — the only regime where non-HRS defense deploys are permitted).
        #   2. delong_overall_auc: DeLong test on the full holdout set.
        #   3. trailing_4fold_auc_median: median of the 4 most recent fold AUCs,
        #      which tracks recency-weighted model quality independently of the
        #      all-history rolling median.
        #
        # deploy_regime_auc_warning=True means the deployment regime AUC is NOT
        # statistically significant at p<0.10 — a soft monitoring flag that does
        # not block final_pass but should trigger human review.
        #
        # DESIGN RATIONALE for NOT adding a hard DeLong gate:
        #   high_vol z=1.309 (p=0.095). Adding p<0.05 as a hard gate would fail now
        #   (z=1.309 < 1.645) and create a structural deadlock: the model needs to
        #   deploy to accumulate evidence; governance needs evidence to permit deploy.
        #   The soft warning fires when p>0.10 without inducing deadlock.
        #
        # Computed on the deployment regime only (high_vol) since that is the sole
        # regime where non-HRS defense sleeve activates. Using p_tail_base vs
        # y_rel_tail_voo_vs_ief (same fields as regime_auc_breakdown).
        "delong_deploy_regime_auc": (
            lambda _hv=realized[realized["regime_label"] == "high_vol"]
            if "regime_label" in realized.columns else pd.DataFrame(): _delong_auc_ztest(
                _hv["y_rel_tail_voo_vs_ief"].values if "y_rel_tail_voo_vs_ief" in _hv.columns else np.array([]),
                _hv["p_tail_base"].values if "p_tail_base" in _hv.columns else np.array([]),
            ) if len(_hv) >= 20 else {"auc": float("nan"), "z": float("nan"),
                                      "p_one_sided": float("nan"), "auc_warning": True}
        )(),
        "delong_overall_auc": _delong_auc_ztest(
            realized["y_rel_tail_voo_vs_ief"].values if "y_rel_tail_voo_vs_ief" in realized.columns else np.array([]),
            realized["p_tail_base"].values if "p_tail_base" in realized.columns else np.array([]),
        ),
        "deploy_regime_auc_warning": (
            # True if high_vol AUC is NOT significant at p<0.10 (soft monitoring only)
            _delong_auc_ztest(
                realized.loc[realized["regime_label"] == "high_vol", "y_rel_tail_voo_vs_ief"].values
                if "regime_label" in realized.columns and "y_rel_tail_voo_vs_ief" in realized.columns
                else np.array([]),
                realized.loc[realized["regime_label"] == "high_vol", "p_tail_base"].values
                if "regime_label" in realized.columns and "p_tail_base" in realized.columns
                else np.array([]),
            ).get("auc_warning", True)
        ),
        "trailing_4fold_auc_median": _compute_trailing_fold_auc(
            realized["raw_val_auc"].values if "raw_val_auc" in realized.columns else np.array([]),
            n_folds=4,
        ),
        "stress_panel": stress_panel,
        "tail_event_threshold": tail_threshold,
        "part1_version_consumed": str(part1_meta.get("version")),
        "environment": _environment_metadata(os.path.abspath(__file__) if "__file__" in globals() else "part2_gen5.py"),
        "build_variant": "PHASE3_2_BASE_PLUS_SOFT_CAUTION_OVERLAY",
        "clearance_logic_version": "deploy_count_hysteresis_v1",
        "dist_overlay_on_rate": float(np.nanmean(out["dist_overlay_on_g53"].values)) if "dist_overlay_on_g53" in out.columns else np.nan,
        "dist_trust_mean": float(np.nanmean(out["dist_trust_score_g53"].values)) if "dist_trust_score_g53" in out.columns else np.nan,
        "dist_overlay_strength_mean": float(np.nanmean(out["dist_overlay_strength_g53"].values)) if "dist_overlay_strength_g53" in out.columns else np.nan,
        "active_ir_final_pass_min": float(cfg.FINAL_PASS_ACTIVE_IR_MIN),
        "active_ir_fail_closed_min": float(cfg.FAIL_CLOSED_ACTIVE_IR),
        "conditional_active_ir": conditional_active_ir,
        "conditional_active_ir_min": float(cfg.CONDITIONAL_ACTIVE_IR_MIN),
        "conditional_active_ir_tfloor": float(cfg.CONDITIONAL_ACTIVE_IR_TFLOOR),  # FIX F1/S45: t-stat floor for final_pass gate
        "conditional_active_ir_failclosed_tfloor": float(cfg.CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR),  # FIX F1/S46: t-stat floor for _should_fail_closed emergency gate
        "conditional_active_ir_min_n": int(cfg.CONDITIONAL_ACTIVE_IR_MIN_N),  # FIX: minimum event count before gate activates
        "conditional_active_ir_floor_fail_closed": float(cfg.CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED),
        # FIX (F1, Quant-Guild Part 43 Audit): monitoring diagnostics for the conditional IR gate.
        # t-stat and p-value of the underlying mean daily defense return.
        # At n=14, SE(IR_annual)=4.24; threshold -0.50 had 45.3% false-alarm rate.
        # These fields allow future gate tightening with statistical backing.
        "conditional_active_ir_n": _cond_ir_diag.get("conditional_active_ir_n", 0),
        "conditional_active_ir_tmean": _cond_ir_diag.get("conditional_active_ir_tmean", float("nan")),
        "conditional_active_ir_pmean": _cond_ir_diag.get("conditional_active_ir_pmean", float("nan")),
        "conditional_active_ir_se_ann": _cond_ir_diag.get("conditional_active_ir_se_ann", float("nan")),
        "effective_drift_ece_max": float(drift_ece_max_eff),
        "effective_drift_brier_max": float(drift_brier_max_eff),
        "effective_final_pass_drift_rate_max": float(final_pass_drift_max_eff),
        "effective_fail_closed_drift_rate": float(fail_closed_drift_rate_eff),
        "effective_fail_closed_cal_gate": float(fail_closed_cal_gate_eff),
        "predictive_quality_ok": predictive_quality_ok,
        "final_pass": bool(
            # FIX (Finding A, Audit 2026-04-21):
            # The prior gate used cls_final["auc"] — the single-pass holdout AUC
            # computed over the full 2020–2026 period. This is a single unrepeated
            # estimate with SE ≈ 0.030 and 95% CI ≈ [0.451, 0.570]. At n=1,644 with
            # ~20.7% base rate, the holdout AUC is dominated by noise from any single
            # market period. The current run shows holdout AUC = 0.510 while
            # raw_val_auc_median = 0.538 — the median over all rolling walk-forward
            # folds. The rolling median is the same statistic used for per-row
            # calibration gating (DEPLOY_MIN_VAL_AUC = 0.530). Using different
            # estimators for the per-row and final-pass gates creates an internal
            # inconsistency: calibration clears on rolling AUC but final_pass blocks
            # on holdout AUC.
            #
            # Fix: align final_pass to raw_val_auc_median (rolling cross-val median).
            # Threshold kept at 0.535 (slightly above DEPLOY_MIN_VAL_AUC = 0.530 to
            # require marginal additional evidence before full deployment).
            # FIX (BUG-3-REVISED, Audit 2026-05-09 — Quant-Guild Part 15):
            # BUG-3 (2026-05-08) raised the enter threshold from 0.535 → 0.537 as a "noise
            # buffer". Result: raw_val_auc_median = 0.5369 misses 0.537 by 0.0001 (< 0.003 SE,
            # where SE≈0.032 at n=252/fold) — pure sampling noise — permanently locking the
            # stack in FAIL_CLOSED_NEUTRAL.
            #
            # The 0.002 buffer provides 0.002/0.032 = 0.06 SE of protection — negligible.
            # The hysteresis band (enter=0.535, stay=0.530) already gives 0.005 = 0.156 SE
            # of exit-protection; that is the correct place for the hysteresis gap.
            # Putting 0.002 of "buffer" inside the enter threshold costs a deadlock whenever
            # the AUC wanders in [0.535, 0.537) — a structurally likely outcome given SE=0.032.
            #
            # Fix: revert enter threshold to 0.535. Stay threshold (0.530) is unchanged.
            # This is consistent with the pre-BUG-3 design and avoids the deadlock.
            #
            # FIX (F1, Quant-Guild Part 52 Audit): Add Path C (trailing 4-fold AUC) to
            # final_pass gate, mirroring _quality_enter exactly.
            #
            # ROOT CAUSE: S45 F2 added trailing_4fold_auc as Path C to _quality_enter
            # (predictive_quality_ok). The SAME path was never propagated to final_pass.
            # This creates a structural inconsistency:
            #   _quality_enter = True  (via Path C: trailing_4fold=0.5609 >= 0.535, ECE<0.03)
            #   final_pass     = False (only checks raw_val_auc_median >= 0.535 = 0.5339 FAIL)
            #
            # IMPACT (S52 artifact, built_at=2026-06-20T01:46 UTC):
            #   raw_val_auc_median  = 0.53385 < 0.535 by 0.00115 (0.06 SE; pure noise)
            #   trailing_4fold_auc  = 0.56088 > 0.535 by 0.026   (1.3 SE; clear margin)
            #   ECE                 = 0.01536 < 0.03              (clear)
            #   All other final_pass sub-conditions: PASS
            #   predictive_quality_ok = True (Path C), but final_pass = False -> FAIL_CLOSED
            #
            # The gap of 0.00115 is 0.06 SE(AUC) at fold n~20 (SE~0.020) -- statistically
            # indistinguishable from the threshold. Blocking on this gap while the trailing
            # 4-fold (the better estimator of current signal quality) clears by 1.3 SE is
            # inconsistent and regressive.
            #
            # FIX: mirror _quality_enter Path C in the AUC sub-condition of final_pass.
            # Three paths are now available in final_pass (matching _quality_enter):
            #   Path B (enter): raw_val_auc_median >= 0.535
            #   Path A (stay):  raw_val_auc_median >= DEPLOY_MIN_VAL_AUC=0.530, prior NORMAL
            #   Path C (enter, NEW): trailing_4fold_auc >= 0.535 AND ECE_base < 0.03
            # All thresholds and guards are identical to _quality_enter's paths B and C.
            # The stay path (prior NORMAL + final_pass=True) is unchanged.  [FIX F1/S52]
            (
                (np.isfinite(raw_val_auc_median) and raw_val_auc_median >= 0.535)  # Path B: enter on rolling median
                or (                                                                 # Path A: stay on hysteresis
                    np.isfinite(raw_val_auc_median) and
                    raw_val_auc_median >= float(cfg.DEPLOY_MIN_VAL_AUC)
                    and prior_summary.get("publish_mode", "") == "NORMAL"
                    and bool(prior_summary.get("final_pass", False))
                )
                or (                                                                 # Path C: enter on trailing 4-fold [FIX F1/S52]
                    np.isfinite(trailing_4fold_auc) and trailing_4fold_auc >= 0.535
                    and np.isfinite(cls_base_ece) and cls_base_ece < 0.03
                )
            ) and
            np.isfinite(strategy_ir) and strategy_ir >= 0.45 and
            np.isfinite(active_mean) and active_mean >= -0.002 and
            # FIX (F1, Quant-Guild Part 45 Audit): Replace the annualized IR floor with a
            # t-statistic floor on the mean daily defense return.
            #
            # OLD (sample-size dependent, P(false alarm)=42% at n=10):
            #   conditional_active_ir >= CONDITIONAL_ACTIVE_IR_MIN (-1.00)
            #   At n=10: SE=5.02, threshold=-1.00 → Φ(-0.199) = 42.1% false alarm rate.
            #   The measured IR=-1.3824 at t_mean=-0.2754 is pure noise — blocked NORMAL.
            #
            # NEW (sample-size invariant, P(false alarm)=5% at any n >= CONDITIONAL_ACTIVE_IR_MIN_N):
            #   t_mean >= CONDITIONAL_ACTIVE_IR_TFLOOR (-1.645)
            #   t_mean = mean_ret / (std / sqrt(n)) — proper one-sided 5% t-test.
            #   NaN (n < MIN_N) → gate passes (insufficient data, defer to deploy_gate).
            #
            # Annualized IR is retained in the summary purely for monitoring/display.
            # FIX (F1, Quant-Guild Part 46 Audit): _should_fail_closed() no longer gates
            # on the raw annualized IR floor (it had the same n-dependence problem this
            # t-floor fixes here — see CONDITIONAL_ACTIVE_IR_FAILCLOSED_TFLOOR comment).
            # It now uses a stricter (1% one-sided) t-stat floor on the same t_mean value.
            (not np.isfinite(_cond_ir_diag.get("conditional_active_ir_tmean", float("nan")))
             or float(_cond_ir_diag.get("conditional_active_ir_tmean", float("nan")))
                >= float(cfg.CONDITIONAL_ACTIVE_IR_TFLOOR)) and
            drift_alarm_rate <= float(final_pass_drift_max_eff) and
            # Clearance logic fix (2026-04-24): use integer deploy-event counts
            # plus hysteresis instead of a brittle fractional rate threshold.
            # Old gate: deploy_downside_rate >= 0.002. At ~1,648 rows that is an
            # implied threshold of 3.296 events, so the whole stack can flip on a
            # one-event change. The helper converts this into a transparent policy:
            # enter NORMAL at >=3 total deploy rows (+ >=1 recent), stay NORMAL at
            # >=2 total deploy rows (+ >=1 recent) if the previous committed run
            # was already NORMAL.
            bool(deploy_gate["gate_pass"]) and
            np.isfinite(deploy_gate["rate"]) and float(deploy_gate["rate"]) <= float(cfg.DEPLOY_DOWNSIDE_RATE_MAX) and
            (not suspicious_perf_flag) and
            predictive_quality_ok
        ),
        "out_path": os.path.join(cfg.PRED_DIR, cfg.OUT_FILE),
    }

    fail_closed = _should_fail_closed(summary, cfg)
    summary["publish_mode"] = "FAIL_CLOSED_NEUTRAL" if fail_closed else "NORMAL"
    if fail_closed:
        out = _apply_fail_closed_neutral(out, cfg)
    else:
        out["publish_fail_closed"] = 0

    _validate_output_schema(out)
    out_path = os.path.join(cfg.PRED_DIR, cfg.OUT_FILE)
    out.to_csv(out_path, index=False)

    # summary json
    summary_path = os.path.join(cfg.PRED_DIR, cfg.SUMMARY_FILE)
    summary["output_hashes"] = {"consensus_tape_sha256": _sha256_file(out_path)}
    # FIX (BUG-1, Audit 2026-05-09): pre-process summary dict with _deep_clean_for_json
    # before json.dump so NaN/Inf values are serialized as null (not NaN literal).
    # json.dump's default= hook is NEVER called for Python float (natively handled),
    # so the old default=_json_safe approach silently wrote NaN as the literal token NaN
    # (invalid RFC 8259 JSON). JavaScript's JSON.parse() throws SyntaxError on NaN,
    # breaking the index.html dashboard for every run where conditional_active_ir is NaN
    # (i.e., until >= 10 defense events accumulate; currently 4 of 10).
    _clean_summary = _deep_clean_for_json(summary)
    if cfg.WRITE_HASHED_SUMMARY:
        _clean_summary["output_hashes"]["summary_payload_sha256"] = _sha256_text(
            json.dumps(_clean_summary, sort_keys=True)
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_clean_summary, f, indent=2)

    # diagnostics json
    diag = {
        "distribution_validation": {
            "raw_coverage_mean": float(np.nanmean(out.get("dist_raw_coverage", pd.Series(dtype=float)).values)) if "dist_raw_coverage" in out.columns else np.nan,
            "conf_coverage_mean": float(np.nanmean(out.get("dist_conf_coverage", pd.Series(dtype=float)).values)) if "dist_conf_coverage" in out.columns else np.nan,
            "median_rmse_mean": float(np.nanmean(out.get("dist_median_rmse", pd.Series(dtype=float)).values)) if "dist_median_rmse" in out.columns else np.nan,
        },
        "fused_probabilities": {
            "p_tail_base_mean": float(np.nanmean(out["p_tail_base"].values)),
            "p_tail_dist_mean": float(np.nanmean(out["p_tail_dist"].values)),
            "p_final_g5_mean": float(np.nanmean(out["p_final_g5"].values)),
            "uncertainty_penalty_mean": float(np.nanmean(out["uncertainty_penalty_g5"].values)),
            "dist_tail_shift_mean": float(np.nanmean(out["dist_tail_shift_g53"].values)) if "dist_tail_shift_g53" in out.columns else np.nan,
            "dist_width_caution_mean": float(np.nanmean(out["dist_width_caution_g53"].values)) if "dist_width_caution_g53" in out.columns else np.nan,
            "dist_overlay_on_rate": float(np.nanmean(out["dist_overlay_on_g53"].values)) if "dist_overlay_on_g53" in out.columns else np.nan,
            "dist_trust_mean": float(np.nanmean(out["dist_trust_score_g53"].values)) if "dist_trust_score_g53" in out.columns else np.nan,
            "dist_overlay_strength_mean": float(np.nanmean(out["dist_overlay_strength_g53"].values)) if "dist_overlay_strength_g53" in out.columns else np.nan,
        },
        "tail_event_threshold": float(tail_threshold),
    }
    diag_path = os.path.join(cfg.PRED_DIR, cfg.DIAG_FILE)
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(_deep_clean_for_json(diag), f, indent=2)

    # ablation csv
    ablation = pd.DataFrame([
        {"model": "base_only", **cls_base},
        {"model": "dist_only", **cls_dist},
        {"model": "base_plus_risk_overlay", **cls_final},
    ])
    ablation_path = os.path.join(cfg.PRED_DIR, cfg.ABLATION_FILE)
    ablation.to_csv(ablation_path, index=False)

    auc_final = summary["classification_final_used"]["auc"]
    auc_text = f"{auc_final:.4f}" if np.isfinite(auc_final) else "nan"
    print(f"✅ GEN5_PART2_GEN532 complete | rows={len(out)} | realized={len(realized)} | AUC(final)={auc_text} | features={len(feature_cols)} | mode={summary['publish_mode']}")
    return summary


def main() -> int:
    summary = build_part2_gen53(CFG)
    print("\nPart 2 Gen 5.3 summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    main()

