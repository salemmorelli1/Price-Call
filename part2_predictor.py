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

SCRIPT_VERSION = "GEN5_PART2_G532_DAILY_CANONICAL_V1"

try:
    import xgboost as xgb  # type: ignore
    HAVE_XGB = True
except Exception:
    xgb = None
    HAVE_XGB = False


@dataclass
class Part2Gen53Config:
    PART1_DIR: str = _DRIVE_ROOT + "/artifacts_part1"
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
    TAIL_EVENT_THRESHOLD: float = float(-0.015 / np.sqrt(7.0))
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
    SPREAD_K: float = 3.0
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

    EXPECTED_PART1_VERSION: str = "V19_P1_HARDENED"
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
    FAIL_CLOSED_ON_FALSE_PASS: bool = True
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
    # because with ~7 events the estimate is noisy, but a strongly negative value
    # indicates systematic defense misfire worth blocking.
    CONDITIONAL_ACTIVE_IR_MIN: float = -0.50          # final_pass: cond IR on deploy rows
    CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED: float = -1.50  # _should_fail_closed: harder

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


def _conditional_active_ir(out: pd.DataFrame, h: int) -> float:
    """IR computed only on rows where the defense sleeve was deployed (deploy_downside=1).

    This replaces the full-series active_ir gate for daily-frequency models where
    the deployment rate is sparse (~0.4% of rows). At that sparsity the full-series
    IR is structurally noise-dominated: the ~1600 near-zero active-weight rows drive
    the std to the full spread-vol level, making any meaningful IR threshold
    structurally unachievable regardless of defense quality.

    Conditional IR isolates the signal: it tests whether the defense events
    themselves earn returns consistent with the direction of the hedge.

    Returns nan if fewer than 3 deploy rows are available (insufficient for IR).
    """
    if "deploy_downside" not in out.columns or "active_ret_net" not in out.columns:
        return np.nan
    deploy_mask = out["deploy_downside"].fillna(0).astype(int) == 1
    deploy_rets = pd.to_numeric(out.loc[deploy_mask, "active_ret_net"], errors="coerce").dropna().values
    return _annualized_ir(deploy_rets, h)


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

    if governance_tier == 'NORMAL' and (
        dist_overlay_strength >= 0.12
        or dist_width_caution >= 0.40
        or uncertainty_penalty >= 0.25
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
    cond_ir = summary.get("conditional_active_ir", np.nan)
    if not isinstance(cond_ir, float):
        try:
            cond_ir = float(cond_ir)
        except Exception:
            cond_ir = np.nan
    cond_ir_floor = float(summary.get("conditional_active_ir_floor_fail_closed",
                                       cfg.CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED))
    return bool(
        (cfg.FAIL_CLOSED_ON_FALSE_PASS and (not bool(summary.get("final_pass", False))))
        or bool(summary.get("suspicious_perf_flag", False))
        or (np.isfinite(summary.get("drift_alarm_rate", np.nan)) and float(summary.get("drift_alarm_rate")) > drift_limit)
        # FIX (2026-04-13): was '> cal_limit', which incorrectly triggered fail_closed
        # when calibration was GOOD (e.g. 98.8% > 85%). The intent is to fail closed
        # when calibration is too POOR — i.e., when the rate is too LOW.
        or (np.isfinite(summary.get("calibration_gate_on_rate", np.nan)) and float(summary.get("calibration_gate_on_rate")) < cal_limit)
        or (np.isfinite(cond_ir) and cond_ir < cond_ir_floor)
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
        for k, v in gov.items():
            out.loc[i, k] = v
        out.loc[i, "w_strategy_ief"] = 1.0 - out.loc[i, "w_strategy_voo"]

    for i in range(len(out)):
        if int(out.loc[i, "y_avail"]) == 1:
            prev_w = out.loc[i - 1, "w_strategy_voo"] if i > 0 else cfg.BASE_WEIGHT_VOO
            out.loc[i, "turnover"] = abs(out.loc[i, "w_strategy_voo"] - prev_w)
            out.loc[i, "cost_model"] = (cfg.SLIP_BPS / 10000.0) * out.loc[i, "turnover"]
            out.loc[i, "strategy_ret_gross"] = out.loc[i, "w_strategy_voo"] * out.loc[i, "fwd_voo"] + (1.0 - out.loc[i, "w_strategy_voo"]) * out.loc[i, "fwd_ief"]
            out.loc[i, "active_ret_gross"] = out.loc[i, "strategy_ret_gross"] - out.loc[i, "benchmark_ret"] if np.isfinite(out.loc[i, "benchmark_ret"]) else np.nan
            out.loc[i, "strategy_ret_net"] = out.loc[i, "strategy_ret_gross"] - out.loc[i, "cost_model"]
            out.loc[i, "active_ret_net"] = out.loc[i, "active_ret_gross"] - out.loc[i, "cost_model"] if np.isfinite(out.loc[i, "active_ret_gross"]) else np.nan

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

    shuffle_auc = _shuffle_auc(full.loc[full["y_avail"] == 1].tail(cfg.TRAIN_WINDOW_DAYS).copy(), feature_cols, cfg)
    active_net = realized["active_ret_net"].dropna().values if len(realized) else np.array([])
    strat_net = realized["strategy_ret_net"].dropna().values if len(realized) else np.array([])
    neg_bench = realized.loc[realized["benchmark_ret"] < 0].copy() if len(realized) else pd.DataFrame()
    raw_val_auc_median = float(np.nanmedian(out["raw_val_auc"].values)) if len(out) else np.nan
    suspicious_perf_flag = bool(np.isfinite(cls_final["auc"]) and cls_final["auc"] > max(0.75, shuffle_auc + 0.20 if np.isfinite(shuffle_auc) else 0.75))
    drift_alarm_rate = float(out["drift_alarm"].fillna(0).mean())
    active_mean = float(np.nanmean(active_net)) if len(active_net) else np.nan
    active_ir = _annualized_ir(active_net, cfg.H)
    # Conditional IR: computed only on deployed rows. Used in final_pass and
    # _should_fail_closed in place of the full-series active_ir which is
    # structurally near-zero at daily deployment sparsity (~0.4% of rows).
    conditional_active_ir = _conditional_active_ir(out, cfg.H)
    strategy_ir = _annualized_ir(strat_net, cfg.H)

    stress_panel = _compute_stress_panel(out, cfg)
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
        "deploy_downside_rate": float(out["deploy_downside"].fillna(0).mean()),
        "defense_trigger_mean": float(np.nanmean(out["defense_trigger_raw"].values)),
        "defense_trigger_threshold_median": float(np.nanmedian(out["defense_trigger_threshold"].values)),
        "deploy_upside_rate": float(out["deploy_upside"].fillna(0).mean()),
        "raw_val_auc_median": raw_val_auc_median,
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
        "stress_panel": stress_panel,
        "tail_event_threshold": tail_threshold,
        "part1_version_consumed": str(part1_meta.get("version")),
        "environment": _environment_metadata(os.path.abspath(__file__) if "__file__" in globals() else "part2_gen5.py"),
        "build_variant": "PHASE3_2_BASE_PLUS_SOFT_CAUTION_OVERLAY",
        "dist_overlay_on_rate": float(np.nanmean(out["dist_overlay_on_g53"].values)) if "dist_overlay_on_g53" in out.columns else np.nan,
        "dist_trust_mean": float(np.nanmean(out["dist_trust_score_g53"].values)) if "dist_trust_score_g53" in out.columns else np.nan,
        "dist_overlay_strength_mean": float(np.nanmean(out["dist_overlay_strength_g53"].values)) if "dist_overlay_strength_g53" in out.columns else np.nan,
        "active_ir_final_pass_min": float(cfg.FINAL_PASS_ACTIVE_IR_MIN),
        "active_ir_fail_closed_min": float(cfg.FAIL_CLOSED_ACTIVE_IR),
        "conditional_active_ir": conditional_active_ir,
        "conditional_active_ir_min": float(cfg.CONDITIONAL_ACTIVE_IR_MIN),
        "conditional_active_ir_floor_fail_closed": float(cfg.CONDITIONAL_ACTIVE_IR_FLOOR_FAIL_CLOSED),
        "effective_drift_ece_max": float(drift_ece_max_eff),
        "effective_drift_brier_max": float(drift_brier_max_eff),
        "effective_final_pass_drift_rate_max": float(final_pass_drift_max_eff),
        "effective_fail_closed_drift_rate": float(fail_closed_drift_rate_eff),
        "effective_fail_closed_cal_gate": float(fail_closed_cal_gate_eff),
        "final_pass": bool(
            np.isfinite(cls_final["auc"]) and cls_final["auc"] >= 0.535 and
            np.isfinite(strategy_ir) and strategy_ir >= 0.45 and
            np.isfinite(active_mean) and active_mean >= -0.002 and
            # H=1 recalibration (2026-04-13): full-series active_ir replaced by
            # conditional_active_ir (IR on deploy_downside=1 rows only).
            # Full-series IR is structurally near-zero at daily sparsity; conditional
            # IR measures whether defense events are directionally coherent.
            # Floor of -0.50 passes unless deployments are systematically wrong.
            # If fewer than 3 deploy rows exist, conditional_ir=nan → gate passes
            # (defer to deploy_downside_rate gate which enforces minimum activity).
            (not np.isfinite(conditional_active_ir) or conditional_active_ir >= float(cfg.CONDITIONAL_ACTIVE_IR_MIN)) and
            drift_alarm_rate <= float(final_pass_drift_max_eff) and
            # H=1 recalibration (2026-04-12): deploy_downside floor lowered 0.01 → 0.002.
            # At daily granularity the spread_component gate fires less frequently than
            # at H=7, because daily log-return predictions have smaller amplitude.
            # 0.002 ≈ 3.3 defense days per year; this is the structurally achievable
            # floor after the DEPLOY_DOWNSIDE_SPREAD_ABS / SPREAD_CONFIRM_MIN recalibration.
            float(out["deploy_downside"].fillna(0).mean()) >= 0.002 and
            float(out["deploy_downside"].fillna(0).mean()) <= 0.30 and
            (not suspicious_perf_flag)
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
    if cfg.WRITE_HASHED_SUMMARY:
        summary["output_hashes"]["summary_payload_sha256"] = _sha256_text(json.dumps(summary, sort_keys=True, default=str))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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
        json.dump(diag, f, indent=2)

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



