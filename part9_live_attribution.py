#!/usr/bin/env python3
# @title PART 9 — Live Attribution & Statistical Significance
# =============================================================================
# Industry-grade live performance analytics for PriceCallProject v2
#
# This is the part that makes the system TRUSTWORTHY.
# No backtest metric matters. Only this file tells you if the model is real.
#
# Responsibilities:
#   1. Consume the prediction log (live realized trades only)
#   2. Compute statistically valid live performance metrics
#   3. Diebold-Mariano test: model vs naive benchmark
#   4. Factor attribution: how much of return is explained by beta, duration, etc.
#   5. Model health diagnostics: feature drift, concept drift, calibration
#   6. Stopping rules: when to suspend live trading
#
# AUDIT CHANGELOG (Quant-Guild Part 8 session, 2026-04)
# ──────────────────────────────────────────────────────
# Finding A (CRITICAL): estimated_annual_edge_bps overstated ~370×.
#   The previous formula computed the raw signed spread return as if the entire
#   portfolio were deployed long-short.  The actual active weight is ~0.04%
#   (dead-banded most days), so the true weight-scaled edge is ~3.2 bps/yr, not
#   1,180 bps/yr.  Fix: source estimated_annual_edge_bps from Part 2 summary
#   active_ret_net_mean (already weight-scaled); fall back to spread-based
#   formula only if Part 2 summary is absent, and flag it as unscaled.
#
# Finding B (IMPORTANT): tail_threshold applied as single scalar across all rows.
#   Part 1 uses a rolling 63-day 20th-percentile quantile as the label threshold,
#   not a fixed -0.015.  Each prediction_log row already carries the correct
#   per-row dynamic threshold in the 'tail_threshold' column written by Part 3.
#   The previous code extracted only the last value (iloc[-1]) and applied it
#   uniformly. Fix: current-protocol evidence requires the exact stored threshold
#   and applies it element-wise; pre-schema rows are excluded from this cohort.
# =============================================================================
from __future__ import annotations

import os
import dataclasses
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

from artifact_integrity import (
    LEGACY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    current_evidence_mask,
    write_json_strict,
)

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Part9Config:
    version: str = "V2_DAILY_CANONICAL"
    predlog_path: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part3/prediction_log.csv"
    part2_tape_path: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part2_g532/predictions/g532_final_consensus_tape.csv"
    part6_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part6"
    out_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part9"
    part8_cost_path: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part8/execution_cost_tape.csv"
    # FIX (BUG-1, Audit 2026-05-08): Part 1 feature artifacts needed for feature drift detection.
    # detect_feature_drift() was defined but never called — feature drift stopping rules were
    # permanently inoperative. Adding part1_dir so generate_live_report() can load X_features.parquet.
    part1_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part1"
    horizon: int = 1                    # CHANGE: 1-day forecast horizon

    # Statistical thresholds
    t_stat_min: float = 2.0             # Minimum t-stat for edge significance
    t_stat_suspend: float = -1.5        # Suspend live trading below this
    max_drawdown_hard_stop: float = 0.15  # 15% max drawdown → hard stop
    min_live_n: int = 60                # CHANGE: 3 months daily obs for minimum significance

    # Calibration thresholds
    ece_warn: float = 0.08
    ece_suspend: float = 0.15
    brier_warn: float = 0.20
    brier_suspend: float = 0.25

    # Feature drift thresholds (Kolmogorov-Smirnov test)
    drift_ks_pvalue_warn: float = 0.05    # Flag if p < 0.05
    drift_ks_pvalue_suspend: float = 0.01 # Suspend if p < 0.01


CFG = Part9Config()

def _resolve_root() -> str:
    candidates = []
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/content/drive/MyDrive/PriceCallProject"))
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd())
    seen = set()
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        s = str(rp)
        if s == "/content" or s in seen:
            continue
        seen.add(s)
        if rp.exists():
            return s
    return str(Path.cwd().resolve())


def _abs_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((Path(_resolve_root()) / path).resolve())



# ============================================================
# Statistical testing
# ============================================================

def _delong_se_auc(y: np.ndarray, p: np.ndarray) -> float:
    """DeLong variance estimator for AUC (Wilcoxon-Mann-Whitney form).

    Standard reference: DeLong, DeLong & Clarke-Pearson (1988).
    This is strictly larger than the plug-in formula sqrt(AUC*(1-AUC)/n)
    because it accounts for the correlation structure of the U-statistic.
    Under H0 (AUC=0.5) the approximation reduces to sqrt(1/(12*n1) + 1/(12*n0)).

    Returns SE under the *observed* AUC, not under H0.  For a t-stat we want
    the SE of (AUC - 0.5), which DeLong provides directly.
    """
    pos_mask = y.astype(bool)
    neg_mask = ~pos_mask
    n1 = int(pos_mask.sum())
    n0 = int(neg_mask.sum())
    if n1 < 2 or n0 < 2:
        # Fall back to plug-in when one class is too small for DeLong
        auc_est = float(np.mean(p[pos_mask].reshape(-1, 1) > p[neg_mask].reshape(1, -1)))
        return float(np.sqrt(max(auc_est * (1.0 - auc_est), 1e-12) / max(n1 + n0, 1)))
    p_pos = p[pos_mask]
    p_neg = p[neg_mask]
    # Structural components: V10[i] = P(score_pos_i > score_neg), V01[j] = P(score_pos > score_neg_j)
    V10 = np.array([float(np.mean(pi > p_neg) + 0.5 * np.mean(pi == p_neg)) for pi in p_pos])
    V01 = np.array([float(np.mean(p_pos > pj) + 0.5 * np.mean(p_pos == pj)) for pj in p_neg])
    auc = float(np.mean(V10))
    s10 = float(np.var(V10, ddof=1))
    s01 = float(np.var(V01, ddof=1))
    var_auc = s10 / n1 + s01 / n0
    return float(np.sqrt(max(var_auc, 1e-12)))


def t_stat_sign_accuracy(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    base_rate: float | np.ndarray,
    *,
    min_class_n: int = 5,
    n_perm: int = 2000,
) -> Dict:
    """Describe classification quality and run guarded permutation inference.

    ``base_rate`` may be a scalar or a per-observation causal forecast.  The
    latter is required for historical scoring because each decision date has a
    different training-window prevalence.  Directional significance is never
    reported until *both* classes meet ``min_class_n``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    br = np.asarray(base_rate, dtype=float)
    if br.ndim == 0:
        br = np.full(y.shape, float(br), dtype=float)
    else:
        br = np.broadcast_to(br, y.shape).astype(float, copy=False)
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(br)
    y, p, br = y[finite], p[finite], br[finite]
    n = len(y)

    actual_event = y >= 0.5
    n_positive = int(actual_event.sum())
    n_negative = int(n - n_positive)
    inference_eligible = bool(n_positive >= min_class_n and n_negative >= min_class_n)

    _null = {"n": n, "accuracy": np.nan, "balanced_accuracy": np.nan,
             "matthews_corrcoef": np.nan,
             "accuracy_null": "class-balanced label permutation",
             "t_stat_accuracy": np.nan, "p_value_accuracy": np.nan,
             "p_value_accuracy_better": np.nan, "p_value_accuracy_worse": np.nan,
             "auc": np.nan, "t_stat_auc": np.nan,
             "p_value_auc_better": np.nan, "p_value_auc_worse": np.nan,
             "brier": np.nan, "brier_null": np.nan, "brier_skill_score": np.nan,
             "n_positive": n_positive, "n_negative": n_negative,
             "minimum_class_count": int(min_class_n),
             "inference_eligible": inference_eligible,
             "inference_status": "eligible" if inference_eligible else "insufficient_class_counts",
             "better_than_null_acc_5pct": False, "better_than_null_acc_1pct": False,
             "worse_than_null_acc_5pct": False, "worse_than_null_acc_1pct": False,
             "better_than_null_auc_5pct": False, "better_than_null_auc_1pct": False,
             "worse_than_null_auc_5pct": False, "worse_than_null_auc_1pct": False}

    if n < 2:
        return {**_null, "n": n}

    # Brier skill score vs base rate
    brier_model = float(np.mean((y - p) ** 2))
    brier_null = float(np.mean((y - br) ** 2))
    bss = 1.0 - brier_model / (brier_null + 1e-10)

    # Raw accuracy is retained for description, but the inferential null is
    # balanced accuracy. A majority-class forecast must score 0.5, not ~80%.
    pred_event = p >= br
    correct = (pred_event == actual_event).astype(float)
    accuracy = float(correct.mean())
    balanced_accuracy = (
        float(balanced_accuracy_score(actual_event, pred_event))
        if n_positive and n_negative else np.nan
    )
    mcc = float(matthews_corrcoef(actual_event, pred_event))

    # Deterministic label permutations preserve event prevalence and produce
    # directional, finite-sample p-values.  They are descriptive only until the
    # prespecified class-count guard is met.
    rng = np.random.default_rng(42)
    perm_scores = np.empty(n_perm, dtype=float)
    perm_auc = np.empty(n_perm, dtype=float)
    if n_positive and n_negative:
        for i in range(n_perm):
            y_perm = rng.permutation(actual_event)
            perm_scores[i] = balanced_accuracy_score(y_perm, pred_event)
            perm_auc[i] = roc_auc_score(y_perm, p)
    else:
        perm_scores.fill(np.nan)
        perm_auc.fill(np.nan)
    perm_mean = float(np.mean(perm_scores)) if np.isfinite(balanced_accuracy) else np.nan
    perm_std = float(np.std(perm_scores, ddof=1)) if np.isfinite(balanced_accuracy) else np.nan
    t = (
        float((balanced_accuracy - perm_mean) / perm_std)
        if inference_eligible and np.isfinite(perm_std) and perm_std > 1e-12 else np.nan
    )
    p_acc_better = (
        float((1 + np.sum(perm_scores >= balanced_accuracy)) / (n_perm + 1))
        if inference_eligible else np.nan
    )
    p_acc_worse = (
        float((1 + np.sum(perm_scores <= balanced_accuracy)) / (n_perm + 1))
        if inference_eligible else np.nan
    )

    # AUC — requires at least one sample from each class
    if n_positive and n_negative:
        auc = float(roc_auc_score(y.astype(int), p))
    else:
        auc = np.nan

    # AUC t-stat using DeLong SE (Finding #9)
    # SE is meaningless when AUC itself is NaN or when n is too small.
    if inference_eligible and np.isfinite(auc):
        se_auc = _delong_se_auc(y.astype(int), p)
        t_auc  = float((auc - 0.5) / se_auc) if se_auc > 0 else np.nan
    else:
        t_auc = np.nan

    p_auc_better = (
        float((1 + np.sum(perm_auc >= auc)) / (n_perm + 1))
        if inference_eligible and np.isfinite(auc) else np.nan
    )
    p_auc_worse = (
        float((1 + np.sum(perm_auc <= auc)) / (n_perm + 1))
        if inference_eligible and np.isfinite(auc) else np.nan
    )

    # Directional significance flags: these encode whether the model is
    # significantly BETTER or WORSE than the null, rather than collapsing both
    # directions into a single ambiguous "significant" indicator.
    better_acc_5 = bool(inference_eligible and p_acc_better <= 0.05)
    better_acc_1 = bool(inference_eligible and p_acc_better <= 0.01)
    worse_acc_5 = bool(inference_eligible and p_acc_worse <= 0.05)
    worse_acc_1 = bool(inference_eligible and p_acc_worse <= 0.01)
    better_auc_5 = bool(inference_eligible and p_auc_better <= 0.05)
    better_auc_1 = bool(inference_eligible and p_auc_better <= 0.01)
    worse_auc_5 = bool(inference_eligible and p_auc_worse <= 0.05)
    worse_auc_1 = bool(inference_eligible and p_auc_worse <= 0.01)

    return {
        "n": n,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "matthews_corrcoef": mcc,
        "accuracy_null": "class-balanced label permutation",
        "permutation_count": n_perm,
        "t_stat_accuracy": t,
        "p_value_accuracy": p_acc_better,
        "p_value_accuracy_better": p_acc_better,
        "p_value_accuracy_worse": p_acc_worse,
        "auc": auc,
        "t_stat_auc": t_auc,
        "p_value_auc_better": p_auc_better,
        "p_value_auc_worse": p_auc_worse,
        "brier": brier_model,
        "brier_null": brier_null,
        "brier_skill_score": float(bss),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "minimum_class_count": int(min_class_n),
        "inference_eligible": inference_eligible,
        "inference_status": "eligible" if inference_eligible else "insufficient_class_counts",
        "better_than_null_acc_5pct": better_acc_5,
        "better_than_null_acc_1pct": better_acc_1,
        "worse_than_null_acc_5pct": worse_acc_5,
        "worse_than_null_acc_1pct": worse_acc_1,
        "better_than_null_auc_5pct": better_auc_5,
        "better_than_null_auc_1pct": better_auc_1,
        "worse_than_null_auc_5pct": worse_auc_5,
        "worse_than_null_auc_1pct": worse_auc_1,
    }


def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_benchmark: np.ndarray,
    h: int = 1,
) -> Dict:
    """
    Diebold-Mariano test: Is model's forecast error significantly smaller than benchmark?

    Benchmark options:
    1. Naive: always predict base rate (p = 0.20)
    2. Historical average: predict rolling 63-day average

    H0: Model and benchmark have equal forecast accuracy
    H1: Model is significantly better (one-tailed)

    For weekly predictions (h=1 in our rebalanced dataset), standard DM applies.
    """
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_benchmark, dtype=float)
    finite = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[finite], e2[finite]
    n = len(e1)

    if n < 8:
        return {"n": n, "dm_stat": np.nan, "p_value": np.nan}

    # Loss differential
    d = e1**2 - e2**2  # Squared error loss
    d_bar = d.mean()
    d_var = d.var(ddof=1)

    if d_var <= 0:
        return {"n": n, "dm_stat": 0.0, "p_value": 0.5, "model_better": False}

    # Harvey, Leybourne & Newbold (1997) small-sample correction to the DM statistic.
    # HLN multiply the raw DM statistic by sqrt((n + 1 - 2h + h(h-1)/n) / n).
    # At h=1 this reduces to sqrt((n-1)/n), which is negligible for large n but
    # reduces the rejection rate for the small live samples encountered early in
    # deployment.
    hln_correction = float(np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / max(n, 1)) / n, 1e-12)))
    dm_raw = float(d_bar / np.sqrt(d_var / n))
    dm = dm_raw * hln_correction
    # One-tailed: positive DM = model worse, negative = model better
    p_value = float(stats.t.sf(-dm, df=n - 1))  # P(model better)

    return {
        "n": n,
        "dm_stat": dm,
        "p_value": p_value,
        "model_better": bool(dm < 0),
        "significant_5pct": bool(p_value < 0.05),
        "significant_1pct": bool(p_value < 0.01),
        "mean_loss_model": float(np.mean(e1**2)),
        "mean_loss_benchmark": float(np.mean(e2**2)),
        "mse_reduction_pct": float(100 * (1 - np.mean(e1**2) / np.mean(e2**2))),
    }


def ece_score(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        idx = (p >= edges[i]) & (p < (1.0 if i == bins - 1 else edges[i + 1]))
        if idx.sum() == 0:
            continue
        ece += (idx.sum() / len(y)) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


# ============================================================
# Feature drift detection
# ============================================================

def detect_feature_drift(
    historical_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    recent_window: int = 63,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test for each feature:
    H0: recent distribution = historical distribution
    Low p-value → feature has drifted → model may be stale
    """
    results = []
    hist = historical_features.dropna()
    rec = recent_features.dropna().tail(recent_window)

    for col in hist.columns:
        if col not in rec.columns:
            continue
        h_vals = hist[col].dropna().values
        r_vals = rec[col].dropna().values
        if len(h_vals) < 20 or len(r_vals) < 5:
            continue
        ks_stat, ks_pval = stats.ks_2samp(h_vals, r_vals)
        results.append({
            "feature": col,
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "hist_mean": float(h_vals.mean()),
            "recent_mean": float(r_vals.mean()),
            "mean_shift": float(r_vals.mean() - h_vals.mean()),
            "drifted": bool(ks_pval < 0.05),
            "severely_drifted": bool(ks_pval < 0.01),
        })

    df = pd.DataFrame(results).sort_values("ks_pval")
    n_drifted = int((df["drifted"]).sum()) if len(df) else 0
    print(f"[Part 9] Feature drift: {n_drifted}/{len(df)} features drifted (KS p<0.05)")
    return df


# ============================================================
# Performance attribution
# ============================================================

def factor_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Decompose strategy returns into:
    1. Beta to benchmark (market exposure)
    2. Alpha (residual after market)
    3. Factor loadings (if factor_returns provided: value, momentum, quality)
    4. Information Ratio of pure alpha
    """
    r = strategy_returns.dropna()
    b = benchmark_returns.reindex(r.index).dropna()
    idx = r.index.intersection(b.index)
    r, b = r[idx], b[idx]

    if len(r) < 8:
        return {"n": len(r), "alpha": np.nan, "beta": np.nan, "ir": np.nan}

    # OLS: r = alpha + beta * b + epsilon
    slope, intercept, r_val, p_val, se = stats.linregress(b.values, r.values)
    alpha_daily = float(intercept)
    beta = float(slope)
    r_squared = float(r_val ** 2)

    # Annualize alpha
    # FIX: was 252.0 / 7 (inherited from H=7 weekly system). At H=1 daily
    # rebalancing each row is one trading day, so annualization is simply 252.
    periods_per_year = 252.0
    alpha_ann = alpha_daily * periods_per_year

    # IR of pure alpha
    residuals = r.values - (alpha_daily + beta * b.values)
    ir_alpha = float(residuals.mean() / (residuals.std(ddof=1) + 1e-10) * np.sqrt(periods_per_year))

    result = {
        "n": len(r),
        "alpha_daily": alpha_daily,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "r_squared": r_squared,
        "ir_alpha": ir_alpha,
        "t_stat_alpha": float(intercept / (se + 1e-10)),
        "tracking_error_ann": float(residuals.std(ddof=1) * np.sqrt(periods_per_year)),
        "info_ratio": float(r.mean() / (r.std(ddof=1) + 1e-10) * np.sqrt(periods_per_year)),
    }

    # Maximum drawdown
    cumret = (1 + r).cumprod()
    rolling_max = cumret.expanding().max()
    drawdown = (cumret - rolling_max) / rolling_max
    result["max_drawdown"] = float(drawdown.min())
    result["current_drawdown"] = float(drawdown.iloc[-1]) if len(drawdown) > 0 else np.nan

    return result


# ============================================================
# Stopping rules
# ============================================================

def evaluate_stopping_rules(
    live_stats: Dict,
    calibration_stats: Dict,
    drift_df: pd.DataFrame,
    cfg: Part9Config,
) -> Dict:
    """
    Automated stopping rules. Returns:
    - status: ACTIVE | WARN | SUSPEND
    - reasons: list of triggered conditions
    """
    reasons = []
    status = "ACTIVE"

    n = live_stats.get("n", 0)
    if n < cfg.min_live_n:
        return {
            "status": "IMMATURE",
            "reasons": [f"Only {n} live observations ({cfg.min_live_n} required)"],
        }
    if not bool(live_stats.get("inference_eligible", False)):
        return {
            "status": "IMMATURE",
            "reasons": [
                "Class-count gate not met: "
                f"positive={live_stats.get('n_positive', 0)}, "
                f"negative={live_stats.get('n_negative', 0)}, "
                f"minimum each={live_stats.get('minimum_class_count', 5)}"
            ],
        }

    # 1. Statistical edge check
    t_auc = live_stats.get("t_stat_auc", np.nan)
    if np.isfinite(t_auc) and t_auc < cfg.t_stat_suspend:
        reasons.append(f"AUC t-stat={t_auc:.2f} < {cfg.t_stat_suspend} SUSPEND THRESHOLD")
        status = "SUSPEND"
    elif np.isfinite(t_auc) and t_auc < cfg.t_stat_min:
        reasons.append(f"AUC t-stat={t_auc:.2f} < {cfg.t_stat_min} WARN THRESHOLD")
        if status == "ACTIVE":
            status = "WARN"

    # 2. Drawdown check
    max_dd = live_stats.get("max_drawdown", np.nan)
    if np.isfinite(max_dd) and abs(max_dd) > cfg.max_drawdown_hard_stop:
        reasons.append(f"Max drawdown={max_dd:.1%} > {cfg.max_drawdown_hard_stop:.1%} HARD STOP")
        status = "SUSPEND"

    # 3. Calibration check
    ece = calibration_stats.get("ece", np.nan)
    if np.isfinite(ece) and ece > cfg.ece_suspend:
        reasons.append(f"ECE={ece:.4f} > {cfg.ece_suspend} SUSPEND")
        status = "SUSPEND"
    elif np.isfinite(ece) and ece > cfg.ece_warn:
        reasons.append(f"ECE={ece:.4f} > {cfg.ece_warn} WARN")
        if status == "ACTIVE":
            status = "WARN"

    # 4. Feature drift check
    if len(drift_df) > 0:
        severe_drifts = drift_df[drift_df.get("severely_drifted", False)]["feature"].tolist() if "severely_drifted" in drift_df.columns else []
        if len(severe_drifts) > 2:
            reasons.append(f"Feature drift: {len(severe_drifts)} features severely drifted: {severe_drifts[:3]}")
            if status == "ACTIVE":
                status = "WARN"

    # Daily-specific: annual TC drag vs edge check
    annual_drag = live_stats.get("annual_tc_drag_bps", np.nan)
    est_edge_bps = live_stats.get("estimated_annual_edge_bps", np.nan)
    if np.isfinite(annual_drag) and np.isfinite(est_edge_bps) and annual_drag > est_edge_bps:
        reasons.append(f"TC drag ({annual_drag:.0f} bps/yr) exceeds estimated edge ({est_edge_bps:.0f} bps/yr)")
        if status == "ACTIVE":
            status = "WARN"

    if not reasons:
        reasons.append("All checks passed")

    return {"status": status, "reasons": reasons}


# ============================================================
# Main report
# ============================================================


def generate_live_report(cfg: Part9Config) -> Dict:
    """Read prediction log and generate the full live performance report."""
    if not os.path.exists(cfg.predlog_path):
        return {"error": f"Prediction log not found at {cfg.predlog_path}"}

    predlog = pd.read_csv(cfg.predlog_path)
    if "decision_date" in predlog.columns:
        predlog["decision_date"] = pd.to_datetime(predlog["decision_date"], errors="coerce")

    # Use deployment_mode for operational filtering when available (Part 3 now
    # writes both publish_mode and deployment_mode separately). Fall back to
    # publish_mode for logs written before this schema change.
    mode_col = "deployment_mode" if "deployment_mode" in predlog.columns else "publish_mode"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "total_predictions_all_cohorts": len(predlog),
        "total_predictions": len(predlog),
        "n_live_realized": 0,
        "status": "IMMATURE",
        "schema_mode_col": mode_col,
    }

    voo_real_col = next((c for c in ["px_voo_realized", "voo_realized"] if c in predlog.columns), None)
    ief_real_col = next((c for c in ["px_ief_realized", "ief_realized"] if c in predlog.columns), None)
    if voo_real_col is None or ief_real_col is None:
        return {"error": "No realized price columns found in prediction log", "n_live_realized": 0}

    # Exclude rows flagged as legacy H=7 entries. These were written by the old
    # weekly-horizon pipeline; their target_dates have already passed without
    # correct realized prices being backfilled, and including them in the live
    # realized set would pollute attribution with stale / wrong-horizon data.
    if "horizon_legacy" in predlog.columns:
        predlog = predlog[predlog["horizon_legacy"].fillna(0).astype(int) == 0].copy()

    # FIX (F1, Quant-Guild S58 Audit): Re-read prediction_log from disk before
    # computing n_live_realized. Part 9 runs at ~9:35 AM ET during the pipeline;
    # backfill runs at 4:00 PM ET and adds realized prices AFTER Part 9 has already
    # read the predlog. Because Part 9 is the last part before Part 10, and the
    # backfill runs after the pipeline commits, this file on disk may already contain
    # realized prices from a prior backfill session that were not captured in the
    # CI restore step (race condition with the concurrency group).
    #
    # FIX: before computing n_live_realized, re-read the predlog from disk. If the
    # on-disk version has MORE realized rows than the in-memory version (which was
    # loaded from the restored checkpoint), use the disk version. This is safe because:
    #   - We only switch to the disk version if it has MORE data (monotone guard)
    #   - Exception-safe: any disk read failure falls through to the in-memory version
    #   - Closes the 1-run staleness gap without any change to pipeline ordering  [FIX F1/S58]
    try:
        _predlog_disk = pd.read_csv(cfg.predlog_path)
        if "decision_date" in _predlog_disk.columns:
            _predlog_disk["decision_date"] = pd.to_datetime(_predlog_disk["decision_date"], errors="coerce")
        _voo_r = next((c for c in [voo_real_col] if c in _predlog_disk.columns), voo_real_col)
        _ief_r = next((c for c in [ief_real_col] if c in _predlog_disk.columns), ief_real_col)
        if _voo_r in _predlog_disk.columns and _ief_r in _predlog_disk.columns:
            _n_disk = int((_predlog_disk[_voo_r].notna() & _predlog_disk[_ief_r].notna()).sum())
            _n_mem = int((predlog[voo_real_col].notna() & predlog[ief_real_col].notna()).sum()) if voo_real_col in predlog.columns else 0
            if _n_disk > _n_mem:
                print(
                    f"[Part 9] FIX F1/S58: disk predlog has {_n_disk} realized rows vs "
                    f"{_n_mem} in memory — using disk version (prior backfill detected).  [S58 F1]"
                )
                predlog = _predlog_disk
                # Re-acquire column refs after predlog reassignment
                voo_real_col = next((c for c in ["px_voo_realized", "voo_realized"] if c in predlog.columns), voo_real_col)
                ief_real_col = next((c for c in ["px_ief_realized", "ief_realized"] if c in predlog.columns), ief_real_col)
    except Exception as _p9_disk_exc:
        print(f"[Part 9] FIX F1/S58: disk re-read failed ({_p9_disk_exc}) — using in-memory predlog.")

    # A methodological change starts a new evidence cohort.  Old rows remain in
    # the ledger for auditability, but they cannot promote the current model.
    if "model_protocol_version" not in predlog.columns:
        predlog["model_protocol_version"] = LEGACY_PROTOCOL_VERSION
    predlog["model_protocol_version"] = predlog["model_protocol_version"].fillna(
        LEGACY_PROTOCOL_VERSION
    ).astype(str)
    if "evidence_eligible" not in predlog.columns:
        predlog["evidence_eligible"] = 0
    if "horizon_legacy" in predlog.columns:
        predlog = predlog[
            pd.to_numeric(predlog["horizon_legacy"], errors="coerce").fillna(0).astype(int).eq(0)
        ].copy()
    current = predlog["model_protocol_version"].eq(PROTOCOL_VERSION)
    all_realized = predlog[voo_real_col].notna() & predlog[ief_real_col].notna()
    eligible_current = current_evidence_mask(predlog)
    report.update({
        "evidence_cohort": PROTOCOL_VERSION,
        "legacy_predictions": int((~current).sum()),
        "legacy_realized_rows": int((~current & all_realized).sum()),
        "current_cohort_predictions": int(current.sum()),
        "current_cohort_eligible_predictions": int(eligible_current.sum()),
    })
    predlog = predlog[eligible_current].copy()

    realized = predlog[predlog[voo_real_col].notna() & predlog[ief_real_col].notna()].copy()
    n_live = len(realized)
    report["n_live_realized"] = n_live

    if n_live < 2:
        report["message"] = (
            f"Only {n_live} eligible realized predictions are available in the "
            f"{PROTOCOL_VERSION} cohort. "
            f"Accumulate {cfg.min_live_n} before making statistical inferences. "
            f"At daily rebalancing, this takes roughly {cfg.min_live_n / 21:.1f} months."
        )

    voo_pred_col = next((c for c in ["px_voo_call_1d", "px_voo_call_7d"] if c in realized.columns), None)
    if voo_pred_col:
        errors = (pd.to_numeric(realized[voo_pred_col], errors="coerce") - pd.to_numeric(realized[voo_real_col], errors="coerce")).dropna()
        if len(errors):
            report["voo_mae_live"] = float(errors.abs().mean())
            report["voo_rmse_live"] = float(np.sqrt((errors ** 2).mean()))

    live_stats = {}
    req_cols = {"p_final_cal", "px_voo_t", "px_ief_t", voo_real_col, ief_real_col}
    if req_cols.issubset(set(realized.columns)):
        # FIX (Finding B, Audit 2026-04): Part 1 uses a rolling 63-day 20th-percentile
        # quantile as the tail label threshold, not a fixed value.  Each prediction_log
        # row carries the dynamic threshold that was active when Part 1 built the label
        # for that date (stored in the 'tail_threshold' column written by Part 3).
        # The previous code extracted tail_series.iloc[-1] (the most-recent value only)
        # and applied that single scalar to ALL live rows.  For early rows in the log
        # the rolling quantile will have been different, producing inconsistent y_live
        # labels. Current-protocol evidence requires the exact per-row value.
        if "tail_threshold" not in realized.columns:
            raise RuntimeError("Current-protocol evidence is missing the row-level tail_threshold field.")
        thr_series = pd.to_numeric(realized["tail_threshold"], errors="coerce")
        if thr_series.isna().any():
            raise RuntimeError("Current-protocol evidence contains a non-numeric tail_threshold value.")

        br_series = pd.to_numeric(
            realized.get("base_rate", pd.Series([0.20] * len(realized), index=realized.index)),
            errors="coerce",
        ).fillna(0.20)
        br_values = br_series.to_numpy(dtype=float)
        base_rate = float(br_values[-1]) if len(br_values) else 0.20
        pred_p_raw = pd.to_numeric(realized["p_final_cal"], errors="coerce").values
        # FIX (Finding 14, Audit 2026-04-21): use the Platt-recalibrated probability
        # p_regime_recal (written by Part 3) when it is available and well-populated.
        # p_regime_recal is regime-conditional Platt-scaled and is the best-calibrated
        # probability estimate the system produces. The original code exclusively used
        # p_final_cal, leaving the recalibration entirely unused in statistical inference.
        _using_recal = False
        if "p_regime_recal" in realized.columns:
            recal_series = pd.to_numeric(realized["p_regime_recal"], errors="coerce")
            if recal_series.notna().mean() >= 0.5:
                pred_p = recal_series.values
                _using_recal = True
                print("[Part 9] Using p_regime_recal (Platt-scaled + blended) for live statistics.")
            else:
                pred_p = pred_p_raw
        else:
            pred_p = pred_p_raw
        spread_real = (
            pd.to_numeric(realized[voo_real_col], errors="coerce").values / pd.to_numeric(realized["px_voo_t"], errors="coerce").values - 1.0
        ) - (
            pd.to_numeric(realized[ief_real_col], errors="coerce").values / pd.to_numeric(realized["px_ief_t"], errors="coerce").values - 1.0
        )
        # FIX (Finding B): per-row threshold applied element-wise
        y_live = (spread_real < thr_series.values).astype(float)
        m = np.isfinite(pred_p) & np.isfinite(y_live) & np.isfinite(br_values)
        if m.sum() >= 2:
            live_stats = t_stat_sign_accuracy(y_live[m], pred_p[m], br_values[m])
            # Report the median threshold across live rows for transparency
            live_stats["tail_threshold"] = float(np.median(thr_series.values))
            live_stats["base_rate"] = base_rate
            live_stats["probability_source"] = "p_regime_recal" if _using_recal else "p_final_cal"
            active_rets = -spread_real[m] * np.sign(pred_p[m] - br_values[m])
            if len(active_rets) >= 2:
                live_stats["mean_active_return"] = float(np.mean(active_rets))
                # FIX (Finding A, Audit 2026-04): the raw spread return used here overstates
                # estimated_annual_edge_bps by ~370× because it ignores the active weight
                # (~0.04% average when the dead-band is active).  The weight-scaled active
                # return is already computed correctly in Part 2 and stored in
                # part2_g532_summary.json as active_ret_net_mean.  Use that value as the
                # primary source; fall back to the (overstated) spread-based formula only
                # if the Part 2 summary is unavailable, and flag it as unscaled.
                p2_summary_path = cfg.part2_tape_path.replace(
                    "g532_final_consensus_tape.csv", "part2_g532_summary.json"
                )
                _edge_set = False
                if os.path.exists(p2_summary_path):
                    try:
                        with open(p2_summary_path, "r", encoding="utf-8") as _f:
                            _p2s = json.load(_f)
                        _arm = float(_p2s.get("active_ret_net_mean", np.nan))
                        if np.isfinite(_arm):
                            live_stats["estimated_annual_edge_bps"] = float(_arm * 252 * 10000.0)
                            live_stats["estimated_annual_edge_source"] = "part2_active_ret_net_mean"
                            _edge_set = True
                    except Exception:
                        pass
                if not _edge_set:
                    # Fallback: spread-based; warn that it is not weight-scaled.
                    live_stats["estimated_annual_edge_bps"] = float(
                        np.mean(active_rets) * 252 * 10000.0
                    )
                    live_stats["estimated_annual_edge_source"] = "spread_return_unscaled_fallback"
            report["classification_stats_live"] = live_stats
            report["calibration_live"] = {
                "ece": ece_score(y_live[m], pred_p[m]),
                "brier": float(np.mean((y_live[m] - pred_p[m]) ** 2)),
            }

            # FIX (BUG-8, Audit 2026-05-11 — Quant-Guild Part 24):
            # Add base-model comparison track so ensemble value-add can be measured live.
            # When p_regime_recal is the primary signal, also evaluate p_final_cal (raw base)
            # against the same y_live labels. This gives the AUC/accuracy of the raw base
            # model in isolation, making it possible to attribute: how much of the live AUC
            # comes from the base model vs. the blend + Platt recalibration.
            # Without this, it is impossible to verify the ensemble is adding value once
            # live data accumulates — you could have high p_regime_recal AUC entirely due
            # to the base model, with the blend/recal contributing noise.
            if _using_recal:
                m_raw = np.isfinite(pred_p_raw) & np.isfinite(y_live)
                if m_raw.sum() >= 2:
                    base_stats = t_stat_sign_accuracy(y_live[m_raw], pred_p_raw[m_raw], br_values[m_raw])
                    base_stats["tail_threshold"] = float(np.median(thr_series.values))
                    base_stats["base_rate"] = base_rate
                    base_stats["probability_source"] = "p_final_cal"
                    report["classification_stats_live_base_model"] = base_stats
                    print(
                        f"[Part 9] Base model (p_final_cal) live AUC={base_stats.get('auc', np.nan):.4f} | "
                        f"Blended+Platt (p_regime_recal) live AUC={live_stats.get('auc', np.nan):.4f}"
                    )

    if os.path.exists(cfg.part2_tape_path):
        tape = pd.read_csv(cfg.part2_tape_path)
        if {"p_final_cal", "y_rel_tail_voo_vs_ief"}.issubset(set(tape.columns)):
            tape_real = tape[tape.get("y_avail", 1) == 1].dropna(subset=["p_final_cal", "y_rel_tail_voo_vs_ief"])
            if len(tape_real) >= 2:
                if "base_rate" in tape_real.columns:
                    base_rate_bt = pd.to_numeric(tape_real["base_rate"], errors="coerce").fillna(0.20).values
                    report["backtest_baseline_contract"] = "per-row causal training-window prevalence"
                else:
                    base_rate_bt = float(tape_real["y_rel_tail_voo_vs_ief"].mean())
                    report["backtest_baseline_contract"] = "legacy full-sample fallback; not promotion eligible"
                report["classification_stats_backtest"] = t_stat_sign_accuracy(
                    tape_real["y_rel_tail_voo_vs_ief"].values,
                    tape_real["p_final_cal"].values,
                    base_rate_bt,
                )

    if os.path.exists(cfg.part8_cost_path):
        try:
            cost_df = pd.read_csv(cfg.part8_cost_path)
            if not cost_df.empty:
                last_cost = cost_df.iloc[-1]
                report["annual_tc_drag_bps"] = float(last_cost.get("annual_tc_drag_bps", np.nan))
                if live_stats and "estimated_annual_edge_bps" in live_stats:
                    live_stats["annual_tc_drag_bps"] = report["annual_tc_drag_bps"]
        except Exception:
            pass

    # FIX (BUG-1, Audit 2026-05-08): detect_feature_drift() was defined but NEVER called.
    # generate_live_report() was passing pd.DataFrame() (empty) to evaluate_stopping_rules,
    # making the entire KS-test feature drift stopping rule permanently inoperative.
    # Fix: load X_features.parquet from Part 1 and actually call detect_feature_drift().
    # Historical = full feature history; recent = trailing 63 trading days (~3 months).
    _drift_df = pd.DataFrame()
    _x_features_path = os.path.join(cfg.part1_dir, "X_features.parquet")
    if os.path.exists(_x_features_path):
        try:
            _x_all = pd.read_parquet(_x_features_path)
            _x_all.index = pd.to_datetime(_x_all.index, errors="coerce")
            _x_all = _x_all.dropna(how="all").sort_index()
            if len(_x_all) >= 90:
                # historical = everything except last 63 rows; recent = last 63 rows
                _x_hist = _x_all.iloc[:-63]
                _x_rec = _x_all.iloc[-63:]
                _drift_df = detect_feature_drift(_x_hist, _x_rec, recent_window=63)
                report["feature_drift"] = {
                    "n_features_tested": len(_drift_df),
                    "n_drifted_5pct": int(_drift_df["drifted"].sum()) if "drifted" in _drift_df.columns else 0,
                    "n_severely_drifted_1pct": int(_drift_df["severely_drifted"].sum()) if "severely_drifted" in _drift_df.columns else 0,
                    "most_drifted": _drift_df.head(3)[["feature", "ks_stat", "ks_pval", "mean_shift"]].to_dict(orient="records") if len(_drift_df) else [],
                }
        except Exception as _drift_exc:
            print(f"[Part 9] Feature drift detection failed: {_drift_exc}")

    stopping = evaluate_stopping_rules(
        live_stats,
        report.get("calibration_live", {}),
        _drift_df,
        cfg,
    )
    report["health_status"] = stopping["status"]
    report["health_reasons"] = stopping["reasons"]

    print("=" * 70)
    print("PART 9 — Live Attribution Report")
    print("=" * 70)
    print(f"Live realized: {n_live} / {len(predlog)} predictions")
    print(f"Health status: {report['health_status']}")
    for reason in stopping["reasons"]:
        print(f"  • {reason}")

    if live_stats:
        print(f"\nLive classification (N={live_stats['n']}):")
        print(f"  AUC:         {live_stats.get('auc', np.nan):.4f}  (t={live_stats.get('t_stat_auc', np.nan):.2f})")
        print(f"  Direction:   {live_stats.get('accuracy', np.nan):.2%}  (t={live_stats.get('t_stat_accuracy', np.nan):.2f})")
        print(f"  BSS:         {live_stats.get('brier_skill_score', np.nan):.4f}")

    return report

def main() -> int:
    cfg = Part9Config()
    cfg = dataclasses.replace(cfg, predlog_path=_abs_path(cfg.predlog_path))
    cfg = dataclasses.replace(cfg, part2_tape_path=_abs_path(cfg.part2_tape_path))
    cfg = dataclasses.replace(cfg, part6_dir=_abs_path(cfg.part6_dir))
    cfg = dataclasses.replace(cfg, out_dir=_abs_path(cfg.out_dir))
    cfg = dataclasses.replace(cfg, part8_cost_path=_abs_path(cfg.part8_cost_path))
    cfg = dataclasses.replace(cfg, part1_dir=_abs_path(cfg.part1_dir))
    os.makedirs(cfg.out_dir, exist_ok=True)

    report = generate_live_report(cfg)

    out_path = os.path.join(cfg.out_dir, "live_attribution_report.json")
    write_json_strict(out_path, report)

    print(f"\n✅ PART 9 COMPLETE → {out_path}")
    return 0


if __name__ == "__main__":
    main()
    
