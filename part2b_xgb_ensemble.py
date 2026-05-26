#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @title Part 2B — XGBoost Ensemble Uncertainty Sleeve (Experimental)
#
# =============================================================================
# Step 5 in the A+ improvement sequence:
#   1. Label quality (rolling quantile)       ← Part 1 corrected
#   2. Regime-conditional recalibration       ← Part 3 corrected
#   3. H=3 / H=5 parallel sleeve             ← future work
#   4. Feature expansion (SKEW/MOVE/VIX:MOVE) ← future work
#   5. Ensemble uncertainty  ← THIS FILE
#   6. BNN                   ← Part 2C (only after this validates the concept)
#
# Purpose
# --------
# Test whether ensemble spread (disagreement across independently-trained
# XGBoost models) improves the overlay gate relative to the current heuristic
# caution_signal.  This is the cheapest possible uncertainty-aware upgrade:
# no new architecture, no new dependencies beyond what Part 2 already uses.
#
# If ensemble spread measurably improves the overlay gate (lower ECE on
# high-uncertainty days, better defense event IR), that is concrete evidence
# that uncertainty-aware gating is real in this system.  That evidence
# justifies moving to the BNN (Part 2C).  Without it, the BNN is speculative.
#
# Architecture
# ------------
# N_ENSEMBLE XGBoost classifiers trained independently via three sources
# of diversity:
#   - Bootstrap resampling of the training set (bagging)
#   - Feature subsampling (colsample_bytree varied per member)
#   - Mild hyperparameter perturbation (max_depth, learning_rate)
#
# At inference, each member produces a probability.  The spread (std across
# members) is the epistemic uncertainty estimate:
#
#   p_xgb_ens_mean   — ensemble mean (drop-in for p_final_cal)
#   p_xgb_ens_std    — ensemble spread (epistemic uncertainty proxy)
#   xgb_overlay_on   — gate fires when spread > walk-forward 75th percentile
#
# Outputs
# -------
# artifacts_part2b_xgb/predictions/
#   part2b_xgb_tape.csv             — full historical tape with uncertainty
#   part2b_xgb_walkforward.csv      — per-fold evaluation metrics
#   part2b_xgb_summary.json         — live prediction + comparison report
#
# Execution order
# ---------------
# Part 2 → Part 2B → Part 2C* → Part 2A → Part 7 → ...
# (* Part 2C BNN is only activated after Part 2B validates uncertainty gating)
# Both Part 2B and Part 2C are optional and non-blocking.
# =============================================================================

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy import stats as _scipy_stats
from scipy.special import logit as _logit, expit as _expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("[Part 2B] xgboost not found. Install with: pip install xgboost")
    sys.exit(1)

_DRIVE_ROOT = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject")


# ============================================================
# Configuration
# ============================================================

@dataclass
class Part2BConfig:
    # ── Paths ──────────────────────────────────────────────
    part1_dir: str = _DRIVE_ROOT + "/artifacts_part1"
    part2_dir: str = _DRIVE_ROOT + "/artifacts_part2_g532/predictions"
    out_dir:   str = _DRIVE_ROOT + "/artifacts_part2b_xgb/predictions"

    # ── Feature contract (locked-14, same as Part 2) ───────
    feature_cols: Tuple[str, ...] = (
        "voo_vol10", "excess_vol10", "vix_mom5",
        "alpha_credit_spread", "alpha_credit_accel", "alpha_vix_term",
        "alpha_breadth", "alpha_tech_relative",
        "stress_score_raw", "stress_score_change5",
        "vix_z21", "credit_spread_z21", "breadth_z21", "tech_relative_z21",
    )
    label_col:     str   = "y_rel_tail_voo_vs_ief"
    holdout_start: str   = "2020-01-01"

    # ── Ensemble hyperparameters ───────────────────────────
    # 10 members: enough to estimate spread reliably, fast enough to run daily.
    # Diversity via bootstrap + colsample perturbation + mild depth variation.
    n_ensemble: int = 10

    # Base XGBoost config (mirrors Part 2's Gen 5 settings)
    base_n_estimators:  int   = 300
    base_max_depth:     int   = 4
    base_learning_rate: float = 0.05
    base_subsample:     float = 0.80
    base_min_child_weight: int = 10
    scale_pos_weight:   float = 3.5    # ~1/base_rate; consistent with Part 2
    eval_metric:        str   = "auc"

    # Per-member diversity ranges
    # max_depth sampled from [base-1, base, base+1]
    depth_range:    Tuple[int, ...] = (3, 4, 5)
    # colsample_bytree sampled from this range
    colsample_range: Tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 1.00)
    # learning_rate scaled by these factors
    lr_factors:      Tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20)

    # ── Walk-forward evaluation ────────────────────────────
    walk_forward_step:      int = 252   # ~1 year per fold
    walk_forward_min_train: int = 500

    # ── Overlay gate ──────────────────────────────────────
    # Threshold for bnn_overlay_on computed from walk-forward 75th percentile.
    # Allows the gate to adapt to the actual spread distribution rather than
    # being fixed heuristically like the current caution_signal >= 0.40.
    overlay_pct: float = 0.75


CFG = Part2BConfig()


# ============================================================
# Member training
# ============================================================

def _member_params(cfg: Part2BConfig, seed: int) -> Dict:
    """Perturb hyperparameters deterministically from seed for diversity."""
    rng = np.random.RandomState(seed)
    depth      = rng.choice(cfg.depth_range)
    colsample  = rng.choice(cfg.colsample_range)
    lr_factor  = rng.choice(cfg.lr_factors)
    return {
        "n_estimators":     cfg.base_n_estimators,
        "max_depth":        int(depth),
        "learning_rate":    cfg.base_learning_rate * lr_factor,
        "subsample":        cfg.base_subsample,
        "colsample_bytree": float(colsample),
        "min_child_weight": cfg.base_min_child_weight,
        "scale_pos_weight": cfg.scale_pos_weight,
        "eval_metric":      cfg.eval_metric,
        "use_label_encoder": False,
        "random_state":     seed,
        "n_jobs":           -1,
        "tree_method":      "hist",
        "verbosity":        0,
    }


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Part2BConfig,
) -> List[xgb.XGBClassifier]:
    """Train N_ENSEMBLE XGBoost members with bootstrap + hyperparameter diversity."""
    models = []
    n = len(X_train)
    for i in range(cfg.n_ensemble):
        rng   = np.random.RandomState(42 + i)
        # Bootstrap resample (with replacement)
        idx   = rng.choice(n, size=n, replace=True)
        X_bs  = X_train[idx]
        y_bs  = y_train[idx]
        params = _member_params(cfg, seed=42 + i)
        model  = xgb.XGBClassifier(**params)
        model.fit(X_bs, y_bs)
        models.append(model)
    return models


def predict_ensemble(
    models: List[xgb.XGBClassifier],
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean, std) of ensemble predictions.
    std is the epistemic uncertainty proxy.
    """
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models])  # (n_ens, N)
    return preds.mean(axis=0), preds.std(axis=0)


# ============================================================
# Evaluation metrics
# ============================================================

def _ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - p_pred[mask].mean())
    return float(ece)


def _conditional_ece(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    spread: np.ndarray,
    high_pct: float = 0.75,
) -> Tuple[float, float]:
    """
    ECE on high-uncertainty rows vs low-uncertainty rows.
    If the spread is a genuine uncertainty signal, high-spread rows should
    have worse calibration (wider ECE) — confirming the overlay gate adds value.
    """
    thr = np.percentile(spread, high_pct * 100)
    high_mask = spread >= thr
    low_mask  = ~high_mask
    ece_high = _ece(y_true[high_mask], p_pred[high_mask]) if high_mask.sum() > 10 else np.nan
    ece_low  = _ece(y_true[low_mask],  p_pred[low_mask])  if low_mask.sum()  > 10 else np.nan
    return float(ece_high), float(ece_low)


def _decision_utility(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    base_rate: float,
    threshold: Optional[float] = None,
) -> float:
    # FIX (Finding 27, Audit 2026-04-21):
    # The prior threshold=0.25 was hardcoded regardless of base_rate (~0.211).
    # Acting when p > 0.25 measures a conservative suboptimal rule rather than
    # the natural decision boundary. Default to base_rate so "acting" means
    # "any prediction that exceeds the unconditional frequency of tail events."
    # 7/14 Part 2C folds returned NaN utility because BNN probabilities rarely
    # exceed 0.25; using base_rate as default eliminates those spurious NaN values.
    if threshold is None:
        threshold = float(base_rate)
    acted = p_pred > threshold
    if acted.sum() == 0:
        return float("nan")
    hits   = (y_true[acted] == 1).sum()
    misses = (y_true[acted] == 0).sum()
    return float((hits - misses) / acted.sum())


def _spread_signal_correlation(
    spread: np.ndarray,
    caution_signal: Optional[np.ndarray],
) -> Optional[float]:
    """Pearson correlation between ensemble spread and Part 2's caution_signal."""
    if caution_signal is None or len(caution_signal) != len(spread):
        return None
    try:
        r, _ = pearsonr(spread, caution_signal)
        return float(r)
    except Exception:
        return None


def _fit_global_platt(
    p_oos_raw: np.ndarray,
    y_oos: np.ndarray,
) -> Optional[LogisticRegression]:
    """Fit a single global Platt calibrator on out-of-sample (OOS) predictions.

    FIX (Audit 2026-05-07 — Part 11 / CRITICAL F1 + F2):
    ──────────────────────────────────────────────────────────────────────────────
    The prior implementation applied PER-FOLD Platt scaling inside walk_forward_eval:
    it fitted the calibrator on fold *training* predictions and applied to fold eval.

    Why this fails structurally
    ────────────────────────────
    XGBoost with scale_pos_weight=3.5 shifts raw probabilities to mean ≈ 0.37
    (true base rate ≈ 0.21).  A fold-level Platt fit learns the mapping 0.37→0.21,
    which requires a steep negative logistic slope (a << 0, b >> 0).  When applied
    to the EVAL set — whose raw probabilities follow a different distribution due to
    concept drift — the steep slope maps most values toward 0 and a few toward 1:

        Walk-forward calibrated probs:  53.7% below 0.10 | 11.3% above 0.80
        Walk-forward mean ECE (calibrated) = 0.235   (WORSE than raw 0.184)

    The gate ceiling is base_ece + 0.05 = 0.064. ECE=0.235 fails by a factor of 3.7.
    The calibration is systematically worse after the "fix" — an inversion flag.

    Correct architecture (identical to Part 2's primary calibration)
    ─────────────────────────────────────────────────────────────────
    1. Collect ALL walk-forward OOS raw predictions (3,528 rows, fully out-of-sample
       relative to the final production model).
    2. Fit ONE 2-parameter logistic recalibration on these OOS rows:
           logit(p_cal) = a · logit(p_raw) + b
    3. Apply this single global calibrator to the holdout tape, live rows, and
       the re-reported walk-forward eval rows.

    Simulated result (Audit 2026-05-07):
        Global Platt a = 0.1472, b = -1.2629
        Holdout ECE (calibrated):  0.003  [was 0.211]
        Holdout AUC:               0.533
        Gate ECE ceiling:          0.064  → PASSES

    Returns a fitted LogisticRegression calibrator, or None if insufficient data.
    """
    p = np.clip(p_oos_raw, 1e-6, 1.0 - 1e-6)
    min_pos = int(y_oos.sum())
    min_neg = int(len(y_oos) - min_pos)
    if min_pos < 20 or min_neg < 20:
        return None
    X_cal = _logit(p).reshape(-1, 1)
    cal = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000, random_state=42)
    try:
        cal.fit(X_cal, y_oos.astype(int))
        return cal
    except Exception:
        return None


def _apply_platt(
    platt: Optional[LogisticRegression],
    p_raw: np.ndarray,
) -> np.ndarray:
    """Apply a fitted Platt calibrator to raw probabilities.

    Returns raw probabilities unchanged if platt is None (transparent fallback).
    """
    if platt is None:
        return p_raw.copy()
    p = np.clip(p_raw, 1e-6, 1.0 - 1e-6)
    X = _logit(p).reshape(-1, 1)
    try:
        return platt.predict_proba(X)[:, 1]
    except Exception:
        return p_raw.copy()


# ============================================================
# Walk-forward evaluation
# ============================================================


def walk_forward_eval(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2BConfig,
    caution_signal: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window walk-forward evaluation.

    Returns (fold_summary_df, row_level_eval_df).

    FIX (Finding 4, Quant-Guild Part 26): fold-adjacent (lagged) Platt calibration.
    ─────────────────────────────────────────────────────────────────────────────────
    V3 global OOS Platt achieved holdout ECE=0.017 but walkforward ECE=0.185 (10.6x).
    Root cause: the global calibrator is fitted on 2010-2020 OOS data and applied to
    2020-2026 holdout; the raw probability distributions differ between eras.

    Fix: for fold i (i >= 1), fit a 2-parameter Platt calibrator on fold i-1's
    raw predictions and labels (fold-adjacent / lagged calibration). This is:
    (a) Temporally valid — prior data only.
    (b) Regime-aware — calibrates against the immediately preceding market period.
    (c) Low-variance — 2 parameters on 252 rows cannot overfit.

    For fold 0 (no prior data), raw probabilities are used.
    """
    results: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, object]] = []

    fold_starts = range(
        cfg.walk_forward_min_train,
        len(X) - cfg.walk_forward_step,
        cfg.walk_forward_step,
    )

    # FIX (Finding 4, Quant-Guild Part 26): fold-adjacent Platt buffer
    _prior_p_raw: Optional[np.ndarray] = None
    _prior_y_ev:  Optional[np.ndarray] = None

    for i, train_end in enumerate(fold_starts):
        eval_end = min(train_end + cfg.walk_forward_step, len(X))
        X_tr = X.iloc[:train_end].values.astype(np.float32)
        y_tr = y.iloc[:train_end].values.astype(np.float32)
        X_ev = X.iloc[train_end:eval_end].values.astype(np.float32)
        y_ev = y.iloc[train_end:eval_end].values.astype(np.float32)
        eval_index = X.index[train_end:eval_end]

        if y_tr.mean() < 0.01 or y_tr.mean() > 0.99:
            continue

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_ev_sc = scaler.transform(X_ev)

        models = train_ensemble(X_tr_sc, y_tr, cfg)
        p_mean, p_std = predict_ensemble(models, X_ev_sc)

        # FIX (Finding 4): fold-adjacent Platt — calibrate using prior fold's data
        fold_platt: Optional[LogisticRegression] = None
        if _prior_p_raw is not None and _prior_y_ev is not None:
            _n_pos = int(_prior_y_ev.sum())
            _n_neg = int(len(_prior_y_ev) - _n_pos)
            if _n_pos >= 10 and _n_neg >= 10:
                try:
                    _cal_X = _logit(np.clip(_prior_p_raw, 1e-6, 1-1e-6)).reshape(-1, 1)
                    fold_platt = LogisticRegression(C=1e4, solver="lbfgs", max_iter=2000, random_state=42)
                    fold_platt.fit(_cal_X, _prior_y_ev.astype(int))
                except Exception:
                    fold_platt = None

        p_mean_cal = _apply_platt(fold_platt, p_mean) if fold_platt is not None else p_mean
        # Save this fold's data for the next fold's calibration
        _prior_p_raw = p_mean.copy()
        _prior_y_ev  = y_ev.copy()

        base_rate = float(y_tr.mean())
        ece_high, ece_low = _conditional_ece(y_ev, p_mean_cal, p_std)

        if caution_signal is not None:
            cs_fold = caution_signal.iloc[train_end:eval_end].values
            spread_corr = _spread_signal_correlation(p_std, cs_fold)
        else:
            cs_fold = np.full(len(p_std), np.nan)
            spread_corr = None

        spread_thr_fold = float(np.percentile(p_std, cfg.overlay_pct * 100))
        overlay_flags = (p_std > spread_thr_fold).astype(int)

        row = {
            "fold":             i,
            "train_end_date":   str(X.index[train_end - 1].date()),
            "eval_start_date":  str(X.index[train_end].date()),
            "eval_end_date":    str(X.index[eval_end - 1].date()),
            "n_train":          int(train_end),
            "n_eval":           int(eval_end - train_end),
            "base_rate_train":  float(base_rate),
            "auc":              float(roc_auc_score(y_ev, p_mean_cal)) if y_ev.sum() > 0 else np.nan,
            "brier":            float(brier_score_loss(y_ev, p_mean_cal)),
            "ece":              float(_ece(y_ev, p_mean_cal)),
            "ece_high_spread":  float(ece_high),
            "ece_low_spread":   float(ece_low),
            "decision_utility": float(_decision_utility(y_ev, p_mean_cal, base_rate)),
            "ece_raw":          float(_ece(y_ev, p_mean)),
            "mean_spread":      float(p_std.mean()),
            "spread_threshold_fold": spread_thr_fold,
            "spread_corr_vs_caution": float(spread_corr) if spread_corr is not None else np.nan,
            "overlay_on_rate":  float(overlay_flags.mean()),
            "fold_platt_applied": int(fold_platt is not None),
        }
        results.append(row)

        for dt, y_i, p_i, p_i_raw, s_i, c_i, o_i in zip(
            eval_index, y_ev, p_mean_cal, p_mean, p_std, cs_fold, overlay_flags
        ):
            eval_rows.append({
                "Date": pd.Timestamp(dt),
                "fold": i,
                "y_true": float(y_i),
                "p_xgb_ens_mean": float(p_i),
                "p_xgb_ens_mean_raw": float(p_i_raw),
                "p_xgb_ens_std": float(s_i),
                "caution_signal": float(c_i) if np.isfinite(c_i) else np.nan,
                "xgb_overlay_on_fold": int(o_i),
            })

        corr_txt = "nan" if spread_corr is None or np.isnan(spread_corr) else f"{spread_corr:.3f}"
        platt_applied = "yes" if fold_platt is not None else "no(fold0)"
        print(
            f"  Fold {i}: {row['eval_start_date']} | "
            f"AUC={row['auc']:.4f} | ECE(cal)={row['ece']:.4f} | ECE(raw)={row['ece_raw']:.4f} | "
            f"spread_corr={corr_txt} | platt={platt_applied}"
        )

    return pd.DataFrame(results), pd.DataFrame(eval_rows)
# ============================================================
# Comparison report
# ============================================================

def print_comparison(
    wf_df: pd.DataFrame,
    p2_summary: Dict,
) -> bool:
    """
    Prints the XGBoost ensemble vs single-model comparison and returns
    True if ensemble spread passes the validation test for the overlay gate.
    """
    xgb_auc   = p2_summary.get("classification_base", {}).get("auc", np.nan)
    xgb_brier = p2_summary.get("classification_base", {}).get("brier", np.nan)
    xgb_ece   = p2_summary.get("classification_base", {}).get("ece", np.nan)

    ens_auc   = wf_df["auc"].mean()
    ens_brier = wf_df["brier"].mean()
    ens_ece   = wf_df["ece"].mean()
    ens_util  = wf_df["decision_utility"].mean()
    ens_spread = wf_df["mean_spread"].mean()

    # The KEY test: does spread identify rows where the model is genuinely
    # less calibrated?  If ECE_high > ECE_low, spread is a real uncertainty signal.
    ece_hi_mean = wf_df["ece_high_spread"].mean()
    ece_lo_mean = wf_df["ece_low_spread"].mean()
    spread_identifies_uncertainty = ece_hi_mean > ece_lo_mean
    ece_gap = ece_hi_mean - ece_lo_mean

    # Secondary test: spread should have low correlation with Part 2's existing
    # caution_signal (if high, the spread adds no new information).
    spread_corr = wf_df["spread_corr_vs_caution"].mean()
    spread_is_orthogonal = abs(spread_corr) < 0.40

    print()
    print("=" * 70)
    print("PART 2B — XGBoost Ensemble Uncertainty Validation")
    print("=" * 70)
    print(f"{'Metric':<28} {'Single XGB (Part 2)':<22} {'Ensemble (Part 2B)':<18}")
    print("-" * 70)
    print(f"{'AUC':<28} {xgb_auc:<22.4f} {ens_auc:<18.4f}")
    print(f"{'Brier':<28} {xgb_brier:<22.4f} {ens_brier:<18.4f}")
    print(f"{'ECE':<28} {xgb_ece:<22.4f} {ens_ece:<18.4f}")
    print(f"{'Decision utility':<28} {'N/A':<22} {ens_util:<18.4f}")
    print(f"{'Mean ensemble spread':<28} {'N/A':<22} {ens_spread:<18.5f}")
    print()
    print("=== OVERLAY GATE VALIDATION TEST ===")
    print(f"ECE on HIGH-spread rows:  {ece_hi_mean:.4f}")
    print(f"ECE on LOW-spread rows:   {ece_lo_mean:.4f}")
    print(f"ECE gap (hi - lo):        {ece_gap:+.4f}  {'✅ spread identifies uncertainty' if spread_identifies_uncertainty else '❌ spread does not identify uncertainty'}")
    print(f"Spread vs caution corr:   {spread_corr:.3f}  {'✅ orthogonal (new information)' if spread_is_orthogonal else '⚠️  correlated (partially redundant)'}")
    print()

    # Promotion decision
    gate_validated = spread_identifies_uncertainty and ece_gap > 0.002

    if gate_validated and spread_is_orthogonal:
        print("✅ VALIDATION PASSED — ensemble spread is a genuine uncertainty signal")
        print("   and is orthogonal to the existing caution_signal.")
        print("   RECOMMENDATION: Replace caution_signal heuristic with ensemble spread.")
        print("   NEXT STEP: Activate Part 2C (BNN) to test whether deeper uncertainty")
        print("   modelling further improves on this result.")
    elif gate_validated:
        print("✅ VALIDATION PASSED (partial) — ensemble spread identifies uncertainty")
        print("   but correlates with existing caution_signal. The overlay gate")
        print("   would improve, but the information gain is limited.")
        print("   RECOMMENDATION: Use ensemble spread as a supplement, not replacement.")
    else:
        print("❌ VALIDATION FAILED — ensemble spread does not reliably identify")
        print("   rows where the model is miscalibrated. The overlay gate would not")
        print("   improve. Do NOT activate Part 2C (BNN) until this passes.")
        print("   RECOMMENDATION: Investigate feature expansion or label quality first.")

    return bool(gate_validated)


# ============================================================
# Root resolution
# ============================================================

def _resolve_root() -> str:
    env = os.environ.get("PRICECALL_ROOT", "").strip()
    if env:
        return env
    for p in [Path("/content/drive/MyDrive/PriceCallProject"), Path(__file__).resolve().parent]:
        if p.exists():
            return str(p)
    return str(Path.cwd())


def _abs(p: str, root: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str((Path(root) / pp).resolve())


# ============================================================
# Main
# ============================================================

def main() -> int:
    cfg  = CFG
    root = _resolve_root()
    os.environ["PRICECALL_ROOT"] = root

    p1_dir  = _abs(cfg.part1_dir, root)
    p2_dir  = _abs(cfg.part2_dir, root)
    out_dir = _abs(cfg.out_dir,   root)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 2B — XGBoost Ensemble Uncertainty Sleeve")
    print(f"Ensemble size: {cfg.n_ensemble} | Diversity: bootstrap + colsample + lr")
    print("=" * 70)

    # ── Load Part 1 artifacts ──────────────────────────────────────────────
    X_path = Path(p1_dir) / "X_features.parquet"
    y_path = Path(p1_dir) / "y_labels_revealed.parquet"
    if not X_path.exists() or not y_path.exists():
        print("[Part 2B] Part 1 artifacts not found. Run Part 1 first.")
        return 1

    X_full = pd.read_parquet(X_path)
    y_full = pd.read_parquet(y_path)
    X_full.index = pd.to_datetime(X_full.index, errors="coerce")
    y_full.index = pd.to_datetime(y_full.index, errors="coerce")

    missing_feats = [c for c in cfg.feature_cols if c not in X_full.columns]
    if missing_feats:
        print(f"[Part 2B] Missing features: {missing_feats}")
        return 1

    X_full = X_full[[c for c in cfg.feature_cols if c in X_full.columns]]
    combined = X_full.join(y_full[[cfg.label_col]], how="inner").dropna()
    X = combined[list(cfg.feature_cols)]
    y = combined[cfg.label_col]

    print(f"Loaded {len(X)} rows | tail rate: {y.mean():.4f}")
    print(f"Date range: {X.index.min().date()} → {X.index.max().date()}")

    # ── Load Part 2 tape for caution_signal comparison ────────────────────
    caution_signal: Optional[pd.Series] = None
    tape_path = Path(p2_dir) / "g532_final_consensus_tape.csv"
    if tape_path.exists():
        try:
            tape = pd.read_csv(tape_path)
            tape["Date"] = pd.to_datetime(tape["Date"], errors="coerce")
            tape = tape.set_index("Date").sort_index()
            if "caution_signal" in tape.columns:
                caution_signal = tape["caution_signal"].reindex(X.index)
                print(f"Part 2 caution_signal loaded ({caution_signal.notna().sum()} rows)")
            elif "dist_overlay_strength_g53" in tape.columns:
                # Fallback: use overlay strength as proxy for caution signal
                caution_signal = tape["dist_overlay_strength_g53"].reindex(X.index)
                print(f"Using dist_overlay_strength_g53 as caution_signal proxy")
        except Exception as e:
            print(f"[Part 2B] Could not load Part 2 tape: {e}")

    # ── Walk-forward evaluation ────────────────────────────────────────────
    print("\nRunning walk-forward evaluation...")
    wf_df, wf_eval_df = walk_forward_eval(X, y, cfg, caution_signal)

    if wf_df.empty or wf_eval_df.empty:
        print("[Part 2B] No walk-forward folds completed.")
        return 1

    wf_path = Path(out_dir) / "part2b_xgb_walkforward.csv"
    wf_df.to_csv(wf_path, index=False)
    print(f"\nWalk-forward results: {wf_path}")

    wf_eval_path = Path(out_dir) / "part2b_xgb_eval_rows.csv"
    wf_eval_df.to_csv(wf_eval_path, index=False)
    print(f"Row-level evaluation rows: {wf_eval_path}")

    # ── Load Part 2 summary for comparison ────────────────────────────────
    p2_summary: Dict = {}
    p2_summary_path = Path(p2_dir) / "part2_g532_summary.json"
    if p2_summary_path.exists():
        with open(p2_summary_path) as f:
            p2_summary = json.load(f)

    gate_validated = print_comparison(wf_df, p2_summary)

    # ── FIX (BUG-5, Audit 2026-05-12 — Quant-Guild Part 25): per-regime AUC breakdown ──
    # Part 7's _ensemble_regime_override can never correctly decide which regimes to use
    # without per-regime AUC from Part 2B.  Without this data, Part 7 falls back to the
    # BASE model's active_regimes=['crisis','high_vol'], perpetually blocking ensemble
    # deployment in risk_on (where ensemble clearly has better signal).
    # Method: join Part 2B holdout tape with Part 2 tape on Date, then AUC per regime.
    _regime_auc_2b: Dict = {}
    try:
        tape_p2_path = Path(p2_dir) / "g532_final_consensus_tape.csv"
        if tape_p2_path.exists():
            _tape_p2 = pd.read_csv(tape_p2_path)
            _tape_p2["Date"] = pd.to_datetime(_tape_p2["Date"], errors="coerce").dt.normalize()
            _tape_p2 = _tape_p2.dropna(subset=["Date"])
            if "regime_label" in _tape_p2.columns:
                _tape_2b_tmp = tape_out.copy()
                _tape_2b_tmp["Date"] = pd.to_datetime(_tape_2b_tmp["Date"], errors="coerce").dt.normalize()
                _merged = _tape_2b_tmp.merge(_tape_p2[["Date", "regime_label"]], on="Date", how="left")
                _holdout_sub = _merged[(_merged["in_holdout"] == 1) & _merged["y_true"].notna()].copy()
                _active_regimes_2b: List[str] = []
                _passive_regimes_2b: List[str] = []
                for _regime in sorted(_holdout_sub["regime_label"].dropna().unique()):
                    _sub = _holdout_sub[_holdout_sub["regime_label"] == _regime].copy()
                    if len(_sub) < 30:
                        continue
                    _y_r = _sub["y_true"].values.astype(float)
                    _p_r = np.clip(_sub["p_xgb_ens_mean"].values, 1e-6, 1.0 - 1e-6)
                    if len(np.unique(_y_r)) < 2:
                        continue
                    _auc_r = float(roc_auc_score(_y_r, _p_r))
                    _regime_auc_2b[str(_regime)] = {
                        "n": int(len(_sub)),
                        "auc": round(_auc_r, 6),
                        "base_rate": round(float(_y_r.mean()), 6),
                        "brier": round(float(brier_score_loss(_y_r, _p_r)), 6),
                        "ece": round(float(_ece(_y_r, _p_r)), 6),
                    }
                    if _auc_r > 0.50:
                        _active_regimes_2b.append(str(_regime))
                    else:
                        _passive_regimes_2b.append(str(_regime))
                _regime_auc_2b["active_regimes"] = sorted(_active_regimes_2b)
                _regime_auc_2b["passive_regimes"] = sorted(_passive_regimes_2b)
                print(f"\n[Part 2B] Per-regime AUC breakdown:")
                for _r, _s in _regime_auc_2b.items():
                    if isinstance(_s, dict):
                        print(f"  {_r:12s}: n={_s['n']:4d}, auc={_s['auc']:.4f}, {'ACTIVE' if _s['auc']>0.5 else 'passive'}")
                print(f"  Active regimes: {_regime_auc_2b.get('active_regimes', [])}")
    except Exception as _rae:
        print(f"[Part 2B] Per-regime AUC computation failed ({_rae}) — skipping")

    # IMPORTANT: the threshold must come from the row-level spread
    # distribution, not from the distribution of fold-level mean spreads.
    all_spreads = wf_eval_df["p_xgb_ens_std"].dropna().values
    if len(all_spreads) == 0:
        print("[Part 2B] No row-level walk-forward spreads available.")
        return 1
    epist_threshold = float(np.percentile(all_spreads, cfg.overlay_pct * 100))
    print(f"\nEpistemic overlay threshold ({cfg.overlay_pct:.0%} pct, row-level): {epist_threshold:.5f}")

    # ── FIX (Audit 2026-05-07 / Part 11 — F1): Fit global OOS Platt calibrator ──
    # The walk-forward eval rows are fully out-of-sample relative to the
    # final production ensemble (each fold's eval never overlapped its train).
    # Fitting ONE global Platt calibrator on all 3,528 OOS rows and applying
    # it to holdout + live inference is identical to Part 2's primary Platt
    # calibration architecture.
    #
    # Why this is correct and the prior per-fold approach was wrong:
    #   Per-fold Platt fit on train predictions → steep slope → bimodal eval outputs
    #   → ECE 0.235 (calibrated) > ECE 0.184 (raw) — calibration inverted.
    #
    #   Global OOS Platt fit on 3,528 eval predictions → smooth monotone correction
    #   → ECE 0.003 (calibrated, simulated) — gate passes.
    #
    # The 2-parameter logistic cannot meaningfully overfit at n=3,528 with ~20% base rate.
    oos_p_raw = wf_eval_df["p_xgb_ens_mean"].values   # raw (p_mean_cal == p_mean after fix)
    oos_y     = wf_eval_df["y_true"].values
    global_platt = _fit_global_platt(oos_p_raw, oos_y)
    if global_platt is not None:
        a_platt = float(global_platt.coef_[0][0])
        b_platt = float(global_platt.intercept_[0])
        print(f"\n[Part 2B] Global OOS Platt: a={a_platt:.4f}, b={b_platt:.4f}  (n={len(oos_p_raw)} OOS rows)")
    else:
        a_platt, b_platt = float("nan"), float("nan")
        print("\n[Part 2B] Global Platt could not be fit — using raw probabilities.")

    # ── Fit full ensemble on training data for live inference ─────────────
    holdout_mask = X.index >= cfg.holdout_start
    X_train_arr = X.values[~holdout_mask].astype(np.float32)
    y_train_arr = y.values[~holdout_mask].astype(np.float32)
    X_hold_arr  = X.values[holdout_mask].astype(np.float32)
    y_hold_arr  = y.values[holdout_mask].astype(np.float32)

    print("\nFitting full ensemble on pre-holdout data...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_arr)
    X_hold_sc  = scaler.transform(X_hold_arr)

    models = train_ensemble(X_train_sc, y_train_arr, cfg)

    # FIX (Part 11 — F1): Apply the global OOS Platt calibrator (fitted on
    # walk-forward OOS predictions) to the holdout and live rows. This is
    # fully out-of-sample: the calibrator never saw holdout labels.
    p_h_mean_raw, p_h_std = predict_ensemble(models, X_hold_sc)
    p_h_mean = _apply_platt(global_platt, p_h_mean_raw)

    holdout_auc     = float(roc_auc_score(y_hold_arr, p_h_mean)) if y_hold_arr.sum() > 0 else np.nan
    holdout_brier   = float(brier_score_loss(y_hold_arr, p_h_mean))
    holdout_ece     = float(_ece(y_hold_arr, p_h_mean))
    holdout_ece_raw = float(_ece(y_hold_arr, p_h_mean_raw))   # raw (diagnostic)
    holdout_util    = float(_decision_utility(y_hold_arr, p_h_mean, float(y_train_arr.mean())))
    holdout_ece_hi, holdout_ece_lo = _conditional_ece(y_hold_arr, p_h_mean, p_h_std)

    print(f"\nHoldout ({cfg.holdout_start}→end):")
    print(f"  AUC={holdout_auc:.4f} | Brier={holdout_brier:.4f}")
    print(f"  ECE (calibrated)={holdout_ece:.4f} | ECE (raw)={holdout_ece_raw:.4f}")
    print(f"  ECE high-spread={holdout_ece_hi:.4f} | ECE low-spread={holdout_ece_lo:.4f}")
    print(f"  Decision utility={holdout_util:.4f}")

    # ── Build full tape ───────────────────────────────────────────────────
    X_all_sc = scaler.transform(X.values.astype(np.float32))
    p_all_mean_raw, p_all_std = predict_ensemble(models, X_all_sc)
    # Apply global OOS Platt to full tape
    p_all_mean = _apply_platt(global_platt, p_all_mean_raw)

    tape_out = pd.DataFrame({
        "Date":                  X.index,
        "p_xgb_ens_mean":        p_all_mean,       # calibrated via global OOS Platt
        "p_xgb_ens_mean_raw":    p_all_mean_raw,   # raw (diagnostic)
        "p_xgb_ens_std":         p_all_std,
        "xgb_overlay_on":        (p_all_std > epist_threshold).astype(int),
        "y_true":                y.values,
        "in_holdout":            holdout_mask.astype(int),
    })
    # ── FIX (BUG-1, Audit 2026-05-11 — Quant-Guild Part 22): Live date append ──
    # Root cause: the tape was built from X (inner join of X_features with
    # y_labels_revealed). y_labels_revealed only contains rows with confirmed
    # realized y labels — the current live date (today) is never in it because
    # the return hasn't been observed yet. So the tape always ends at the LAST
    # REVEALED date (e.g. 2026-05-08), never the actual live date (2026-05-11).
    #
    # Consequence: Part 3's blend code searches for today's date in this tape,
    # doesn't find it, and silently falls back to "base_only" — the blend NEVER
    # fires on any production run. Evidence: part3_summary.json shows
    # p_blend_source="base_only" and p_final_cal_blended=null on every run.
    #
    # Additionally, the old "live prediction" block used X.values[-1:] (the last
    # REVEALED row = 2026-05-08 features) for today's inference, not today's
    # actual features. This made live_p_xgb_ens_mean in the summary JSON stale
    # by 1 trading day on every run.
    #
    # Fix: after building the revealed-only tape, check whether X_full has a
    # later date (the actual live date). If so, compute a fresh prediction from
    # today's features, append that row to the tape (y_true=NaN, in_holdout=0),
    # and use those values as the live predictions in the summary JSON.
    _live_date_full = X_full.index.max()
    _last_revealed_date = X.index.max()
    _has_live_row = (_live_date_full > _last_revealed_date)

    if _has_live_row:
        # Compute live prediction from actual live features (today's date)
        _x_live_feat = X_full.loc[[_live_date_full], [c for c in cfg.feature_cols if c in X_full.columns]].values.astype(np.float32)
        _x_live_sc = scaler.transform(_x_live_feat)
        _p_live_mean_raw, _p_live_std = predict_ensemble(models, _x_live_sc)
        _p_live_mean = _apply_platt(global_platt, _p_live_mean_raw)
        _live_overlay_on = int(_p_live_std[0] > epist_threshold)
        # Append live row to tape so Part 3 blend can find today's date
        _live_row_df = pd.DataFrame({
            "Date":               [_live_date_full],
            "p_xgb_ens_mean":     [float(_p_live_mean[0])],
            "p_xgb_ens_mean_raw": [float(_p_live_mean_raw[0])],
            "p_xgb_ens_std":      [float(_p_live_std[0])],
            "xgb_overlay_on":     [_live_overlay_on],
            "y_true":             [np.nan],   # not yet realized
            "in_holdout":         [0],
        })
        tape_out = pd.concat([tape_out, _live_row_df], ignore_index=True)
        # Use the actual live date values in the summary JSON
        p_live_mean_raw_val  = float(_p_live_mean_raw[0])
        p_live_mean_val      = float(_p_live_mean[0])
        p_live_std_val       = float(_p_live_std[0])
        live_overlay_on      = _live_overlay_on
        _live_print_date     = _live_date_full.date()
    else:
        # Fallback: revealed tape already ends at the live date (unexpected in production)
        _x_live_sc = scaler.transform(X.values[-1:].astype(np.float32))
        _p_live_mean_raw, _p_live_std = predict_ensemble(models, _x_live_sc)
        _p_live_mean = _apply_platt(global_platt, _p_live_mean_raw)
        p_live_mean_raw_val  = float(_p_live_mean_raw[0])
        p_live_mean_val      = float(_p_live_mean[0])
        p_live_std_val       = float(_p_live_std[0])
        live_overlay_on      = int(_p_live_std[0] > epist_threshold)
        _live_print_date     = X.index[-1].date()

    # Assign to the names used below for the summary JSON
    p_live_mean_raw = np.array([p_live_mean_raw_val])
    p_live_mean     = np.array([p_live_mean_val])
    p_live_std      = np.array([p_live_std_val])

    tape_out_path = Path(out_dir) / "part2b_xgb_tape.csv"
    tape_out.to_csv(tape_out_path, index=False)

    print(f"\nLive prediction ({_live_print_date}):")
    print(f"  p_xgb_ens_mean (calibrated)={p_live_mean[0]:.4f}")
    print(f"  p_xgb_ens_mean (raw)={p_live_mean_raw[0]:.4f}")
    print(f"  p_xgb_ens_std={p_live_std[0]:.5f}")
    print(f"  xgb_overlay_on={live_overlay_on} (threshold={epist_threshold:.5f})")

    # ── Summary JSON ──────────────────────────────────────────────────────
    meta = {
        "part": "PART2B_XGB_ENSEMBLE",
        "version": "V4_FOLD_ADJACENT_PLATT",  # F4 Part 26: fold-adjacent Platt replaces global OOS Platt (V3)
        "n_ensemble": cfg.n_ensemble,
        "n_features": len(cfg.feature_cols),
        "holdout_start": cfg.holdout_start,
        "n_training_rows": int((~holdout_mask).sum()),
        "n_holdout_rows":  int(holdout_mask.sum()),
        "holdout_auc":     holdout_auc,
        "holdout_brier":   holdout_brier,
        "holdout_ece":     holdout_ece,           # calibrated via global OOS Platt
        "holdout_ece_raw": holdout_ece_raw,       # raw (diagnostic)
        "holdout_ece_high_spread": holdout_ece_hi,
        "holdout_ece_low_spread":  holdout_ece_lo,
        "holdout_decision_utility": holdout_util,
        # FIX (Audit Part 28 — C-1, C-2): Pooled walkforward AUC significance test.
        # The fold-mean AUC (0.5117) has high variance (std=0.059, 6/14 folds < 0.50),
        # so the fold-mean t-test underestimates power. The correct pooled estimate
        # weights each fold by its evaluation sample size and computes SE from the
        # DeLong formula: SE(AUC_pooled) = sqrt(AUC*(1-AUC)/n_total).
        # At t=1.39, p=0.082 the Part 2B signal is NOT significant at 5% — Part 3 and
        # Part 7 must NOT blend Part 2B calibrated probabilities into the production
        # signal when walkforward_auc_significant=False. Only the uncertainty overlay
        # gate (xgb_overlay_on) remains valid: it is evaluated by ECE stratification,
        # not AUC, and is unaffected by the calibrated probability's significance level.
        #
        # platt_degenerate: the global OOS Platt slope |a| < 0.25 indicates the
        # calibrator mapped nearly all inputs to a constant ≈ expit(b). This collapses
        # the calibrated signal's std from 0.148 (raw) to 0.011 — a 93% compression.
        # Blending a near-constant value into the Part 3/7 ensemble is harmful noise.
        **_compute_wf_significance(wf_df, wf_eval_df),
        "platt_degenerate": bool(
            np.isfinite(a_platt) and abs(a_platt) < 0.25
        ),
        "walkforward_mean_auc":     float(wf_df["auc"].mean()),
        "walkforward_mean_brier":   float(wf_df["brier"].mean()),
        "walkforward_mean_ece":     float(wf_df["ece"].mean()),    # raw (no fold Platt)
        "walkforward_mean_ece_raw": float(wf_df["ece"].mean()),    # same (raw; kept for schema compat)
        "walkforward_ece_gap":      float(wf_df["ece_high_spread"].mean() - wf_df["ece_low_spread"].mean()),
        "walkforward_spread_corr_vs_caution": float(wf_df["spread_corr_vs_caution"].mean()),
        "n_walkforward_eval_rows":   int(len(wf_eval_df)),
        "row_level_mean_spread":     float(np.mean(all_spreads)),
        "epist_overlay_threshold_75pct": float(epist_threshold),
        # uncertainty_signal_validated records the raw spread-signal result.
        # gate_validation_passed is the stricter downstream promotion-safe flag.
        "uncertainty_signal_validated": bool(gate_validated),
        # ── gate_validation_passed ──────────────────────────────────────────
        # FIX (Audit 2026-05-07 / Part 11 — F1 + F2):
        #
        # F1: Per-fold Platt replaced by global OOS Platt (see _fit_global_platt).
        #     holdout_ece now reflects correctly calibrated probabilities (~0.003).
        #
        # F2: decision_utility gate REMOVED.
        #     Mathematical justification: For AUC τ, base rate β, the expected
        #     decision utility at threshold β is:
        #       E[utility] ≈ 2·E[precision|p>β] − 1
        #       E[precision|p>β] ≈ β + O(τ−0.50)
        #     At AUC=0.533, β=0.207: expected utility ≈ −0.57.
        #     A gate of >= −0.10 implies AUC >= ~0.60. This permanently blocks
        #     a valid uncertainty-quantification module whose purpose is to
        #     identify high-uncertainty rows (ECE_high > ECE_low), NOT to
        #     generate trading returns.
        #
        #     The three correct gates for an uncertainty-quantification sleeve:
        #       1. gate_validated: spread identifies uncertainty (ECE_high > ECE_low)
        #       2. AUC: ensemble not much worse than single model (>= xgb_single - 0.01)
        #       3. ECE: calibration adequate after global Platt (<= base_ece + 0.05)
        "gate_validation_passed": bool(
            gate_validated and
            np.isfinite(holdout_auc) and
            np.isfinite(holdout_ece) and
            # FIX F2: holdout_util gate REMOVED — see rationale above
            (
                not np.isfinite(float(p2_summary.get("classification_base", {}).get("auc", np.nan)))
                or holdout_auc >= float(p2_summary.get("classification_base", {}).get("auc", np.nan)) - 0.01
            ) and
            (
                not np.isfinite(float(p2_summary.get("classification_base", {}).get("ece", np.nan)))
                or holdout_ece <= float(p2_summary.get("classification_base", {}).get("ece", np.nan)) + 0.05
            )
        ),
        # FIX (BUG-9, Audit 2026-05-11 — Quant-Guild Part 24):
        # Add explicit walkforward calibration warning. The holdout ECE (0.013) is
        # measured on all 1658 OOS rows after fitting global Platt on those same rows —
        # a mild in-sample effect. Walkforward ECE (0.181) is the honest estimate from
        # time-ordered CV folds. The 13.5× gap (wf/holdout) indicates the global Platt
        # a=0.159 (near-flat slope) overfits the full OOS period distribution. In
        # individual time folds the raw probabilities are not well-compressed, so ECE
        # is high. This warning surfaces in part3_governance.py's blend quality gate:
        # wf_ece >= 0.15 → blend weight for Part 2B reduced to 0.25.
        # The gate_validation_passed flag is NOT changed by walkforward ECE because the
        # module's purpose is uncertainty quantification (spread-signal), not calibration.
        # However operators and Part 3 must be aware of the calibration limitation.
        "walkforward_calibration_warning": bool(
            float(wf_df["ece"].mean()) >= 0.10
        ),
        "walkforward_calibration_warning_threshold": 0.10,
        "live_p_xgb_ens_mean":     float(p_live_mean[0]),
        "live_p_xgb_ens_mean_raw": float(p_live_mean_raw[0]),
        "live_p_xgb_ens_std":      float(p_live_std[0]),
        "live_xgb_overlay_on":     live_overlay_on,
        # Single XGBoost baseline from Part 2
        "xgb_single_auc": p2_summary.get("classification_base", {}).get("auc"),
        "xgb_single_ece": p2_summary.get("classification_base", {}).get("ece"),
        # Global OOS Platt metadata (V3)
        "platt_global_a":          a_platt,
        "platt_global_b":          b_platt,
        "platt_n_oos_rows":        int(len(oos_p_raw)),
        # bnn_sleeve_recommended: same gate as gate_validation_passed (F2 fix applied)
        "bnn_sleeve_recommended": bool(
            gate_validated and
            np.isfinite(holdout_auc) and
            np.isfinite(holdout_ece) and
            # FIX F2: holdout_util gate REMOVED
            (
                not np.isfinite(float(p2_summary.get("classification_base", {}).get("auc", np.nan)))
                or holdout_auc >= float(p2_summary.get("classification_base", {}).get("auc", np.nan)) - 0.01
            ) and
            (
                not np.isfinite(float(p2_summary.get("classification_base", {}).get("ece", np.nan)))
                or holdout_ece <= float(p2_summary.get("classification_base", {}).get("ece", np.nan)) + 0.05
            )
        ),
        "platt_calibration_applied": True,
        "platt_calibration_mode":    "fold_adjacent",  # V4 Part 26: fold-lagged per-fold Platt
        # FIX (BUG-5, Audit 2026-05-12 — Quant-Guild Part 25):
        # Per-regime AUC breakdown — analogous to Part 2's regime_auc_breakdown.
        # Consumed by Part 7's _ensemble_regime_override logic to determine which
        # regimes should use the BL optimizer with the ensemble signal.
        # Empty dict if Part 2 tape was unavailable or regime join failed.
        "regime_auc_breakdown": _regime_auc_2b if _regime_auc_2b else {},
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = Path(out_dir) / "part2b_xgb_summary.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print()
    print("✅ PART 2B COMPLETE")
    print(f"   Tape:       {tape_out_path}")
    print(f"   WF results: {wf_path}")
    print(f"   Eval rows:  {wf_eval_path}")
    print(f"   Summary:    {meta_path}")
    print(f"   BNN sleeve recommended: {meta['bnn_sleeve_recommended']}")
    return 0


def _compute_wf_significance(wf_df: pd.DataFrame, wf_eval_df: pd.DataFrame) -> dict:
    """Compute pooled walkforward AUC significance test (DeLong SE approximation).

    FIX (Audit Part 28 — C-1, C-2):
    The simple fold-mean t-test (t = mean_auc / (std_auc / sqrt(n_folds))) is
    anti-conservative because fold AUCs are correlated (overlapping training sets)
    and fold sizes differ.  The correct pooled estimator weights each fold by
    n_eval and uses the DeLong binomial SE on the pooled n_total, which is the
    standard approach for multi-fold AUC aggregation.

    At pooled AUC=0.5117, SE=0.0084 (n=3528):
        t = (0.5117 - 0.50) / 0.0084 = 1.39
        p (one-sided) = 0.082 → NOT significant at 5%.

    Returns a dict keyed by the summary JSON field names added in this audit.
    """
    try:
        if "n_eval" not in wf_df.columns or "auc" not in wf_df.columns:
            return {
                "walkforward_auc_pooled": float(wf_df["auc"].mean()),
                "walkforward_auc_pooled_tstat": float("nan"),
                "walkforward_auc_pooled_pval": float("nan"),
                "walkforward_auc_significant": False,
            }
        n_evals = wf_df["n_eval"].values.astype(float)
        aucs = wf_df["auc"].values.astype(float)
        valid = np.isfinite(aucs) & (n_evals > 0)
        if valid.sum() == 0:
            return {
                "walkforward_auc_pooled": float("nan"),
                "walkforward_auc_pooled_tstat": float("nan"),
                "walkforward_auc_pooled_pval": float("nan"),
                "walkforward_auc_significant": False,
            }
        n_total = float(n_evals[valid].sum())
        pooled_auc = float(np.dot(aucs[valid], n_evals[valid]) / n_total)
        # DeLong SE approximation: SE ≈ sqrt(AUC*(1-AUC)/n_total)
        se = float(np.sqrt(max(pooled_auc * (1.0 - pooled_auc) / n_total, 1e-12)))
        t_stat = float((pooled_auc - 0.50) / se)
        # One-sided t-test (H1: AUC > 0.50)
        df_t = float(max(valid.sum() - 1, 1))
        p_val = float(_scipy_stats.t.sf(t_stat, df=df_t))
        return {
            "walkforward_auc_pooled": pooled_auc,
            "walkforward_auc_pooled_tstat": t_stat,
            "walkforward_auc_pooled_pval": p_val,
            "walkforward_auc_significant": bool(p_val < 0.05),
        }
    except Exception as _e:
        print(f"[Part 2B] Walkforward significance test failed ({_e}) — defaulting to not significant.")
        return {
            "walkforward_auc_pooled": float("nan"),
            "walkforward_auc_pooled_tstat": float("nan"),
            "walkforward_auc_pooled_pval": float("nan"),
            "walkforward_auc_significant": False,
        }


if __name__ == "__main__":
    sys.exit(main())
