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
    threshold: float = 0.25,
) -> float:
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

    Returns
    -------
    (fold_summary_df, row_level_eval_df)

    The row-level dataframe is the important object for the production overlay
    threshold.  The threshold should be learned from the distribution of
    predictive spread across evaluation rows, not from the distribution of
    fold-level mean spreads.
    """
    results: List[Dict[str, float]] = []
    eval_rows: List[Dict[str, object]] = []

    fold_starts = range(
        cfg.walk_forward_min_train,
        len(X) - cfg.walk_forward_step,
        cfg.walk_forward_step,
    )

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

        base_rate = float(y_tr.mean())
        ece_high, ece_low = _conditional_ece(y_ev, p_mean, p_std)

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
            "auc":              float(roc_auc_score(y_ev, p_mean)) if y_ev.sum() > 0 else np.nan,
            "brier":            float(brier_score_loss(y_ev, p_mean)),
            "ece":              float(_ece(y_ev, p_mean)),
            "ece_high_spread":  float(ece_high),
            "ece_low_spread":   float(ece_low),
            "decision_utility": float(_decision_utility(y_ev, p_mean, base_rate)),
            "mean_spread":      float(p_std.mean()),
            "spread_threshold_fold": spread_thr_fold,
            "spread_corr_vs_caution": float(spread_corr) if spread_corr is not None else np.nan,
            "overlay_on_rate":  float(overlay_flags.mean()),
        }
        results.append(row)

        for dt, y_i, p_i, s_i, c_i, o_i in zip(eval_index, y_ev, p_mean, p_std, cs_fold, overlay_flags):
            eval_rows.append({
                "Date": pd.Timestamp(dt),
                "fold": i,
                "y_true": float(y_i),
                "p_xgb_ens_mean": float(p_i),
                "p_xgb_ens_std": float(s_i),
                "caution_signal": float(c_i) if np.isfinite(c_i) else np.nan,
                "xgb_overlay_on_fold": int(o_i),
            })

        corr_txt = "nan" if spread_corr is None or np.isnan(spread_corr) else f"{spread_corr:.3f}"
        print(
            f"  Fold {i}: {row['eval_start_date']} | "
            f"AUC={row['auc']:.4f} | ECE_hi={row['ece_high_spread']:.4f} | "
            f"ECE_lo={row['ece_low_spread']:.4f} | "
            f"spread_corr={corr_txt}"
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

    # ── Compute walk-forward-derived epistemic threshold ──────────────────
    # IMPORTANT: the threshold must come from the row-level spread
    # distribution, not from the distribution of fold-level mean spreads.
    all_spreads = wf_eval_df["p_xgb_ens_std"].dropna().values
    if len(all_spreads) == 0:
        print("[Part 2B] No row-level walk-forward spreads available.")
        return 1
    epist_threshold = float(np.percentile(all_spreads, cfg.overlay_pct * 100))
    print(f"\nEpistemic overlay threshold ({cfg.overlay_pct:.0%} pct, row-level): {epist_threshold:.5f}")

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

    # Holdout metrics
    p_h_mean, p_h_std = predict_ensemble(models, X_hold_sc)
    holdout_auc  = float(roc_auc_score(y_hold_arr, p_h_mean)) if y_hold_arr.sum() > 0 else np.nan
    holdout_brier = float(brier_score_loss(y_hold_arr, p_h_mean))
    holdout_ece   = float(_ece(y_hold_arr, p_h_mean))
    holdout_util  = float(_decision_utility(y_hold_arr, p_h_mean, float(y_train_arr.mean())))
    holdout_ece_hi, holdout_ece_lo = _conditional_ece(y_hold_arr, p_h_mean, p_h_std)

    print(f"\nHoldout ({cfg.holdout_start}→end):")
    print(f"  AUC={holdout_auc:.4f} | Brier={holdout_brier:.4f} | ECE={holdout_ece:.4f}")
    print(f"  ECE high-spread={holdout_ece_hi:.4f} | ECE low-spread={holdout_ece_lo:.4f}")
    print(f"  Decision utility={holdout_util:.4f}")

    # ── Build full tape ───────────────────────────────────────────────────
    X_all_sc = scaler.transform(X.values.astype(np.float32))
    p_all_mean, p_all_std = predict_ensemble(models, X_all_sc)

    tape_out = pd.DataFrame({
        "Date":               X.index,
        "p_xgb_ens_mean":     p_all_mean,
        "p_xgb_ens_std":      p_all_std,
        "xgb_overlay_on":     (p_all_std > epist_threshold).astype(int),
        "y_true":             y.values,
        "in_holdout":         holdout_mask.astype(int),
    })
    tape_out_path = Path(out_dir) / "part2b_xgb_tape.csv"
    tape_out.to_csv(tape_out_path, index=False)

    # ── Live prediction (latest row) ──────────────────────────────────────
    x_live_sc = scaler.transform(X.values[-1:].astype(np.float32))
    p_live_mean, p_live_std = predict_ensemble(models, x_live_sc)
    live_overlay_on = int(p_live_std[0] > epist_threshold)

    print(f"\nLive prediction ({X.index[-1].date()}):")
    print(f"  p_xgb_ens_mean={p_live_mean[0]:.4f}")
    print(f"  p_xgb_ens_std={p_live_std[0]:.5f}")
    print(f"  xgb_overlay_on={live_overlay_on} (threshold={epist_threshold:.5f})")

    # ── Summary JSON ──────────────────────────────────────────────────────
    meta = {
        "part": "PART2B_XGB_ENSEMBLE",
        "version": "V1_BOOTSTRAP_ENSEMBLE",
        "n_ensemble": cfg.n_ensemble,
        "n_features": len(cfg.feature_cols),
        "holdout_start": cfg.holdout_start,
        "n_training_rows": int((~holdout_mask).sum()),
        "n_holdout_rows":  int(holdout_mask.sum()),
        "holdout_auc":     holdout_auc,
        "holdout_brier":   holdout_brier,
        "holdout_ece":     holdout_ece,
        "holdout_ece_high_spread": holdout_ece_hi,
        "holdout_ece_low_spread":  holdout_ece_lo,
        "holdout_decision_utility": holdout_util,
        "walkforward_mean_auc":     float(wf_df["auc"].mean()),
        "walkforward_mean_brier":   float(wf_df["brier"].mean()),
        "walkforward_mean_ece":     float(wf_df["ece"].mean()),
        "walkforward_ece_gap":      float(wf_df["ece_high_spread"].mean() - wf_df["ece_low_spread"].mean()),
        "walkforward_spread_corr_vs_caution": float(wf_df["spread_corr_vs_caution"].mean()),
        "n_walkforward_eval_rows":   int(len(wf_eval_df)),
        "row_level_mean_spread":     float(np.mean(all_spreads)),
        "epist_overlay_threshold_75pct": float(epist_threshold),
        "gate_validation_passed":   bool(gate_validated),
        "live_p_xgb_ens_mean":  float(p_live_mean[0]),
        "live_p_xgb_ens_std":   float(p_live_std[0]),
        "live_xgb_overlay_on":  live_overlay_on,
        # Single XGBoost baseline from Part 2
        "xgb_single_auc":   p2_summary.get("classification_base", {}).get("auc"),
        "xgb_single_ece":   p2_summary.get("classification_base", {}).get("ece"),
        "bnn_sleeve_recommended": bool(gate_validated),
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
    print(f"   BNN sleeve recommended: {gate_validated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
