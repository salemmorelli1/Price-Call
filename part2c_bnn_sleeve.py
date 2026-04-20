#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @title Part 2C — Bayesian Neural Network Parallel Sleeve (Experimental)
#
# =============================================================================
# Experimental BNN parallel sleeve for PriceCallProject.
#
# This is NOT a replacement for Part 2 or Part 2B (XGBoost ensemble).  It runs
# alongside both and writes its own artifact directory.  Models are compared on:
#   AUC, Brier, ECE, and decision utility.
# Neither replaces the other until the live evidence supports a decision.
#
# Architecture: Deep Ensemble + MC Dropout
# -----------------------------------------
# A deep ensemble of N_ENSEMBLE small MLPs (2 hidden layers, dropout).
# At inference time, dropout stays ON and each model samples N_MC_SAMPLES
# forward passes.  This gives:
#
#   p_bnn_mean        — posterior mean probability (drop-in for p_final_cal)
#   p_bnn_epistemic   — std across ensemble members (model disagreement)
#   p_bnn_aleatoric   — mean within-model MC dropout std (data noise)
#   p_bnn_total_std   — combined uncertainty = sqrt(epist² + aleat²)
#
# p_bnn_epistemic is the overlay gate signal: when it is large, the ensemble
# members disagree — the model is uncertain about this input and the defense
# signal should be treated with lower confidence.
#
# Weight prior: tight N(0, 0.1) via L2 regularisation on all weights.
# This is the correct inductive bias for a problem where AUC ≈ 0.54 and
# most days carry near-zero signal.
#
# Execution order:
#   Part 0 → Part 6 → Part 1 → Part 2 → Part 2B → Part 2C → Part 2A → Part 7 → ...
# Part 2C reads the same Part 1 artifacts as Part 2 and writes to
# artifacts_part2c_bnn/predictions/.
#
# Evaluation
# -----------
# After the walk-forward holdout is scored, Part 2C prints a side-by-side
# comparison against the XGBoost baseline from Part 2's summary JSON.
# The comparison covers AUC, Brier, ECE, and a simple decision utility
# metric (expected calibrated P&L from the defense signal).
#
# Priority note
# -------------
# This is step 6 in the A+ sequence.  Part 2B (XGBoost ensemble) is step 5
# and must validate uncertainty-aware gating before this is activated.
# Part 2C is only worth running if part2b_xgb_summary.json reports
# bnn_sleeve_recommended: true.  Check that file before running this.
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

warnings.filterwarnings("ignore")

# ── Optional PyTorch import ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    print("[Part 2C] PyTorch not found.  Install with: pip install torch")
    print("[Part 2C] Falling back to sklearn MLPClassifier ensemble (no MC dropout).")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_DRIVE_ROOT = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject")


# ============================================================
# Configuration
# ============================================================

@dataclass
class Part2CConfig:
    # ── Paths ──────────────────────────────────────────────
    part1_dir: str = _DRIVE_ROOT + "/artifacts_part1"
    part2_dir: str = _DRIVE_ROOT + "/artifacts_part2_g532/predictions"
    out_dir:   str = _DRIVE_ROOT + "/artifacts_part2c_bnn/predictions"

    # ── Feature contract ───────────────────────────────────
    # Must match Part 1's locked-14 feature schema exactly.
    feature_cols: Tuple[str, ...] = (
        "voo_vol10", "excess_vol10", "vix_mom5",
        "alpha_credit_spread", "alpha_credit_accel", "alpha_vix_term",
        "alpha_breadth", "alpha_tech_relative",
        "stress_score_raw", "stress_score_change5",
        "vix_z21", "credit_spread_z21", "breadth_z21", "tech_relative_z21",
    )
    label_col:    str = "y_rel_tail_voo_vs_ief"
    holdout_start: str = "2020-01-01"

    # ── Ensemble hyperparameters ───────────────────────────
    n_ensemble:   int = 10          # number of independently-trained models
    n_mc_samples: int = 100         # MC dropout samples per inference call
    hidden_dim_1: int = 64          # first hidden layer width
    hidden_dim_2: int = 32          # second hidden layer width
    dropout_rate: float = 0.20      # dropout probability (kept ON at inference)

    # ── Training ───────────────────────────────────────────
    # Tight L2 weight prior (weight_decay) encodes N(0, 0.1) beliefs.
    # At AUC ≈ 0.54 the signal is genuinely small; strong regularisation
    # prevents the network from fitting noise.
    lr:           float = 1e-3
    weight_decay: float = 1e-2      # L2 coefficient — tight prior
    n_epochs:     int = 200
    batch_size:   int = 128
    patience:     int = 20          # early stopping patience (val loss)
    val_frac:     float = 0.15      # fraction of training data for val set

    # ── Walk-forward evaluation ────────────────────────────
    # Each fold trains on all data up to fold_end, evaluates on the next
    # walk_forward_step trading days.  The folds never look forward.
    walk_forward_step: int = 252    # evaluate every ~1 year
    walk_forward_min_train: int = 500  # minimum rows before first fold


CFG = Part2CConfig()


# ============================================================
# PyTorch model (preferred)
# ============================================================

class _BayesianMLP(nn.Module):
    """
    Small 2-hidden-layer MLP with MC Dropout.

    Keeping dropout ON at inference time (model.train()) gives us
    approximate Bayesian inference via variational dropout.  Sampling
    N forward passes produces a distribution over p(y=1 | x, D).
    """
    def __init__(self, n_features: int, hidden_1: int, hidden_2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )
        # Tight weight initialisation (consistent with N(0,0.1) prior)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


def _train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Part2CConfig,
    seed: int,
) -> "_BayesianMLP":
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_train)
    n_val = max(1, int(n * cfg.val_frac))
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)

    pos_weight = torch.tensor([(1 - y_tr.mean()) / (y_tr.mean() + 1e-9)])
    loss_fn = nn.BCELoss()

    model = _BayesianMLP(
        X_train.shape[1], cfg.hidden_dim_1, cfg.hidden_dim_2, cfg.dropout_rate
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True
    )

    best_val_loss = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation loss (eval mode: dropout OFF for loss tracking)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch(
    models: List["_BayesianMLP"],
    X: np.ndarray,
    n_mc: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean, epistemic_std, aleatoric_std) for each row.

    epistemic_std  = std across ensemble members  (model uncertainty)
    aleatoric_std  = mean of within-model MC std  (data noise)
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    all_member_means = []
    all_member_stds = []

    for model in models:
        model.train()   # keep dropout ON
        with torch.no_grad():
            mc_samples = torch.stack([model(X_t) for _ in range(n_mc)])  # (n_mc, N)
        member_mean = mc_samples.mean(0).numpy()
        member_std  = mc_samples.std(0).numpy()
        all_member_means.append(member_mean)
        all_member_stds.append(member_std)

    member_means = np.stack(all_member_means)   # (n_ensemble, N)
    member_stds  = np.stack(all_member_stds)    # (n_ensemble, N)

    mean         = member_means.mean(axis=0)
    epistemic    = member_means.std(axis=0)      # disagreement across models
    aleatoric    = member_stds.mean(axis=0)      # average within-model noise
    return mean, epistemic, aleatoric


# ============================================================
# Sklearn fallback (no PyTorch)
# ============================================================

def _train_sklearn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Part2CConfig,
    seed: int,
) -> CalibratedClassifierCV:
    mlp = MLPClassifier(
        hidden_layer_sizes=(cfg.hidden_dim_1, cfg.hidden_dim_2),
        activation="relu",
        alpha=cfg.weight_decay,
        max_iter=cfg.n_epochs,
        random_state=seed,
        early_stopping=True,
        validation_fraction=cfg.val_frac,
        n_iter_no_change=cfg.patience,
    )
    cal = CalibratedClassifierCV(mlp, method="isotonic", cv=3)
    cal.fit(X_train, y_train)
    return cal


def _predict_sklearn(
    models: list,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models])
    mean      = preds.mean(axis=0)
    epistemic = preds.std(axis=0)
    aleatoric = np.zeros_like(mean)          # no MC dropout available
    return mean, epistemic, aleatoric


# ============================================================
# Calibration metrics
# ============================================================

def _ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def _decision_utility(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    base_rate: float,
    threshold: float = 0.25,
    win_scale: float = 1.0,
    loss_scale: float = 1.0,
) -> float:
    """
    Simple defense decision utility.
    When p_pred > threshold, we act defensively (reduce VOO).
    Utility = sum over rows where we act defensively:
        +win_scale  if y_true == 1 (tail event occurred, defense was right)
        -loss_scale if y_true == 0 (no tail, defense was opportunity cost)
    Normalised by number of decisions made.
    """
    acted = p_pred > threshold
    if acted.sum() == 0:
        return float("nan")
    hits = (y_true[acted] == 1).sum()
    misses = (y_true[acted] == 0).sum()
    utility = (hits * win_scale - misses * loss_scale) / acted.sum()
    return float(utility)


# ============================================================
# Walk-forward evaluation
# ============================================================

def walk_forward_eval(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2CConfig,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.
    Returns a DataFrame with one row per fold containing evaluation metrics.
    No fold ever uses future data during training.
    """
    dates = X.index
    n = len(X)
    scaler_global = StandardScaler()

    results = []
    fold_starts = range(
        cfg.walk_forward_min_train,
        n - cfg.walk_forward_step,
        cfg.walk_forward_step,
    )

    for i, train_end in enumerate(fold_starts):
        eval_end = min(train_end + cfg.walk_forward_step, n)
        X_tr_raw = X.iloc[:train_end].values.astype(np.float32)
        y_tr     = y.iloc[:train_end].values.astype(np.float32)
        X_ev_raw = X.iloc[train_end:eval_end].values.astype(np.float32)
        y_ev     = y.iloc[train_end:eval_end].values.astype(np.float32)

        if y_tr.mean() < 0.01 or y_tr.mean() > 0.99:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_ev = scaler.transform(X_ev_raw)

        # Train ensemble
        if HAVE_TORCH:
            models = [
                _train_torch_model(X_tr, y_tr, cfg, seed=42 + j)
                for j in range(cfg.n_ensemble)
            ]
            p_mean, p_epist, p_aleat = _predict_torch(models, X_ev, cfg.n_mc_samples)
        else:
            models = [
                _train_sklearn_model(X_tr, y_tr, cfg, seed=42 + j)
                for j in range(cfg.n_ensemble)
            ]
            p_mean, p_epist, p_aleat = _predict_sklearn(models, X_ev)

        base_rate = float(y_tr.mean())
        row = {
            "fold":              i,
            "train_end_date":    str(dates[train_end - 1].date()),
            "eval_start_date":   str(dates[train_end].date()),
            "eval_end_date":     str(dates[eval_end - 1].date()),
            "n_train":           int(train_end),
            "n_eval":            int(eval_end - train_end),
            "base_rate_train":   float(base_rate),
            "auc":               float(roc_auc_score(y_ev, p_mean)) if y_ev.sum() > 0 else np.nan,
            "brier":             float(brier_score_loss(y_ev, p_mean)),
            "ece":               float(_ece(y_ev, p_mean)),
            "decision_utility":  float(_decision_utility(y_ev, p_mean, base_rate)),
            "mean_epistemic_std": float(p_epist.mean()),
            "mean_aleatoric_std": float(p_aleat.mean()),
            "overlay_on_rate":   float((p_epist > np.percentile(p_epist, 75)).mean()),
        }
        results.append(row)
        print(
            f"  Fold {i}: train_end={row['train_end_date']} | "
            f"AUC={row['auc']:.4f} | Brier={row['brier']:.4f} | "
            f"ECE={row['ece']:.4f} | utility={row['decision_utility']:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# Live inference
# ============================================================

def fit_full_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Part2CConfig,
) -> Tuple[list, StandardScaler]:
    """Fit ensemble on full available data for live prediction."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    if HAVE_TORCH:
        models = [
            _train_torch_model(X_scaled, y_train, cfg, seed=100 + j)
            for j in range(cfg.n_ensemble)
        ]
    else:
        models = [
            _train_sklearn_model(X_scaled, y_train, cfg, seed=100 + j)
            for j in range(cfg.n_ensemble)
        ]
    return models, scaler


def predict_live(
    models: list,
    scaler: StandardScaler,
    x_live: np.ndarray,
    cfg: Part2CConfig,
) -> Dict[str, float]:
    """Single-row live prediction with full uncertainty decomposition."""
    x_scaled = scaler.transform(x_live.reshape(1, -1))
    if HAVE_TORCH:
        mean, epist, aleat = _predict_torch(models, x_scaled, cfg.n_mc_samples)
    else:
        mean, epist, aleat = _predict_sklearn(models, x_scaled)
    total = float(np.sqrt(epist[0] ** 2 + aleat[0] ** 2))
    return {
        "p_bnn_mean":      float(mean[0]),
        "p_bnn_epistemic": float(epist[0]),
        "p_bnn_aleatoric": float(aleat[0]),
        "p_bnn_total_std": total,
        # Overlay gate: fire when epistemic uncertainty is in top quartile
        # of historical distribution (replaces caution_signal >= 0.40)
        "bnn_overlay_on":  int(epist[0] > 0.0),   # threshold set from wf eval
    }


# ============================================================
# Comparison report
# ============================================================

def print_comparison(
    wf_df: pd.DataFrame,
    part2_summary: Dict,
) -> None:
    xgb_auc    = part2_summary.get("classification_base", {}).get("auc", np.nan)
    xgb_brier  = part2_summary.get("classification_base", {}).get("brier", np.nan)
    xgb_ece    = part2_summary.get("classification_base", {}).get("ece", np.nan)

    bnn_auc    = wf_df["auc"].mean()
    bnn_brier  = wf_df["brier"].mean()
    bnn_ece    = wf_df["ece"].mean()
    bnn_util   = wf_df["decision_utility"].mean()
    bnn_epist  = wf_df["mean_epistemic_std"].mean()

    print()
    print("=" * 68)
    print("PART 2C — BNN vs XGBoost Comparison (walk-forward holdout)")
    print("=" * 68)
    print(f"{'Metric':<24} {'XGBoost (Part 2)':<22} {'BNN Ensemble':<20}")
    print("-" * 68)
    print(f"{'AUC':<24} {xgb_auc:<22.4f} {bnn_auc:<20.4f}")
    print(f"{'Brier':<24} {xgb_brier:<22.4f} {bnn_brier:<20.4f}")
    print(f"{'ECE':<24} {xgb_ece:<22.4f} {bnn_ece:<20.4f}")
    print(f"{'Decision utility':<24} {'N/A':<22} {bnn_util:<20.4f}")
    print(f"{'Mean epistemic std':<24} {'N/A':<22} {bnn_epist:<20.4f}")
    print("-" * 68)

    # Promotion recommendation
    auc_improvement  = bnn_auc - xgb_auc
    brier_improvement = xgb_brier - bnn_brier
    ece_improvement   = xgb_ece - bnn_ece

    improvements = sum([
        auc_improvement > 0.005,
        brier_improvement > 0.002,
        ece_improvement > 0.001,
        bnn_util > 0.0,
    ])

    print()
    if improvements >= 3:
        print("✅ RECOMMENDATION: BNN wins on ≥3 metrics. Consider promoting to")
        print("   parallel-primary sleeve alongside XGBoost.")
    elif improvements >= 2:
        print("⚠️  RECOMMENDATION: BNN mixed. Run 200+ live rows before deciding.")
    else:
        print("❌ RECOMMENDATION: XGBoost dominates. Keep BNN as experimental only.")
    print()
    print(f"  AUC Δ:    {auc_improvement:+.4f}")
    print(f"  Brier Δ:  {-brier_improvement:+.4f} (negative = BNN worse)")
    print(f"  ECE Δ:    {-ece_improvement:+.4f} (negative = BNN worse)")
    print()


# ============================================================
# Root resolution (matches Part 2 pattern)
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
    cfg = CFG
    root = _resolve_root()
    os.environ["PRICECALL_ROOT"] = root

    p1_dir  = _abs(cfg.part1_dir,  root)
    p2_dir  = _abs(cfg.part2_dir,  root)
    out_dir = _abs(cfg.out_dir,    root)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 2C — Bayesian Neural Network Parallel Sleeve")
    back = "PyTorch (MC Dropout)" if HAVE_TORCH else "sklearn MLP Ensemble (no MC dropout)"
    print(f"Backend: {back}")
    print(f"Ensemble size: {cfg.n_ensemble} | MC samples: {cfg.n_mc_samples if HAVE_TORCH else 'N/A'}")
    print("=" * 70)

    # ── Load Part 1 artifacts ──────────────────────────────────────────────
    X_path = Path(p1_dir) / "X_features.parquet"
    y_path = Path(p1_dir) / "y_labels_revealed.parquet"
    if not X_path.exists() or not y_path.exists():
        print("[Part 2C] Part 1 artifacts not found. Run Part 1 first.")
        return 1

    X_full = pd.read_parquet(X_path)
    y_full = pd.read_parquet(y_path)
    X_full.index = pd.to_datetime(X_full.index, errors="coerce")
    y_full.index = pd.to_datetime(y_full.index, errors="coerce")

    # Align
    X_full = X_full[[c for c in cfg.feature_cols if c in X_full.columns]]
    missing_feats = [c for c in cfg.feature_cols if c not in X_full.columns]
    if missing_feats:
        print(f"[Part 2C] Missing features from Part 1 contract: {missing_feats}")
        return 1

    combined = X_full.join(y_full[[cfg.label_col]], how="inner").dropna()
    X = combined[list(cfg.feature_cols)]
    y = combined[cfg.label_col]

    print(f"Loaded {len(X)} aligned rows | tail rate: {y.mean():.4f}")
    print(f"Date range: {X.index.min().date()} → {X.index.max().date()}")

    # ── Walk-forward evaluation ────────────────────────────────────────────
    print("\nRunning walk-forward evaluation...")
    wf_df = walk_forward_eval(X, y, cfg)

    if wf_df.empty:
        print("[Part 2C] No walk-forward folds completed.")
        return 1

    wf_path = Path(out_dir) / "part2c_walkforward.csv"
    wf_df.to_csv(wf_path, index=False)
    print(f"\nWalk-forward results written: {wf_path}")

    # ── Load Part 2 summary for comparison ────────────────────────────────
    p2_summary: Dict = {}
    p2_summary_path = Path(p2_dir) / "part2_g532_summary.json"
    if p2_summary_path.exists():
        with open(p2_summary_path) as f:
            p2_summary = json.load(f)

    print_comparison(wf_df, p2_summary)

    # ── Set epistemic threshold from walk-forward distribution ────────────
    # The top-quartile threshold for the overlay gate is computed from the
    # walk-forward evaluation data rather than being fixed heuristically.
    epist_threshold = float(wf_df["mean_epistemic_std"].quantile(0.75)) \
        if len(wf_df) >= 4 else 0.05
    print(f"Epistemic uncertainty overlay threshold (75th pct): {epist_threshold:.5f}")

    # ── Fit full model on all available data ──────────────────────────────
    print("\nFitting full ensemble on complete dataset for live inference...")
    X_all = X.values.astype(np.float32)
    y_all = y.values.astype(np.float32)

    # Holdout split: train on pre-holdout, score on holdout
    holdout_mask = X.index >= cfg.holdout_start
    X_train_arr = X_all[~holdout_mask]
    y_train_arr = y_all[~holdout_mask]
    X_hold_arr  = X_all[holdout_mask]
    y_hold_arr  = y_all[holdout_mask]

    models, scaler = fit_full_model(X_train_arr, y_train_arr, cfg)

    # Score holdout
    if HAVE_TORCH:
        X_hold_sc = scaler.transform(X_hold_arr)
        p_h_mean, p_h_epist, p_h_aleat = _predict_torch(models, X_hold_sc, cfg.n_mc_samples)
    else:
        X_hold_sc = scaler.transform(X_hold_arr)
        p_h_mean, p_h_epist, p_h_aleat = _predict_sklearn(models, X_hold_sc)

    holdout_auc   = float(roc_auc_score(y_hold_arr, p_h_mean)) if y_hold_arr.sum() > 0 else np.nan
    holdout_brier = float(brier_score_loss(y_hold_arr, p_h_mean))
    holdout_ece   = float(_ece(y_hold_arr, p_h_mean))
    holdout_util  = float(_decision_utility(y_hold_arr, p_h_mean, float(y_train_arr.mean())))

    print(f"\nHoldout ({cfg.holdout_start}→end):")
    print(f"  AUC={holdout_auc:.4f} | Brier={holdout_brier:.4f} | "
          f"ECE={holdout_ece:.4f} | utility={holdout_util:.4f}")

    # ── Live inference on latest feature row ──────────────────────────────
    live_row = X_all[-1:]
    live_result = predict_live(models, scaler, live_row, cfg)
    # Apply walk-forward-derived epistemic threshold
    live_result["bnn_overlay_on"] = int(
        live_result["p_bnn_epistemic"] > epist_threshold
    )
    live_result["bnn_epist_threshold"] = float(epist_threshold)

    print(f"\nLive prediction (latest row {X.index[-1].date()}):")
    for k, v in live_result.items():
        print(f"  {k}: {v}")

    # ── Build full tape with BNN predictions ──────────────────────────────
    if HAVE_TORCH:
        X_all_sc = scaler.transform(X_all)
        p_all_mean, p_all_epist, p_all_aleat = _predict_torch(
            models, X_all_sc, cfg.n_mc_samples
        )
    else:
        X_all_sc = scaler.transform(X_all)
        p_all_mean, p_all_epist, p_all_aleat = _predict_sklearn(models, X_all_sc)

    tape = pd.DataFrame({
        "Date":              X.index,
        "p_bnn_mean":        p_all_mean,
        "p_bnn_epistemic":   p_all_epist,
        "p_bnn_aleatoric":   p_all_aleat,
        "p_bnn_total_std":   np.sqrt(p_all_epist**2 + p_all_aleat**2),
        "bnn_overlay_on":    (p_all_epist > epist_threshold).astype(int),
        "y_true":            y_all,
        "in_holdout":        holdout_mask.values.astype(int),
    })
    tape_path = Path(out_dir) / "part2c_bnn_tape.csv"
    tape.to_csv(tape_path, index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────
    meta = {
        "part": "PART2C_BNN",
        "version": "V1_DEEP_ENSEMBLE_MC_DROPOUT",
        "backend": "pytorch" if HAVE_TORCH else "sklearn",
        "n_ensemble": cfg.n_ensemble,
        "n_mc_samples": cfg.n_mc_samples if HAVE_TORCH else 0,
        "hidden_dims": [cfg.hidden_dim_1, cfg.hidden_dim_2],
        "dropout_rate": cfg.dropout_rate,
        "weight_decay": cfg.weight_decay,
        "n_features": len(cfg.feature_cols),
        "feature_cols": list(cfg.feature_cols),
        "holdout_start": cfg.holdout_start,
        "n_training_rows": int((~holdout_mask).sum()),
        "n_holdout_rows": int(holdout_mask.sum()),
        "holdout_auc": holdout_auc,
        "holdout_brier": holdout_brier,
        "holdout_ece": holdout_ece,
        "holdout_decision_utility": holdout_util,
        "walkforward_mean_auc": float(wf_df["auc"].mean()),
        "walkforward_mean_brier": float(wf_df["brier"].mean()),
        "walkforward_mean_ece": float(wf_df["ece"].mean()),
        "walkforward_mean_utility": float(wf_df["decision_utility"].mean()),
        "epist_overlay_threshold_75pct": float(epist_threshold),
        "live_p_bnn_mean": live_result["p_bnn_mean"],
        "live_p_bnn_epistemic": live_result["p_bnn_epistemic"],
        "live_bnn_overlay_on": live_result["bnn_overlay_on"],
        # XGBoost baseline from Part 2 for quick comparison
        "xgb_baseline_auc": p2_summary.get("classification_base", {}).get("auc"),
        "xgb_baseline_ece": p2_summary.get("classification_base", {}).get("ece"),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = Path(out_dir) / "part2c_bnn_summary.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print()
    print("✅ PART 2C COMPLETE")
    print(f"   Tape:       {tape_path}")
    print(f"   WF results: {wf_path}")
    print(f"   Summary:    {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
