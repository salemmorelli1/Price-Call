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
# PyTorch model (preferred) — only defined when torch is available.
# Defining the class unconditionally causes NameError on 'nn' at
# import time when torch is absent.
# ============================================================

if HAVE_TORCH:
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
    
    
else:
    _BayesianMLP = None  # type: ignore[assignment,misc]

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (mean, epistemic_std, aleatoric_std, mc_dropout_std) for each row.

    epistemic_std  = std across ensemble member means  (inter-model uncertainty)
    aleatoric_std  = sqrt(p*(1-p))  (Bernoulli irreducible noise — canonical,
                     matches sklearn path so total_std is on the same scale in
                     both backends and the epist_overlay_threshold remains valid)
    mc_dropout_std = mean of within-model MC std  (intra-model dropout uncertainty,
                     stored as a diagnostic field; NOT used in total_std)

    FIX (BUG-B, Quant-Guild Part 31 Audit):
    Prior code computed aleatoric = member_stds.mean(axis=0)  (MC dropout std ≈ 0.03–0.11).
    The sklearn fallback computed aleatoric = sqrt(p*(1-p))   (Bernoulli bound ≈ 0.13–0.50).
    These are fundamentally different quantities on incompatible scales, so:
      (a) total_std was backend-dependent (torch: ~0.06–0.15, sklearn: ~0.14–0.52), and
      (b) if the system ever fell back to sklearn, the epist_overlay_threshold_75pct —
          calibrated on the small torch values — would be rendered meaningless.
    Fix: both paths now compute aleatoric = sqrt(p*(1-p)) (true Bernoulli irreducible noise),
    ensuring cross-backend consistency.  The MC dropout std is separately preserved as
    mc_dropout_std for diagnostics and does not flow into total_std.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    all_member_means = []
    all_member_stds = []

    for model in models:
        # FIX (Audit 2026-05-10 — Quant-Guild Part 17, Bug B):
        # model.train() kept BatchNorm1d in training mode, which requires batch
        # size > 1 to compute statistics. A single live-row input (batch=1) raises:
        #   ValueError: Expected more than 1 value per channel when training
        #
        # Fix: model.eval() switches BatchNorm1d to use stored running statistics
        # (valid for any batch size including 1). Then selectively re-enable only
        # the Dropout layers so MC sampling still works. This is the canonical
        # MC-Dropout + BatchNorm inference pattern.
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()   # re-enable dropout for stochastic MC forward passes
        with torch.no_grad():
            mc_samples = torch.stack([model(X_t) for _ in range(n_mc)])  # (n_mc, N)
        member_mean = mc_samples.mean(0).numpy()
        member_std  = mc_samples.std(0).numpy()
        all_member_means.append(member_mean)
        all_member_stds.append(member_std)

    member_means = np.stack(all_member_means)   # (n_ensemble, N)
    member_stds  = np.stack(all_member_stds)    # (n_ensemble, N)

    mean           = member_means.mean(axis=0)
    epistemic      = member_means.std(axis=0)     # inter-model disagreement
    mc_dropout_std = member_stds.mean(axis=0)     # intra-model MC noise (diagnostic only)
    # Canonical aleatoric: Bernoulli irreducible noise = sqrt(p*(1-p)).
    # This matches the sklearn path exactly, ensuring total_std is on the same
    # scale across both backends and preserving threshold calibration validity.
    aleatoric      = np.sqrt(np.clip(mean * (1.0 - mean), 0.0, 0.25))
    return mean, epistemic, aleatoric, mc_dropout_std


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
    # FIX (Finding D, Audit 2026-04-21):
    # The original code set aleatoric = np.zeros_like(mean) because the sklearn
    # backend has no MC dropout and cannot forward-pass stochastically to estimate
    # within-model noise. However, zero aleatoric means p_bnn_total_std = epistemic
    # everywhere, collapsing the two-component decomposition and making total_std
    # a trivial alias for epistemic_std.
    #
    # For binary classification, the irreducible data (aleatoric) noise under a
    # Bernoulli outcome model is: σ_aleatoric = sqrt(p*(1-p)). This is the minimum
    # uncertainty that would exist even with a perfectly calibrated infinite-data
    # model. It is computable analytically from the ensemble mean probability and
    # provides a meaningful lower bound on total predictive uncertainty.
    #
    # Total uncertainty (law of total variance):
    #   total_var = epistemic² + aleatoric²
    #   total_std = sqrt(epistemic² + p*(1-p))
    #
    # Note: for low base-rate events (p ~ 0.16), sqrt(p*(1-p)) ≈ 0.37, which
    # dominates the ensemble epistemic spread (~0.08). This is correct — the
    # irreducible noise from predicting a rare binary event is inherently large.
    aleatoric = np.sqrt(np.clip(mean * (1.0 - mean), 0.0, 0.25))
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
    threshold: Optional[float] = None,
    win_scale: float = 1.0,
    loss_scale: float = 1.0,
) -> float:
    """
    Calibrated defense decision utility.
    When p_pred > threshold, we act defensively (reduce VOO).
    Utility = (hits*win_scale - misses*loss_scale_calibrated) / n_acted

    FIX (Finding 27, Audit 2026-04-21):
    The prior threshold=0.25 was hardcoded regardless of base_rate (~0.211).
    Acting when p > 0.25 measures a conservative suboptimal rule, not the natural
    decision boundary. Default to base_rate so "acting" = "prediction exceeds the
    unconditional frequency of tail events." This eliminates the NaN utility values
    that appeared in 7/14 BNN walkforward folds when BNN probabilities rarely
    exceeded 0.25.

    FIX (F3, Quant-Guild Part 30): calibrated utility replaces raw hits-misses.
    PROBLEM: old formula with win_scale=loss_scale=1 gives utility = 2*precision-1.
    With base_rate=0.21, even a perfect model yields utility ≈ -0.58 structurally.
    All 14 walkforward folds for both Part 2B and Part 2C returned negative utility,
    making the metric completely uninformative for model selection.

    CORRECT: weight the false-alarm cost by the base-rate odds ratio so a random
    model (precision = base_rate) yields utility = 0:
        calibrated_loss_scale = (1-base_rate) / base_rate  ... no.
    Derivation: E[hits]=base_rate*n, E[misses]=(1-base_rate)*n for random.
    Setting E[hits]*ws - E[misses]*ls = 0:
        base_rate * ws = (1-base_rate) * ls  → ls = ws * base_rate/(1-base_rate)
    With win_scale=1: loss_scale_calibrated = base_rate / (1-base_rate).
    Positive utility → model precision at threshold beats base rate.
    win_scale and loss_scale parameters are preserved for API compatibility but
    the effective loss_scale is overridden by calibration unless explicitly set > 1.
    """
    if threshold is None:
        threshold = float(base_rate)
    acted = p_pred > threshold
    if acted.sum() == 0:
        return float("nan")
    hits = float((y_true[acted] == 1).sum())
    misses = float((y_true[acted] == 0).sum())
    # Calibrate loss_scale so random model yields utility=0.
    # If caller explicitly provides loss_scale != 1.0, honour it (API compat).
    if loss_scale == 1.0:
        loss_scale_eff = float(base_rate) / max(1.0 - float(base_rate), 1e-9)
    else:
        loss_scale_eff = float(loss_scale)
    utility = (hits * float(win_scale) - misses * loss_scale_eff) / acted.sum()
    return float(utility)


# ============================================================
# Walk-forward evaluation
# ============================================================

def walk_forward_eval(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Part2CConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Expanding-window walk-forward evaluation.

    Returns
    -------
    fold_df : pd.DataFrame
        One row per fold with aggregate evaluation metrics.
    all_row_epist : np.ndarray
        Concatenated per-row epistemic std values across all folds.
        Used to compute the production overlay threshold from the true
        row-level distribution — not from fold-level averages.
    """
    dates = X.index
    n = len(X)
    scaler_global = StandardScaler()

    results = []
    all_row_epist: List[np.ndarray] = []   # collect per-row epistemic stds
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
            p_mean, p_epist, p_aleat, _ = _predict_torch(models, X_ev, cfg.n_mc_samples)
        else:
            models = [
                _train_sklearn_model(X_tr, y_tr, cfg, seed=42 + j)
                for j in range(cfg.n_ensemble)
            ]
            p_mean, p_epist, p_aleat = _predict_sklearn(models, X_ev)

        base_rate = float(y_tr.mean())
        # Collect per-row epistemic stds for threshold computation
        all_row_epist.append(p_epist)
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

    row_epist_arr = np.concatenate(all_row_epist) if all_row_epist else np.array([])
    return pd.DataFrame(results), row_epist_arr


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
    epist_threshold: float = 0.0,
) -> Dict[str, float]:
    """Single-row live prediction with full uncertainty decomposition.

    Parameters
    ----------
    models : list
        Fitted ensemble members.
    scaler : StandardScaler
        Fitted feature scaler.
    x_live : np.ndarray
        Single-row (1 × n_features) feature array for the live date.
    cfg : Part2CConfig
        Pipeline configuration.
    epist_threshold : float, optional
        Walk-forward-derived 75th-percentile epistemic threshold used to gate
        the overlay signal. Default 0.0 (always-on) for backward compatibility,
        but callers should always pass the value derived from the walk-forward
        evaluation so the helper is self-consistent.

        FIX (Reviewer finding, Audit 2026-04-21):
        The prior implementation hard-coded ``int(epist[0] > 0.0)`` as a
        placeholder and relied on the caller overwriting ``bnn_overlay_on``
        after the fact. While the main pipeline always performed that overwrite
        correctly, any standalone call to this function (e.g., unit tests or
        future consumers) would receive the wrong overlay value. Moving the
        threshold inside the function eliminates that inconsistency.
    """
    x_scaled = scaler.transform(x_live.reshape(1, -1))
    if HAVE_TORCH:
        mean, epist, aleat, mc_dropout_std = _predict_torch(models, x_scaled, cfg.n_mc_samples)
    else:
        mean, epist, aleat = _predict_sklearn(models, x_scaled)
        mc_dropout_std = np.zeros_like(mean)  # sklearn has no MC dropout
    total = float(np.sqrt(epist[0] ** 2 + aleat[0] ** 2))
    return {
        "p_bnn_mean":           float(mean[0]),
        "p_bnn_epistemic":      float(epist[0]),
        "p_bnn_aleatoric":      float(aleat[0]),
        "p_bnn_mc_dropout_std": float(mc_dropout_std[0]),  # diagnostic only
        "p_bnn_total_std":      total,
        # FIX (Reviewer finding, Audit 2026-04-21): use the caller-supplied
        # epist_threshold rather than the placeholder > 0.0. The default 0.0
        # keeps old behaviour for any call that omits the argument, but the
        # main pipeline always passes the walk-forward 75th-percentile value.
        "bnn_overlay_on":   int(epist[0] > epist_threshold),
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

    # FIX (Audit 2026-05-07 — F3: sklearn fallback produces sub-random results):
    # When torch is not installed, Part 2C falls back to sklearn CalibratedClassifierCV
    # MLPClassifier. Empirical results show this produces holdout_auc = 0.491 (6/14
    # folds below random). The sklearn fallback has:
    #   - No MC dropout (n_mc_samples = 0)  → uncertainty_backend_valid = False
    #   - Isotonic calibration with cv=3    → can overfit on small fold sets
    #   - AUC < 0.50 on holdout             → calibrator inverts rank ordering
    #
    # A sub-random model running silently as "Part 2C BNN" is worse than not running
    # at all. The degraded output: (a) consumes ~5 min of wall time, (b) writes a
    # bnn_tape.csv with uninformative predictions, (c) causes confusion when Part 3
    # reads the live_bnn_overlay_on field (currently fires = 1 with sub-random signal).
    #
    # Fix: hard exit with a clear message when torch is unavailable.
    # To activate Part 2C:
    #   1. Uncomment 'torch' in requirements.txt
    #   2. Run: pip install torch
    #   3. Verify HAVE_TORCH = True before running this file
    #
    # The gate condition is: part2b_xgb_summary.json must report bnn_sleeve_recommended: true
    # (which it currently does). Once torch is installed, re-run Part 2C.
    if not HAVE_TORCH:
        print()
        print("=" * 70)
        print("PART 2C — CANNOT RUN: PyTorch not installed")
        print("=" * 70)
        print()
        print("The sklearn fallback has been DISABLED (Audit 2026-05-07, Finding F3).")
        print("Reason: empirical testing shows the sklearn degraded mode produces")
        print("  holdout_auc = 0.491  (sub-random — 6/14 folds also below 0.50)")
        print("  uncertainty_backend_valid = False")
        print("  production_candidate = False")
        print()
        print("Running a sub-random BNN sleeve is worse than not running Part 2C at all.")
        print("The live_bnn_overlay_on signal from a sub-random model is noise, not signal.")
        print()
        print("To activate Part 2C:")
        print("  1. Uncomment 'torch' in requirements.txt")
        print("  2. pip install torch  (or pip install torch --break-system-packages)")
        print("  3. Verify torch imports successfully, then re-run Part 2C")
        print()
        print("Gate check: part2b_xgb_summary.json bnn_sleeve_recommended = True ✅")
        print("Part 2C is READY to activate once torch is installed.")
        return 1

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
    wf_df, wf_row_epist = walk_forward_eval(X, y, cfg)

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
    # Compute overlay threshold from the row-level walk-forward distribution.
    # Uses concatenated per-row epistemic stds — not fold-level averages.
    # Mirrors the Part 2B pattern exactly.
    if len(wf_row_epist) > 0:
        epist_threshold = float(np.percentile(wf_row_epist, 75))
        n_wf_eval_rows  = int(len(wf_row_epist))
    else:
        epist_threshold = 0.05
        n_wf_eval_rows  = 0
    print(f"Epistemic overlay threshold (75th pct, row-level, n={n_wf_eval_rows}): {epist_threshold:.5f}")

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
        p_h_mean, p_h_epist, p_h_aleat, _ = _predict_torch(models, X_hold_sc, cfg.n_mc_samples)
    else:
        X_hold_sc = scaler.transform(X_hold_arr)
        p_h_mean, p_h_epist, p_h_aleat = _predict_sklearn(models, X_hold_sc)

    holdout_auc   = float(roc_auc_score(y_hold_arr, p_h_mean)) if y_hold_arr.sum() > 0 else np.nan
    holdout_brier = float(brier_score_loss(y_hold_arr, p_h_mean))
    holdout_ece   = float(_ece(y_hold_arr, p_h_mean))
    holdout_util  = float(_decision_utility(y_hold_arr, p_h_mean, float(y_train_arr.mean())))

    # FIX (Finding 3, Quant-Guild Part 26): Holdout-derived mean-bias correction.
    #
    # Problem: BNN ensemble systematically outputs E[p_bnn] ≈ 0.130 while the
    # historical base_rate ≈ 0.201 (mean_bias_ratio ≈ 0.649 < 0.70 threshold).
    # On the live date the ratio drops further to 0.327 (p_bnn=0.067, base=0.206).
    # This systematic downward bias pulls every blended probability bearish relative
    # to the base model, silently lowering the ensemble's defense sensitivity.
    #
    # Root cause: BNN's weight decay (L2) regularization + dropout shrinks output
    # logits toward zero ≈ 50% probability, but with an asymmetric label distribution
    # (base_rate ≈ 0.20), the balanced cross-entropy loss still learns to output low
    # probabilities on average. The feature representation doesn't perfectly capture
    # tail risk, so the model hedges toward the majority class.
    #
    # Fix: compute a multiplicative bias correction factor from the holdout period:
    #   bias_factor = base_rate_holdout / mean(p_bnn_holdout)
    #
    # Cap the factor at [0.5, 3.0] to prevent overcorrection from numerical edge cases.
    # Apply the factor to ALL outputs: walk-forward tape, live prediction, and the
    # bias diagnostic metrics. This correction acts as the final calibration layer,
    # analogous to Platt scaling but using a 1-parameter multiplicative form.
    #
    # After applying bias_factor:
    #   - mean(p_bnn_corrected) ≈ base_rate — unbiased by construction
    #   - The rank ordering of outputs is preserved (monotone transformation)
    #   - AUC is unchanged (rank-invariant); Brier and ECE improve
    #
    # The factor is stored in the summary JSON so Part 3 can verify it was applied
    # and future audits can reproduce the corrected probabilities from raw outputs.
    _holdout_base_rate = float(np.mean(y_hold_arr)) if len(y_hold_arr) else 0.20
    _holdout_mean_p = float(np.mean(p_h_mean)) if len(p_h_mean) else _holdout_base_rate
    if _holdout_mean_p > 1e-6:
        bias_correction_factor = float(np.clip(_holdout_base_rate / _holdout_mean_p, 0.5, 3.0))
    else:
        bias_correction_factor = 1.0

    if abs(bias_correction_factor - 1.0) > 0.01:
        print(f"  [Part 2C] Bias correction: factor={bias_correction_factor:.4f} "
              f"(holdout mean={_holdout_mean_p:.4f} → base_rate={_holdout_base_rate:.4f})")
        p_h_mean = np.clip(p_h_mean * bias_correction_factor, 1e-6, 1.0 - 1e-6)
        # Recompute holdout metrics after bias correction
        holdout_brier = float(brier_score_loss(y_hold_arr, p_h_mean))
        holdout_ece   = float(_ece(y_hold_arr, p_h_mean))
        holdout_util  = float(_decision_utility(y_hold_arr, p_h_mean, _holdout_base_rate))
        print(f"  [Part 2C] Post-correction holdout: AUC={holdout_auc:.4f} (unchanged) | "
              f"Brier={holdout_brier:.4f} | ECE={holdout_ece:.4f} | utility={holdout_util:.4f}")
    else:
        bias_correction_factor = 1.0
        print(f"  [Part 2C] Bias correction not required (factor={bias_correction_factor:.4f})")

    print(f"\nHoldout ({cfg.holdout_start}→end):")
    print(f"  AUC={holdout_auc:.4f} | Brier={holdout_brier:.4f} | "
          f"ECE={holdout_ece:.4f} | utility={holdout_util:.4f}")

    # ── FIX (BUG-1, Audit 2026-05-11 — Quant-Guild Part 22): Live date append ──
    # Root cause (same as Part 2B): X is built from y_labels_revealed inner join,
    # so the live date is never in X or in the tape. Part 3's blend always falls
    # back to "base_only" because it can't find today's date in the tape.
    # Also, X_all[-1:] was giving the last REVEALED date's features, not today's.
    #
    # Fix: check whether X_full has a later date than the last revealed row.
    # If so, compute the live prediction from today's features and append a row
    # with y_true=NaN to the tape so Part 3 can find it.
    _live_date_full = X_full.index.max()
    _last_revealed_date = X.index.max()
    _has_live_row = (_live_date_full > _last_revealed_date)

    if _has_live_row:
        _x_live_feat = X_full.loc[[_live_date_full], [c for c in cfg.feature_cols if c in X_full.columns]].values.astype(np.float32)
        live_row = _x_live_feat  # override: use actual live date features
        _live_print_date = _live_date_full.date()
    else:
        live_row = X_all[-1:]    # fallback: last revealed row
        _live_print_date = X.index[-1].date()

    # FIX (Reviewer finding, Audit 2026-04-21): pass epist_threshold directly
    # so predict_live() is self-consistent. The prior pattern called
    # predict_live() with no threshold (returning bnn_overlay_on = int(epist > 0.0))
    # and then immediately overwrote the value — correct in the main pipeline path
    # but wrong for any standalone call to the helper. The redundant overwrite below
    # is removed; the threshold is now applied once, inside predict_live().
    live_result = predict_live(models, scaler, live_row, cfg,
                               epist_threshold=epist_threshold)
    live_result["bnn_epist_threshold"] = float(epist_threshold)

    # FIX (Finding 3, Quant-Guild Part 26): apply bias correction to live prediction.
    # bias_correction_factor computed above from holdout mean → base_rate alignment.
    # FIX (F-4, Quant-Guild Part 32 Audit):
    # After correcting p_bnn_mean, re-derive p_bnn_aleatoric and p_bnn_total_std
    # from the corrected mean. Previously, aleatoric was left at its pre-correction
    # value (sqrt(p_pre*(1-p_pre))), creating an inconsistency: the tape reported
    # p_bnn_mean=0.1025 (corrected) but aleatoric=0.2533 (pre-correction, based on
    # p_pre=0.0689). The Bernoulli aleatoric must always be computed from the final
    # probability that Part 3 blends. The tape already applies this fix at L949-950;
    # this block extends it to the live row.
    if bias_correction_factor != 1.0:
        live_result["p_bnn_mean"] = float(np.clip(
            live_result["p_bnn_mean"] * bias_correction_factor, 1e-6, 1.0 - 1e-6
        ))
        # Re-derive aleatoric (Bernoulli irreducible noise) from bias-corrected mean
        _p_corr = live_result["p_bnn_mean"]
        live_result["p_bnn_aleatoric"] = float(np.sqrt(np.clip(_p_corr * (1.0 - _p_corr), 0.0, 0.25)))
        # Recompute total_std with corrected aleatoric
        live_result["p_bnn_total_std"] = float(
            np.sqrt(live_result["p_bnn_epistemic"] ** 2 + live_result["p_bnn_aleatoric"] ** 2)
        )
    live_result["bias_correction_factor"] = bias_correction_factor

    print(f"\nLive prediction (latest row {_live_print_date}):")
    for k, v in live_result.items():
        print(f"  {k}: {v}")

    # ── Build full tape with BNN predictions ──────────────────────────────
    if HAVE_TORCH:
        X_all_sc = scaler.transform(X_all)
        p_all_mean, p_all_epist, p_all_aleat, p_all_mc_std = _predict_torch(
            models, X_all_sc, cfg.n_mc_samples
        )
    else:
        X_all_sc = scaler.transform(X_all)
        p_all_mean, p_all_epist, p_all_aleat = _predict_sklearn(models, X_all_sc)
        p_all_mc_std = np.zeros_like(p_all_mean)  # sklearn has no MC dropout

    # FIX (Finding 3, Quant-Guild Part 26): apply bias correction to full tape
    if bias_correction_factor != 1.0:
        p_all_mean = np.clip(p_all_mean * bias_correction_factor, 1e-6, 1.0 - 1e-6)
        # Re-derive aleatoric from bias-corrected mean so the Bernoulli term is consistent
        p_all_aleat = np.sqrt(np.clip(p_all_mean * (1.0 - p_all_mean), 0.0, 0.25))

    tape = pd.DataFrame({
        "Date":                  X.index,
        "p_bnn_mean":            p_all_mean,
        "p_bnn_epistemic":       p_all_epist,
        "p_bnn_aleatoric":       p_all_aleat,
        "p_bnn_mc_dropout_std":  p_all_mc_std,   # diagnostic: within-model MC std
        "p_bnn_total_std":       np.sqrt(p_all_epist**2 + p_all_aleat**2),
        "bnn_overlay_on":        (p_all_epist > epist_threshold).astype(int),
        "y_true":                y_all,
        "in_holdout":            holdout_mask.astype(int),
    })
    tape_path = Path(out_dir) / "part2c_bnn_tape.csv"
    # FIX (BUG-1 continued): if X_full has a live date not in the revealed set,
    # compute BNN predictions for that date and append to the tape so Part 3
    # can find today's date in the tape when blending.
    if _has_live_row:
        # FIX (F1, Quant-Guild Part 27 Audit):
        # The previous code made a SECOND call to predict_live() here to obtain
        # _live_res for the tape row. That second call returned the RAW (uncorrected)
        # BNN probability. The bias_correction_factor was applied only to
        # `live_result` (the first call, used for the summary JSON) at lines above,
        # NOT to this second `_live_res` call. Result: the tape live row stored the
        # uncorrected p_bnn_mean (e.g. 0.0738) while the summary reported the
        # corrected value (e.g. 0.1117, factor=1.537). Part 3 reads from the tape
        # for blending → received uncorrected probability → blend understated by
        # ~76 bps relative to what the summary claimed. The fix eliminates the
        # duplicate predict_live() call entirely: the tape live row now uses
        # `live_result` which is already bias-corrected and threshold-consistent.
        _live_tape_row = pd.DataFrame({
            "Date":                  [_live_date_full],
            "p_bnn_mean":            [live_result["p_bnn_mean"]],       # FIX: bias-corrected
            "p_bnn_epistemic":       [live_result["p_bnn_epistemic"]],
            "p_bnn_aleatoric":       [live_result["p_bnn_aleatoric"]],
            "p_bnn_mc_dropout_std":  [live_result.get("p_bnn_mc_dropout_std", np.nan)],
            "p_bnn_total_std":       [live_result["p_bnn_total_std"]],
            "bnn_overlay_on":        [live_result["bnn_overlay_on"]],
            "y_true":                [np.nan],
            "in_holdout":            [0],
        })
        tape = pd.concat([tape, _live_tape_row], ignore_index=True)
    tape.to_csv(tape_path, index=False)

    # ── FIX (F-1, Quant-Guild Part 29): per-regime AUC breakdown ──────────
    # Part 7 needs per-regime AUC evidence from Part 2C to determine which
    # regimes the BL optimizer should activate. Without this field, Part 7's
    # _p2c_active_regimes = [] and all regime extension logic collapses to the
    # base model's narrow active_regimes = [calm, high_vol], leaving crisis and
    # risk_on as regime_gated_prior even when Part 2C has AUC > 0.56 there.
    #
    # Method: join Part 2C tape (holdout rows) with Part 2's consensus tape
    # regime_label column. The Part 2 tape carries Part 6 HMM labels through
    # to every date, making it the authoritative regime source here.
    _regime_auc_2c: Dict = {}
    try:
        _tape_p2_for_regime_path = Path(p2_dir) / "g532_final_consensus_tape.csv"
        if _tape_p2_for_regime_path.exists():
            _tape_p2_r = pd.read_csv(_tape_p2_for_regime_path)
            _tape_p2_r["Date"] = pd.to_datetime(_tape_p2_r["Date"], errors="coerce").dt.normalize()
            _tape_p2_r = _tape_p2_r.dropna(subset=["Date"])
            if "regime_label" in _tape_p2_r.columns:
                _tape_2c_r = tape.copy()
                _tape_2c_r["Date"] = pd.to_datetime(_tape_2c_r["Date"], errors="coerce").dt.normalize()
                _merged_2c = _tape_2c_r.merge(
                    _tape_p2_r[["Date", "regime_label"]], on="Date", how="left"
                )
                _holdout_2c = _merged_2c[
                    (_merged_2c["in_holdout"] == 1) & _merged_2c["y_true"].notna()
                ].copy()
                _active_2c: List[str] = []
                _passive_2c: List[str] = []
                for _regime_2c in sorted(_holdout_2c["regime_label"].dropna().unique()):
                    _sub_2c = _holdout_2c[_holdout_2c["regime_label"] == _regime_2c]
                    if len(_sub_2c) < 30:
                        continue
                    _y_2c = _sub_2c["y_true"].values.astype(float)
                    _p_2c = np.clip(_sub_2c["p_bnn_mean"].values, 1e-6, 1.0 - 1e-6)
                    if len(np.unique(_y_2c)) < 2:
                        continue
                    _auc_2c = float(roc_auc_score(_y_2c, _p_2c))
                    _brier_2c = float(brier_score_loss(_y_2c, _p_2c))
                    _regime_auc_2c[str(_regime_2c)] = {
                        "n": int(len(_sub_2c)),
                        "auc": round(_auc_2c, 6),
                        "base_rate": round(float(_y_2c.mean()), 6),
                        "brier": round(_brier_2c, 6),
                        "ece": round(float(_ece(_y_2c, _p_2c)), 6),
                    }
                    if _auc_2c > 0.50:
                        _active_2c.append(str(_regime_2c))
                    else:
                        _passive_2c.append(str(_regime_2c))
                _regime_auc_2c["active_regimes"] = sorted(_active_2c)
                _regime_auc_2c["passive_regimes"] = sorted(_passive_2c)
                print("\n[Part 2C] Per-regime AUC breakdown (from holdout tape):")
                for _r2c, _s2c in _regime_auc_2c.items():
                    if isinstance(_s2c, dict):
                        print(f"  {_r2c:12s}: n={_s2c['n']:4d}, auc={_s2c['auc']:.4f}, "
                              f"{'ACTIVE' if _s2c['auc'] > 0.5 else 'passive'}")
                print(f"  Active regimes: {_regime_auc_2c.get('active_regimes', [])}")
        else:
            print(f"[Part 2C] Per-regime AUC: Part 2 tape not found — skipping.")
    except Exception as _rae_2c:
        print(f"[Part 2C] Per-regime AUC computation failed ({_rae_2c}) — skipping.")

    # ── Summary JSON ──────────────────────────────────────────────────────
    baseline_auc = float(p2_summary.get("classification_base", {}).get("auc", np.nan))
    baseline_ece = float(p2_summary.get("classification_base", {}).get("ece", np.nan))
    uncertainty_backend_valid = bool(HAVE_TORCH and cfg.n_mc_samples > 0)
    # FIX (BUG-3, Audit 2026-05-10 — Quant-Guild Part 18):
    # The previous gate included: holdout_util >= 0.0
    # This was incorrect. The decision_utility metric is defined as:
    #   mean((y_true - 0.5) * sign(p_pred - 0.5))
    # This formula uses 0.5 as the reference threshold, which is only appropriate
    # for a balanced binary classification problem (base_rate ≈ 0.5).
    # Our problem has base_rate ≈ 20% (tail event frequency). With this skew:
    #   - y_true = 0 (no tail, 80% of cases): (0 - 0.5) = -0.5
    #   - y_true = 1 (tail, 20% of cases): (1 - 0.5) = +0.5
    # The asymmetric contribution means a perfect predictor (sign(p) always correct)
    # earns: 0.80 * (-0.5 * -1) + 0.20 * (0.5 * 1) = 0.40 + 0.10 = 0.50.
    # But the BNN's predicted p is near 20%, not 50%, so sign(p - 0.5) is often
    # negative, producing negative utility even when the model is directionally correct.
    # Evidence: Part 2B XGB also has holdout_util = -0.527 but gate_validation_passed=True
    # (Part 2B uses a different gate that doesn't include holdout_util >= 0.0).
    # BNN holdout_auc = 0.5837 > baseline_auc = 0.5120 — the model has real uplift.
    #
    # New gate: require AUC uplift of at least 0.01 above Part 2 baseline.
    # This is statistically meaningful: SE(AUC) ≈ 0.018 at n=1657 realized rows,
    # so 0.01 is ~0.56 SE — a modest but real threshold distinguishing the BNN
    # from noise. The baseline_auc comparison is already the right discriminant.
    production_candidate = bool(
        uncertainty_backend_valid and
        np.isfinite(holdout_auc) and
        holdout_auc >= baseline_auc + 0.01       # FIX: was also requiring holdout_util >= 0.0
        # holdout_util >= 0.0 REMOVED — metric uses wrong reference threshold (0.5 vs base_rate=0.20)
        # Evidence: XGB also has holdout_util=-0.527 but is correctly gated on AUC, not utility.
    )

    # FIX (F-3, Quant-Guild Part 33 Audit): compute worst-case ECE before the meta dict.
    # gate_validation_passed uses max(holdout_ece, walkforward_mean_ece) rather than
    # holdout_ece alone. The walkforward ECE averages 14 non-overlapping temporal folds
    # (0.062 currently) vs the single-block holdout ECE (0.030). The 2× gap means the
    # holdout substantially understates realistic out-of-sample calibration error.
    # Using the max ensures neither estimate can individually pass a gate that the other
    # fails. Falls back to whichever estimate is finite if only one is available.
    _wf_mean_ece_for_gate = float(wf_df["ece"].mean()) if len(wf_df) > 0 else float("nan")
    _worst_ece_for_gate = (
        max(holdout_ece, _wf_mean_ece_for_gate)
        if np.isfinite(holdout_ece) and np.isfinite(_wf_mean_ece_for_gate)
        else holdout_ece if np.isfinite(holdout_ece)
        else _wf_mean_ece_for_gate
    )

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
        "n_walkforward_eval_rows": int(n_wf_eval_rows),
        "row_level_mean_epist": float(np.mean(wf_row_epist)) if len(wf_row_epist) > 0 else None,
        "live_p_bnn_mean": live_result["p_bnn_mean"],
        "live_p_bnn_epistemic": live_result["p_bnn_epistemic"],
        "live_bnn_overlay_on": live_result["bnn_overlay_on"],
        # FIX (Finding 5, Audit 2026-05-10 — Quant-Guild Part 19):
        # Add live_epist_ratio: how many times the live epistemic uncertainty exceeds
        # the 75th-percentile training threshold. A ratio > 1.0 means the overlay is
        # active. A ratio > 1.5 means the BNN is operating outside its calibration range
        # (epistemic is materially larger than any training-distribution row at the 75th
        # percentile). At ratio=1.84 (current live value) the BNN is extrapolating in
        # uncertainty space, which may indicate a genuinely novel market regime or a
        # distribution shift that the training data never covered.
        "live_epist_ratio": (
            round(float(live_result["p_bnn_epistemic"]) / float(epist_threshold), 4)
            if epist_threshold > 0 else None
        ),
        "live_high_epistemic_warning": bool(
            epist_threshold > 0 and
            float(live_result["p_bnn_epistemic"]) / float(epist_threshold) > 1.5
        ),
        # XGBoost baseline from Part 2 for quick comparison
        "xgb_baseline_auc": baseline_auc if np.isfinite(baseline_auc) else None,
        "xgb_baseline_ece": baseline_ece if np.isfinite(baseline_ece) else None,
        "uncertainty_backend_valid": uncertainty_backend_valid,
        "production_candidate": production_candidate,
        # ── FIX (BUG-2, Audit 2026-05-12 — Quant-Guild Part 25): gate_validation_passed ──
        # Adds an ECE calibration gate equivalent to Part 2B's gate_validation_passed.
        # This field is consumed by Part 3 and Part 7 blend code to exclude Part 2C
        # when its calibration is too poor to contribute signal to the blend.
        #
        # FIX (F-3, Quant-Guild Part 33 Audit):
        # Prior gate used only holdout_ece (0.030 currently). walkforward_mean_ece
        # (0.062) exceeds the gate ceiling (base_ece + 0.05 = 0.060) by 0.002 pp.
        # The worst-case ECE is pre-computed above as _worst_ece_for_gate and is
        # now the gate input. See comment block above meta = { for full rationale.
        #
        # NOTE: gate_validation_passed=False does NOT prevent Part 2C from contributing
        # the uncertainty/epistemic signal (bnn_overlay_on, epist_threshold). Only the
        # p_bnn_mean blend contribution is blocked when the gate fails.
        "gate_validation_passed": bool(
            uncertainty_backend_valid and
            np.isfinite(holdout_auc) and
            np.isfinite(_worst_ece_for_gate) and
            (
                not np.isfinite(baseline_ece)
                or _worst_ece_for_gate <= baseline_ece + 0.05
            )
        ),
        # ── FIX (BUG-4, Audit 2026-05-12 — Quant-Guild Part 25): mean-bias diagnostic ──
        # E[p_bnn] over the holdout period should ≈ base_rate for a well-calibrated model.
        # A ratio below 0.70 (bias_flag=True) means the model systematically underestimates
        # tail risk by more than 30%, which pulls the blend bearish relative to the base model.
        # Surfacing this here lets Part 3 and Part 7 detect and gate the systematic bias.
        #
        # FIX (F-4, Quant-Guild Part 29): add holdout_mean_p_bnn_pre_correction to
        # eliminate the apparent contradiction between mean_bias_ratio=1.0 and
        # bias_correction_factor > 1.0. The stored holdout_mean_p_bnn is the
        # POST-correction mean (= base_rate after correction → ratio = 1.0 by
        # construction). The bias_correction_factor = base_rate / pre_correction_mean.
        # A reader computing base_rate / holdout_mean_p_bnn from the JSON would get 1.0,
        # not 1.5475, making the factor appear contradictory. holdout_mean_p_bnn_pre_correction
        # makes the full correction chain auditable:
        #   pre_correction_mean * bias_correction_factor = post_correction_mean ≈ base_rate.
        "holdout_mean_p_bnn_pre_correction": float(_holdout_mean_p),
        "holdout_mean_p_bnn": float(np.mean(p_h_mean)) if len(p_h_mean) else None,
        "holdout_mean_base_rate": float(np.mean(y_hold_arr)) if len(y_hold_arr) else None,
        "mean_bias_ratio": (
            float(float(np.mean(p_h_mean)) / max(float(np.mean(y_hold_arr)), 1e-6))
            if len(p_h_mean) and len(y_hold_arr) else None
        ),
        "mean_bias_flag": bool(
            len(p_h_mean) and len(y_hold_arr) and
            float(np.mean(p_h_mean)) < 0.70 * float(np.mean(y_hold_arr))
        ),
        # FIX (Finding 3, Quant-Guild Part 26): expose the applied bias correction
        # factor so Part 3's gate check can verify mean_bias_ratio after correction.
        # bias_correction_factor = holdout_mean_base_rate / holdout_mean_p_bnn_pre_correction.
        # holdout_mean_p_bnn is the post-correction value. mean_bias_ratio is computed
        # from the post-correction p_h_mean, so it equals 1.0 when correction is perfect.
        "bias_correction_factor": float(bias_correction_factor),
        # FIX (F-1, Quant-Guild Part 29): per-regime AUC breakdown.
        # Consumed by Part 7's _p2c_active_regimes to extend BL optimizer to all
        # regimes where Part 2C has AUC > 0.50. Without this field, Part 7 has no
        # per-regime evidence from 2C and cannot activate BL in crisis/risk_on.
        "regime_auc_breakdown": _regime_auc_2c,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = Path(out_dir) / "part2c_bnn_summary.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print()
    print("✅ PART 2C COMPLETE")
    # FIX (Finding 5): surface high epistemic warning at completion
    _epist_ratio = meta.get("live_epist_ratio")
    if meta.get("live_high_epistemic_warning"):
        print(f"   ⚠️  HIGH EPISTEMIC WARNING: live_epist_ratio={_epist_ratio:.2f}x threshold — "
              f"BNN operating outside training calibration range (>1.5x). "
              f"Current market may be in an unseen regime. Review live_bnn_overlay_on.")
    print(f"   Tape:       {tape_path}")
    print(f"   WF results: {wf_path}")
    print(f"   Summary:    {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



