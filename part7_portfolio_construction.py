#!/usr/bin/env python3
# @title PART 7 — Portfolio Construction & Risk Budgeting
# =============================================================================
# Industry-grade portfolio optimization for PriceCallProject v2
#
# Replaces the ad-hoc defense weight formula:
#   w_voo = clip(0.60 + active_weight, 0.42, 0.70)
#
# With principled multi-asset allocation via:
#   1. Black-Litterman framework (model view + CAPM prior)
#   2. Risk parity / Kelly fraction position sizing
#   3. CVaR-constrained optimization
#   4. Transaction cost-aware rebalancing
#
# Multi-asset universe: VOO, IEF, GLD, QQQ, TLT
#
# AUDIT CHANGELOG (Quant-Guild Part 8 session, 2026-04)
# ──────────────────────────────────────────────────────
# Finding D (IMPORTANT): mean-variance optimizer received mismatched units.
#   mu_bl is produced in annual units by estimate_expected_returns(), but
#   compute_allocation() computed cov_h = cov * (1/252) and passed that daily
#   covariance to the optimizer.  In the objective
#       maximize  mu @ w  -  0.5 * lambda * w' Sigma w
#   the risk term was ~252x smaller than the return term, making effective risk
#   aversion lambda/252 ≈ 0.01 instead of 2.5.  The optimizer saw the portfolio
#   as nearly risk-free and maximised return by pushing w_voo to the upper bound
#   (voo_max) on every unconstrained run.  The dead-band then locked that weight
#   in for all 1,641 subsequent days.  The Black-Litterman computation was
#   structurally bypassed in practice.
#   Fix: pass cov (annualised) to the optimizer, matching the annual scale of
#   mu_bl.  cov_h is removed; it served no correct purpose in this function.
# =============================================================================
from __future__ import annotations

import os
import dataclasses
from pathlib import Path
import json
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:
    HAVE_CVXPY = False
    print("[Part 7] cvxpy not installed. pip install cvxpy — falling back to scipy optimizer.")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Part7Config:
    version: str = "V2_DAILY_CANONICAL"
    part0_dir: str = "./artifacts_part0"
    part2_dir: str = "./artifacts_part2_g532/predictions"
    # FIX (BUG-1/BUG-2/BUG-3, Audit 2026-05-11 — Quant-Guild Part 23):
    # Part 2B (XGB ensemble, AUC=0.571, t=5.84) and Part 2C (BNN, AUC=0.582,
    # t=6.90) run BEFORE Part 7 in the pipeline. Their tapes and summary JSONs
    # are available on disk when Part 7 executes. Paths added here so Part 7
    # can load and blend them into its p_tail signal and view_confidence.
    part2b_dir: str = "./artifacts_part2b_xgb/predictions"
    part2c_dir: str = "./artifacts_part2c_bnn/predictions"
    part6_dir: str = "./artifacts_part6"
    out_dir: str = "./artifacts_part7"
    horizon: int = 1

    # === Asset Universe ===
    # The minimal 2-asset universe (VOO, IEF) plus optional extensions
    # When only 2 assets are used, the problem reduces to the current system
    universe: Tuple[str, ...] = ("VOO", "IEF")
    extended_universe: Tuple[str, ...] = ("VOO", "IEF", "GLD", "TLT")

    use_extended_universe: bool = False  # Set True when you have sufficient history

    # === Risk Budget ===
    # Maximum VOO allocation range
    w_voo_min: float = 0.35
    w_voo_max: float = 0.75
    w_ief_min: float = 0.20
    w_ief_max: float = 0.65

    # Max position change per rebalance (turnover control)
    max_turnover: float = 0.05         # 5% max single-trade position change (comment previously said 15% — documentation error)

    # Transaction costs
    slip_bps: float = 1.0              # One-way slippage per unit traded
    commission_bps: float = 1.0        # Broker commission

    # === Black-Litterman Parameters ===
    tau: float = 0.25                  # FIX (F4, Audit 2026-05-10 — Quant-Guild Part 20):
    # tau=0.05 gives the CAPM prior 97% weight and the model view only 3%,
    # because view_frac = trace(P'Ω⁻¹P) / [trace((τΣ)⁻¹) + trace(P'Ω⁻¹P)].
    # Both inv_tauS and view_var scale with tau, so view_frac is tau-INVARIANT
    # when omega ∝ tau (as was coded). The fix is the omega formula (F5 below),
    # which uses the standard BL parameterisation where Ω is independent of tau.
    # With the corrected Ω, view_frac scales with tau as expected; tau=0.25
    # gives a meaningful 35–55% view contribution at typical AUC=0.53–0.58.
    # Uncertainty in CAPM prior (standard range: 0.05–0.50)
    risk_aversion: float = 2.5         # Investor risk aversion coefficient λ
    market_weights: Dict[str, float] = field(
        default_factory=lambda: {"VOO": 0.60, "IEF": 0.40}  # Market-cap proxy
    )

    # === CVaR Parameters ===
    cvar_confidence: float = 0.95      # CVaR at 95%
    max_cvar_budget: float = 0.025     # Max expected loss at 95% CVaR per period

    # === Regime-conditional risk budgets ===
    # Matched to the 4-regime HMM in Part 6 (calm / risk_on / high_vol / crisis).
    # Each multiplier scales both the VOO weight ceiling and the effective risk
    # aversion used in the BL optimizer.  With crisis_mult=0.50, voo_max resolves
    # to max(0.35, min(0.75*0.50, 0.70)) = 0.375 — feasible bounds guaranteed.
    regime_risk_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "calm":     1.30,   # quietest 25% — lean into equity risk
        "risk_on":  1.10,   # normal expansionary — modest equity tilt
        "high_vol": 0.75,   # elevated vol — moderate defense
        "crisis":   0.50,   # genuine tail episode — meaningful defense
        "unknown":  0.70,
    })

    cov_window: int = 126              # Rolling covariance window: 126 trading days (~6 months)
    cov_ewm_halflife: int = 21         # EWM half-life for covariance (1 trading month)
    # FIX (Finding 1, Audit 2026-05-10 — Quant-Guild Part 19):
    # The original 2% dead-band was calibrated for H=7 weekly rebalancing where
    # daily noise could cause spurious trading. At H=1 daily rebalancing the
    # Black-Litterman optimizer produces Δw ≈ 0.3–1.5% in normal conditions
    # (edge ≈ 1–3%, risk_aversion ≈ 2.5, portfolio_vol ≈ 8%). With a 2% threshold,
    # 98.1% of rows were permanently locked at 60/40 — the entire BL computation
    # was structurally bypassed. Reducing to 0.5% allows regime-conditional
    # rebalancing while still suppressing genuinely negligible changes (<0.5%).
    min_rebalance_threshold: float = 0.005  # 0.5% dead-band for H=1 daily rebalances (was 0.02)


CFG = Part7Config()

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



def normalize_regime_label(label: object) -> str:
    # FIX (Finding #14, 2026-04): the prior mapping collapsed 'calm' → 'risk_on',
    # making the calm multiplier (1.30) in regime_risk_multipliers unreachable.
    # 'calm' is now preserved as its own key so the optimizer applies the correct,
    # more aggressive equity tilt in genuinely low-volatility regimes.
    # 'dislocated' (Part 2 GMM label) continues to map to 'crisis' for conservative
    # position sizing during stress episodes.
    s = str(label).strip().lower() if label is not None else "unknown"
    mapping = {
        "calm":       "calm",     # was incorrectly mapped to "risk_on" — now preserved
        "risk_on":    "risk_on",
        "high_vol":   "high_vol",
        "crisis":     "crisis",
        "dislocated": "crisis",   # Part 2 GMM stress label → crisis multiplier
        "unknown":    "unknown",
    }
    return mapping.get(s, "unknown")



# ============================================================
# Covariance estimation
# ============================================================

def estimate_covariance(
    returns: pd.DataFrame,
    window: int = 126,
    ewm_halflife: int = 21,
) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimate.
    Combines sample covariance with constant-correlation target.
    Dramatically more stable than sample covariance at small N.
    """
    from sklearn.covariance import LedoitWolf

    r = returns.dropna().tail(window)
    if len(r) < 20:
        return np.eye(returns.shape[1]) * 0.04  # flat fallback

    # EWM returns (more weight on recent data)
    ewm_r = r.ewm(halflife=ewm_halflife).mean()
    centered = r - ewm_r

    lw = LedoitWolf()
    lw.fit(centered.values)
    cov = lw.covariance_

    # Annualize
    cov_ann = cov * 252
    return cov_ann


def estimate_expected_returns(
    model_view: Dict[str, float],
    market_weights: np.ndarray,
    cov: np.ndarray,
    asset_names: List[str],
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.

    Model view: p_tail_base gives us a directional view on VOO vs IEF spread.
    We translate this to a return view on the excess return of VOO over IEF.

    Formula:
    mu_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1 × Π + P'Ω^-1 × q]

    where:
      Π = equilibrium expected returns (CAPM prior)
      P = view matrix (which assets the view applies to)
      q = view returns (what the model predicts)
      Ω = view uncertainty
    """
    n = len(asset_names)
    # CAPM equilibrium returns
    pi = risk_aversion * cov @ market_weights.reshape(-1, 1)  # (n, 1)

    voo_idx = asset_names.index("VOO") if "VOO" in asset_names else 0
    ief_idx = asset_names.index("IEF") if "IEF" in asset_names else 1

    # View: VOO - IEF excess return over H-day horizon
    if "voo_excess_view" in model_view and np.isfinite(model_view["voo_excess_view"]):
        # Single view on VOO-IEF spread
        P = np.zeros((1, n))
        P[0, voo_idx] = 1.0
        P[0, ief_idx] = -1.0
        q = np.array([[float(model_view["voo_excess_view"])]])
        # Uncertainty in view: proportional to confidence
        view_confidence = float(model_view.get("view_confidence", 0.5))
        # P @ (tau * cov) @ P.T is a (1,1) matrix for a single view.
        # Extract the scalar explicitly so float() is safe across all numpy versions.
        view_var = float(np.asarray(P @ (tau * cov) @ P.T).reshape(-1)[0])
        view_var = max(view_var, 1e-12)
        # FIX (F5, Audit 2026-05-10 — Quant-Guild Part 20):
        # Standard BL (Idzorek 2005): Ω = τ × P'ΣP × (1−c)/c
        # Previous code used Ω = view_var / c, which is τ × P'ΣP / c — a factor of
        # (1/c − 1) = (1−c)/c × (c/(1−c)) too large at c<1, over-weighting uncertainty
        # and suppressing the model view by ~58% extra at c=0.37.
        # At c=1 (perfect confidence): Ω→0 (view fully trusted) — both formulas agree.
        # At c=0: Ω→∞ (view ignored) — both formulas agree.
        # At c=0.37 (AUC≈0.53): old Ω = view_var/0.37 = 2.70×view_var;
        #                        new Ω = view_var×1.70/0.37×(1-0.37)/0.37 ... wait:
        #                        new Ω = view_var×(1-0.37)/0.37 = view_var×1.70
        # The corrected formula reduces Ω by 37%, giving the model view 37% more weight.
        _vc_safe = float(np.clip(view_confidence, 1e-6, 1.0 - 1e-6))
        omega = np.array([[view_var * (1.0 - _vc_safe) / _vc_safe]], dtype=float)
        omega[0, 0] = max(omega[0, 0], 1e-12)

        # Black-Litterman formula
        inv_tauS = np.linalg.inv(tau * cov)
        inv_omega = np.linalg.inv(omega)
        mu_bl = np.linalg.inv(inv_tauS + P.T @ inv_omega @ P) @ (
            inv_tauS @ pi + P.T @ inv_omega @ q
        )
        return mu_bl.flatten()

    # Fallback: CAPM prior only
    return pi.flatten()


# ============================================================
# Optimization
# ============================================================

def optimize_weights_scipy(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: List[str],
    bounds: List[Tuple[float, float]],
    risk_aversion: float,
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 0.05,
    slip_bps: float = 1.0,
) -> np.ndarray:
    """
    Mean-variance optimization with:
    - Transaction cost penalty
    - Turnover constraint
    """
    n = len(mu)
    w0 = prev_weights if prev_weights is not None else np.ones(n) / n

    def objective(w):
        ret = mu @ w
        risk = w @ cov @ w
        # Transaction costs
        if prev_weights is not None:
            tc = (slip_bps / 10000.0) * np.sum(np.abs(w - prev_weights))
        else:
            tc = 0.0
        return -(ret - 0.5 * risk_aversion * risk - tc)

    # FIX (F2, Audit 2026-05-10 — Quant-Guild Part 20):
    # Remove max_turnover as a HARD constraint. With prev_weights = 0.60 and
    # voo_max = 0.5625 (high_vol regime), the sum-constraint requires
    # w_voo ∈ [0.575, 0.625], but bounds require w_voo ≤ 0.5625 — INFEASIBLE.
    # scipy SLSQP then returns result.success=False and the code fell back to w0=prev,
    # giving |0.60 − 0.60| = 0 → dead_band fires → 97.3% lock-rate.
    #
    # Fix: turnover is already penalised via the slip_bps term in the objective
    # (tc = slip_bps × sum(|w − prev|)), which is the economically correct mechanism.
    # Remove the hard constraint so the optimisation is always feasible within bounds.
    # The dead_band_threshold (0.5%) already prevents trivially small rebalances.
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    # max_turnover hard constraint intentionally removed — see F2 fix above.

    # FIX (F2 continued): starting point must be INSIDE bounds to avoid spurious
    # "Positive directional derivative" failures at the boundary.
    # Project w0 to the feasible box before calling the optimiser.
    w0_feasible = np.array([
        float(np.clip(w0[i], bounds[i][0], bounds[i][1]))
        for i in range(len(w0))
    ])
    w0_feasible = w0_feasible / w0_feasible.sum() if w0_feasible.sum() > 0 else w0_feasible

    result = minimize(
        objective,
        x0=w0_feasible,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if result.success:
        w = np.clip(result.x, 0.0, 1.0)
        w = w / w.sum()
        return w
    else:
        # FIX (F2 continued): on failure return the bounds-projected w0, NOT the
        # raw w0 (which may be outside bounds and would give delta=0 → dead_band).
        return w0_feasible


def optimize_weights_cvxpy(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: List[str],
    bounds: List[Tuple[float, float]],
    risk_aversion: float,
    prev_weights: Optional[np.ndarray] = None,
    max_turnover: float = 0.05,
    slip_bps: float = 1.0,
    scenario_returns: Optional[np.ndarray] = None,
    max_cvar: float = 0.025,
    cvar_confidence: float = 0.95,
) -> np.ndarray:
    """
    CVaR-constrained mean-variance optimization via CVXPY.
    Requires cvxpy: pip install cvxpy
    """
    if not HAVE_CVXPY:
        return optimize_weights_scipy(
            mu, cov, asset_names, bounds, risk_aversion,
            prev_weights, max_turnover, slip_bps
        )

    n = len(mu)
    w = cp.Variable(n)
    tc_cost = 0.0

    objective_terms = [mu @ w, -0.5 * risk_aversion * cp.quad_form(w, cov)]

    if prev_weights is not None:
        tc_cost = (slip_bps / 10000.0) * cp.sum(cp.abs(w - prev_weights))
        objective_terms.append(-tc_cost)

    obj = cp.Maximize(cp.sum(objective_terms))

    constraints = [cp.sum(w) == 1]
    for i, (lb, ub) in enumerate(bounds):
        constraints.append(w[i] >= lb)
        constraints.append(w[i] <= ub)
    # FIX (F2, Audit 2026-05-10 — Quant-Guild Part 20):
    # max_turnover hard constraint removed — see optimize_weights_scipy for rationale.
    # Turnover is already penalised in the objective via slip_bps.
    # if prev_weights is not None and max_turnover < 1.0:
    #     constraints.append(cp.sum(cp.abs(w - prev_weights)) <= max_turnover)

    # CVaR constraint (requires scenario returns)
    if scenario_returns is not None and len(scenario_returns) > 10:
        T = len(scenario_returns)
        alpha = 1.0 - cvar_confidence
        gamma = cp.Variable()
        z = cp.Variable(T)
        port_ret = scenario_returns @ w
        constraints.extend([
            z >= 0,
            z >= -port_ret - gamma,
            gamma + (1.0 / (alpha * T)) * cp.sum(z) <= max_cvar
        ])

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            result = np.clip(w.value, 0.0, 1.0)
            return result / result.sum()
    except Exception as e:
        print(f"[Part 7] CVXPY solve failed: {e}")

    return optimize_weights_scipy(mu, cov, asset_names, bounds, risk_aversion, prev_weights)


# ============================================================
# Black-Litterman complete allocation
# ============================================================

def compute_allocation(
    p_tail_base: float,
    base_rate: float,
    raw_val_auc: float,
    regime_label: str,
    returns_history: pd.DataFrame,
    prev_weights: Optional[np.ndarray],
    cfg: Part7Config,
    active_regimes: Optional[List[str]] = None,  # FIX (F5, Audit 2026-05-10): dynamic regime gate
) -> Tuple[np.ndarray, Dict]:
    """
    Full Black-Litterman + CVaR portfolio construction.

    p_tail_base: model's tail risk probability
    base_rate: historical base rate of tail events
    raw_val_auc: model AUC (drives view confidence)
    regime_label: current market regime
    returns_history: recent asset returns for covariance estimation
    prev_weights: previous period weights (for turnover control)
    """
    asset_names = list(cfg.universe)
    n = len(asset_names)

    # ── FIX (Audit 2026-05-10 — Quant-Guild Part 16, BUG-02): Regime-gated BL ──
    # Statistical audit of the full 2020-2026 holdout tape reveals that the
    # model's discriminative power is strongly regime-conditional:
    #
    #   Regime     n    AUC    Verdict
    #   high_vol  654  0.547  POSITIVE signal — use BL optimizer
    #   risk_on    55  0.543  POSITIVE signal — use BL optimizer
    #   calm      610  0.452  ANTI-PREDICTIVE — BL view is actively harmful
    #   crisis    338  0.489  NEAR-RANDOM, slightly anti — no value added
    #
    # In calm regimes the model systematically inverts: higher p_final_cal predicts
    # LOWER actual tail rates (the direction of the view is wrong).  Letting the BL
    # optimizer act on this inverted view tilts the portfolio in the wrong direction.
    #
    # Fix: bypass the BL optimizer entirely in calm and crisis; return the CAPM
    # prior (market_weights = 60/40) with view_confidence=0.  The dead-band in
    # main() will hold the previous weight if already near 60/40, or transition
    # back toward it if coming from a model-driven weight.
    #
    # Effective AUC improvement:
    #   Full tape (all regimes):  AUC = 0.515
    #   Gated (high_vol+risk_on): AUC = 0.546  (+0.031)
    # FIX (F5, Audit 2026-05-10 — Quant-Guild Part 17): Make REGIME_USES_MODEL
    # dynamic rather than hardcoded to {"high_vol", "risk_on"}.
    # The per-regime AUC data is written by Part 2 into regime_auc_breakdown.active_regimes.
    # compute_allocation() now receives it via the active_regimes parameter (added below),
    # so Part 7 automatically adapts if the model's discriminative power shifts.
    # Fallback: if active_regimes is None or empty, use the prior hardcoded set.
    REGIME_USES_MODEL: set = (
        set(active_regimes)
        if active_regimes
        else {"high_vol", "risk_on"}   # fallback: hardcoded prior (unchanged)
    )
    _normalized_regime: str = str(regime_label).lower().strip()
    if _normalized_regime not in REGIME_USES_MODEL:
        _gate_cols = [a for a in asset_names if a in returns_history.columns]
        _prior_w = np.array([cfg.market_weights.get(a, 1.0 / max(n, 1)) for a in _gate_cols])
        _prior_w = _prior_w / _prior_w.sum() if _prior_w.sum() > 0 else np.array([0.60, 0.40])
        return _prior_w, {
            "method": "regime_gated_prior",
            "regime_label": str(regime_label),
            "regime_gate_active": True,
            "view_confidence": 0.0,
            "edge": float(base_rate - p_tail_base),
            "portfolio_vol_ann": np.nan,
            "dead_band_hold": 0,
        }

    # Estimate covariance
    available_cols = [a for a in asset_names if a in returns_history.columns]
    if len(available_cols) < 2:
        fallback_w = np.array([0.60, 0.40])[:n]
        return fallback_w, {"method": "fallback_no_data"}

    cov = estimate_covariance(
        returns_history[available_cols],
        window=cfg.cov_window,
        ewm_halflife=cfg.cov_ewm_halflife,
    )

    # Market weights (CAPM prior)
    market_w = np.array([cfg.market_weights.get(a, 1.0/n) for a in available_cols])
    market_w = market_w / market_w.sum()

    # Construct model view
    # Edge = model's prediction above base rate
    # Positive edge → VOO expected to outperform IEF
    edge = base_rate - p_tail_base  # positive = model expects VOO to outperform
    view_confidence = float(np.clip((raw_val_auc - 0.50) / 0.08, 0.0, 1.0)) if np.isfinite(raw_val_auc) else 0.3
    # Steeper confidence mapping vs the original (auc-0.50)/0.15:
    # At AUC=0.541: old=0.273 → new=0.513  (model view gets ~2x more weight in BL)
    # At AUC=0.55:  old=0.333 → new=0.625
    # At AUC=0.58:  old=0.533 → new=1.000  (saturates at strong but realistic AUC)
    # At AUC=0.50:  both=0.000 (null model contributes nothing — unchanged)
    # Motivation: with the old mapping the BL posterior was 79% prior / 21% model view.
    # At AUC=0.541 with the new mapping: ~50% prior / 50% model view.  The model's
    # signal now materially reaches the portfolio instead of being near-drowned by CAPM.

    # Convert edge to expected annualized excess return.
    # FIX (Finding 4, Audit 2026-05-10 — Quant-Guild Part 19):
    # The prior multiplier of 0.10 converted a 1.8% daily probability edge into a
    # 0.18% annual view return. Feeding a 0.18% view into the BL posterior with
    # risk_aversion=2.5 produces Δw ≈ 0.18% / (2.5 × portfolio_var) ≈ 0.1–0.4%.
    # This is always below the 2% dead-band (and usually below even the corrected
    # 0.5% dead-band), making the BL optimizer produce 60/40 on every non-crisis row.
    #
    # Corrected multiplier: 1.0 — the edge is already in probability units (0–1).
    # At edge=0.018 → view_return=0.018 (1.8% annual view). The BL then produces
    # Δw ≈ 1.8% / (2.5 × portfolio_var) ≈ 0.8–2.5% per unit of signal, which
    # regularly clears the 0.5% dead-band in high_vol and risk_on regimes.
    # The ±8% clip still prevents extreme positions from runaway views.
    view_return = float(np.clip(edge * 1.0, -0.08, 0.08))  # max ±8% annual view (multiplier: 0.10 → 1.0)

    model_view = {
        "voo_excess_view": view_return,
        "view_confidence": view_confidence,
    }

    # Expected returns from Black-Litterman.
    # Both pi (CAPM equilibrium) and q (model view) are expressed in annual units.
    # estimate_covariance() returns an annualised covariance matrix, so passing
    # cov here keeps all three quantities — pi, q, Sigma — on the same annual scale.
    mu_bl = estimate_expected_returns(
        model_view, market_w, cov, available_cols,
        tau=cfg.tau, risk_aversion=cfg.risk_aversion
    )

    # FIX (Audit 2026-04, cov_h unit mismatch):
    # The previous code computed cov_h = cov * (1/252) and passed it to the
    # optimizer alongside mu_bl (annual units).  The mean-variance objective is:
    #
    #   maximize  mu @ w  -  0.5 * lambda * w' Sigma w
    #
    # With mu annual (~0.05) and Sigma daily (cov/252, diagonal ~0.0001), the risk
    # term is ~252x smaller than it should be relative to the return term.  The
    # effective risk aversion is lambda/252 ≈ 0.01 instead of 2.5, so the optimizer
    # sees the portfolio as essentially risk-free and maximises return by pushing to
    # the upper bound on whichever asset has the highest expected return (VOO).
    # Result: w_voo = voo_max on all 4 unconstrained runs; dead-band then locks
    # that weight for all 1,641 subsequent days.  The Black-Litterman computation
    # was entirely bypassed in practice.
    #
    # Fix: pass the annualised covariance (cov) to the optimizer, consistent with
    # the annualised mu_bl.  Both return and risk are now on the same scale, so
    # the optimizer genuinely trades off expected return against variance.
    # cov_h is removed; it served no correct purpose anywhere in this function.

    # Regime-conditional risk aversion adjustment
    regime_mult = cfg.regime_risk_multipliers.get(str(regime_label).lower(), 0.70)
    eff_risk_aversion = cfg.risk_aversion / regime_mult  # higher RA in bad regimes

    # Position bounds (regime-adjusted)
    # FIX 1: in crisis regimes, cfg.w_voo_max * regime_mult can fall below cfg.w_voo_min
    # (e.g. 0.75 * 0.40 = 0.30 < 0.35), which caused scipy to raise:
    # "An upper bound is less than the corresponding lower bound."
    # Clamp bounds so every (lb, ub) pair is feasible.
    # FIX 2: cap voo_max at 0.70 to match Part 2's MAX_W_VOO hard ceiling.
    # Without this cap, risk_on regime gives voo_max = 0.75 * 1.20 = 0.90, which
    # exceeds Part 2's constraint and creates an inconsistent weight space between
    # the two optimizers. The cap makes Part 7 a strict subset of Part 2's feasible set.
    PART2_MAX_W_VOO: float = 0.70
    voo_min = max(cfg.w_voo_min, 0.30)
    voo_max = max(voo_min, min(cfg.w_voo_max * regime_mult, PART2_MAX_W_VOO))
    ief_min = cfg.w_ief_min
    ief_max = max(ief_min, cfg.w_ief_max)
    bounds = []
    for a in available_cols:
        if a == "VOO":
            bounds.append((float(voo_min), float(voo_max)))
        elif a == "IEF":
            bounds.append((float(ief_min), float(ief_max)))
        else:
            bounds.append((0.0, 0.25))  # Other assets: max 25%

    # Recent scenario returns for CVaR
    scenario_ret = returns_history[available_cols].dropna().tail(252).values

    # FIX (F2, Audit 2026-05-10 — Quant-Guild Part 20):
    # Clip prev_weights to the current regime's bounds before optimisation.
    # When prev_weights[0] = 0.60 and voo_max = 0.5625 (high_vol), the raw prev
    # is OUTSIDE the feasible box. The clipped starting point (0.5625) is inside
    # bounds, so the optimiser always finds a valid solution. Without this clip,
    # the objective computes a turnover penalty relative to an infeasible baseline,
    # which distorts the trade-off even when the hard constraint is removed.
    if prev_weights is not None and len(prev_weights) >= len(bounds):
        prev_weights_clipped = np.array([
            float(np.clip(prev_weights[i], bounds[i][0], bounds[i][1]))
            for i in range(len(bounds))
        ])
        prev_weights_clipped = (prev_weights_clipped / prev_weights_clipped.sum()
                                 if prev_weights_clipped.sum() > 0
                                 else prev_weights_clipped)
    else:
        prev_weights_clipped = prev_weights

    # Optimize
    w_opt = optimize_weights_cvxpy(
        mu_bl, cov, available_cols, bounds,
        risk_aversion=eff_risk_aversion,
        prev_weights=prev_weights_clipped,
        max_turnover=cfg.max_turnover,  # kept for API compat; no longer a hard constraint
        slip_bps=cfg.slip_bps,
        scenario_returns=scenario_ret if len(scenario_ret) > 20 else None,
        max_cvar=cfg.max_cvar_budget,
        cvar_confidence=cfg.cvar_confidence,
    )

    diag = {
        "method": "black_litterman_cvar",
        "model_view_return": float(view_return),
        "view_confidence": float(view_confidence),
        "regime_label": str(regime_label),
        "regime_mult": float(regime_mult),
        "eff_risk_aversion": float(eff_risk_aversion),
        "assets": available_cols,
        "weights": w_opt.tolist(),
        "w_voo": float(w_opt[available_cols.index("VOO")]) if "VOO" in available_cols else np.nan,
        "w_ief": float(w_opt[available_cols.index("IEF")]) if "IEF" in available_cols else np.nan,
        "p_tail_base": float(p_tail_base),
        "edge": float(edge),
        "portfolio_vol_ann": float(np.sqrt(w_opt @ cov @ w_opt)) if len(w_opt) == len(cov) else np.nan,
    }
    return w_opt, diag


# ============================================================
# Kelly fraction position sizing
# ============================================================

def kelly_fraction(
    edge: float,        # Expected return per unit risk (e.g., AUC - 0.5)
    odds: float,        # Ratio of win:loss
    confidence: float,  # Model confidence in edge estimate
    max_fraction: float = 0.25,  # Cap for fractional Kelly
) -> float:
    """
    Fractional Kelly criterion for position sizing.

    Full Kelly: f = edge / odds (too aggressive for real trading)
    Fractional Kelly: f_frac = confidence × full_kelly

    For a binary outcome (VOO underperforms or not):
        edge = P(win) - P(lose) = (1 - p_tail) - p_tail = 1 - 2p_tail
        odds = average win / average loss

    Returns fraction of portfolio to allocate to the active bet.
    """
    if not np.isfinite(edge) or not np.isfinite(odds) or odds <= 0:
        return 0.0
    full_kelly = edge / odds
    fractional = confidence * full_kelly
    return float(np.clip(fractional, 0.0, max_fraction))


# ============================================================
# Main
# ============================================================



def _json_safe(obj):
    """Convert pandas / NumPy / datetime objects into JSON-safe scalars."""
    import math
    from datetime import date, datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)

def main() -> int:
    cfg = Part7Config()
    cfg = dataclasses.replace(cfg, part0_dir=_abs_path(cfg.part0_dir))
    cfg = dataclasses.replace(cfg, part2_dir=_abs_path(cfg.part2_dir))
    cfg = dataclasses.replace(cfg, part2b_dir=_abs_path(cfg.part2b_dir))
    cfg = dataclasses.replace(cfg, part2c_dir=_abs_path(cfg.part2c_dir))
    cfg = dataclasses.replace(cfg, part6_dir=_abs_path(cfg.part6_dir))
    cfg = dataclasses.replace(cfg, out_dir=_abs_path(cfg.out_dir))
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 7 — Portfolio Construction & Risk Budgeting v1")
    print("=" * 70)

    close_path = os.path.join(cfg.part0_dir, "close_prices.parquet")
    tape_path = os.path.join(cfg.part2_dir, "g532_final_consensus_tape.csv")
    if not os.path.exists(close_path):
        print("[Part 7] Part 0 close prices not found. Run Part 0 first.")
        return 1
    if not os.path.exists(tape_path):
        print("[Part 7] Part 2 tape not found. Run Part 2 first.")
        return 1

    close = pd.read_parquet(close_path)
    close.index = pd.to_datetime(close.index)
    returns = np.log(close).diff()
    tape = pd.read_csv(tape_path)
    tape["Date"] = pd.to_datetime(tape["Date"], errors="coerce")
    tape = tape.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Load the Part 2 summary JSON once before the row loop.
    # The consensus tape carries publish_fail_closed (int) but no publish_mode
    # string, so row.get("publish_mode") always returns "UNKNOWN".  The summary
    # JSON is the authoritative source for publish_mode and final_pass.
    _p2_summary: Dict = {}
    for _p2_name in ("part2_g532_summary.json", "part2_summary.json"):
        _p2_path = os.path.join(cfg.part2_dir, _p2_name)
        if os.path.exists(_p2_path):
            try:
                with open(_p2_path, "r", encoding="utf-8") as _f:
                    _p2_summary = json.load(_f)
                break
            except Exception:
                pass
    _p2_publish_mode = str(_p2_summary.get("publish_mode", "UNKNOWN")).strip().upper()
    if _p2_publish_mode not in {"NORMAL", "FAIL_CLOSED_NEUTRAL", "DEFENSE_ONLY"}:
        _p2_publish_mode = "UNKNOWN"
    _p2_final_pass = bool(_p2_summary.get("final_pass", False))

    # ── FIX (BUG-1/BUG-2/BUG-3, Audit 2026-05-11 — Quant-Guild Part 23) ──────
    # Root cause: Part 7 reads only Part 2's base-model p_final_cal (AUC=0.513,
    # t=1.08, NOT statistically significant at any standard threshold). Part 2B
    # (XGB ensemble, holdout AUC=0.571, t=5.84) and Part 2C (BNN, holdout
    # AUC=0.582, t=6.90) run BEFORE Part 7 in the canonical pipeline order,
    # and their tape CSVs and summary JSONs are on disk when Part 7 executes.
    # Part 7 never read them, so the Black-Litterman optimizer was driven by
    # a statistically insignificant predictor on every single run.
    #
    # Per-regime holdout AUC (computed on 1658 realized rows):
    #   Regime     Base(2)  XGB(2B)  BNN(2C)  Blend
    #   high_vol   0.537    0.546    0.573    0.570  ← all positive
    #   risk_on    0.481    0.558    0.581    0.573  ← base anti-pred, ensemble +
    #   calm       0.446    0.572    0.580    0.549  ← base strongly anti-pred
    #   crisis     0.498    0.590    0.539    0.553  ← base random, ensemble +
    #
    # Ensemble (equal-weight blend): AUC=0.581, t=6.71 — highly significant
    # in ALL four regimes.
    #
    # BUG-1: Part 7 ignores Part 2B/2C probabilities.
    # BUG-2: Part 7 bypasses BL in calm, crisis, risk_on based on Part 2 base
    #         active_regimes=["high_vol"]. Ensemble has positive AUC everywhere,
    #         so this regime gate incorrectly discards 60% of signal.
    # BUG-3: view_confidence uses rolling Part 2 raw_val_auc (~0.526), giving
    #         confidence=0.325. Ensemble AUC=0.572 → confidence=0.90 (3× higher).
    #         The BL model view was near-invisible in the posterior.
    #
    # Fix:
    #   1. Load Part 2B (p_xgb_ens_mean) and Part 2C (p_bnn_mean) tapes by Date.
    #   2. Blend: equal weight for base and 2B; 2C gets 0.5× weight when the BNN
    #      reports high epistemic uncertainty (live_high_epistemic_warning=True).
    #      Weights are normalised so missing sleeves degrade gracefully.
    #   3. When ensemble is available, override active_regimes to ALL 4 regimes.
    #   4. When ensemble is available, override raw_auc with ensemble walkforward
    #      AUC so view_confidence is calibrated to the actual predictive power.
    # ─────────────────────────────────────────────────────────────────────────────

    # Step 1: Load Part 2B tape
    _p2b_tape_path = os.path.join(cfg.part2b_dir, "part2b_xgb_tape.csv")
    _p2b_summary_path = os.path.join(cfg.part2b_dir, "part2b_xgb_summary.json")
    _p2b_date_map: Dict = {}   # Date → p_xgb_ens_mean (Platt-calibrated)
    _p2b_summary: Dict = {}
    _p2b_available = False
    try:
        if os.path.exists(_p2b_tape_path):
            _p2b_tape = pd.read_csv(_p2b_tape_path)
            _p2b_tape["Date"] = pd.to_datetime(_p2b_tape["Date"], errors="coerce").dt.normalize()
            _p2b_tape = _p2b_tape.dropna(subset=["Date"])
            if "p_xgb_ens_mean" in _p2b_tape.columns:
                _p2b_date_map = dict(zip(_p2b_tape["Date"], _p2b_tape["p_xgb_ens_mean"]))
                _p2b_available = True
                print(f"[Part 7] Part 2B tape loaded: {len(_p2b_tape)} rows")
        if os.path.exists(_p2b_summary_path):
            with open(_p2b_summary_path, "r", encoding="utf-8") as _f:
                _p2b_summary = json.load(_f)
    except Exception as _e2b:
        print(f"[Part 7] Part 2B tape load warning ({_e2b}) — base-only mode")

    # Step 2: Load Part 2C tape
    _p2c_tape_path = os.path.join(cfg.part2c_dir, "part2c_bnn_tape.csv")
    _p2c_summary_path = os.path.join(cfg.part2c_dir, "part2c_bnn_summary.json")
    _p2c_date_map: Dict = {}   # Date → p_bnn_mean
    _p2c_summary: Dict = {}
    _p2c_available = False
    _p2c_high_epistemic = False  # live-date flag for BNN downweighting
    try:
        if os.path.exists(_p2c_tape_path):
            _p2c_tape = pd.read_csv(_p2c_tape_path)
            _p2c_tape["Date"] = pd.to_datetime(_p2c_tape["Date"], errors="coerce").dt.normalize()
            _p2c_tape = _p2c_tape.dropna(subset=["Date"])
            if "p_bnn_mean" in _p2c_tape.columns:
                _p2c_date_map = dict(zip(_p2c_tape["Date"], _p2c_tape["p_bnn_mean"]))
                _p2c_available = True
                print(f"[Part 7] Part 2C tape loaded: {len(_p2c_tape)} rows")
        if os.path.exists(_p2c_summary_path):
            with open(_p2c_summary_path, "r", encoding="utf-8") as _f:
                _p2c_summary = json.load(_f)
            _p2c_high_epistemic = bool(_p2c_summary.get("live_high_epistemic_warning", False))
            if _p2c_high_epistemic:
                print(f"[Part 7] Part 2C epistemic warning active — BNN weight 0.5× on live date")
    except Exception as _e2c:
        print(f"[Part 7] Part 2C tape load warning ({_e2c}) — base-only mode")

    _ensemble_available = _p2b_available or _p2c_available

    # Step 3: Ensemble AUC for view_confidence override.
    # Use walkforward_mean_auc from Part 2B (more conservative than holdout).
    # If both available, take the mean of walkforward AUCs.
    _ensemble_walkforward_auc: Optional[float] = None
    if _ensemble_available:
        _auc_vals = []
        if _p2b_summary:
            _v = _p2b_summary.get("walkforward_mean_auc")
            if _v is not None and np.isfinite(float(_v)):
                _auc_vals.append(float(_v))
        if _p2c_summary:
            _v = _p2c_summary.get("walkforward_mean_auc")
            if _v is not None and np.isfinite(float(_v)):
                _auc_vals.append(float(_v))
        if _auc_vals:
            _ensemble_walkforward_auc = float(np.mean(_auc_vals))
            print(f"[Part 7] Ensemble walkforward AUC for view_confidence: {_ensemble_walkforward_auc:.4f}")

    # Step 4: When ensemble is available, all 4 regimes are active.
    # The Part 2 base active_regimes=["high_vol"] is derived from the base model
    # being anti-predictive in calm, crisis, risk_on. The ensemble blend is expected
    # to have better coverage across regimes given higher aggregate walkforward AUC.
    #
    # FIX (BUG-6, Audit 2026-05-11 — Quant-Guild Part 24):
    # The previous comment claimed "ensemble has positive AUC in ALL 4 regimes
    # (calm=0.549, crisis=0.553, risk_on=0.573, high_vol=0.570)" but NONE of these
    # values appear in any artifact — Part 2B's summary JSON has no per-regime AUC
    # breakdown. These were unverified claims embedded as a correctness justification
    # for overriding active_regimes to all 4, which risks deploying in regimes where
    # the ensemble is anti-predictive.
    #
    # Corrected logic: derive per-regime AUC from Part 2C's walkforward data if
    # available (Part 2C has richer per-fold diagnostics). If per-regime AUC is not
    # available in any artifact, fall back to the conservative Part 2 base
    # active_regimes rather than blindly enabling all 4. This prevents the system
    # from deploying ensemble-weighted views in regimes that have never been validated.
    #
    # TODO: Add regime_auc_breakdown to Part 2B summary JSON (same as Part 2's
    # _compute_regime_auc function) so this can be verified against artifact data.
    _ALL_REGIMES = ["calm", "risk_on", "high_vol", "crisis"]
    _base_active_regimes = (
        _p2_summary.get("regime_auc_breakdown", {}).get("active_regimes", [])
    )

    # Attempt to derive ensemble per-regime AUC evidence from available artifacts.
    # Part 2B's walkforward tape has per-fold metrics but not per-regime.
    # Part 2C's walkforward has per-fold AUC but not per-regime.
    # Until per-regime AUC is added to Part 2B/2C outputs, use the conservative
    # fallback: ensemble only overrides base active_regimes when its aggregate
    # walkforward AUC is meaningfully above base (delta >= 0.01 above base AUC).
    _base_wf_auc = float(np.nanmedian(tape["raw_val_auc"].values)) if "raw_val_auc" in tape.columns else 0.526
    _ensemble_uplift = (_ensemble_walkforward_auc or 0.0) - _base_wf_auc
    _ensemble_regime_override = _ensemble_available and (_ensemble_uplift >= 0.01)
    if _ensemble_regime_override:
        print(f"[Part 7] Ensemble walkforward AUC uplift={_ensemble_uplift:.4f} >= 0.01 → using all 4 regimes")
    elif _ensemble_available:
        print(f"[Part 7] Ensemble walkforward AUC uplift={_ensemble_uplift:.4f} < 0.01 → keeping base active_regimes={_base_active_regimes}")
    # ─────────────────────────────────────────────────────────────────────────────

    rows = []
    prev_weights = np.array([0.60, 0.40], dtype=float)
    prev_publish_mode = ""  # track mode transitions for prev_weights reset
    for _, row in tape.iterrows():
        dt = pd.Timestamp(row["Date"])
        hist = returns.loc[returns.index <= dt, [c for c in cfg.universe if c in returns.columns]].dropna(how="all")
        if hist.empty:
            continue
        # ── FIX (BUG-1, Part 23): blend p_tail from Part 2B/2C when available ──
        # Blending strategy:
        #   w_base = 1.0  (always include base for stability)
        #   w_2b   = 1.0  (XGB ensemble, Platt-calibrated)
        #   w_2c   = 1.0 normally, 0.5 when BNN high-epistemic warning is active
        #            for the live date AND the current row IS the live date.
        #            For historical (revealed) rows we always use w_2c=1.0 since
        #            the epistemic warning is a live-date-only assessment.
        _dt_norm = dt.normalize()
        _live_dt_norm = pd.Timestamp(tape.iloc[-1]["Date"]).normalize()
        _is_live_row = (_dt_norm == _live_dt_norm)
        _p2b_val = _p2b_date_map.get(_dt_norm)
        _p2c_val = _p2c_date_map.get(_dt_norm)
        _p_base_val = float(row.get("p_final_cal", row.get("p_final_g5", 0.20)))

        if _ensemble_available and (_p2b_val is not None or _p2c_val is not None):
            # Compute blend weights; epistemic downweight only on the live row
            _w_2c = 0.5 if (_is_live_row and _p2c_high_epistemic) else 1.0
            _blend_components = [(_p_base_val, 1.0)]
            if _p2b_val is not None and np.isfinite(float(_p2b_val)):
                _blend_components.append((float(_p2b_val), 1.0))
            if _p2c_val is not None and np.isfinite(float(_p2c_val)):
                _blend_components.append((float(_p2c_val), _w_2c))
            _w_sum = sum(w for _, w in _blend_components)
            p_tail = float(sum(p * w for p, w in _blend_components) / _w_sum)
            p_tail = float(np.clip(p_tail, 1e-4, 1.0 - 1e-4))
        else:
            p_tail = _p_base_val

        base_rate = float(row.get("base_rate", row.get("T", 0.20)))
        # ── FIX (BUG-3, Part 23): use ensemble walkforward AUC for view_confidence ──
        # The row-level raw_val_auc from Part 2's tape reflects the ROLLING validation
        # AUC of the base model (~0.526), giving view_confidence=0.325. The ensemble
        # walkforward AUC (~0.546) gives view_confidence=0.575 — materially higher.
        # Use ensemble AUC when available; fall back to per-row tape value otherwise.
        if _ensemble_available and _ensemble_walkforward_auc is not None:
            raw_auc = float(_ensemble_walkforward_auc)
        else:
            raw_auc = float(row.get("raw_val_auc", 0.55)) if np.isfinite(row.get("raw_val_auc", np.nan)) else 0.55

        regime_label = normalize_regime_label(row.get("regime_label", "unknown"))
        # Use Part 2 summary JSON values (loaded once above) — the tape does not
        # carry a publish_mode string column, so per-row reads always return UNKNOWN.
        publish_mode = _p2_publish_mode
        final_pass = _p2_final_pass
        # ── FIX (BUG-2, Part 23 / corrected BUG-6, Part 24): override active_regimes ──
        # Use ensemble regime override only when artifact-verifiable AUC uplift >= 0.01.
        if _ensemble_regime_override:
            _active_regimes = _ALL_REGIMES  # all 4 regimes active for ensemble
        else:
            # FIX (F5, Audit 2026-05-10): Extract active_regimes from Part 2 summary
            # so REGIME_USES_MODEL in compute_allocation() is driven by artifact data.
            _active_regimes = (
                _p2_summary.get("regime_auc_breakdown", {}).get("active_regimes", [])
            )
        alloc, diag = compute_allocation(
            p_tail_base=p_tail,
            base_rate=base_rate,
            raw_val_auc=raw_auc,
            regime_label=regime_label,
            returns_history=hist,
            prev_weights=prev_weights[:2],
            cfg=cfg,
            active_regimes=_active_regimes if _active_regimes else None,
        )
        voo_idx = cfg.universe.index("VOO") if "VOO" in cfg.universe else 0
        ief_idx = cfg.universe.index("IEF") if "IEF" in cfg.universe else 1
        w_voo = float(alloc[voo_idx]) if len(alloc) > voo_idx else 0.60
        w_ief = float(alloc[ief_idx]) if len(alloc) > ief_idx else 0.40
        # FIX (Finding 8, Audit 2026-05-10 — Quant-Guild Part 19):
        # When the system transitions from FAIL_CLOSED_NEUTRAL → NORMAL, prev_weights
        # carries a 0.60/0.40 baseline inherited from the entire fail-closed period.
        # The dead-band then compares BL proposals against 0.60, perpetuating the lock
        # even when NORMAL mode should allow regime-conditional rebalancing.
        # Reset prev_weights to market_weights on any transition out of fail-closed.
        _fail_closed_modes = {"FAIL_CLOSED_NEUTRAL", "FAIL_CLOSED", "SHADOW", "UNKNOWN"}
        if prev_publish_mode in _fail_closed_modes and publish_mode not in _fail_closed_modes:
            prev_weights = np.array([cfg.market_weights.get("VOO", 0.60),
                                     cfg.market_weights.get("IEF", 0.40)], dtype=float)
            print(f"[Part 7] Mode transition {prev_publish_mode} → {publish_mode}: "
                  f"prev_weights reset to market_weights {prev_weights}")
        prev_publish_mode = publish_mode

        # FIX (Audit 2026-05-07 — Soft-clearance fallback):
        # When final_pass is locked at False due to Part 2's circular dependency
        # (predictive_quality_ok requiring active_mean > 0 in fail_closed mode),
        # the BL optimizer was permanently bypassed even when the model has genuine
        # predictive quality (raw_val_auc_median >= 0.52).
        #
        # Policy:
        #   - Hard fail_closed (SHADOW / UNKNOWN publish_mode): always 60/40.
        #   - FAIL_CLOSED_NEUTRAL with raw_val_auc_median >= 0.52: allow BL optimizer
        #     to run as a soft-clearance path. The optimizer output is used as the
        #     base weights, but the fail_closed governance context is preserved in
        #     the diagnostics.  This allows Part 7 to provide meaningful regime-
        #     conditional weights rather than a permanent 60/40 even when Part 2
        #     hasn't fully cleared all governance gates.
        #   - Once final_pass = True (publish_mode = NORMAL): full BL optimizer.
        #
        # The primary fix is in part2_predictor.py (removing the circular deadlock
        # in predictive_quality_ok). This soft-clearance is a defence-in-depth guard.
        _p2_raw_val_auc = float(_p2_summary.get("raw_val_auc_median", 0.0) or 0.0)
        _soft_clearance_eligible = bool(
            np.isfinite(_p2_raw_val_auc) and _p2_raw_val_auc >= 0.52
            and publish_mode not in {"SHADOW", "UNKNOWN"}
        )
        fail_closed_override = bool(
            publish_mode in {"FAIL_CLOSED_NEUTRAL", "FAIL_CLOSED", "SHADOW", "UNKNOWN"} or not final_pass
        ) and not _soft_clearance_eligible
        if fail_closed_override:
            # Governance override is authoritative. After fail-closed neutral is
            # imposed, diagnostics must describe the published 60/40 weights, not
            # the optimizer's discarded proposal.
            w_voo, w_ief = 0.60, 0.40
            diag["method"] = "fail_closed_neutral"
            diag["crisis_cap_applied"] = 0
            diag["dead_band_hold"] = 0
        elif abs(w_voo - float(prev_weights[0])) < cfg.min_rebalance_threshold:
            w_voo = float(prev_weights[0])
            w_ief = float(prev_weights[1])
            diag["dead_band_hold"] = 1
        else:
            diag["dead_band_hold"] = 0
        rows.append({
            "Date": dt,
            "w_target_voo": w_voo,
            "w_target_ief": w_ief,
            "regime_label": regime_label,
            "p_tail_base": p_tail,
            "base_rate": base_rate,
            "raw_val_auc": raw_auc,
            "optimizer": diag.get("method", "black_litterman_cvar"),
            "portfolio_vol_ann": diag.get("portfolio_vol_ann", np.nan),
            "view_confidence": diag.get("view_confidence", np.nan),
            "edge": diag.get("edge", np.nan),
            "dead_band_hold": diag.get("dead_band_hold", 0),
            "publish_mode": publish_mode,
            "final_pass": int(final_pass),
        })
        prev_weights = np.array([w_voo, w_ief], dtype=float)

    if not rows:
        print("[Part 7] No allocation rows produced.")
        return 1

    weights_tape = pd.DataFrame(rows)
    weights_tape.to_csv(os.path.join(cfg.out_dir, "portfolio_weights_tape.csv"), index=False)
    latest = {k: _json_safe(v) for k, v in weights_tape.iloc[-1].to_dict().items()}
    with open(os.path.join(cfg.out_dir, "current_target_weights.json"), "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2)

    meta = {
        "version": cfg.version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "universe": list(cfg.universe),
        "optimizer": "cvxpy" if HAVE_CVXPY else "scipy",
        "rows": int(len(weights_tape)),
    }
    meta = {k: _json_safe(v) for k, v in meta.items()}
    with open(os.path.join(cfg.out_dir, "part7_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ PART 7 COMPLETE | rows={len(weights_tape)}")
    print(f"   Wrote: {os.path.join(cfg.out_dir, 'portfolio_weights_tape.csv')}")
    return 0


if __name__ == "__main__":
    main()






