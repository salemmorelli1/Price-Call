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
    max_turnover: float = 0.05         # 15% max single-trade position change

    # Transaction costs
    slip_bps: float = 1.0              # One-way slippage per unit traded
    commission_bps: float = 1.0        # Broker commission

    # === Black-Litterman Parameters ===
    tau: float = 0.05                  # Uncertainty in CAPM prior (typical: 0.02–0.10)
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
    min_rebalance_threshold: float = 0.02  # 2% dead-band for daily rebalances


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
    s = str(label).strip().lower() if label is not None else "unknown"
    mapping = {
        "calm": "risk_on",
        "risk_on": "risk_on",
        "high_vol": "high_vol",
        "crisis": "crisis",
        "dislocated": "crisis",
        "unknown": "unknown",
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
        omega = np.array([[view_var * (1.0 / max(view_confidence, 0.10))]], dtype=float)

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

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if prev_weights is not None and max_turnover < 1.0:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_turnover - np.sum(np.abs(w - prev_weights))
        })

    result = minimize(
        objective,
        x0=w0,
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
        # Fallback to minimum variance
        return w0


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
    if prev_weights is not None and max_turnover < 1.0:
        constraints.append(cp.sum(cp.abs(w - prev_weights)) <= max_turnover)

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

    # Scale cov to H-day horizon
    cov_h = cov * (cfg.horizon / 252)

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
    # FIX: removed ann_factor = 252/horizon.
    # At H=1, ann_factor=252 caused view_return to always saturate the ±0.08 clip,
    # making edge magnitude carry zero information into the BL posterior.
    # Direct formulation: 10% annualized return per unit of model edge.
    # This is horizon-invariant and preserves edge signal at all frequencies.
    view_return = float(np.clip(edge * 0.10, -0.08, 0.08))  # max ±8% annual view

    model_view = {
        "voo_excess_view": view_return,
        "view_confidence": view_confidence,
    }

    # Expected returns from Black-Litterman
    mu_bl = estimate_expected_returns(
        model_view, market_w, cov_h, available_cols,
        tau=cfg.tau, risk_aversion=cfg.risk_aversion
    )

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

    # Optimize
    w_opt = optimize_weights_cvxpy(
        mu_bl, cov_h, available_cols, bounds,
        risk_aversion=eff_risk_aversion,
        prev_weights=prev_weights,
        max_turnover=cfg.max_turnover,
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


def main() -> int:
    cfg = Part7Config()
    cfg = dataclasses.replace(cfg, part0_dir=_abs_path(cfg.part0_dir))
    cfg = dataclasses.replace(cfg, part2_dir=_abs_path(cfg.part2_dir))
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

    rows = []
    prev_weights = np.array([0.60, 0.40], dtype=float)
    for _, row in tape.iterrows():
        dt = pd.Timestamp(row["Date"])
        hist = returns.loc[returns.index <= dt, [c for c in cfg.universe if c in returns.columns]].dropna(how="all")
        if hist.empty:
            continue
        p_tail = float(row.get("p_final_cal", row.get("p_final_g5", 0.20)))
        base_rate = float(row.get("base_rate", row.get("T", 0.20)))
        raw_auc = float(row.get("raw_val_auc", 0.55)) if np.isfinite(row.get("raw_val_auc", np.nan)) else 0.55
        regime_label = normalize_regime_label(row.get("regime_label", "unknown"))
        # Use Part 2 summary JSON values (loaded once above) — the tape does not
        # carry a publish_mode string column, so per-row reads always return UNKNOWN.
        publish_mode = _p2_publish_mode
        final_pass = _p2_final_pass
        alloc, diag = compute_allocation(
            p_tail_base=p_tail,
            base_rate=base_rate,
            raw_val_auc=raw_auc,
            regime_label=regime_label,
            returns_history=hist,
            prev_weights=prev_weights[:2],
            cfg=cfg,
        )
        voo_idx = cfg.universe.index("VOO") if "VOO" in cfg.universe else 0
        ief_idx = cfg.universe.index("IEF") if "IEF" in cfg.universe else 1
        w_voo = float(alloc[voo_idx]) if len(alloc) > voo_idx else 0.60
        w_ief = float(alloc[ief_idx]) if len(alloc) > ief_idx else 0.40
        if publish_mode in {"FAIL_CLOSED_NEUTRAL", "FAIL_CLOSED", "SHADOW", "UNKNOWN"} or not final_pass:
            w_voo, w_ief = 0.60, 0.40
            diag["method"] = "fail_closed_neutral"
        if abs(w_voo - float(prev_weights[0])) < cfg.min_rebalance_threshold:
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
    latest = weights_tape.iloc[-1].to_dict()
    with open(os.path.join(cfg.out_dir, "current_target_weights.json"), "w") as f:
        json.dump(latest, f, indent=2, default=str)

    meta = {
        "version": cfg.version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "universe": list(cfg.universe),
        "optimizer": "cvxpy" if HAVE_CVXPY else "scipy",
        "rows": int(len(weights_tape)),
    }
    with open(os.path.join(cfg.out_dir, "part7_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ PART 7 COMPLETE | rows={len(weights_tape)}")
    print(f"   Wrote: {os.path.join(cfg.out_dir, 'portfolio_weights_tape.csv')}")
    return 0


if __name__ == "__main__":
    main()


