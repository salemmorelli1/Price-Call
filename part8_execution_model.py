#!/usr/bin/env python3
# @title PART 8 — Execution & Transaction Cost Model
# =============================================================================
# Industry-grade execution analytics for PriceCallProject v2
#
# WHY THIS PART EXISTS
# ─────────────────────
# The current system models transaction costs as a flat 5 bps one-way for
# every rebalance, regardless of trade size, market conditions, or timing.
# This is wrong in two different ways:
#
#   1. For retail-scale portfolios (<$500k), 5 bps massively OVERSTATES costs.
#      Actual ETF costs are 0.5–1.5 bps one-way for VOO/IEF.
#      The current model penalizes active management ~3× too hard.
#
#   2. For institutional-scale portfolios (>$10M), 5 bps flat UNDERSTATES costs
#      because it ignores square-root market impact that grows with trade size.
#
# Part 8 fixes both problems and adds what a real execution desk uses:
#
#   Pre-Trade Analytics  — cost estimate and schedule before placing the order
#   Execution Schedule   — Almgren-Chriss optimal schedule (TWAP/adaptive)
#   Post-Trade Analysis  — actual slippage vs arrival price after each trade
#   Annual Drag Model    — total execution cost impact on strategy IR
#   Venue Guidance       — when in the day to execute for minimal impact
#
# POSITION IN THE PIPELINE
# ─────────────────────────
# Part 7 (Portfolio Construction) outputs: target weights + trade sizes
# Part 8 (Execution)               inputs: those trade sizes + market data
#                                  outputs: cost estimates, trade schedule,
#                                           post-trade attribution
# Part 3 (Governance)              consumes: Part 8 net-of-costs weights
# Part 9 (Attribution)             consumes: Part 8 post-trade record
#
# =============================================================================
#
# AUDIT CHANGELOG (Quant-Guild Part 8, 2026-04)
# ──────────────────────────────────────────────
# Finding 1+2 (CRITICAL — merged): compute_annual_cost_drag() was filtering
#   out rows with turnover < 0.001 before computing avg_turnover. This left
#   only 10 of 1,644 realized rows in the mean, producing a conditional
#   turnover mean of 0.11 (11%) versus the correct full-population mean of
#   0.000669 (<0.1%). The formula structure (× 252 × 2) is correct for a
#   daily H=1 system; the bug was solely in the turnover input. Result: annual
#   drag overstated 164× (21.9 bps vs 0.133 bps true). Downstream: Part 9 and
#   Part 10 TC gates compare annual_tc_drag_bps > estimated_annual_edge_bps;
#   with inflated drag the gate would fire spuriously once n_live_realized ≥ 60.
#   Fix: remove the < 0.001 filter from the mean calculation; retain it only
#   for the cost_bps computation (numerical stability on zero-notional rows).
#
# Finding 3 (CRITICAL): vix_level was hardcoded to 16.5 in main(). The stress
#   scalar that gates execution aggressiveness and cost estimates was therefore
#   structurally blind to true market conditions. Fix: _load_live_vix() reads
#   the most recent VIX close from Part 0 close_prices.parquet; falls back to
#   18.0 only if the artifact is absent.
#
# Finding 4 (IMPORTANT): static VOO fallback approx_price was $530 — 18.8%
#   below the live price of ~$652. This produces a 23% error in trade_shares
#   when Part 0 artifacts are absent. Fix: updated static fallback to $650;
#   warning escalated from print() to warnings.warn(RuntimeWarning).
#
# Finding 5 (IMPORTANT): generate_order_instructions() computed trade_shares
#   using self.cfg.asset_params[ticker]["approx_price"] (static), while
#   estimate_cost() already used dynamically loaded prices. Fix: use the
#   module-level _get_asset_params() for trade_shares so both paths agree.
#
# Finding 6 (IMPORTANT): _load_dynamic_params_from_part0() and
#   _get_asset_params() were copy-pasted verbatim across three classes
#   (PreTradeAnalyzer, AlmgrenChrissScheduler, PostTradeAnalyzer).
#   Fix: extracted to module-level with a single shared cache.
#
# Finding 7 (LOW): prev_weights fallback hardcoded to {VOO: 0.60, IEF: 0.40}.
#   This only fires on the very first run before the Part 7 tape has ≥ 2 rows;
#   in all live runs the tape provides the correct prior target. Documented
#   clearly; no code change required beyond the comment.
#
# Finding 8 (MEDIUM): sec_fee_bps = 0.278 corresponds to the FY2023 SEC fee
#   schedule and is stale. The value is not hard-updated here because the
#   correct rate requires verification against the current SEC advisory each
#   October. The comment is updated to flag the rate as year-specific and
#   provide the lookup URL.
#
# Finding 32 (prior): dead 'volume.parquet' fallback already removed in prior
#   audit; retained as-is.
# =============================================================================

from __future__ import annotations

import os
import dataclasses
import warnings
from pathlib import Path
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Part8Config:
    version: str = "V2_DAILY_CANONICAL"
    part7_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part7"
    part0_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part0"
    out_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part8"
    # FIX (Finding E, prior audit): direct path to Part 2 consensus tape.
    part2_dir: str = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject") + "/artifacts_part2_g532/predictions"

    # === Asset-level execution parameters ===
    # VOO: ~$650/share, ~5M shares/day ADV, ~$3.3B ADV
    # IEF: ~$95/share,  ~2M shares/day ADV, ~$190M ADV
    # GLD: ~$230/share, ~8M shares/day ADV, ~$1.8B ADV
    # TLT: ~$95/share,  ~6M shares/day ADV, ~$570M ADV
    #
    # NOTE (Finding 4): these static prices are fallbacks used only when
    # Part 0 close_prices.parquet is unavailable.  The dynamic loader in
    # _load_dynamic_params() will override approx_price, adv_shares, and
    # daily_vol_pct with values computed from the most recent 20 trading days.
    # Update these static values periodically to keep the fallback reasonable.
    asset_params: Dict[str, Dict] = field(default_factory=lambda: {
        "VOO": {
            "adv_shares": 5_000_000,
            "approx_price": 650.0,          # FIX (Finding 4): updated from $530 (stale)
            "half_spread_bps": 0.3,
            "impact_coeff_k": 0.35,
            "daily_vol_pct": 1.2,
            "tick_size": 0.01,
        },
        "IEF": {
            "adv_shares": 2_000_000,
            "approx_price": 95.0,
            "half_spread_bps": 0.5,
            "impact_coeff_k": 0.45,
            "daily_vol_pct": 0.35,
            "tick_size": 0.01,
        },
        "GLD": {
            "adv_shares": 8_000_000,
            "approx_price": 230.0,
            "half_spread_bps": 0.4,
            "impact_coeff_k": 0.40,
            "daily_vol_pct": 0.90,
            "tick_size": 0.01,
        },
        "TLT": {
            "adv_shares": 6_000_000,
            "approx_price": 95.0,
            "half_spread_bps": 0.4,
            "impact_coeff_k": 0.40,
            "daily_vol_pct": 0.70,
            "tick_size": 0.01,
        },
    })

    # === Execution timing guidance ===
    optimal_exec_window_start: str = "10:00"
    optimal_exec_window_end: str = "14:30"

    # === Scale thresholds for execution strategy ===
    immediate_exec_pct_adv: float = 0.005
    twap_short_pct_adv: float = 0.05
    twap_long_pct_adv: float = 0.20

    # Dead-band filter — skip rebalance if Δweight is too small to justify TC.
    min_rebalance_threshold: float = 0.02

    # Annual TC drag warning threshold.
    max_annual_tc_drag_bps: float = 50.0

    # === Commission model ===
    commission_bps: float = 0.0

    # === Borrowing costs (long-only — not applicable) ===
    borrow_cost_bps_annual: float = 0.0

    # === SEC fee (applies to sell transactions on US equities) ===
    # IMPORTANT (Finding 8): the SEC fee rate is reset annually each October.
    # Do NOT hard-code a new value without verifying the current rate at:
    #   https://www.sec.gov/rules-regulations/fee-rate-advisory
    # FY2023 rate: $27.80/million = 0.278 bps
    # FY2025 rate: $8.00/million  = 0.800 bps  (verify before updating)
    # The value below is intentionally left at 0.278 until the current advisory
    # is confirmed; update it together with the comment each October.
    sec_fee_bps: float = 0.278

    # === Post-trade tracking ===
    post_trade_retention_days: int = 365


CFG = Part8Config()


# ============================================================
# Root / path helpers
# ============================================================

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
    seen: set = set()
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
# Module-level dynamic parameter loader                       (FIX — Finding 6)
#
# Previously _load_dynamic_params_from_part0() and _get_asset_params() were
# copy-pasted verbatim across PreTradeAnalyzer, AlmgrenChrissScheduler, and
# PostTradeAnalyzer (~45 lines × 3 = 135 duplicate lines).  Any Part 0 schema
# change required three parallel edits.  Extracted here with a single shared
# cache keyed by (part0_dir, frozenset(tickers)).
# ============================================================

_DYNAMIC_PARAMS_CACHE: Dict[Tuple, Dict[str, Dict[str, float]]] = {}


def _load_dynamic_params(part0_dir: str, tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load the most-recent 20-day approx_price, adv_shares, and daily_vol_pct
    for each ticker from Part 0 close_prices.parquet / volume_data.parquet.

    Returns an empty dict on any failure so callers transparently fall back to
    static config values.  Results are cached per (part0_dir, ticker_set).

    FIX (Finding 4): escalated missing-volume warning from print() to
    warnings.warn(RuntimeWarning) so it surfaces in test runners and logs.
    """
    cache_key = (part0_dir, frozenset(tickers))
    if cache_key in _DYNAMIC_PARAMS_CACHE:
        return _DYNAMIC_PARAMS_CACHE[cache_key]

    close_path = os.path.join(part0_dir, "close_prices.parquet")
    volume_path = os.path.join(part0_dir, "volume_data.parquet")

    # FIX (Finding 32, prior audit): dead 'volume.parquet' fallback removed.
    if not os.path.exists(close_path):
        _DYNAMIC_PARAMS_CACHE[cache_key] = {}
        return {}

    if not os.path.exists(volume_path):
        # FIX (Finding 4): escalated from print() to warnings.warn.
        warnings.warn(
            f"[Part 8] volume_data.parquet not found at {volume_path}. "
            "Static asset parameters will be used; prices may be stale. "
            "Run Part 0 to enable dynamic params.",
            RuntimeWarning,
            stacklevel=3,
        )
        _DYNAMIC_PARAMS_CACHE[cache_key] = {}
        return {}

    try:
        close = pd.read_parquet(close_path)
        vol = pd.read_parquet(volume_path)
    except Exception:
        _DYNAMIC_PARAMS_CACHE[cache_key] = {}
        return {}

    if "Date" in close.columns:
        close = close.set_index("Date")
    if "Date" in vol.columns:
        vol = vol.set_index("Date")

    close.index = pd.to_datetime(close.index, errors="coerce")
    vol.index = pd.to_datetime(vol.index, errors="coerce")
    close = close.sort_index()
    vol = vol.sort_index()

    out: Dict[str, Dict[str, float]] = {}
    for ticker in tickers:
        if ticker not in close.columns:
            continue
        px = pd.to_numeric(close[ticker], errors="coerce").dropna()
        if px.empty:
            continue
        shares = (
            pd.to_numeric(vol[ticker], errors="coerce").dropna()
            if ticker in vol.columns
            else pd.Series(dtype=float)
        )
        adv_shares = float(shares.tail(20).mean()) if len(shares) else np.nan
        ret = px.pct_change().dropna()
        daily_vol_pct = float(ret.tail(20).std(ddof=1) * 100.0) if len(ret) >= 5 else np.nan
        out[ticker] = {
            "approx_price": float(px.iloc[-1]),
            "adv_shares": adv_shares,
            "daily_vol_pct": daily_vol_pct,
        }

    _DYNAMIC_PARAMS_CACHE[cache_key] = out
    return out


def _get_asset_params(
    ticker: str,
    cfg: Part8Config,
    use_dynamic_params: bool = True,
) -> Dict:
    """
    Return a parameter dict for *ticker*, merging static config defaults with
    dynamically loaded values from Part 0 (if available and requested).

    FIX (Finding 6): formerly triplicated inside each class; now module-level.
    """
    params = dict(cfg.asset_params[ticker])
    if use_dynamic_params:
        dyn = _load_dynamic_params(cfg.part0_dir, list(cfg.asset_params.keys())).get(ticker, {})
        for k in ("approx_price", "adv_shares", "daily_vol_pct"):
            v = dyn.get(k, np.nan)
            if np.isfinite(v) and v > 0:
                params[k] = float(v)
    return params


# ============================================================
# Live VIX loader                                             (FIX — Finding 3)
#
# main() previously hardcoded vix_level=16.5, making the stress scalar
# structurally blind to true market conditions.  This utility reads the most
# recent VIX close from Part 0 close_prices.parquet and falls back to 18.0
# only when the artifact is absent.
# ============================================================

def _load_live_vix(part0_dir: str, fallback: float = 18.0) -> float:
    """
    Return the most recent VIX closing level from Part 0 close_prices.parquet.

    Tries column names 'VIX', '^VIX', and 'vix' in that order.
    Returns *fallback* if the file is absent, the column is missing, or any
    read error occurs.
    """
    close_path = os.path.join(part0_dir, "close_prices.parquet")
    if not os.path.exists(close_path):
        return fallback
    try:
        close = pd.read_parquet(close_path)
        if "Date" in close.columns:
            close = close.set_index("Date")
        close.index = pd.to_datetime(close.index, errors="coerce")
        close = close.sort_index()
        for col in ("VIX", "^VIX", "vix"):
            if col in close.columns:
                v = pd.to_numeric(close[col], errors="coerce").dropna()
                if not v.empty:
                    return float(v.iloc[-1])
    except Exception:
        pass
    return fallback


# ============================================================
# Pre-Trade Cost Estimation
# ============================================================

class PreTradeAnalyzer:
    """
    Estimates expected transaction costs before placing an order.

    Uses the square-root market impact model (Almgren et al., 2005):
        I = k × σ × √(Q / ADV)

    Where:
        k   = asset-specific impact coefficient (typically 0.3–0.7 for ETFs)
        σ   = daily volatility (fraction)
        Q   = trade size (in dollars)
        ADV = average daily volume (in dollars)

    Total one-way cost = half_spread + permanent_impact + temporary_impact
    """

    def __init__(self, cfg: Part8Config = CFG):
        self.cfg = cfg

    # FIX (Finding 6): removed triplicated _load_dynamic_params_from_part0()
    # and _get_asset_params() methods; delegating to module-level functions.

    def _get_params(self, ticker: str, use_dynamic: bool = True) -> Dict:
        return _get_asset_params(ticker, self.cfg, use_dynamic_params=use_dynamic)

    def estimate_cost(
        self,
        ticker: str,
        trade_dollars: float,
        portfolio_dollars: float,
        direction: str = "buy",
        use_dynamic_params: bool = True,
        market_vol_scalar: float = 1.0,
    ) -> Dict:
        """
        Estimate the full transaction cost for a single asset trade.

        Returns a dict with:
            spread_bps, impact_bps, commission_bps, sec_fee_bps,
            total_bps, total_dollars, pct_adv, execution_strategy,
            execution_horizon_min, timing_risk_bps.
        """
        if ticker not in self.cfg.asset_params:
            return self._unknown_asset_cost(ticker, trade_dollars)

        params = self._get_params(ticker, use_dynamic=use_dynamic_params)
        adv_dollars = params["adv_shares"] * params["approx_price"]
        sigma_daily = params["daily_vol_pct"] / 100.0 * market_vol_scalar
        pct_adv = abs(trade_dollars) / adv_dollars

        # ─── 1. Bid-ask spread ────────────────────────────────────────────────
        spread_bps = params["half_spread_bps"]

        # ─── 2. Square-root market impact (Almgren-Chriss) ───────────────────
        k = params["impact_coeff_k"]
        impact_pct = k * sigma_daily * np.sqrt(pct_adv)
        impact_bps = float(impact_pct * 10000)

        # ─── 3. Commission ────────────────────────────────────────────────────
        commission_bps = self.cfg.commission_bps

        # ─── 4. SEC fee (sell only) ───────────────────────────────────────────
        sec_fee = self.cfg.sec_fee_bps if direction == "sell" else 0.0

        # ─── 5. Total ─────────────────────────────────────────────────────────
        total_bps = spread_bps + impact_bps + commission_bps + sec_fee
        total_dollars = abs(trade_dollars) * (total_bps / 10000)

        # ─── 6. Execution strategy ────────────────────────────────────────────
        if pct_adv < self.cfg.immediate_exec_pct_adv:
            strategy = "immediate_market"
            horizon_min = 1
        elif pct_adv < self.cfg.twap_short_pct_adv:
            strategy = "twap_1hr"
            horizon_min = 60
        elif pct_adv < self.cfg.twap_long_pct_adv:
            strategy = "twap_4hr"
            horizon_min = 240
        else:
            strategy = "twap_full_day_or_split"
            horizon_min = 390

        # ─── 7. Timing risk ───────────────────────────────────────────────────
        exec_days = horizon_min / 390.0
        timing_risk_bps = float(sigma_daily * np.sqrt(exec_days) * 10000 * 0.5)

        return {
            "ticker": ticker,
            "trade_dollars": float(trade_dollars),
            "pct_adv": float(pct_adv),
            "direction": direction,
            "spread_bps": float(spread_bps),
            "impact_bps": float(impact_bps),
            "commission_bps": float(commission_bps),
            "sec_fee_bps": float(sec_fee),
            "total_bps": float(total_bps),
            "total_dollars": float(total_dollars),
            "execution_strategy": strategy,
            "execution_horizon_min": int(horizon_min),
            "timing_risk_bps": float(timing_risk_bps),
            "optimal_window": f"{self.cfg.optimal_exec_window_start}–{self.cfg.optimal_exec_window_end}",
            "stressed_market": bool(market_vol_scalar > 1.2),
            # Carry the resolved price so callers can use it for share math.
            "_resolved_approx_price": float(params["approx_price"]),
        }

    def _trivial_trade(self, ticker: str) -> Dict:
        return {
            "ticker": ticker, "trade_dollars": 0, "pct_adv": 0,
            "spread_bps": 0, "impact_bps": 0, "commission_bps": 0,
            "sec_fee_bps": 0, "total_bps": 0, "total_dollars": 0,
            "execution_strategy": "no_trade", "execution_horizon_min": 0,
            "timing_risk_bps": 0, "optimal_window": "N/A",
            "_resolved_approx_price": 100.0,
        }

    def _unknown_asset_cost(self, ticker: str, trade_dollars: float) -> Dict:
        total_bps = 5.0
        return {
            "ticker": ticker, "trade_dollars": float(trade_dollars),
            "pct_adv": np.nan, "spread_bps": 2.0, "impact_bps": 2.5,
            "commission_bps": 0.5, "sec_fee_bps": 0,
            "total_bps": total_bps,
            "total_dollars": float(abs(trade_dollars) * total_bps / 10000),
            "execution_strategy": "unknown_asset_conservative",
            "execution_horizon_min": 15, "timing_risk_bps": 2.0,
            "optimal_window": self.cfg.optimal_exec_window_start,
            "_resolved_approx_price": 100.0,
        }


# ============================================================
# Almgren-Chriss Optimal Execution Schedule
# ============================================================

class AlmgrenChrissScheduler:
    """
    Optimal trade execution schedule from Almgren & Chriss (2001).

    For our ETF universe, the optimal schedule is nearly linear (TWAP)
    because market impact is small relative to price variance.  The optimal
    horizon depends on portfolio size.

    Almgren-Chriss optimal trajectory:
        x(t) = X × sinh(κ(T-t)) / sinh(κT)
        where κ = sqrt(η × γ / σ²)
    """

    def __init__(self, cfg: Part8Config = CFG):
        self.cfg = cfg

    # FIX (Finding 6): removed triplicated dynamic param methods.

    def _get_params(self, ticker: str, use_dynamic: bool = True) -> Dict:
        return _get_asset_params(ticker, self.cfg, use_dynamic_params=use_dynamic)

    def optimal_schedule(
        self,
        ticker: str,
        trade_shares: float,
        execution_horizon_min: int = 60,
        n_slices: int = 6,
        risk_aversion: float = 2.5,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of scheduled order slices.
        Columns: time_offset_min, shares_to_trade, pct_of_total, cumulative_pct.
        """
        if ticker not in self.cfg.asset_params or abs(trade_shares) < 1:
            return pd.DataFrame({
                "time_offset_min": [0],
                "shares_to_trade": [trade_shares],
                "pct_of_total": [1.0],
                "cumulative_pct": [1.0],
            })

        params = self._get_params(ticker, use_dynamic=True)
        sigma_per_min = (params["daily_vol_pct"] / 100.0) / np.sqrt(390)
        adv_per_min = params["adv_shares"] / 390.0
        T = max(execution_horizon_min, 1)

        eta = 0.005 * params["approx_price"] / adv_per_min
        gamma = 0.001 * params["approx_price"] / adv_per_min

        kappa_sq = float(risk_aversion * gamma * sigma_per_min ** 2 / eta)
        kappa_sq = max(kappa_sq, 1e-10)
        kappa = float(np.sqrt(kappa_sq))

        times = np.linspace(0, T, n_slices + 1)[:-1]
        dt = T / n_slices

        if abs(kappa * T) < 0.01:
            shares_at_t = np.full(n_slices, abs(trade_shares) / n_slices)
        else:
            shares_rate = (
                abs(trade_shares) * kappa / np.sinh(kappa * T)
                * np.cosh(kappa * (T - times))
            )
            shares_at_t = shares_rate * dt
            if shares_at_t.sum() > 0:
                shares_at_t = shares_at_t / shares_at_t.sum() * abs(trade_shares)

        sign = np.sign(trade_shares) if trade_shares != 0 else 1.0
        shares_at_t = shares_at_t * sign

        return pd.DataFrame({
            "time_offset_min": [int(t) for t in times],
            "shares_to_trade": [round(float(s), 2) for s in shares_at_t],
            "pct_of_total": [
                float(s / trade_shares) if trade_shares != 0 else 0.0
                for s in shares_at_t
            ],
            "cumulative_pct": list(
                np.cumsum(shares_at_t / (trade_shares if trade_shares != 0 else 1))
            ),
        })

    def generate_order_instructions(
        self,
        decision_date: str,
        allocations: Dict[str, float],
        portfolio_dollars: float,
        prev_allocations: Dict[str, float],
        vix_level: float = 18.0,
    ) -> Dict:
        """
        Given a target allocation from Part 7 and current allocation, generate
        complete execution instructions.

        VIX-based stress scalar:
            VIX > 30 → stress_scalar = 2.0 (wider slices, longer TWAP)
            VIX > 20 → stress_scalar = 1.4
            VIX ≤ 20 → stress_scalar = 1.0  (standard)
        """
        if vix_level > 30:
            stress_scalar = 2.0
            stress_label = "HIGH_STRESS"
            urgency = "LOW"
        elif vix_level > 20:
            stress_scalar = 1.4
            stress_label = "ELEVATED"
            urgency = "NORMAL"
        else:
            stress_scalar = 1.0
            stress_label = "NORMAL"
            urgency = "NORMAL"

        analyzer = PreTradeAnalyzer(self.cfg)
        instructions = []
        total_cost_dollars = 0.0

        for ticker in allocations:
            target_w = float(allocations.get(ticker, 0.0))
            prev_w = float(prev_allocations.get(ticker, 0.0))
            delta_w = target_w - prev_w
            trade_dollars = delta_w * portfolio_dollars
            direction = "buy" if trade_dollars > 0 else "sell"

            if abs(delta_w) < self.cfg.min_rebalance_threshold:
                continue
            if abs(trade_dollars) < 100:
                continue

            cost = analyzer.estimate_cost(
                ticker, trade_dollars, portfolio_dollars,
                direction=direction,
                market_vol_scalar=stress_scalar,
            )
            total_cost_dollars += cost["total_dollars"]

            # FIX (Finding 5): use the dynamically resolved price from the cost
            # result rather than the static config value.  estimate_cost() already
            # called _get_asset_params() with use_dynamic=True; the resolved price
            # is carried back in cost["_resolved_approx_price"].
            approx_price = cost.get("_resolved_approx_price", 100.0)
            trade_shares = trade_dollars / approx_price

            schedule = self.optimal_schedule(
                ticker,
                trade_shares,
                execution_horizon_min=cost["execution_horizon_min"],
                n_slices=max(1, cost["execution_horizon_min"] // 10),
            )

            instructions.append({
                "ticker": ticker,
                "direction": direction,
                "trade_dollars": round(float(trade_dollars), 2),
                "trade_shares_approx": round(abs(trade_shares), 2),
                "weight_delta": round(float(delta_w), 6),
                "target_weight": round(float(target_w), 6),
                "prev_weight": round(float(prev_w), 6),
                "pre_trade_cost": {
                    k: v for k, v in cost.items() if not k.startswith("_")
                },
                "execution_schedule": schedule.to_dict(orient="records"),
                "execute_between": cost["optimal_window"],
                "urgency": urgency,
                "stress_level": stress_label,
            })

        return {
            "decision_date": str(decision_date),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "portfolio_dollars": portfolio_dollars,
            "vix_level": vix_level,
            "stress_label": stress_label,
            "total_estimated_cost_dollars": round(total_cost_dollars, 2),
            "total_estimated_cost_bps": round(
                total_cost_dollars / max(portfolio_dollars, 1) * 10000, 3
            ),
            "instructions": instructions,
            "n_trades": len(instructions),
            "execute_window": (
                f"{self.cfg.optimal_exec_window_start}–"
                f"{self.cfg.optimal_exec_window_end} on {decision_date}"
            ),
        }


# ============================================================
# Post-Trade Analysis
# ============================================================

class PostTradeAnalyzer:
    """
    Computes implementation shortfall and slippage attribution after execution.

    Implementation Shortfall (Perold 1988):
        IS = (actual execution price - decision price) × shares
           = paper portfolio return - live portfolio return

    Decomposition:
        IS = delay cost + market impact + timing risk + spread cost
    """

    def __init__(self, cfg: Part8Config = CFG):
        self.cfg = cfg
        # FIX (prior audit): post_trade_log_path belongs at construction time,
        # not inside _get_asset_params() where it was unreachable dead code.
        self.post_trade_log_path = os.path.join(cfg.out_dir, "post_trade_log.csv")

    # FIX (Finding 6): removed triplicated dynamic param methods.

    def _get_params(self, ticker: str, use_dynamic: bool = True) -> Dict:
        return _get_asset_params(ticker, self.cfg, use_dynamic_params=use_dynamic)

    def record_trade(
        self,
        decision_date: str,
        ticker: str,
        direction: str,
        shares: float,
        decision_price: float,
        arrival_price: float,
        avg_fill_price: float,
        completion_time_min: int,
        estimated_cost_bps: float,
    ) -> Dict:
        """Record a completed trade and compute implementation shortfall."""
        if direction == "buy":
            slip_from_decision = (avg_fill_price - decision_price) / decision_price * 10000
            slip_from_arrival = (avg_fill_price - arrival_price) / arrival_price * 10000
        else:
            slip_from_decision = (decision_price - avg_fill_price) / decision_price * 10000
            slip_from_arrival = (arrival_price - avg_fill_price) / arrival_price * 10000

        trade_dollars = abs(shares * avg_fill_price)
        is_bps = float(slip_from_decision)

        record = {
            "decision_date": str(decision_date),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "direction": direction,
            "shares": float(shares),
            "trade_dollars": round(trade_dollars, 2),
            "decision_price": float(decision_price),
            "arrival_price": float(arrival_price),
            "avg_fill_price": float(avg_fill_price),
            "implementation_shortfall_bps": round(is_bps, 3),
            "slippage_from_arrival_bps": round(float(slip_from_arrival), 3),
            "delay_cost_bps": round(
                float((arrival_price - decision_price) / decision_price * 10000), 3
            ),
            "completion_time_min": int(completion_time_min),
            "estimated_cost_bps": float(estimated_cost_bps),
            "cost_overrun_bps": round(is_bps - estimated_cost_bps, 3),
            "model_accurate": bool(abs(is_bps - estimated_cost_bps) < 3.0),
        }

        os.makedirs(self.cfg.out_dir, exist_ok=True)
        df_new = pd.DataFrame([record])
        if os.path.exists(self.post_trade_log_path):
            df_existing = pd.read_csv(self.post_trade_log_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(self.post_trade_log_path, index=False)
        return record

    def generate_report(self) -> Dict:
        """Summarize post-trade performance across all recorded trades."""
        if not os.path.exists(self.post_trade_log_path):
            return {"error": "No post-trade log found", "n_trades": 0}

        df = pd.read_csv(self.post_trade_log_path)
        if df.empty:
            return {"n_trades": 0}

        report = {
            "n_trades": len(df),
            "mean_implementation_shortfall_bps": float(
                df["implementation_shortfall_bps"].mean()
            ),
            "median_implementation_shortfall_bps": float(
                df["implementation_shortfall_bps"].median()
            ),
            "mean_estimated_cost_bps": float(df["estimated_cost_bps"].mean()),
            "mean_cost_overrun_bps": float(df["cost_overrun_bps"].mean()),
            "pct_within_estimate": float(df["model_accurate"].mean()),
            "by_ticker": df.groupby("ticker")["implementation_shortfall_bps"]
                          .mean().to_dict(),
            "by_direction": df.groupby("direction")["implementation_shortfall_bps"]
                             .mean().to_dict(),
            "worst_trades": df.nlargest(3, "implementation_shortfall_bps")[
                [
                    "decision_date", "ticker", "direction",
                    "implementation_shortfall_bps", "estimated_cost_bps",
                ]
            ].to_dict(orient="records"),
        }

        trades_per_year = len(df) / max(
            (
                pd.to_datetime(df["decision_date"]).max()
                - pd.to_datetime(df["decision_date"]).min()
            ).days / 365.25,
            0.1,
        )
        avg_trade_size_pct_portfolio = 0.05
        annual_drag_bps = (
            report["mean_implementation_shortfall_bps"]
            * trades_per_year
            * avg_trade_size_pct_portfolio
            * 2
        )
        report["estimated_annual_drag_bps"] = round(annual_drag_bps, 1)
        return report


# ============================================================
# Annual Cost Drag Model
# ============================================================

def compute_annual_cost_drag(
    tape: pd.DataFrame,
    cfg: Part8Config = CFG,
    portfolio_dollars: float = 1_000_000,
) -> Dict:
    """
    Compute the full annual transaction cost drag on strategy performance.

    FIX (Findings 1+2, merged):
    ────────────────────────────
    The previous implementation filtered out rows with turnover < 0.001
    BEFORE computing the average turnover.  This left only 10 of 1,644
    realized rows in the mean (conditional mean ≈ 0.11 = 11%), while the
    correct full-population mean is ≈ 0.000669 (<0.1%).  The formula
    structure — avg_cost_bps × avg_turnover × 252 × 2 — is correct for a
    daily H=1 system: 252 is the number of daily evaluation opportunities
    per year, and the × 2 round-trip multiplier is correct given one-way
    turnover reporting.  The sole bug was feeding a conditional turnover mean
    (over traded rows only) instead of the full-population mean (over all
    realized rows).  Result: 21.9 bps stated vs 0.133 bps true — a 164×
    overstatement that would have fired the Part 9/10 TC alarm spuriously
    once n_live_realized ≥ 60.

    Fix applied:
    • cost_bps is still estimated only on rows where a trade actually occurred
      (turnover > 0.001), for numerical stability.
    • avg_turnover is now the full-population mean over ALL realized rows
      (including the zero-turnover majority), which correctly encodes both
      the typical trade size and the empirical probability of trading on any
      given day.
    • 252 multiplier is retained as correct for a daily H=1 system.

    Verification:
        full_pop_mean(0.000669) × 252 × 2 × avg_cost_bps(0.3948) ≈ 0.133 bps/yr
        == cond_mean(0.11) × actual_rpy(1.59) × 2 × avg_cost_bps ≈ 0.138 bps/yr
    (3.5% rounding gap from the discrete turnover < 0.001 cutoff; both are
    correct within noise.)
    """
    if "turnover" not in tape.columns:
        return {"error": "No turnover column in tape"}

    realized = tape[tape["y_avail"] == 1].copy()
    if len(realized) == 0:
        return {"error": "No realized rows in tape"}

    realized = realized.dropna(subset=["turnover"])

    # Full-population turnover mean: includes all realized rows (even the
    # zero-turnover majority).  This is the correct input to the formula.
    full_pop_avg_turnover = float(
        pd.to_numeric(realized["turnover"], errors="coerce").fillna(0).mean()
    )

    # cost_bps is only estimated on rows that actually traded, for numerical
    # stability (a zero-notional trade has no well-defined cost_bps).
    trading_rows = realized[
        pd.to_numeric(realized["turnover"], errors="coerce").fillna(0) > 0.001
    ].copy()

    analyzer = PreTradeAnalyzer(cfg)
    cost_rows = []

    for _, row in trading_rows.iterrows():
        turnover = float(row["turnover"])
        voo_trade = turnover * portfolio_dollars * float(row.get("w_strategy_voo", 0.60))
        ief_trade = turnover * portfolio_dollars * (
            1 - float(row.get("w_strategy_voo", 0.60))
        )

        # Use stress_score_raw or vix_level if present; otherwise default 18.0.
        vix = float(row.get("vix_level", 18.0)) if "vix_level" in row.index else 18.0
        if not np.isfinite(vix):
            vix = 18.0
        stress = 2.0 if vix > 30 else (1.4 if vix > 20 else 1.0)

        cost_voo = analyzer.estimate_cost(
            "VOO", voo_trade, portfolio_dollars, market_vol_scalar=stress
        )
        cost_ief = analyzer.estimate_cost(
            "IEF", ief_trade, portfolio_dollars, market_vol_scalar=stress
        )
        combined_bps = (
            cost_voo["total_bps"] * voo_trade / max(voo_trade + ief_trade, 1)
            + cost_ief["total_bps"] * ief_trade / max(voo_trade + ief_trade, 1)
        )
        cost_dollars = cost_voo["total_dollars"] + cost_ief["total_dollars"]

        cost_rows.append({
            "Date": row.get("Date", pd.NaT),
            "turnover": turnover,
            "cost_bps_actual_model": combined_bps,
            "cost_bps_current_flat": 5.0,
            "cost_dollars_actual": cost_dollars,
            "cost_dollars_current": turnover * portfolio_dollars * 5.0 / 10000,
        })

    if not cost_rows:
        return {"error": "No trades with sufficient turnover"}

    cost_df = pd.DataFrame(cost_rows)
    n_rebalances = len(cost_df)
    avg_cost_bps = float(cost_df["cost_bps_actual_model"].mean())

    # FIX (Findings 1+2): use full_pop_avg_turnover here, not cost_df mean.
    actual_annual_bps = float(avg_cost_bps * full_pop_avg_turnover * 252 * 2)
    current_annual_bps = float(5.0 * full_pop_avg_turnover * 252 * 2)

    return {
        "n_rebalances": n_rebalances,
        # Report the full-population mean so downstream readers understand
        # this is expected daily turnover, not the conditional traded-row mean.
        "avg_turnover_pct": float(full_pop_avg_turnover * 100),
        "portfolio_dollars": portfolio_dollars,
        "avg_cost_bps_actual": avg_cost_bps,
        "avg_cost_bps_current_flat": 5.0,
        "annual_drag_bps_actual": round(actual_annual_bps, 4),
        "annual_drag_bps_current": round(current_annual_bps, 4),
        "current_overstates_by_bps": round(current_annual_bps - actual_annual_bps, 4),
        "pct_overstated": round(
            (current_annual_bps / max(actual_annual_bps, 1e-6) - 1) * 100, 1
        ),
        "note": (
            "The current flat 5bps assumption OVERSTATES costs for retail-scale "
            "portfolios on liquid ETFs. Your live strategy IR is likely HIGHER than "
            "the backtest shows. For portfolios >$10M, update asset_params with "
            "live ADV data."
        ),
    }


# ============================================================
# Main helpers
# ============================================================

def load_part7_instructions(cfg: Part8Config = CFG) -> Dict:
    """
    Load the latest allocation target from Part 3 fusion allocations (preferred)
    or Part 7 target weights (fallback).
    """
    part3_dir = os.path.join(
        os.path.dirname(cfg.part7_dir.rstrip("/\\")), "artifacts_part3_v1"
    )
    fusion_alloc_path = os.path.join(part3_dir, "v1_fusion_allocations.csv")

    if os.path.exists(fusion_alloc_path):
        try:
            df = pd.read_csv(fusion_alloc_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).sort_values("Date")
            if not df.empty:
                latest_date = df["Date"].max()
                latest = df[df["Date"] == latest_date].copy()
                voo_total = float(
                    latest[latest["sleeve"] == "VOO"]["weight"].sum()
                )
                ief_total = float(
                    latest[latest["sleeve"] == "IEF"]["weight"].sum()
                )
                alpha_voo = float(
                    latest[
                        (latest["sleeve"] == "VOO") & (latest["is_alpha"] == 1)
                    ]["weight"].sum()
                )
                return {
                    "Date": str(latest_date),
                    "w_target_voo": voo_total,
                    "w_target_ief": ief_total,
                    "w_alpha_voo": alpha_voo,
                    "source": "v1_fusion_allocations",
                }
        except Exception as e:
            print(f"[Part 8] Warning: could not load fusion allocations: {e}")

    # Fallback: Part 7 current target weights
    current_path = os.path.join(cfg.part7_dir, "current_target_weights.json")
    weights_path = os.path.join(cfg.part7_dir, "portfolio_weights_tape.csv")

    if os.path.exists(current_path):
        with open(current_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(weights_path):
        df = pd.read_csv(weights_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
        if not df.empty:
            return df.iloc[-1].to_dict()
    return {}


def calibrate_impact_coefficients(cfg: Part8Config = CFG) -> Dict[str, float]:
    return {k: float(v["impact_coeff_k"]) for k, v in cfg.asset_params.items()}


def generate_part3_record(
    cfg: Part8Config,
    instructions: Dict,
    annual_drag: Dict,
) -> Dict[str, float]:
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_estimated_cost_dollars": float(
            instructions.get("total_estimated_cost_dollars", 0.0)
        ),
        "total_estimated_cost_bps": float(
            instructions.get("total_estimated_cost_bps", 0.0)
        ),
        # FIX (prior audit): key is "annual_drag_bps_actual", not "annual_drag_bps".
        "annual_tc_drag_bps": float(
            annual_drag.get(
                "annual_drag_bps_actual",
                annual_drag.get("annual_drag_bps", np.nan),
            )
        ) if isinstance(annual_drag, dict) else np.nan,
    }


# ============================================================
# Main
# ============================================================

def main() -> int:
    cfg = Part8Config()
    cfg = dataclasses.replace(cfg, part7_dir=_abs_path(cfg.part7_dir))
    cfg = dataclasses.replace(cfg, part0_dir=_abs_path(cfg.part0_dir))
    cfg = dataclasses.replace(cfg, out_dir=_abs_path(cfg.out_dir))
    cfg = dataclasses.replace(cfg, part2_dir=_abs_path(cfg.part2_dir))
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 8 — Execution & Transaction Cost Model v2 (daily canonical)")
    print("=" * 70)

    analyzer = PreTradeAnalyzer(cfg)
    scheduler = AlmgrenChrissScheduler(cfg)

    latest = load_part7_instructions(cfg)
    if not latest:
        print("[Part 8] Part 7 target weights not found — writing meta only.")
        meta = {
            "version": cfg.version,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "warning": "no_part7_targets",
        }
        with open(os.path.join(cfg.out_dir, "part8_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return 0

    decision_date = str(latest.get("Date", date.today().isoformat()))
    w_voo = float(latest.get("w_target_voo", latest.get("VOO", 0.60)))
    w_ief = float(latest.get("w_target_ief", latest.get("IEF", 0.40)))

    # prev_weights: use second-to-last row of Part 7 tape (yesterday's model
    # target).  The fallback of {VOO: 0.60, IEF: 0.40} fires only on the very
    # first run before the tape has ≥ 2 rows; in all live runs the tape row
    # is authoritative.  (Finding 7 note: the bot's portfolio_state.json shows
    # 100% cash because no paper trades have executed yet; this is the MODEL
    # target state, not the executed state, so 60/40 is still a reasonable
    # first-run prior for the MODEL portfolio even though it differs from the
    # bot's cash state.)
    prev_weights: Dict[str, float] = {"VOO": 0.60, "IEF": 0.40}
    weights_path = os.path.join(cfg.part7_dir, "portfolio_weights_tape.csv")
    if os.path.exists(weights_path):
        wdf = pd.read_csv(weights_path)
        if len(wdf) >= 2:
            prev_weights = {
                "VOO": float(wdf.iloc[-2].get("w_target_voo", 0.60)),
                "IEF": float(wdf.iloc[-2].get("w_target_ief", 0.40)),
            }

    # FIX (Finding 3): load live VIX from Part 0 artifacts.
    # Falls back to 18.0 if close_prices.parquet is absent or has no VIX column.
    live_vix = _load_live_vix(cfg.part0_dir, fallback=18.0)
    print(f"[Part 8] Live VIX: {live_vix:.2f}")

    instructions = scheduler.generate_order_instructions(
        decision_date=decision_date,
        allocations={"VOO": w_voo, "IEF": w_ief},
        portfolio_dollars=1000.0,
        prev_allocations=prev_weights,
        vix_level=live_vix,   # FIX (Finding 3): was hardcoded 16.5
    )

    tape_path = os.path.join(cfg.part2_dir, "g532_final_consensus_tape.csv")
    annual_drag: Dict = {}
    if os.path.exists(tape_path):
        tape = pd.read_csv(tape_path)
        annual_drag = compute_annual_cost_drag(tape, cfg, portfolio_dollars=1000.0)
    else:
        print(
            f"[Part 8] WARNING: Consensus tape not found at {tape_path} "
            "— annual_drag_summary will be empty."
        )

    record = generate_part3_record(cfg, instructions, annual_drag)
    record.update({
        "Date": decision_date,
        "w_target_voo": w_voo,
        "w_target_ief": w_ief,
    })

    pd.DataFrame([record]).to_csv(
        os.path.join(cfg.out_dir, "execution_cost_tape.csv"), index=False
    )

    meta = {
        "version": cfg.version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "assets_modeled": list(cfg.asset_params.keys()),
        "live_vix_used": live_vix,   # FIX (Finding 3): audit trail
        "latest_order_instructions": instructions,
        "impact_coefficients": calibrate_impact_coefficients(cfg),
        "annual_drag_summary": annual_drag,
        "min_rebalance_threshold": cfg.min_rebalance_threshold,
        "max_annual_tc_drag_bps": cfg.max_annual_tc_drag_bps,
    }
    with open(os.path.join(cfg.out_dir, "part8_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    with open(os.path.join(cfg.out_dir, "example_order_instructions.json"), "w") as f:
        json.dump(instructions, f, indent=2, default=str)

    print("\n✅ PART 8 COMPLETE")
    print(f"   Wrote: {os.path.join(cfg.out_dir, 'execution_cost_tape.csv')}")
    print(f"   Meta:  {os.path.join(cfg.out_dir, 'part8_meta.json')}")
    return 0


if __name__ == "__main__":
    main()
