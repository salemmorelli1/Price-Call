#!/usr/bin/env python3
# @title Part 2A Gen 6

# part2a21_alpha.py
# =============================================================================
# PROJECT: VOO vs IEF Daily Price Call
# STAGE:   Part 2A
# VERSION: 21.0 (Gen 5.3.2 soft-caution alpha, current production family)
# =============================================================================

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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Part2A21Config:
    root: str = "."
    stale_days_max: int = 14
    z_min_history: int = 20
    smooth_window: int = 3

    # Alpha construction
    entry_z: float = 0.50
    full_z: float = 2.50
    max_abs_alpha: float = 0.15
    vol_shrink_k: float = 4.00
    drift_mult: float = 0.60
    turnover_cost_per_unit: float = 0.0005  # 5 bps / unit turnover

    # Governance multipliers
    gov_mult_normal: float = 1.00
    gov_mult_caution: float = 0.60
    gov_mult_defensive: float = 0.25

    # Gen 5.3.2 soft-caution integration
    overlay_entry_z_bump_max: float = 0.35
    overlay_agreement_min_base: float = 0.50
    overlay_agreement_bump_max: float = 0.10
    overlay_agreement_min_cap: float = 0.75

    caution_shrink_overlay_w: float = 0.25
    caution_shrink_width_w: float = 0.20
    caution_shrink_unc_w: float = 0.10
    caution_shrink_floor: float = 0.55

    overlay_hard_veto_width: float = 0.995
    overlay_hard_veto_uncertainty: float = 0.995

    # Output layout
    out_primary: str = _DRIVE_ROOT + "/artifacts_part2a_alpha"
    out_predictions_subdir: str = "predictions"

    fn_positions: str = "part2a21_alpha_positions.csv"
    fn_eligibility: str = "part2a21_alpha_eligibility.csv"
    fn_summary_csv: str = "part2a21_alpha_summary.csv"
    fn_summary_json: str = "part2a21_alpha_summary.json"
    fn_summary_tape: str = "part2a21_alpha_summary_tape.csv"

    alias_positions: Tuple[str, ...] = (
        "alpha_positions.csv",
        "part2a_alpha_positions.csv",
        "part2a21_alpha_positions.csv",
    )
    alias_eligibility: Tuple[str, ...] = (
        "alpha_eligibility.csv",
        "part2a_alpha_eligibility.csv",
        "part2a21_alpha_eligibility.csv",
    )
    alias_summary_csv: Tuple[str, ...] = (
        "alpha_summary.csv",
        "part2a_alpha_summary.csv",
        "part2a21_alpha_summary.csv",
    )
    alias_summary_json: Tuple[str, ...] = (
        "alpha_summary.json",
        "part2a_alpha_summary.json",
        "part2a21_alpha_summary.json",
    )
    alias_summary_tape: Tuple[str, ...] = (
        "alpha_summary_tape.csv",
        "part2a_alpha_summary_tape.csv",
        "part2a21_alpha_summary_tape.csv",
    )

    tape_glob_patterns: Tuple[str, ...] = (
        "artifacts_part2_g532/predictions/g532_final_consensus_tape.csv",
        "artifacts_part2_g53*/predictions/*final_consensus_tape.csv",
        "artifacts_part2_g5*/predictions/*final_consensus_tape.csv",
        "artifacts_part2*/predictions/*final_consensus_tape.csv",
        "artifacts_part2*/predictions/*.csv",
        "artifacts_part3*/v1_final_production_tape.csv",
    )


CFG = Part2A21Config()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _annualized_ir(x: Iterable[float], h_reb: int = 1) -> float:
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = np.nanstd(x, ddof=1)
    if (not np.isfinite(sd)) or (sd <= 0):
        return np.nan
    return float(np.nanmean(x) / sd * np.sqrt(252.0 / h_reb))


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    return None


def _discover_latest_by_glob(root: Path, patterns: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(pat))
    out = [p for p in out if p.is_file()]
    out = sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _copy_alias(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def _expanding_zscore(s: pd.Series, min_history: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.expanding(min_periods=min_history).mean().shift(1)
    sd = s.expanding(min_periods=min_history).std(ddof=0).shift(1)
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-3.0, 3.0)


def _normalize_governance_tier(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.upper().fillna("NORMAL")
    x = x.where(x.isin(["NORMAL", "CAUTION", "DEFENSIVE"]), "NORMAL")
    return x


def discover_canonical_tape(root: Path) -> Path:
    tape_candidates = _discover_latest_by_glob(root, CFG.tape_glob_patterns)
    preferred = [p for p in tape_candidates if "final_consensus_tape" in p.name.lower()]
    if preferred:
        return preferred[0]
    if tape_candidates:
        return tape_candidates[0]

    local_fallbacks = [
        Path("/mnt/data/g532_final_consensus_tape.csv"),
        Path("/mnt/data/g53_final_consensus_tape.csv"),
        Path("/mnt/data/g52_final_consensus_tape.csv"),
        Path("/mnt/data/v77_final_consensus_tape.csv"),
        Path("/mnt/data/v1_final_production_tape.csv"),
    ]
    for p in local_fallbacks:
        if p.exists():
            return p

    raise FileNotFoundError("No canonical Part 2 tape found.")


def load_tape(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Canonical tape is empty: {path}")

    date_col = _pick_col(df, ["Date", "date", "ds", "timestamp"])
    if date_col is None:
        raise ValueError("Canonical tape missing a date column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    numeric_candidates = [
        "p_final_cal", "p0", "alpha_scale", "excess_vol10",
        "alpha_tech_relative", "alpha_breadth", "alpha_credit_spread", "qqq_r1",
        "fwd_voo", "fwd_ief", "y_avail", "is_live", "drift_alarm",
        "deploy_downside", "high_risk_state", "publish_fail_closed",
        "dist_overlay_on_g53", "dist_overlay_strength_g53",
        "dist_trust_score_g53", "dist_width_caution_g53",
        "dist_tail_shift_g53", "uncertainty_penalty_g5",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "governance_tier" not in df.columns:
        df["governance_tier"] = "NORMAL"
    df["governance_tier"] = _normalize_governance_tier(df["governance_tier"])

    if "regime_label" not in df.columns:
        df["regime_label"] = "unknown"
    df["regime_label"] = df["regime_label"].astype(str).fillna("unknown")

    for c, default in {
        "dist_overlay_on_g53": 0.0,
        "dist_overlay_strength_g53": 0.0,
        "dist_trust_score_g53": 0.0,
        "dist_width_caution_g53": 0.0,
        "dist_tail_shift_g53": 0.0,
        "uncertainty_penalty_g5": 0.0,
    }.items():
        if c not in df.columns:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)

    return df


def _resolve_qqq_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("qqq_r1", "qqq_ret_1d"):
        if c in df.columns:
            return c
    return None


def _read_part2_summary(tape_source: Path) -> Dict:
    """Attempt to read Part 2's summary JSON from the canonical sibling path.

    The tape lives at  <dir>/g532_final_consensus_tape.csv.
    The summary lives at <dir>/part2_g532_summary.json.
    We also try the common alternate name part2_summary.json.
    Returns an empty dict if nothing is found so callers degrade gracefully.
    """
    parent = tape_source.parent
    candidates = [
        parent / "part2_g532_summary.json",
        parent / "part2_summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def build_alpha_positions(
    df: pd.DataFrame, tape_source: Path, part2_summary: Optional[Dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
    x = df.copy()

    qqq_col = _resolve_qqq_col(x)
    required = [
        "alpha_tech_relative", "alpha_breadth",
        "alpha_credit_spread", "excess_vol10"
    ]
    missing = [c for c in required if c not in x.columns]
    if qqq_col is None:
        missing.append("qqq_r1|qqq_ret_1d")
    if missing:
        raise ValueError(f"Canonical tape missing required alpha inputs: {missing}")

    z_tech = _expanding_zscore(x["alpha_tech_relative"], CFG.z_min_history)
    z_breadth = _expanding_zscore(x["alpha_breadth"], CFG.z_min_history)
    z_credit = _expanding_zscore(x["alpha_credit_spread"], CFG.z_min_history)
    z_qqq_mr = -_expanding_zscore(x[qqq_col], CFG.z_min_history)

    score_raw = (
        1.00 * z_tech +
        0.50 * z_breadth +
        1.00 * z_credit +
        0.50 * z_qqq_mr
    )
    score_smooth = score_raw.rolling(CFG.smooth_window, min_periods=1).mean()

    comp_df = pd.DataFrame({
        "tech": np.sign(z_tech),
        "breadth": np.sign(z_breadth),
        "credit": np.sign(z_credit),
        "qqq_mr": np.sign(z_qqq_mr),
    })
    pos_count = (comp_df > 0).sum(axis=1)
    neg_count = (comp_df < 0).sum(axis=1)
    nz_count = (comp_df != 0).sum(axis=1)
    component_agreement = np.where(nz_count > 0, np.maximum(pos_count, neg_count) / nz_count, 0.0)

    stale_days = int((pd.Timestamp.today().normalize() - pd.Timestamp(x["Date"].iloc[-1]).normalize()).days)
    not_stale = stale_days <= CFG.stale_days_max

    high_risk = pd.to_numeric(x.get("high_risk_state"), errors="coerce").fillna(0).astype(int) > 0
    dislocated = x["regime_label"].astype(str).str.lower().eq("dislocated")
    deploy_downside = pd.to_numeric(x.get("deploy_downside"), errors="coerce").fillna(0).astype(int) > 0
    drift_alarm = pd.to_numeric(x.get("drift_alarm"), errors="coerce").fillna(0).astype(int) > 0
    publish_fail_closed = pd.to_numeric(x.get("publish_fail_closed"), errors="coerce").fillna(0).astype(int) > 0

    defense_alpha_scale = pd.to_numeric(x.get("alpha_scale"), errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    dist_overlay_on = pd.to_numeric(x.get("dist_overlay_on_g53"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    dist_overlay_strength = pd.to_numeric(x.get("dist_overlay_strength_g53"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    dist_trust = pd.to_numeric(x.get("dist_trust_score_g53"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    dist_width_caution = pd.to_numeric(x.get("dist_width_caution_g53"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    uncertainty_penalty = pd.to_numeric(x.get("uncertainty_penalty_g5"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    caution_raw = (
        CFG.caution_shrink_overlay_w * dist_overlay_strength +
        CFG.caution_shrink_width_w * dist_width_caution +
        CFG.caution_shrink_unc_w * uncertainty_penalty
    ).clip(lower=0.0, upper=1.0)

    entry_z_eff = CFG.entry_z + CFG.overlay_entry_z_bump_max * caution_raw
    agreement_min_eff = (
        CFG.overlay_agreement_min_base + CFG.overlay_agreement_bump_max * caution_raw
    ).clip(lower=CFG.overlay_agreement_min_base, upper=CFG.overlay_agreement_min_cap)

    caution_shrink = (
        1.0 - (
            CFG.caution_shrink_overlay_w * dist_overlay_strength +
            CFG.caution_shrink_width_w * dist_width_caution +
            CFG.caution_shrink_unc_w * uncertainty_penalty
        )
    ).clip(lower=CFG.caution_shrink_floor, upper=1.0)

    score_ready = score_smooth.notna() & (score_smooth > entry_z_eff)
    agreement_ok = pd.Series(component_agreement, index=x.index) >= agreement_min_eff

    # Current part2a21 soft-caution overlay: only extreme and jointly bad uncertainty
    # conditions should zero the sleeve outright. Otherwise caution is handled
    # through entry tightening and gross-alpha shrink.
    overlay_hard_veto = (
        (
            (dist_width_caution >= CFG.overlay_hard_veto_width)
            & (uncertainty_penalty >= CFG.overlay_hard_veto_uncertainty)
            & (dist_overlay_on >= 1.0)
        )
        | publish_fail_closed
    )

    veto_off = ~(high_risk | dislocated | deploy_downside | overlay_hard_veto)
    eligible = score_ready & agreement_ok & veto_off & not_stale

    base_strength = ((score_smooth - entry_z_eff) / np.maximum(1e-9, (CFG.full_z - entry_z_eff))).clip(lower=0.0, upper=1.0)
    vol_shrink = 1.0 / (1.0 + CFG.vol_shrink_k * pd.to_numeric(x["excess_vol10"], errors="coerce").fillna(0.0).abs())
    vol_shrink = vol_shrink.clip(lower=0.10, upper=1.00)

    gov_tier = x["governance_tier"].astype(str).str.upper()
    gov_mult = np.where(
        gov_tier.eq("DEFENSIVE"), CFG.gov_mult_defensive,
        np.where(gov_tier.eq("CAUTION"), CFG.gov_mult_caution, CFG.gov_mult_normal)
    )
    gov_mult = pd.Series(gov_mult, index=x.index, dtype=float)

    drift_mult = pd.Series(np.where(drift_alarm, CFG.drift_mult, 1.00), index=x.index, dtype=float)
    alpha_scale_effective = (defense_alpha_scale * gov_mult * drift_mult * caution_shrink).clip(lower=0.0, upper=1.0)

    alpha_abs = CFG.max_abs_alpha * base_strength * vol_shrink * alpha_scale_effective
    alpha_abs = pd.to_numeric(alpha_abs, errors="coerce").fillna(0.0).clip(lower=0.0, upper=CFG.max_abs_alpha)

    alpha_position = np.where(eligible, alpha_abs, 0.0)
    alpha_leg = np.where(alpha_position > 0, "VOO", "FLAT")

    reasons = []
    for i in range(len(x)):
        if not not_stale:
            reasons.append("stale_tape")
        elif bool(overlay_hard_veto.iloc[i]):
            reasons.append("overlay_hard_veto")
        elif bool(high_risk.iloc[i]):
            reasons.append("high_risk_veto")
        elif bool(dislocated.iloc[i]):
            reasons.append("dislocated_veto")
        elif bool(deploy_downside.iloc[i]):
            reasons.append("deploy_downside_veto")
        elif not bool(score_ready.iloc[i]):
            reasons.append("weak_score")
        elif not bool(agreement_ok.iloc[i]):
            reasons.append("low_agreement")
        else:
            reasons.append("ok")

    positions_df = pd.DataFrame({
        "Date": x["Date"],
        "is_live": pd.to_numeric(x.get("is_live"), errors="coerce").fillna(0).astype(int),
        "prob_voo": pd.to_numeric(x.get("p_final_cal"), errors="coerce"),
        "p0": pd.to_numeric(x.get("p0"), errors="coerce"),
        "alpha_score_raw": pd.to_numeric(score_raw, errors="coerce").round(6),
        "alpha_score": pd.to_numeric(score_smooth, errors="coerce").round(6),
        "component_agreement": pd.to_numeric(component_agreement, errors="coerce").round(6),
        "entry_z_eff": pd.to_numeric(entry_z_eff, errors="coerce").round(6),
        "agreement_min_eff": pd.to_numeric(agreement_min_eff, errors="coerce").round(6),
        "signal_strength": pd.to_numeric(base_strength, errors="coerce").round(6),
        "defense_alpha_scale": pd.to_numeric(defense_alpha_scale, errors="coerce").round(6),
        "governance_mult": pd.to_numeric(gov_mult, errors="coerce").round(6),
        "caution_shrink": pd.to_numeric(caution_shrink, errors="coerce").round(6),
        "alpha_scale_effective": pd.to_numeric(alpha_scale_effective, errors="coerce").round(6),
        "excess_vol10": pd.to_numeric(x.get("excess_vol10"), errors="coerce").round(6),
        "vol_shrink": pd.to_numeric(vol_shrink, errors="coerce").round(6),
        "eligible": eligible.astype(int),
        "eligibility_reason": reasons,
        "alpha_leg": alpha_leg,
        "alpha_position": pd.to_numeric(alpha_position, errors="coerce").round(6),
        "alpha_abs": pd.to_numeric(np.abs(alpha_position), errors="coerce").round(6),
        "w_alpha_voo": pd.to_numeric(alpha_position, errors="coerce").round(6),
        "w_alpha_ief": 0.0,
        "dist_overlay_on_g53": pd.to_numeric(dist_overlay_on, errors="coerce").round(6),
        "dist_overlay_strength_g53": pd.to_numeric(dist_overlay_strength, errors="coerce").round(6),
        "dist_trust_score_g53": pd.to_numeric(dist_trust, errors="coerce").round(6),
        "dist_width_caution_g53": pd.to_numeric(dist_width_caution, errors="coerce").round(6),
        "uncertainty_penalty_g5": pd.to_numeric(uncertainty_penalty, errors="coerce").round(6),
        "prob_source": "part2a21_soft_caution_overlay",
        "tape_source": str(tape_source),
    })

    eligibility_df = positions_df[[
        "Date", "is_live", "eligible", "eligibility_reason",
        "prob_voo", "alpha_score", "component_agreement",
        "entry_z_eff", "agreement_min_eff",
        "signal_strength", "excess_vol10", "alpha_position",
        "dist_overlay_on_g53", "dist_overlay_strength_g53",
        "dist_width_caution_g53", "uncertainty_penalty_g5"
    ]].copy()

    if ("fwd_voo" not in x.columns) or ("fwd_ief" not in x.columns):
        raise ValueError("Canonical tape missing realized forward returns (fwd_voo / fwd_ief).")

    mature = pd.to_numeric(x.get("y_avail"), errors="coerce").fillna(0).astype(int)
    spread = pd.to_numeric(x["fwd_voo"], errors="coerce") - pd.to_numeric(x["fwd_ief"], errors="coerce")
    prev_alpha = pd.Series(alpha_position, index=x.index).shift(1).fillna(0.0)
    alpha_turnover = (pd.Series(alpha_position, index=x.index) - prev_alpha).abs()
    alpha_cost_model = CFG.turnover_cost_per_unit * alpha_turnover * 2.0  # round-trip consistency with Part 3

    rank_ic = np.where(
        (mature > 0) & (positions_df["eligible"] > 0) & (positions_df["alpha_abs"] > 0),
        np.where(np.sign(spread) == 0, 0.0, np.sign(spread)),
        np.nan,
    )

    topk_rel_ret_gross = np.where(
        (mature > 0) & (positions_df["eligible"] > 0) & (positions_df["alpha_abs"] > 0),
        pd.Series(alpha_position, index=x.index).abs() * spread,
        np.nan,
    )
    topk_rel_ret_net = np.where(
        np.isfinite(topk_rel_ret_gross),
        topk_rel_ret_gross - alpha_cost_model,
        np.nan,
    )

    drift_alarm_series = pd.to_numeric(x.get("drift_alarm"), errors="coerce")
    if drift_alarm_series.isna().all():
        drift_alarm_series = pd.to_numeric(x.get("publish_fail_closed"), errors="coerce")
    if drift_alarm_series.isna().all():
        drift_alarm_series = pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    else:
        drift_alarm_series = drift_alarm_series.fillna(0.0)

    alpha_overlay_scale = alpha_scale_effective * vol_shrink

    summary_tape = pd.DataFrame({
        "Date": x["Date"],
        "rank_ic": pd.to_numeric(rank_ic, errors="coerce").round(6),
        "topk_rel_ret_net": pd.to_numeric(topk_rel_ret_net, errors="coerce").round(6),
        "breadth_selected": ((positions_df["alpha_abs"] > 0) & (positions_df["eligible"] > 0)).astype(int),
        "eligible_breadth": positions_df["eligible"].astype(int),
        "gross_alpha_budget_used": positions_df["alpha_abs"].round(6),
        "alpha_drift_alarm": pd.to_numeric(drift_alarm_series, errors="coerce").round(6),
        "alpha_overlay_scale": pd.to_numeric(alpha_overlay_scale, errors="coerce").round(6),
        "is_live": positions_df["is_live"].astype(int),
        "alpha_leg": positions_df["alpha_leg"],
        "alpha_position": positions_df["alpha_position"],
        "alpha_abs": positions_df["alpha_abs"],
        "alpha_turnover": pd.to_numeric(alpha_turnover, errors="coerce").round(6),
        "alpha_cost_model": pd.to_numeric(alpha_cost_model, errors="coerce").round(6),
        "topk_rel_ret_gross": pd.to_numeric(topk_rel_ret_gross, errors="coerce").round(6),
        "realized_spread_voo_minus_ief": pd.to_numeric(spread, errors="coerce").round(6),
        "signal_spread_proxy": positions_df["alpha_position"],
        "prob_voo": positions_df["prob_voo"],
        "p0": positions_df["p0"],
        "eligible": positions_df["eligible"].astype(int),
        "eligibility_reason": positions_df["eligibility_reason"],
        "y_avail": mature.astype(int),
        "dist_overlay_on_g53": positions_df["dist_overlay_on_g53"],
        "dist_overlay_strength_g53": positions_df["dist_overlay_strength_g53"],
        "dist_width_caution_g53": positions_df["dist_width_caution_g53"],
        "uncertainty_penalty_g5": positions_df["uncertainty_penalty_g5"],
        "caution_shrink": positions_df["caution_shrink"],
        "alpha_scale_effective": positions_df["alpha_scale_effective"],
        "positions_source": CFG.fn_positions,
        "eligibility_source": CFG.fn_eligibility,
        "source_tape": str(tape_source),
    })

    mature_hist = summary_tape.dropna(subset=["rank_ic", "topk_rel_ret_net"]).copy()
    realized_dates = int(len(mature_hist))
    mean_rank_ic = float(mature_hist["rank_ic"].mean()) if realized_dates else np.nan
    mean_topk_rel_ret_net = float(mature_hist["topk_rel_ret_net"].mean()) if realized_dates else np.nan
    ir_topk_rel_ret_net = _annualized_ir(mature_hist["topk_rel_ret_net"], h_reb=1) if realized_dates else np.nan
    alpha_drift_alarm_rate = float(pd.to_numeric(summary_tape["alpha_drift_alarm"], errors="coerce").fillna(0).mean())

    latest = positions_df.iloc[-1]
    # Resolve publish_mode and final_pass from Part 2 summary JSON when
    # available.  The consensus tape carries publish_fail_closed (int) but
    # not a publish_mode string, so reading the row always returned "UNKNOWN".
    _p2 = part2_summary or {}
    _p2_publish_mode = str(_p2.get("publish_mode", "UNKNOWN")).strip().upper()
    if _p2_publish_mode not in {"NORMAL", "FAIL_CLOSED_NEUTRAL", "DEFENSE_ONLY"}:
        _p2_publish_mode = "UNKNOWN"
    _p2_final_pass = bool(_p2.get("final_pass", False))

    summary_payload = {
        "version": "GEN5_PART2A21_SOFT_CAUTION_V1",
        "built_at_utc": _now_utc_iso(),
        "final_pass": _p2_final_pass,
        "publish_mode": _p2_publish_mode,
        "source_tape": str(tape_source),
        "rows_total": int(len(positions_df)),
        "realized_dates": realized_dates,
        "first_date": str(pd.Timestamp(positions_df["Date"].iloc[0]).date()),
        "last_date": str(pd.Timestamp(positions_df["Date"].iloc[-1]).date()),
        "stale_days": stale_days,
        "tape_is_stale": bool(not not_stale),
        "mean_rank_ic": None if not np.isfinite(mean_rank_ic) else round(mean_rank_ic, 6),
        "mean_topk_rel_ret_net": None if not np.isfinite(mean_topk_rel_ret_net) else round(mean_topk_rel_ret_net, 6),
        "ir_topk_rel_ret_net": None if not np.isfinite(ir_topk_rel_ret_net) else round(ir_topk_rel_ret_net, 6),
        "alpha_drift_alarm_rate": round(alpha_drift_alarm_rate, 6),
        "eligible_rate": round(float(positions_df["eligible"].mean()), 6),
        "nonzero_alpha_rate": round(float((positions_df["alpha_abs"] > 0).mean()), 6),
        "mean_abs_alpha": round(float(positions_df["alpha_abs"].mean()), 6),
        "max_abs_alpha_seen": round(float(positions_df["alpha_abs"].max()), 6),
        "mean_caution_shrink": round(float(positions_df["caution_shrink"].mean()), 6),
        "mean_alpha_scale_effective": round(float(positions_df["alpha_scale_effective"].mean()), 6),
        "mean_dist_overlay_strength": round(float(positions_df["dist_overlay_strength_g53"].mean()), 6),
        "mean_dist_width_caution": round(float(positions_df["dist_width_caution_g53"].mean()), 6),
        "mean_uncertainty_penalty": round(float(positions_df["uncertainty_penalty_g5"].mean()), 6),
        "overlay_hard_veto_rate": round(float(overlay_hard_veto.mean()), 6),
        "high_risk_veto_rate": round(float(high_risk.mean()), 6),
        "deploy_downside_veto_rate": round(float(deploy_downside.mean()), 6),
        "latest_is_live": int(latest["is_live"]),
        "latest_alpha_leg": str(latest["alpha_leg"]),
        "latest_alpha_position": round(float(_safe_float(latest["alpha_position"], 0.0)), 6),
        "latest_alpha_abs": round(float(_safe_float(latest["alpha_abs"], 0.0)), 6),
        "latest_eligible": bool(int(latest["eligible"]) > 0),
        "latest_reason": str(latest["eligibility_reason"]),
        "latest_alpha_score": round(float(_safe_float(latest["alpha_score"], 0.0)), 6),
        "latest_component_agreement": round(float(_safe_float(latest["component_agreement"], 0.0)), 6),
        "latest_w_alpha_voo": round(float(_safe_float(latest["w_alpha_voo"], 0.0)), 6),
        "latest_w_alpha_ief": 0.0,
        "latest_dist_overlay_strength": round(float(_safe_float(latest["dist_overlay_strength_g53"], 0.0)), 6),
        "latest_dist_width_caution": round(float(_safe_float(latest["dist_width_caution_g53"], 0.0)), 6),
        "latest_uncertainty_penalty": round(float(_safe_float(latest["uncertainty_penalty_g5"], 0.0)), 6),
        "latest_alpha_scale_effective": round(float(_safe_float(latest["alpha_scale_effective"], 0.0)), 6),
    }

    summary_df = pd.DataFrame([summary_payload])
    return positions_df, eligibility_df, summary_df, summary_payload, summary_tape


def output_directories(tape_source: Path) -> List[Path]:
    primary = Path(CFG.out_primary)
    primary_pred = primary / CFG.out_predictions_subdir
    tape_parent = tape_source.parent

    dirs = [primary, primary_pred, tape_parent]
    out: List[Path] = []
    seen = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out


def write_artifacts(
    dirs: List[Path],
    positions_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_payload: Dict,
    summary_tape: pd.DataFrame,
) -> Dict[str, List[str]]:
    written: Dict[str, List[str]] = {
        "positions": [],
        "eligibility": [],
        "summary_csv": [],
        "summary_json": [],
        "summary_tape": [],
    }

    for d in dirs:
        p_positions = d / CFG.fn_positions
        p_elig = d / CFG.fn_eligibility
        p_summary_csv = d / CFG.fn_summary_csv
        p_summary_json = d / CFG.fn_summary_json
        p_summary_tape = d / CFG.fn_summary_tape

        _atomic_write_csv(positions_df, p_positions)
        _atomic_write_csv(eligibility_df, p_elig)
        _atomic_write_csv(summary_df, p_summary_csv)
        _atomic_write_text(p_summary_json, json.dumps(summary_payload, indent=2))
        _atomic_write_csv(summary_tape, p_summary_tape)

        written["positions"].append(str(p_positions))
        written["eligibility"].append(str(p_elig))
        written["summary_csv"].append(str(p_summary_csv))
        written["summary_json"].append(str(p_summary_json))
        written["summary_tape"].append(str(p_summary_tape))

        for alias in CFG.alias_positions:
            dst = d / alias
            _copy_alias(p_positions, dst)
            written["positions"].append(str(dst))
        for alias in CFG.alias_eligibility:
            dst = d / alias
            _copy_alias(p_elig, dst)
            written["eligibility"].append(str(dst))
        for alias in CFG.alias_summary_csv:
            dst = d / alias
            _copy_alias(p_summary_csv, dst)
            written["summary_csv"].append(str(dst))
        for alias in CFG.alias_summary_json:
            dst = d / alias
            _copy_alias(p_summary_json, dst)
            written["summary_json"].append(str(dst))
        for alias in CFG.alias_summary_tape:
            dst = d / alias
            _copy_alias(p_summary_tape, dst)
            written["summary_tape"].append(str(dst))

    return written


def main() -> None:
    root = Path(CFG.root)
    tape_source = discover_canonical_tape(root)
    print(f"✅ CANONICAL TAPE DISCOVERED: {tape_source}")

    tape_df = load_tape(tape_source)
    part2_summary = _read_part2_summary(tape_source)
    positions_df, eligibility_df, summary_df, summary_payload, summary_tape = build_alpha_positions(
        tape_df, tape_source, part2_summary=part2_summary
    )

    out_dirs = output_directories(tape_source)
    written = write_artifacts(
        dirs=out_dirs,
        positions_df=positions_df,
        eligibility_df=eligibility_df,
        summary_df=summary_df,
        summary_payload=summary_payload,
        summary_tape=summary_tape,
    )

    mature_hist = summary_tape.dropna(subset=["rank_ic", "topk_rel_ret_net"]).copy()
    latest = positions_df.iloc[-1]

    mean_rank_ic = float(mature_hist["rank_ic"].mean()) if len(mature_hist) else np.nan
    mean_topk_rel_ret_net = float(mature_hist["topk_rel_ret_net"].mean()) if len(mature_hist) else np.nan
    ir_topk_rel_ret_net = _annualized_ir(mature_hist["topk_rel_ret_net"], h_reb=1) if len(mature_hist) else np.nan
    alpha_drift_alarm_rate = float(pd.to_numeric(summary_tape["alpha_drift_alarm"], errors="coerce").fillna(0).mean())

    print("\n✅ PART 2A.21 SOFT-CAUTION ALPHA WRITTEN")
    print(f"Rows total:                 {len(positions_df)}")
    print(f"Realized dates:             {len(mature_hist)}")
    print(f"Eligible rate:              {positions_df['eligible'].mean():.6f}")
    print(f"Nonzero alpha rate:         {(positions_df['alpha_abs'] > 0).mean():.6f}")
    print(f"Mean abs alpha:             {positions_df['alpha_abs'].mean():.6f}")
    print(f"Mean caution_shrink:        {positions_df['caution_shrink'].mean():.6f}")
    print(f"Mean alpha_scale_effective: {positions_df['alpha_scale_effective'].mean():.6f}")
    print(f"Mean rank_ic:               {mean_rank_ic:.6f}" if np.isfinite(mean_rank_ic) else "Mean rank_ic:               nan")
    print(f"Mean topk_rel_ret_net:      {mean_topk_rel_ret_net:.6f}" if np.isfinite(mean_topk_rel_ret_net) else "Mean topk_rel_ret_net:      nan")
    print(f"IR topk_rel_ret_net:        {ir_topk_rel_ret_net:.6f}" if np.isfinite(ir_topk_rel_ret_net) else "IR topk_rel_ret_net:        nan")
    print(f"Alpha drift alarm rate:     {alpha_drift_alarm_rate:.6f}")
    print(f"Latest Date:                {pd.Timestamp(latest['Date']).date()}")
    print(f"Latest alpha_leg:           {latest['alpha_leg']}")
    print(f"Latest alpha_position:      {float(latest['alpha_position']):.6f}")
    print(f"Metadata final_pass:        {bool(summary_payload['final_pass'])}")

    print("\nArtifact paths:")
    for k, vals in written.items():
        print(f"- {k}:")
        for v in vals:
            print(f"    {v}")

    print("\nSummary JSON preview:")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()



