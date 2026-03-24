#!/usr/bin/env python3
from __future__ import annotations

# =============================================================================
# PROJECT: VOO vs IEF Tail-Risk + Shadow/Live Alpha Fusion
# VERSION: Part 3 V1
#
# PURPOSE:
# - Keep the defense sleeve live-safe and executable.
# - Add alpha promotion state machine:
#       SHADOW -> ELIGIBLE -> LIVE_TRIAL -> LIVE_FUSED
# - Add fusion engine that automatically activates when alpha is promoted.
# - Export clearer alpha-state diagnostics for Part 4:
#       alpha_state_display, alpha_quality_ok, alpha_drift_ok,
#       alpha_trial_gate_open, alpha_fused_gate_open,
#       alpha_promotion_ready, alpha_blocker_text, alpha_blockers_json
# - Preserve 7-day VOO / IEF forecast logging.
#
# INPUTS:
# - Defense tape from Part 2.2:
#     ./artifacts_part2_v77/predictions/v77_final_consensus_tape.csv
#
# - Alpha artifacts from Part 2A.2.1:
#     ./artifacts_part2a_alpha/predictions/part2a21_alpha_positions.csv
#     ./artifacts_part2a_alpha/predictions/part2a21_alpha_summary_tape.csv
#     ./artifacts_part2a_alpha/predictions/part2a21_alpha_eligibility.csv
#     ./artifacts_part2a_alpha/predictions/part2a21_alpha_summary.json
#
# OUTPUTS:
# - ./artifacts_part3_v1/v1_final_production_tape.csv
# - ./artifacts_part3_v1/v1_final_production_governance.csv
# - ./artifacts_part3_v1/v1_fusion_allocations.csv
# - ./artifacts_part3/prediction_log.csv
# =============================================================================

import os
import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ============================================================
# 0) Config
# ============================================================

@dataclass
class Part3V1Config:
    # -------------------------
    # Live safety
    # -------------------------
    LIVE_MODE: bool = True
    MAX_STALENESS_DAYS: int = 3
    H_REB: int = 7

    # -------------------------
    # Artifact paths
    # -------------------------
    DEFENSE_TAPE_CANDIDATES: tuple = (
        "./artifacts_part2_v77/predictions/v77_final_consensus_tape.csv",
        "./v77_final_consensus_tape.csv",
    )

    ALPHA_POS_CANDIDATES: tuple = (
        "./artifacts_part2a_alpha/predictions/part2a21_alpha_positions.csv",
        "./artifacts_part2a_alpha/predictions/part2a2_alpha_positions.csv",
    )

    ALPHA_SUM_CANDIDATES: tuple = (
        "./artifacts_part2a_alpha/predictions/part2a21_alpha_summary_tape.csv",
        "./artifacts_part2a_alpha/predictions/part2a2_alpha_summary_tape.csv",
    )

    ALPHA_ELIG_CANDIDATES: tuple = (
        "./artifacts_part2a_alpha/predictions/part2a21_alpha_eligibility.csv",
    )

    ALPHA_JSON_CANDIDATES: tuple = (
        "./artifacts_part2a_alpha/predictions/part2a21_alpha_summary.json",
        "./artifacts_part2a_alpha/predictions/part2a2_alpha_summary.json",
    )

    OUT_DIR: str = "./artifacts_part3_v1"
    PRED_LOG_PATH: str = "./artifacts_part3/prediction_log.csv"

    # -------------------------
    # Strategic base portfolio
    # -------------------------
    BASE_VOO_W: float = 0.60
    BASE_IEF_W: float = 0.40

    # -------------------------
    # Defense execution logic
    # -------------------------
    EPS_W: float = 0.05
    EPS_P: float = 0.05
    LEGS: int = 2

    SLIP_BASE_BPS: float = 5.0
    SLIP_STRESS_BPS: float = 15.0
    STRESS_VOL_Q: float = 0.80
    MIN_STRESS_HISTORY: int = 20

    KILL_MULT: float = 3.0
    KILL_ABS: float = 0.45
    K_DEF: float = 1.0

    # -------------------------
    # Defense governance
    # -------------------------
    ROLL_WINDOW: int = 52
    SIG_VAR_MIN: float = 1e-4
    SIG_VAR_WIN: int = 26
    MAX_KILL_RATE: float = 0.50
    MAX_TRADE_RATE: float = 0.80

    MAX_ANN_COST_DRAG: float = 0.015
    MAX_EFF_PERIOD_TURNOVER_AVG: float = 0.05

    DEF_PERSIST_K: int = 5
    FRIC_THROTTLE: float = 0.70
    PR_THROTTLE: float = 0.50
    DEF_THROTTLE: float = 0.50

    PR_LIFT_EPS: float = 0.10
    DEF_DES_MIN: float = 0.0
    STALE_EPS: float = 1e-6

    ZERO_K0: int = 3
    ZERO_R0: int = 2
    ZERO_LOW_TH: float = 1.0 / 3.0
    ZERO_HIGH_TH: float = 2.0 / 3.0
    ZERO_SCALE: float = 0.85

    TG_K: int = 3
    TG_LIFT_TH: float = 1.80
    TG_TURNEFF_TH: float = 0.01
    TG_ANNCOSTDRAG_TH: float = 0.0005
    TG_DES_MIN: float = 0.002

    # -------------------------
    # Alpha promotion / fusion
    # -------------------------
    AUTO_PROMOTION_ENABLED: bool = True

    ALPHA_MIN_REALIZED_ELIGIBLE: int = 26
    ALPHA_MIN_REALIZED_TRIAL: int = 52
    ALPHA_MIN_REALIZED_FUSED: int = 78

    ALPHA_MIN_MEAN_RANK_IC: float = 0.00
    ALPHA_MIN_MEAN_TOPK_RET: float = 0.00
    ALPHA_MIN_IR: float = 0.00
    ALPHA_MAX_DRIFT_RATE: float = 0.80

    ALPHA_PROMOTE_PERSIST_K: int = 3
    ALPHA_DEMOTE_PERSIST_K: int = 2

    ALPHA_TRIAL_BUDGET_MULT: float = 0.50
    ALPHA_FUSED_BUDGET_MULT: float = 1.00

    # Alpha is funded from the VOO sleeve
    ALPHA_MAX_GROSS_ABS: float = 0.10
    ALPHA_MAX_SHARE_OF_VOO: float = 0.50
    ALPHA_COST_BPS: float = 5.0

    # -------------------------
    # Misc
    # -------------------------
    BENCH_Q: float = 10.0


CFG = Part3V1Config()


# ============================================================
# 1) Low-level helpers
# ============================================================

def _safe_num(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _annualized_ir(x, H):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = np.nanstd(x, ddof=1)
    if (not np.isfinite(sd)) or (sd <= 0):
        return np.nan
    return float(np.nanmean(x) / sd * np.sqrt(252.0 / H))


def _read_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _expanding_rank(x):
    """
    Causal expanding empirical rank in [0,1].
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    hist = []
    for i, v in enumerate(x):
        hist.append(v)
        arr = np.asarray(hist, dtype=float)
        out[i] = float((arr <= v).mean())
    return out


def _latest_valid_date(df, col="Date"):
    if df is None or len(df) == 0 or col not in df.columns:
        return None
    x = pd.to_datetime(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return pd.Timestamp(x.max()).normalize()


def normalize_alpha_state_for_display(alpha_state: str) -> str:
    mapping = {
        "SHADOW": "SHADOW",
        "ELIGIBLE": "CANDIDATE",
        "LIVE_TRIAL": "LIVE_TRIAL",
        "LIVE_FUSED": "LIVE_FUSED",
        "UNAVAILABLE": "UNAVAILABLE",
    }
    return mapping.get(str(alpha_state), str(alpha_state))


def build_alpha_blocker_summary(
    realized_dates,
    mean_rank_ic,
    mean_topk_rel_ret_net,
    ir_topk_rel_ret_net,
    alpha_drift_alarm_rate,
    final_pass,
    cfg
):
    realized_dates = _safe_int(realized_dates, 0)
    mean_rank_ic = _safe_num(mean_rank_ic)
    mean_topk_rel_ret_net = _safe_num(mean_topk_rel_ret_net)
    ir_topk_rel_ret_net = _safe_num(ir_topk_rel_ret_net)
    alpha_drift_alarm_rate = _safe_num(alpha_drift_alarm_rate)

    final_pass_clean = None
    if pd.notna(final_pass):
        final_pass_clean = bool(final_pass)

    blockers = []

    quality_ok = (
        realized_dates >= int(cfg.ALPHA_MIN_REALIZED_ELIGIBLE) and
        np.isfinite(mean_rank_ic) and (mean_rank_ic > float(cfg.ALPHA_MIN_MEAN_RANK_IC)) and
        np.isfinite(mean_topk_rel_ret_net) and (mean_topk_rel_ret_net > float(cfg.ALPHA_MIN_MEAN_TOPK_RET)) and
        np.isfinite(ir_topk_rel_ret_net) and (ir_topk_rel_ret_net > float(cfg.ALPHA_MIN_IR))
    )

    drift_ok = (
        (not np.isfinite(alpha_drift_alarm_rate)) or
        (alpha_drift_alarm_rate <= float(cfg.ALPHA_MAX_DRIFT_RATE))
    )

    trial_gate_open = realized_dates >= int(cfg.ALPHA_MIN_REALIZED_TRIAL)
    fused_gate_open = realized_dates >= int(cfg.ALPHA_MIN_REALIZED_FUSED)

    if realized_dates < int(cfg.ALPHA_MIN_REALIZED_ELIGIBLE):
        blockers.append(
            f"insufficient_history_for_candidate:{realized_dates}<{cfg.ALPHA_MIN_REALIZED_ELIGIBLE}"
        )

    if (not np.isfinite(mean_rank_ic)) or (mean_rank_ic <= float(cfg.ALPHA_MIN_MEAN_RANK_IC)):
        blockers.append(f"rank_ic_not_positive:{mean_rank_ic}")

    if (not np.isfinite(mean_topk_rel_ret_net)) or (mean_topk_rel_ret_net <= float(cfg.ALPHA_MIN_MEAN_TOPK_RET)):
        blockers.append(f"topk_net_not_positive:{mean_topk_rel_ret_net}")

    if (not np.isfinite(ir_topk_rel_ret_net)) or (ir_topk_rel_ret_net <= float(cfg.ALPHA_MIN_IR)):
        blockers.append(f"ir_not_positive:{ir_topk_rel_ret_net}")

    if np.isfinite(alpha_drift_alarm_rate) and (alpha_drift_alarm_rate > float(cfg.ALPHA_MAX_DRIFT_RATE)):
        blockers.append(
            f"drift_too_high:{alpha_drift_alarm_rate:.4f}>{cfg.ALPHA_MAX_DRIFT_RATE:.4f}"
        )

    if final_pass_clean is False:
        blockers.append("alpha_summary_final_pass_false")

    blocker_text = "NONE" if len(blockers) == 0 else "; ".join(blockers)

    return {
        "alpha_quality_ok": int(quality_ok),
        "alpha_drift_ok": int(drift_ok),
        "alpha_trial_gate_open": int(trial_gate_open),
        "alpha_fused_gate_open": int(fused_gate_open),
        "alpha_promotion_ready": int(quality_ok and drift_ok and trial_gate_open),
        "alpha_blocker_count": int(len(blockers)),
        "alpha_blocker_text": blocker_text,
        "alpha_blockers_json": json.dumps(blockers),
    }


# ============================================================
# 2) Defense tape discovery / standardization
# ============================================================

def get_certified_defense_tape(cfg: Part3V1Config, path_override=None):
    required_base = ["Date", "p0", "excess_vol10"]
    prob_any = [["p_final_cal"], ["p_final_raw"], ["p_final"]]

    def _passes(head: pd.DataFrame) -> bool:
        if not all(c in head.columns for c in required_base):
            return False
        if not any(all(c in head.columns for c in cols) for cols in prob_any):
            return False
        return True

    if path_override is not None:
        if not os.path.exists(path_override):
            raise FileNotFoundError(f"Defense override path not found: {path_override}")
        head = pd.read_csv(path_override, nrows=1)
        if not _passes(head):
            raise ValueError("Defense override tape failed contract checks.")
        print(f"✅ DEFENSE TAPE DISCOVERED (override): {path_override}")
        return pd.read_csv(path_override), path_override

    p = _read_first_existing(cfg.DEFENSE_TAPE_CANDIDATES)
    if p is None:
        raise FileNotFoundError("No certified defense tape found.")

    head = pd.read_csv(p, nrows=1)
    if not _passes(head):
        raise ValueError("Discovered defense tape failed contract checks.")

    print(f"✅ DEFENSE TAPE DISCOVERED: {p}")
    return pd.read_csv(p), p


def standardize_defense_tape(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    for c in ["p0", "excess_vol10"]:
        if c not in df.columns:
            raise ValueError(f"Defense tape missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    if "p_final_cal" in df.columns:
        df["p_final_use"] = pd.to_numeric(df["p_final_cal"], errors="coerce").astype(float)
        df["p_final_src"] = "p_final_cal"
    elif "p_final" in df.columns:
        df["p_final_use"] = pd.to_numeric(df["p_final"], errors="coerce").astype(float)
        df["p_final_src"] = "p_final"
    elif "p_final_raw" in df.columns:
        df["p_final_use"] = pd.to_numeric(df["p_final_raw"], errors="coerce").astype(float)
        df["p_final_src"] = "p_final_raw"
    else:
        raise ValueError("Defense tape missing usable probability column.")

    if "alpha_scale" not in df.columns:
        df["alpha_scale"] = 1.0
    df["alpha_scale"] = (
        pd.to_numeric(df["alpha_scale"], errors="coerce")
        .fillna(1.0)
        .astype(float)
        .clip(0.0, 1.0)
    )

    if "governance_tier" not in df.columns:
        df["governance_tier"] = "NA"
    df["governance_tier"] = df["governance_tier"].astype(str)

    if "is_live" not in df.columns:
        df["is_live"] = 0
    df["is_live"] = pd.to_numeric(df["is_live"], errors="coerce").fillna(0).astype(int)

    num_cols = [
        "px_voo_t", "px_ief_t",
        "fwd_voo_hat", "fwd_ief_hat",
        "fwd_voo_hat_final", "fwd_ief_hat_final",
        "px_voo_call_7d", "px_ief_call_7d",
        "y_voo", "fwd_voo", "fwd_ief", "bench_ret", "excess_ret",
        "ece_roll", "brier_roll", "ece_avail_roll", "brier_avail_roll"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "bench_ret" not in df.columns:
        if "excess_ret" in df.columns:
            df["bench_ret"] = pd.to_numeric(df["excess_ret"], errors="coerce")
        elif {"fwd_voo", "fwd_ief"}.issubset(df.columns):
            df["bench_ret"] = (
                pd.to_numeric(df["fwd_voo"], errors="coerce") -
                pd.to_numeric(df["fwd_ief"], errors="coerce")
            )
        else:
            df["bench_ret"] = np.nan

    return df


def _pick_decision_row(df: pd.DataFrame, cfg: Part3V1Config) -> pd.Series:
    if "is_live" in df.columns and (df["is_live"] == 1).any():
        row = df.loc[df["is_live"] == 1].iloc[-1]
    else:
        row = df.iloc[-1]

    today_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    row_date = pd.Timestamp(row["Date"]).normalize()

    age_days = int((today_utc - row_date).days)

    if cfg.LIVE_MODE and age_days > cfg.MAX_STALENESS_DAYS:
        raise RuntimeError(
            f"STALE TAPE REFUSED: last Date={row_date.date()} age={age_days}d > MAX_STALENESS_DAYS={cfg.MAX_STALENESS_DAYS}"
        )

    if (not cfg.LIVE_MODE) and age_days > cfg.MAX_STALENESS_DAYS:
        print(f"[WARN] Defense tape is stale. last={row_date.date()} age={age_days}d")

    return row


# ============================================================
# 3) Alpha artifact discovery / standardization
# ============================================================

def read_alpha_artifacts(cfg: Part3V1Config):
    pos_path = _read_first_existing(cfg.ALPHA_POS_CANDIDATES)
    sum_path = _read_first_existing(cfg.ALPHA_SUM_CANDIDATES)
    elig_path = _read_first_existing(cfg.ALPHA_ELIG_CANDIDATES)
    json_path = _read_first_existing(cfg.ALPHA_JSON_CANDIDATES)

    alpha_pos = pd.read_csv(pos_path) if pos_path else None
    alpha_sum = pd.read_csv(sum_path) if sum_path else None
    alpha_elig = pd.read_csv(elig_path) if elig_path else None

    alpha_json = None
    if json_path:
        with open(json_path, "r") as f:
            alpha_json = json.load(f)

    if pos_path:
        print(f"✅ ALPHA POSITIONS DISCOVERED: {pos_path}")
    else:
        print("[WARN] Alpha positions not found.")

    if sum_path:
        print(f"✅ ALPHA SUMMARY DISCOVERED: {sum_path}")
    else:
        print("[WARN] Alpha summary not found.")

    if elig_path:
        print(f"✅ ALPHA ELIGIBILITY DISCOVERED: {elig_path}")
    else:
        print("[WARN] Alpha eligibility not found.")

    if json_path:
        print(f"✅ ALPHA SUMMARY JSON DISCOVERED: {json_path}")
    else:
        print("[WARN] Alpha summary JSON not found.")

    return alpha_pos, alpha_sum, alpha_elig, alpha_json, pos_path, sum_path, elig_path, json_path


def standardize_alpha_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None

    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["Date"]).sort_values(
        ["Date", "Ticker"] if "Ticker" in x.columns else ["Date"]
    ).reset_index(drop=True)

    num_cols = [
        "weight", "weight_raw", "selected", "alpha_score",
        "fwd_ret", "px_t", "is_live"
    ]
    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x


def standardize_alpha_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None

    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    num_cols = [
        "rank_ic", "topk_rel_ret_net", "breadth_selected", "eligible_breadth",
        "gross_alpha_budget_used", "alpha_drift_alarm", "alpha_overlay_scale", "is_live"
    ]
    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x


def standardize_alpha_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None

    x = df.copy()
    if "asof_date" in x.columns:
        x["asof_date"] = pd.to_datetime(x["asof_date"], errors="coerce").dt.normalize()
        x = x.dropna(subset=["asof_date"]).sort_values(
            ["asof_date", "Ticker"] if "Ticker" in x.columns else ["asof_date"]
        ).reset_index(drop=True)

    num_cols = ["eligible", "in_elig_set", "n_obs", "hit_rate", "mean_alpha_target", "mean_score"]
    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    return x


# ============================================================
# 4) Forecast helper + prediction log
# ============================================================

def price_call_from_row(row: pd.Series, px_col: str, hat_final_col: str, hat_col: str, call_col: str):
    call = _safe_num(row[call_col]) if call_col in row.index else np.nan
    if np.isfinite(call):
        return float(call), "explicit_call"

    px = _safe_num(row[px_col]) if px_col in row.index else np.nan
    if not np.isfinite(px):
        return np.nan, "missing_px"

    hat_f = _safe_num(row[hat_final_col]) if hat_final_col in row.index else np.nan
    if np.isfinite(hat_f):
        return float(px * np.exp(hat_f)), "hat_final"

    hat = _safe_num(row[hat_col]) if hat_col in row.index else np.nan
    if np.isfinite(hat):
        return float(px * np.exp(hat)), "hat_raw"

    return np.nan, "missing_hat"


def _idx_of_date(df: pd.DataFrame, dt) -> int:
    m = pd.to_datetime(df["Date"]).dt.normalize() == pd.Timestamp(dt).normalize()
    if not m.any():
        return None
    return int(np.where(m.values)[0][-1])


def append_and_update_prediction_log(df0: pd.DataFrame, decision_row: pd.Series, cfg: Part3V1Config):
    os.makedirs(os.path.dirname(cfg.PRED_LOG_PATH), exist_ok=True)

    df0 = df0.copy()
    df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce").dt.normalize()

    decision_date = pd.Timestamp(decision_row["Date"]).normalize()

    p_final_cal = _safe_num(decision_row["p_final_cal"]) if "p_final_cal" in decision_row.index else np.nan
    p0 = _safe_num(decision_row["p0"]) if "p0" in decision_row.index else np.nan
    alpha_scale = _safe_num(decision_row["alpha_scale"]) if "alpha_scale" in decision_row.index else np.nan
    gov_tier = str(decision_row["governance_tier"]) if "governance_tier" in decision_row.index else "NA"

    px_voo_t = _safe_num(decision_row["px_voo_t"]) if "px_voo_t" in decision_row.index else np.nan
    px_ief_t = _safe_num(decision_row["px_ief_t"]) if "px_ief_t" in decision_row.index else np.nan

    px_voo_call, voo_call_src = price_call_from_row(
        decision_row, "px_voo_t", "fwd_voo_hat_final", "fwd_voo_hat", "px_voo_call_7d"
    )
    px_ief_call, ief_call_src = price_call_from_row(
        decision_row, "px_ief_t", "fwd_ief_hat_final", "fwd_ief_hat", "px_ief_call_7d"
    )

    if os.path.exists(cfg.PRED_LOG_PATH):
        log = pd.read_csv(cfg.PRED_LOG_PATH)
        log["decision_date"] = pd.to_datetime(log["decision_date"], errors="coerce").dt.normalize()
    else:
        log = pd.DataFrame(columns=[
            "decision_date", "h_reb",
            "p_final_cal", "p0", "alpha_scale", "governance_tier",
            "px_voo_t", "px_ief_t", "px_voo_call_7d", "px_ief_call_7d",
            "voo_call_src", "ief_call_src",
            "target_date", "px_voo_realized", "px_ief_realized",
            "voo_err", "ief_err", "voo_abs_err", "ief_abs_err", "voo_ape", "ief_ape"
        ])

    payload = {
        "decision_date": decision_date,
        "h_reb": int(cfg.H_REB),
        "p_final_cal": p_final_cal,
        "p0": p0,
        "alpha_scale": alpha_scale,
        "governance_tier": gov_tier,
        "px_voo_t": px_voo_t,
        "px_ief_t": px_ief_t,
        "px_voo_call_7d": px_voo_call,
        "px_ief_call_7d": px_ief_call,
        "voo_call_src": voo_call_src,
        "ief_call_src": ief_call_src,
    }

    m = log["decision_date"] == decision_date
    if m.any():
        i0 = int(np.where(m.values)[0][-1])
        for k, v in payload.items():
            log.loc[i0, k] = v
    else:
        payload.update({
            "target_date": pd.NaT,
            "px_voo_realized": np.nan,
            "px_ief_realized": np.nan,
            "voo_err": np.nan,
            "ief_err": np.nan,
            "voo_abs_err": np.nan,
            "ief_abs_err": np.nan,
            "voo_ape": np.nan,
            "ief_ape": np.nan,
        })
        log = pd.concat([log, pd.DataFrame([payload])], ignore_index=True)

    for i in range(len(log)):
        if pd.notna(log.loc[i, "px_voo_realized"]) and pd.notna(log.loc[i, "px_ief_realized"]):
            continue

        d0 = pd.Timestamp(log.loc[i, "decision_date"]).normalize()
        idx0 = _idx_of_date(df0, d0)
        if idx0 is None:
            continue

        idxT = idx0 + int(log.loc[i, "h_reb"])
        if idxT >= len(df0):
            continue

        target_date = pd.Timestamp(df0.loc[idxT, "Date"]).normalize()
        px_voo_real = _safe_num(df0.loc[idxT, "px_voo_t"]) if "px_voo_t" in df0.columns else np.nan
        px_ief_real = _safe_num(df0.loc[idxT, "px_ief_t"]) if "px_ief_t" in df0.columns else np.nan

        log.loc[i, "target_date"] = target_date
        log.loc[i, "px_voo_realized"] = px_voo_real
        log.loc[i, "px_ief_realized"] = px_ief_real

        pv = _safe_num(log.loc[i, "px_voo_call_7d"])
        pi = _safe_num(log.loc[i, "px_ief_call_7d"])

        if np.isfinite(pv) and np.isfinite(px_voo_real):
            err = px_voo_real - pv
            log.loc[i, "voo_err"] = err
            log.loc[i, "voo_abs_err"] = abs(err)
            log.loc[i, "voo_ape"] = abs(err) / (abs(px_voo_real) + 1e-12)

        if np.isfinite(pi) and np.isfinite(px_ief_real):
            err = px_ief_real - pi
            log.loc[i, "ief_err"] = err
            log.loc[i, "ief_abs_err"] = abs(err)
            log.loc[i, "ief_ape"] = abs(err) / (abs(px_ief_real) + 1e-12)

    log = log.sort_values("decision_date").reset_index(drop=True)
    log.to_csv(cfg.PRED_LOG_PATH, index=False)

    realized = log.dropna(subset=["px_voo_realized", "px_ief_realized"])
    if len(realized) > 0:
        voo_mae = float(realized["voo_abs_err"].mean())
        ief_mae = float(realized["ief_abs_err"].mean())
        voo_mape = float(100.0 * realized["voo_ape"].mean())
        ief_mape = float(100.0 * realized["ief_ape"].mean())
        print(f"[PredLog] realized rows={len(realized)} | VOO MAE={voo_mae:.4f} ({voo_mape:.2f}%) | IEF MAE={ief_mae:.4f} ({ief_mape:.2f}%)")
    else:
        print("[PredLog] realized rows=0 (not enough tape history yet).")

    print(f"[PredLog] path: {cfg.PRED_LOG_PATH}")


# ============================================================
# 5) Defense execution + governance
# ============================================================

def run_defense_execution(df_in: pd.DataFrame, cfg: Part3V1Config) -> pd.DataFrame:
    df = df_in.copy()

    p = df["p_final_use"].values.astype(float)
    p0 = df["p0"].values.astype(float)

    p0_floor = np.zeros(len(df), dtype=float)
    p0_floor[0] = np.nanmedian(p0[: min(len(p0), 20)]) if len(p0) else 0.20
    if not np.isfinite(p0_floor[0]):
        p0_floor[0] = 0.20

    for i in range(1, len(df)):
        p0_floor[i] = np.nanmedian(p0[:i])

    p0_eff = np.maximum(p0, p0_floor)

    vol = df["excess_vol10"].values.astype(float)
    vol_rank = _expanding_rank(vol)

    w_exec = np.zeros(len(df), dtype=float)
    cost_exec = np.zeros(len(df), dtype=float)

    kill_flag = np.zeros(len(df), dtype=int)
    trade_flag = np.zeros(len(df), dtype=int)
    trade_size = np.zeros(len(df), dtype=float)
    target_w_raw = np.zeros(len(df), dtype=float)
    target_w_postkill = np.zeros(len(df), dtype=float)

    tw0 = np.clip(1.0 - cfg.K_DEF * ((p[0] - p0_eff[0]) / (p0_eff[0] + 1e-12)), 0.0, 1.0)
    k0 = (p[0] > cfg.KILL_MULT * p0_eff[0]) or (p[0] > cfg.KILL_ABS)

    target_w_raw[0] = tw0
    target_w_postkill[0] = 0.0 if k0 else tw0
    kill_flag[0] = int(k0)
    w_exec[0] = target_w_postkill[0]

    cur_w = w_exec[0]
    cur_p = p[0]

    for i in range(1, len(df)):
        tw = np.clip(1.0 - cfg.K_DEF * ((p[i] - p0_eff[i]) / (p0_eff[i] + 1e-12)), 0.0, 1.0)
        k = (p[i] > cfg.KILL_MULT * p0_eff[i]) or (p[i] > cfg.KILL_ABS)
        tw_pk = 0.0 if k else tw

        target_w_raw[i] = tw
        target_w_postkill[i] = tw_pk
        kill_flag[i] = int(k)

        do_trade = bool(k) or (abs(tw_pk - cur_w) > cfg.EPS_W and abs(p[i] - cur_p) > cfg.EPS_P)
        if do_trade:
            trade = abs(tw_pk - cur_w)
            slip = cfg.SLIP_STRESS_BPS if (i >= cfg.MIN_STRESS_HISTORY and vol_rank[i] >= cfg.STRESS_VOL_Q) else cfg.SLIP_BASE_BPS
            cost_exec[i] = (slip / 10000.0) * trade * cfg.LEGS
            trade_flag[i] = 1
            trade_size[i] = trade
            cur_w, cur_p = tw_pk, p[i]

        w_exec[i] = cur_w

    df["p0_floor_val"] = p0_floor
    df["p0_eff"] = p0_eff
    df["vol_rank"] = vol_rank

    df["w_exec"] = w_exec
    df["cost_exec"] = cost_exec
    df["kill_flag"] = kill_flag
    df["trade_flag"] = trade_flag
    df["trade_size"] = trade_size
    df["target_w_raw"] = target_w_raw
    df["target_w_postkill"] = target_w_postkill

    return df


def compute_live_proxy_health(df: pd.DataFrame, cfg: Part3V1Config) -> pd.DataFrame:
    out = df.copy()

    base_scale = (
        pd.to_numeric(out["alpha_scale"], errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
        .values.astype(float)
    )

    dw = out["w_exec"].diff().abs().fillna(0.0).values.astype(float)

    turn_eff_inst = cfg.BASE_VOO_W * base_scale * dw
    ann_cost_drag_inst = (
        cfg.BASE_VOO_W * base_scale * out["cost_exec"].values.astype(float)
    ) * (252.0 / cfg.H_REB)

    out["turn_eff_inst"] = turn_eff_inst
    out["turn_eff_roll"] = pd.Series(turn_eff_inst).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    out["ann_cost_drag_inst"] = ann_cost_drag_inst
    out["ann_cost_drag_roll"] = pd.Series(ann_cost_drag_inst).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    pv = pd.Series(out["p_final_use"].values.astype(float)).rolling(
        cfg.SIG_VAR_WIN, min_periods=cfg.SIG_VAR_WIN
    ).var()
    out["p_final_var_roll"] = pv.values

    out["kill_rate_roll"] = pd.Series(out["kill_flag"]).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values
    out["trade_rate_roll"] = pd.Series(out["trade_flag"]).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    fric_ok = (
        (out["turn_eff_roll"].values < cfg.MAX_EFF_PERIOD_TURNOVER_AVG) &
        (out["ann_cost_drag_roll"].values < cfg.MAX_ANN_COST_DRAG)
    )
    sig_ok = np.where(np.isnan(out["p_final_var_roll"].values), True, out["p_final_var_roll"].values >= cfg.SIG_VAR_MIN)
    regime_ok = (
        (out["kill_rate_roll"].values <= cfg.MAX_KILL_RATE) &
        (out["trade_rate_roll"].values <= cfg.MAX_TRADE_RATE)
    )

    out["GovHealthScore_live"] = (fric_ok.astype(float) + sig_ok.astype(float) + regime_ok.astype(float)) / 3.0
    out["live_fric_fail_now"] = (~fric_ok).astype(int)

    return out


def realized_data_available(df: pd.DataFrame) -> bool:
    need = ["y_voo", "fwd_voo", "fwd_ief", "bench_ret"]
    if not all(c in df.columns for c in need):
        return False
    return df[need].notna().any().all()


def build_defense_governance(df_exec: pd.DataFrame, cfg: Part3V1Config):
    df = compute_live_proxy_health(df_exec.copy(), cfg)
    n = len(df)

    has_realized = realized_data_available(df)

    df["GovMode"] = "ACTIVE"
    df["GovReason"] = "OK"

    df["PR_Fails_Streak"] = 0
    df["FRIC_Fails_Streak"] = 0
    df["DEF_Fails_Streak"] = 0
    df["STALE_Streak"] = 0
    df["TooGood_Streak"] = 0

    df["PR_Soft"] = 0
    df["FRIC_Soft"] = 0
    df["DEF_Soft"] = 0
    df["STALE_Mode"] = 0

    df["Lift_End"] = np.nan
    df["dES_End"] = np.nan
    df["TailN_End"] = np.nan
    df["GovHealthScore_realized"] = np.nan

    df["ZeRO_Active_End"] = 0
    df["ZeRO_Scale_End"] = 1.0
    df["TooGood_Alarm_End"] = 0

    df["gov_scale"] = 1.0

    pr_fails = 0
    fric_fails = 0
    def_fails = 0
    stale_fails = 0
    tg_streak = 0

    z_active = False
    z_low = 0
    z_high = 0

    for t in range(n):
        fric_fail = bool(df["live_fric_fail_now"].iloc[t])
        fric_fails = fric_fails + 1 if fric_fail else 0
        df.at[t, "FRIC_Fails_Streak"] = fric_fails

        if has_realized and (t >= cfg.ROLL_WINDOW - 1):
            sl = df.iloc[t - cfg.ROLL_WINDOW + 1: t + 1].copy()

            y = sl["y_voo"].values.astype(float)
            p = sl["p_final_use"].values.astype(float)
            mr = np.isfinite(y) & np.isfinite(p)

            lift = np.nan
            if mr.sum() >= 10 and len(np.unique(y[mr].astype(int))) > 1:
                yv = y[mr].astype(int)
                pv = p[mr]
                prev = yv.mean()
                pr = average_precision_score(yv, pv)
                lift = pr / prev if prev > 0 else np.nan
            df.at[t, "Lift_End"] = lift

            bench = sl["bench_ret"].values.astype(float)
            strat = sl["w_exec"].values.astype(float) * bench - sl["cost_exec"].values.astype(float)

            mb = np.isfinite(bench) & np.isfinite(strat)
            dES = np.nan
            tail_n = np.nan

            if mb.sum() >= 10:
                qv = np.percentile(bench[mb], cfg.BENCH_Q)
                mask = mb & (bench <= qv)
                if mask.sum() >= 3:
                    dES = float(np.mean(strat[mask]) - np.mean(bench[mask]))
                    tail_n = int(mask.sum())

            df.at[t, "dES_End"] = dES
            df.at[t, "TailN_End"] = tail_n

            pr_fail = np.isfinite(lift) and (lift < (1.0 - cfg.PR_LIFT_EPS))
            def_fail = np.isfinite(dES) and (dES < cfg.DEF_DES_MIN)
            st_now = np.isfinite(dES) and (abs(dES) < cfg.STALE_EPS)

            pr_fails = pr_fails + 1 if pr_fail else 0
            def_fails = def_fails + 1 if def_fail else 0
            stale_fails = stale_fails + 1 if st_now else 0

            df.at[t, "PR_Fails_Streak"] = pr_fails
            df.at[t, "DEF_Fails_Streak"] = def_fails
            df.at[t, "STALE_Streak"] = stale_fails

            hc = []
            hc.append(1.0 if (np.isfinite(lift) and lift > 1.0) else 0.0)
            hc.append(1.0 if (np.isfinite(dES) and dES > 0.0) else 0.0)
            hc.append(1.0 if (df["turn_eff_roll"].iloc[t] < cfg.MAX_EFF_PERIOD_TURNOVER_AVG) else 0.0)
            df.at[t, "GovHealthScore_realized"] = float(np.mean(hc))

            tg_now = (
                np.isfinite(lift) and (lift >= cfg.TG_LIFT_TH) and
                (df["turn_eff_roll"].iloc[t] <= cfg.TG_TURNEFF_TH) and
                (df["ann_cost_drag_roll"].iloc[t] <= cfg.TG_ANNCOSTDRAG_TH) and
                np.isfinite(dES) and (dES >= cfg.TG_DES_MIN)
            )
            tg_streak = tg_streak + 1 if tg_now else 0
            df.at[t, "TooGood_Streak"] = tg_streak
            df.at[t, "TooGood_Alarm_End"] = 1 if (tg_streak >= cfg.TG_K) else 0

            stale_mode = stale_fails >= cfg.DEF_PERSIST_K
            pr_soft = (pr_fails >= cfg.DEF_PERSIST_K) and (not stale_mode)
            fric_soft = fric_fails >= cfg.DEF_PERSIST_K
            def_soft = (def_fails >= cfg.DEF_PERSIST_K) and (not stale_mode)

            df.at[t, "STALE_Mode"] = int(stale_mode)
            df.at[t, "PR_Soft"] = int(pr_soft)
            df.at[t, "FRIC_Soft"] = int(fric_soft)
            df.at[t, "DEF_Soft"] = int(def_soft)

            scale = 1.0
            reasons = []

            if stale_mode:
                df.at[t, "GovMode"] = "INERT"
                reasons.append("STALE")
            else:
                df.at[t, "GovMode"] = "ACTIVE"

            if pr_soft:
                scale = min(scale, cfg.PR_THROTTLE)
                reasons.append("PR_THROTTLE")

            if fric_soft:
                scale = min(scale, cfg.FRIC_THROTTLE)
                reasons.append("FRIC_THROTTLE")

            if def_soft:
                scale = min(scale, cfg.DEF_THROTTLE)
                reasons.append("DEF_THROTTLE")

            df.at[t, "gov_scale"] = scale
            df.at[t, "GovReason"] = "OK" if len(reasons) == 0 else "|".join(reasons)

        else:
            fric_soft = fric_fails >= cfg.DEF_PERSIST_K
            df.at[t, "FRIC_Soft"] = int(fric_soft)
            df.at[t, "gov_scale"] = cfg.FRIC_THROTTLE if fric_soft else 1.0
            df.at[t, "GovMode"] = "ACTIVE"
            df.at[t, "GovReason"] = "FRIC_THROTTLE" if fric_soft else "OK"

        ghs = _safe_num(df["GovHealthScore_realized"].iloc[t])
        if not np.isfinite(ghs):
            ghs = _safe_num(df["GovHealthScore_live"].iloc[t])

        if not z_active:
            z_low = (z_low + 1) if (ghs <= cfg.ZERO_LOW_TH) else 0
            if z_low >= cfg.ZERO_K0:
                z_active = True
                z_high = 0
        else:
            z_high = (z_high + 1) if (ghs >= cfg.ZERO_HIGH_TH) else 0
            if z_high >= cfg.ZERO_R0:
                z_active = False
                z_low = 0
                z_high = 0

        df.at[t, "ZeRO_Active_End"] = int(z_active)
        df.at[t, "ZeRO_Scale_End"] = cfg.ZERO_SCALE if z_active else 1.0

    base_scale = (
        pd.to_numeric(df["alpha_scale"], errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
        .values.astype(float)
    )

    df["defense_scale_preZero"] = np.clip(
        base_scale * df["gov_scale"].values.astype(float),
        0.0, 1.0
    )

    df["defense_scale_final"] = np.clip(
        df["defense_scale_preZero"].values.astype(float) *
        df["ZeRO_Scale_End"].values.astype(float),
        0.0, 1.0
    )

    gov_df = df.loc[:, [
        "Date", "GovMode", "GovReason", "gov_scale",
        "defense_scale_preZero", "defense_scale_final",
        "GovHealthScore_live", "GovHealthScore_realized",
        "Lift_End", "dES_End", "TailN_End",
        "PR_Fails_Streak", "FRIC_Fails_Streak", "DEF_Fails_Streak", "STALE_Streak",
        "PR_Soft", "FRIC_Soft", "DEF_Soft", "STALE_Mode",
        "TooGood_Streak", "TooGood_Alarm_End",
        "ZeRO_Active_End", "ZeRO_Scale_End",
        "turn_eff_roll", "ann_cost_drag_roll"
    ]].copy()

    return df, gov_df


# ============================================================
# 6) Alpha promotion state machine
# ============================================================

def build_alpha_state_history(alpha_sum: pd.DataFrame, alpha_json, cfg: Part3V1Config):
    if alpha_sum is None or len(alpha_sum) == 0:
        return pd.DataFrame(columns=[
            "Date", "realized_dates", "mean_rank_ic", "mean_topk_rel_ret_net",
            "ir_topk_rel_ret_net", "alpha_drift_alarm_rate",
            "quality_ok", "drift_ok", "alpha_state", "alpha_state_display",
            "alpha_live", "fusion_live", "budget_mult",
            "alpha_promo_good_streak", "alpha_promo_bad_streak", "alpha_final_pass_json",
            "alpha_quality_ok", "alpha_drift_ok", "alpha_trial_gate_open",
            "alpha_fused_gate_open", "alpha_promotion_ready",
            "alpha_blocker_count", "alpha_blocker_text", "alpha_blockers_json"
        ])

    s = alpha_sum.copy().sort_values("Date").reset_index(drop=True)

    rows = []
    state = "SHADOW"
    good_streak = 0
    bad_streak = 0

    final_pass_json = None
    if isinstance(alpha_json, dict):
        final_pass_json = alpha_json.get("final_pass", None)

    for i in range(len(s)):
        sl = s.iloc[: i + 1].copy()
        sr = sl.dropna(subset=["rank_ic", "topk_rel_ret_net"]).copy()

        realized_dates = int(len(sr))
        mean_ic = float(sr["rank_ic"].mean()) if len(sr) else np.nan
        mean_topk = float(sr["topk_rel_ret_net"].mean()) if len(sr) else np.nan
        ir_topk = _annualized_ir(sr["topk_rel_ret_net"].values, cfg.H_REB) if len(sr) else np.nan

        if "alpha_drift_alarm" in sl.columns:
            drift_rate = float(pd.to_numeric(sl["alpha_drift_alarm"], errors="coerce").fillna(0).mean())
        else:
            drift_rate = np.nan

        quality_ok = (
            realized_dates >= cfg.ALPHA_MIN_REALIZED_ELIGIBLE and
            np.isfinite(mean_ic) and (mean_ic > cfg.ALPHA_MIN_MEAN_RANK_IC) and
            np.isfinite(mean_topk) and (mean_topk > cfg.ALPHA_MIN_MEAN_TOPK_RET) and
            np.isfinite(ir_topk) and (ir_topk > cfg.ALPHA_MIN_IR)
        )

        drift_ok = (not np.isfinite(drift_rate)) or (drift_rate <= cfg.ALPHA_MAX_DRIFT_RATE)

        if cfg.AUTO_PROMOTION_ENABLED and quality_ok and drift_ok:
            good_streak += 1
            bad_streak = 0
        else:
            bad_streak += 1
            good_streak = 0

        if not cfg.AUTO_PROMOTION_ENABLED:
            state = "SHADOW"

        else:
            if state in ["LIVE_FUSED", "LIVE_TRIAL"]:
                if bad_streak >= cfg.ALPHA_DEMOTE_PERSIST_K:
                    state = "ELIGIBLE" if quality_ok else "SHADOW"

            if state in ["SHADOW", "ELIGIBLE"]:
                if quality_ok:
                    state = "ELIGIBLE"

                if (
                    realized_dates >= cfg.ALPHA_MIN_REALIZED_TRIAL and
                    quality_ok and drift_ok and
                    good_streak >= cfg.ALPHA_PROMOTE_PERSIST_K
                ):
                    state = "LIVE_TRIAL"

            if state in ["LIVE_TRIAL", "LIVE_FUSED"]:
                if (
                    realized_dates >= cfg.ALPHA_MIN_REALIZED_FUSED and
                    quality_ok and drift_ok and
                    good_streak >= cfg.ALPHA_PROMOTE_PERSIST_K
                ):
                    state = "LIVE_FUSED"

        alpha_live = state in ["LIVE_TRIAL", "LIVE_FUSED"]
        fusion_live = state in ["LIVE_TRIAL", "LIVE_FUSED"]

        if state == "LIVE_TRIAL":
            budget_mult = cfg.ALPHA_TRIAL_BUDGET_MULT
        elif state == "LIVE_FUSED":
            budget_mult = cfg.ALPHA_FUSED_BUDGET_MULT
        else:
            budget_mult = 0.0

        alpha_diag = build_alpha_blocker_summary(
            realized_dates=realized_dates,
            mean_rank_ic=mean_ic,
            mean_topk_rel_ret_net=mean_topk,
            ir_topk_rel_ret_net=ir_topk,
            alpha_drift_alarm_rate=drift_rate,
            final_pass=final_pass_json,
            cfg=cfg
        )

        rows.append({
            "Date": pd.Timestamp(s.loc[i, "Date"]).normalize(),
            "realized_dates": realized_dates,
            "mean_rank_ic": mean_ic,
            "mean_topk_rel_ret_net": mean_topk,
            "ir_topk_rel_ret_net": ir_topk,
            "alpha_drift_alarm_rate": drift_rate,
            "quality_ok": int(quality_ok),
            "drift_ok": int(drift_ok),
            "alpha_state": state,
            "alpha_state_display": normalize_alpha_state_for_display(state),
            "alpha_live": int(alpha_live),
            "fusion_live": int(fusion_live),
            "budget_mult": float(budget_mult),
            "alpha_promo_good_streak": int(good_streak),
            "alpha_promo_bad_streak": int(bad_streak),
            "alpha_final_pass_json": final_pass_json,
            **alpha_diag
        })

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


# ============================================================
# 7) Fusion engine
# ============================================================

def _alpha_positions_for_date(alpha_pos: pd.DataFrame, dt: pd.Timestamp):
    if alpha_pos is None or len(alpha_pos) == 0:
        return pd.DataFrame()

    x = alpha_pos[alpha_pos["Date"] <= dt].copy()
    if len(x) == 0:
        return pd.DataFrame()

    last_dt = pd.Timestamp(x["Date"].max()).normalize()
    out = x[x["Date"] == last_dt].copy()

    wcol = "weight" if "weight" in out.columns else ("weight_raw" if "weight_raw" in out.columns else None)
    if wcol is None:
        out["alpha_base_weight"] = 0.0
    else:
        out["alpha_base_weight"] = pd.to_numeric(out[wcol], errors="coerce").fillna(0.0)

    if "selected" in out.columns:
        sel = pd.to_numeric(out["selected"], errors="coerce").fillna(0).astype(int)
        if sel.sum() > 0:
            out = out.loc[sel == 1].copy()
        else:
            out = out.loc[out["alpha_base_weight"].abs() > 0].copy()
    else:
        out = out.loc[out["alpha_base_weight"].abs() > 0].copy()

    out = out.loc[out["alpha_base_weight"] > 0].copy()

    return out.sort_values(["alpha_base_weight", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def build_fusion_engine(
    df_def: pd.DataFrame,
    alpha_pos: pd.DataFrame,
    alpha_state_hist: pd.DataFrame,
    cfg: Part3V1Config
):
    df = df_def.copy().sort_values("Date").reset_index(drop=True)

    if len(alpha_state_hist) > 0:
        st = alpha_state_hist.copy().sort_values("Date").reset_index(drop=True)
        df = pd.merge_asof(
            df.sort_values("Date"),
            st.sort_values("Date"),
            on="Date",
            direction="backward"
        )
    else:
        df["realized_dates"] = 0
        df["mean_rank_ic"] = np.nan
        df["mean_topk_rel_ret_net"] = np.nan
        df["ir_topk_rel_ret_net"] = np.nan
        df["alpha_drift_alarm_rate"] = np.nan
        df["quality_ok"] = 0
        df["drift_ok"] = 0
        df["alpha_state"] = "UNAVAILABLE"
        df["alpha_state_display"] = "UNAVAILABLE"
        df["alpha_live"] = 0
        df["fusion_live"] = 0
        df["budget_mult"] = 0.0
        df["alpha_promo_good_streak"] = 0
        df["alpha_promo_bad_streak"] = 0
        df["alpha_final_pass_json"] = None
        df["alpha_quality_ok"] = 0
        df["alpha_drift_ok"] = 0
        df["alpha_trial_gate_open"] = 0
        df["alpha_fused_gate_open"] = 0
        df["alpha_promotion_ready"] = 0
        df["alpha_blocker_count"] = 1
        df["alpha_blocker_text"] = "alpha_artifacts_unavailable"
        df["alpha_blockers_json"] = json.dumps(["alpha_artifacts_unavailable"])

    df["w_defense_voo"] = cfg.BASE_VOO_W * (
        (1.0 - df["defense_scale_final"].values.astype(float)) +
        df["defense_scale_final"].values.astype(float) * df["w_exec"].values.astype(float)
    )
    df["w_defense_voo"] = np.clip(df["w_defense_voo"], 0.0, 1.0)
    df["w_defense_ief"] = 1.0 - df["w_defense_voo"].values.astype(float)

    alloc_rows = []
    alpha_costs = []
    alpha_grosses = []
    core_voo_list = []
    core_ief_list = []
    n_names_list = []
    alpha_ret_gross_list = []

    prev_alpha_weights = {}

    for i in range(len(df)):
        dt = pd.Timestamp(df.loc[i, "Date"]).normalize()
        state = str(df.loc[i, "alpha_state"])
        budget_mult = _safe_num(df.loc[i, "budget_mult"], 0.0)

        day_pos = _alpha_positions_for_date(alpha_pos, dt)
        w_def_voo = _safe_num(df.loc[i, "w_defense_voo"], 0.0)
        w_def_ief = _safe_num(df.loc[i, "w_defense_ief"], 1.0)

        cur_weights = {}
        alpha_ret_gross = 0.0
        source_alpha_date = _latest_valid_date(day_pos, "Date")

        if state in ["LIVE_TRIAL", "LIVE_FUSED"] and len(day_pos) > 0 and budget_mult > 0:
            raw_weights = {
                str(r["Ticker"]): max(_safe_num(r["alpha_base_weight"], 0.0), 0.0) * budget_mult
                for _, r in day_pos.iterrows()
            }
            gross_raw = sum(raw_weights.values())

            gross_cap = min(
                cfg.ALPHA_MAX_GROSS_ABS,
                max(0.0, cfg.ALPHA_MAX_SHARE_OF_VOO * w_def_voo)
            )

            if gross_raw > gross_cap and gross_raw > 0:
                scale = gross_cap / gross_raw
                raw_weights = {k: v * scale for k, v in raw_weights.items()}

            cur_weights = {k: v for k, v in raw_weights.items() if v > 0}

            for _, r in day_pos.iterrows():
                tk = str(r["Ticker"])
                w = cur_weights.get(tk, 0.0)
                if w <= 0:
                    continue

                fr = _safe_num(r["fwd_ret"], np.nan) if "fwd_ret" in day_pos.columns else np.nan
                if np.isfinite(fr):
                    alpha_ret_gross += w * fr

        alpha_gross = float(sum(cur_weights.values()))
        core_voo = max(0.0, w_def_voo - alpha_gross)
        core_ief = w_def_ief

        union = set(prev_alpha_weights) | set(cur_weights)
        alpha_turnover = sum(abs(cur_weights.get(k, 0.0) - prev_alpha_weights.get(k, 0.0)) for k in union)
        alpha_cost_exec = (cfg.ALPHA_COST_BPS / 10000.0) * alpha_turnover * 2.0

        prev_alpha_weights = cur_weights.copy()

        alpha_costs.append(alpha_cost_exec)
        alpha_grosses.append(alpha_gross)
        core_voo_list.append(core_voo)
        core_ief_list.append(core_ief)
        n_names_list.append(len(cur_weights))
        alpha_ret_gross_list.append(alpha_ret_gross)

        # Allocation rows
        alloc_rows.append({
            "Date": dt,
            "Ticker": "VOO_CORE",
            "weight": core_voo,
            "alpha_state": state,
            "alpha_state_display": normalize_alpha_state_for_display(state),
            "fusion_live": int(state in ["LIVE_TRIAL", "LIVE_FUSED"]),
            "budget_mult": budget_mult,
            "source_alpha_date": source_alpha_date,
            "alpha_quality_ok": _safe_int(df.loc[i, "alpha_quality_ok"], 0),
            "alpha_drift_ok": _safe_int(df.loc[i, "alpha_drift_ok"], 0),
            "alpha_trial_gate_open": _safe_int(df.loc[i, "alpha_trial_gate_open"], 0),
            "alpha_fused_gate_open": _safe_int(df.loc[i, "alpha_fused_gate_open"], 0),
            "alpha_promotion_ready": _safe_int(df.loc[i, "alpha_promotion_ready"], 0),
            "alpha_blocker_text": str(df.loc[i, "alpha_blocker_text"]),
        })

        alloc_rows.append({
            "Date": dt,
            "Ticker": "IEF_CORE",
            "weight": core_ief,
            "alpha_state": state,
            "alpha_state_display": normalize_alpha_state_for_display(state),
            "fusion_live": int(state in ["LIVE_TRIAL", "LIVE_FUSED"]),
            "budget_mult": budget_mult,
            "source_alpha_date": source_alpha_date,
            "alpha_quality_ok": _safe_int(df.loc[i, "alpha_quality_ok"], 0),
            "alpha_drift_ok": _safe_int(df.loc[i, "alpha_drift_ok"], 0),
            "alpha_trial_gate_open": _safe_int(df.loc[i, "alpha_trial_gate_open"], 0),
            "alpha_fused_gate_open": _safe_int(df.loc[i, "alpha_fused_gate_open"], 0),
            "alpha_promotion_ready": _safe_int(df.loc[i, "alpha_promotion_ready"], 0),
            "alpha_blocker_text": str(df.loc[i, "alpha_blocker_text"]),
        })

        for tk, w in cur_weights.items():
            alloc_rows.append({
                "Date": dt,
                "Ticker": tk,
                "weight": w,
                "alpha_state": state,
                "alpha_state_display": normalize_alpha_state_for_display(state),
                "fusion_live": int(state in ["LIVE_TRIAL", "LIVE_FUSED"]),
                "budget_mult": budget_mult,
                "source_alpha_date": source_alpha_date,
                "alpha_quality_ok": _safe_int(df.loc[i, "alpha_quality_ok"], 0),
                "alpha_drift_ok": _safe_int(df.loc[i, "alpha_drift_ok"], 0),
                "alpha_trial_gate_open": _safe_int(df.loc[i, "alpha_trial_gate_open"], 0),
                "alpha_fused_gate_open": _safe_int(df.loc[i, "alpha_fused_gate_open"], 0),
                "alpha_promotion_ready": _safe_int(df.loc[i, "alpha_promotion_ready"], 0),
                "alpha_blocker_text": str(df.loc[i, "alpha_blocker_text"]),
            })

    alloc_df = pd.DataFrame(alloc_rows).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    df["alpha_cost_exec"] = alpha_costs
    df["alpha_gross_weight"] = alpha_grosses
    df["w_core_voo"] = core_voo_list
    df["w_core_ief"] = core_ief_list
    df["alpha_n_names"] = n_names_list
    df["alpha_ret_gross"] = alpha_ret_gross_list

    df["defense_cost_port"] = (
        cfg.BASE_VOO_W *
        df["defense_scale_final"].values.astype(float) *
        df["cost_exec"].values.astype(float)
    )

    df["portfolio_turnover_proxy"] = (
        df["trade_size"].values.astype(float) +
        pd.Series(df["alpha_gross_weight"].values.astype(float)).diff().abs().fillna(0.0).values
    )

    if {"fwd_voo", "fwd_ief"}.issubset(df.columns):
        df["portfolio_ret_defense_gross"] = (
            df["w_defense_voo"].values.astype(float) * df["fwd_voo"].values.astype(float) +
            df["w_defense_ief"].values.astype(float) * df["fwd_ief"].values.astype(float)
        )
        df["portfolio_ret_defense_net"] = (
            df["portfolio_ret_defense_gross"].values.astype(float) -
            df["defense_cost_port"].values.astype(float)
        )

        df["portfolio_ret_fused_gross"] = (
            df["w_core_voo"].values.astype(float) * df["fwd_voo"].values.astype(float) +
            df["w_core_ief"].values.astype(float) * df["fwd_ief"].values.astype(float) +
            df["alpha_ret_gross"].values.astype(float)
        )

        df["portfolio_ret_fused_net"] = (
            df["portfolio_ret_fused_gross"].values.astype(float) -
            df["defense_cost_port"].values.astype(float) -
            df["alpha_cost_exec"].values.astype(float)
        )

        df["active_ret_vs_60_40_fused_net"] = (
            df["portfolio_ret_fused_net"].values.astype(float) -
            (
                cfg.BASE_VOO_W * df["fwd_voo"].values.astype(float) +
                cfg.BASE_IEF_W * df["fwd_ief"].values.astype(float)
            )
        )
    else:
        df["portfolio_ret_defense_gross"] = np.nan
        df["portfolio_ret_defense_net"] = np.nan
        df["portfolio_ret_fused_gross"] = np.nan
        df["portfolio_ret_fused_net"] = np.nan
        df["active_ret_vs_60_40_fused_net"] = np.nan

    return df, alloc_df


# ============================================================
# 8) Main
# ============================================================

def main(cfg: Part3V1Config, defense_override=None):
    _ensure_dir(cfg.OUT_DIR)
    os.makedirs(os.path.dirname(cfg.PRED_LOG_PATH), exist_ok=True)

    # --------------------------------------------------------
    # 1) Load defense tape
    # --------------------------------------------------------
    raw_def, defense_source = get_certified_defense_tape(cfg, defense_override)
    df0 = standardize_defense_tape(raw_def)

    decision_row = _pick_decision_row(df0, cfg)

    px_voo_call, voo_src = price_call_from_row(
        decision_row, "px_voo_t", "fwd_voo_hat_final", "fwd_voo_hat", "px_voo_call_7d"
    )
    px_ief_call, ief_src = price_call_from_row(
        decision_row, "px_ief_t", "fwd_ief_hat_final", "fwd_ief_hat", "px_ief_call_7d"
    )

    print(f"Decision-time 7D price call: VOO={px_voo_call:.4f} ({voo_src}) | IEF={px_ief_call:.4f} ({ief_src})")
    print("Part 3 V1 defense_source:", defense_source)
    print("Part 3 V1 last defense Date:", df0["Date"].max(), "| is_live tail:", int(df0.sort_values("Date").tail(1)["is_live"].values[0]))

    # --------------------------------------------------------
    # 2) Defense execution + governance
    # --------------------------------------------------------
    df_exec = run_defense_execution(df0, cfg)
    df_def, gov_df = build_defense_governance(df_exec, cfg)

    # --------------------------------------------------------
    # 3) Alpha artifacts + state machine
    # --------------------------------------------------------
    alpha_pos, alpha_sum, alpha_elig, alpha_json, pos_src, sum_src, elig_src, json_src = read_alpha_artifacts(cfg)
    alpha_pos = standardize_alpha_positions(alpha_pos)
    alpha_sum = standardize_alpha_summary(alpha_sum)
    alpha_elig = standardize_alpha_eligibility(alpha_elig)

    alpha_state_hist = build_alpha_state_history(alpha_sum, alpha_json, cfg)

    if len(alpha_state_hist) > 0:
        latest_alpha = alpha_state_hist.sort_values("Date").tail(1).iloc[0]
        print(
            f"Alpha state latest: {latest_alpha['alpha_state']} "
            f"(display={latest_alpha['alpha_state_display']}) | "
            f"realized_dates={int(latest_alpha['realized_dates'])} | "
            f"budget_mult={latest_alpha['budget_mult']:.2f} | "
            f"drift_rate={_safe_num(latest_alpha['alpha_drift_alarm_rate']):.4f}"
        )
        print(
            f"Alpha diagnostics: quality_ok={int(latest_alpha['alpha_quality_ok'])} | "
            f"drift_ok={int(latest_alpha['alpha_drift_ok'])} | "
            f"trial_gate_open={int(latest_alpha['alpha_trial_gate_open'])} | "
            f"fused_gate_open={int(latest_alpha['alpha_fused_gate_open'])} | "
            f"promotion_ready={int(latest_alpha['alpha_promotion_ready'])}"
        )
        print(f"Alpha blockers: {latest_alpha['alpha_blocker_text']}")
    else:
        print("[WARN] Alpha state history unavailable. Fusion will remain inactive.")

        if (
            int(latest_alpha["alpha_quality_ok"]) == 1 and
            int(latest_alpha["alpha_drift_ok"]) == 1 and
            int(latest_alpha["alpha_trial_gate_open"]) == 1
        ):
            assert str(latest_alpha["alpha_state"]) in ["LIVE_TRIAL", "LIVE_FUSED"], (
                "Alpha gates are open but alpha_state did not promote."
            )

    # --------------------------------------------------------
    # 4) Fusion engine
    # --------------------------------------------------------
    df_final, fusion_alloc = build_fusion_engine(
        df_def=df_def,
        alpha_pos=alpha_pos,
        alpha_state_hist=alpha_state_hist,
        cfg=cfg
    )

    # Merge alpha state columns into governance table
    if len(alpha_state_hist) > 0:
        gov_df = pd.merge_asof(
            gov_df.sort_values("Date"),
            alpha_state_hist.sort_values("Date"),
            on="Date",
            direction="backward"
        )
    else:
        gov_df["realized_dates"] = 0
        gov_df["mean_rank_ic"] = np.nan
        gov_df["mean_topk_rel_ret_net"] = np.nan
        gov_df["ir_topk_rel_ret_net"] = np.nan
        gov_df["alpha_drift_alarm_rate"] = np.nan
        gov_df["quality_ok"] = 0
        gov_df["drift_ok"] = 0
        gov_df["alpha_state"] = "UNAVAILABLE"
        gov_df["alpha_state_display"] = "UNAVAILABLE"
        gov_df["alpha_live"] = 0
        gov_df["fusion_live"] = 0
        gov_df["budget_mult"] = 0.0
        gov_df["alpha_promo_good_streak"] = 0
        gov_df["alpha_promo_bad_streak"] = 0
        gov_df["alpha_final_pass_json"] = None
        gov_df["alpha_quality_ok"] = 0
        gov_df["alpha_drift_ok"] = 0
        gov_df["alpha_trial_gate_open"] = 0
        gov_df["alpha_fused_gate_open"] = 0
        gov_df["alpha_promotion_ready"] = 0
        gov_df["alpha_blocker_count"] = 1
        gov_df["alpha_blocker_text"] = "alpha_artifacts_unavailable"
        gov_df["alpha_blockers_json"] = json.dumps(["alpha_artifacts_unavailable"])

    # --------------------------------------------------------
    # 5) Quick audit prints
    # --------------------------------------------------------
    m_def = df_final.dropna(subset=["portfolio_ret_defense_net"]).copy()
    m_real = df_final.dropna(subset=["portfolio_ret_fused_net", "active_ret_vs_60_40_fused_net"]).copy()

    defense_ir = _annualized_ir(m_def["portfolio_ret_defense_net"].values, cfg.H_REB) if len(m_def) else np.nan
    fused_ir = _annualized_ir(m_real["portfolio_ret_fused_net"].values, cfg.H_REB) if len(m_real) else np.nan
    active_ir = _annualized_ir(m_real["active_ret_vs_60_40_fused_net"].values, cfg.H_REB) if len(m_real) else np.nan
    active_mean = float(m_real["active_ret_vs_60_40_fused_net"].mean()) if len(m_real) else np.nan

    alpha_state_dist = df_final["alpha_state"].value_counts(normalize=True).round(4)
    fusion_live_rate = float(pd.to_numeric(df_final["fusion_live"], errors="coerce").fillna(0).mean())

    print("\n" + "=" * 96)
    print("🏛️  PART 3 V1 AUDIT (Defense Sleeve + Fusion Engine)")
    print("=" * 96)
    print(
        f"Rows: {len(df_final)} | Realized fused rows: {len(m_real)} | "
        f"Fusion live rate: {fusion_live_rate:.2%}"
    )
    print(
        f"Defense IR (net): {defense_ir:.3f} | "
        f"Fused IR (net): {fused_ir:.3f} | "
        f"Active IR vs 60/40: {active_ir:.3f} | "
        f"Active mean: {active_mean:.6f}"
    )
    print("Alpha state distribution:")
    print(alpha_state_dist)
    print("\nAlpha promotion thresholds:")
    print(
        f"Eligible={cfg.ALPHA_MIN_REALIZED_ELIGIBLE} | "
        f"Trial={cfg.ALPHA_MIN_REALIZED_TRIAL} | "
        f"Fused={cfg.ALPHA_MIN_REALIZED_FUSED} | "
        f"Max drift rate={cfg.ALPHA_MAX_DRIFT_RATE:.2f}"
    )

    # --------------------------------------------------------
    # 6) Prediction log
    # --------------------------------------------------------
    append_and_update_prediction_log(df0, decision_row, cfg)

    # --------------------------------------------------------
    # 7) Stamp provenance + write outputs
    # --------------------------------------------------------
    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    df_final["part3_version"] = "V1"
    df_final["defense_tape_source"] = defense_source
    df_final["alpha_positions_source"] = pos_src
    df_final["alpha_summary_source"] = sum_src
    df_final["alpha_eligibility_source"] = elig_src
    df_final["alpha_json_source"] = json_src
    df_final["run_utc"] = run_ts

    gov_df["part3_version"] = "V1"
    gov_df["defense_tape_source"] = defense_source
    gov_df["alpha_positions_source"] = pos_src
    gov_df["alpha_summary_source"] = sum_src
    gov_df["alpha_eligibility_source"] = elig_src
    gov_df["alpha_json_source"] = json_src
    gov_df["run_utc"] = run_ts

    fusion_alloc["part3_version"] = "V1"
    fusion_alloc["run_utc"] = run_ts

    out_tape = os.path.join(cfg.OUT_DIR, "v1_final_production_tape.csv")
    out_gov = os.path.join(cfg.OUT_DIR, "v1_final_production_governance.csv")
    out_alloc = os.path.join(cfg.OUT_DIR, "v1_fusion_allocations.csv")

    df_final.to_csv(out_tape, index=False)
    gov_df.to_csv(out_gov, index=False)
    fusion_alloc.to_csv(out_alloc, index=False)

    # --------------------------------------------------------
    # 8) Sanity checks
    # --------------------------------------------------------
    t = pd.read_csv(out_tape)
    t["Date"] = pd.to_datetime(t["Date"])
    assert t["Date"].max() == pd.to_datetime(df0["Date"].max()).normalize()

    if "is_live" in t.columns:
        assert int(t.sort_values("Date").iloc[-1]["is_live"]) == 1

    alloc = pd.read_csv(out_alloc)
    alloc["Date"] = pd.to_datetime(alloc["Date"])
    alloc_chk = alloc.groupby("Date", as_index=False)["weight"].sum()
    if len(alloc_chk):
        max_dev = float(np.max(np.abs(alloc_chk["weight"].values - 1.0)))
        print(f"Fusion allocation sum-to-one max deviation: {max_dev:.8f}")

    print(f"\n✅ PART 3 V1 WRITTEN")
    print(f"   Tape:        {out_tape}")
    print(f"   Governance:  {out_gov}")
    print(f"   Allocations: {out_alloc}")
    print(f"   Prediction log: {cfg.PRED_LOG_PATH}")
    print("   Alpha is only made live when the state machine reaches LIVE_TRIAL or LIVE_FUSED.")
    print("   Fusion is funded from the VOO sleeve, not from IEF.")
    print("   UI-facing alpha label uses CANDIDATE instead of ambiguous ELIGIBLE.")


if __name__ == "__main__":
    main(CFG)
    
