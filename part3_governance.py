#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# -----------------------------
# Project root
# -----------------------------
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd()


# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class V14Config:
    # live safety
    LIVE_MODE: bool = True
    MAX_STALENESS_DAYS: int = 3
    H_REB: int = 7
    BASE_VOO_W: float = 0.60

    # execution gating
    EPS_W: float = 0.05
    EPS_P: float = 0.05
    LEGS: int = 2

    # slippage
    SLIP_BASE_BPS: float = 5.0
    SLIP_STRESS_BPS: float = 15.0
    STRESS_VOL_Q: float = 0.80
    MIN_STRESS_HISTORY: int = 20

    # kill switch
    KILL_MULT: float = 3.0
    KILL_ABS: float = 0.45

    # weight transform
    K_DEF: float = 1.0

    # alpha policy
    ALPHA_STAR: float = 0.45
    ALPHA_CAP: float = 0.50
    ALPHA_FLOOR: float = 0.00

    # governance windowing
    ROLL_WINDOW: int = 52

    # realized governance thresholds
    PR_LIFT_EPS: float = 0.10
    STALE_EPS: float = 1e-6
    MIN_TAIL_N_ROLL: int = 5

    # friction caps
    MAX_ANN_COST_DRAG: float = 0.015
    MAX_EFF_PERIOD_TURNOVER_AVG: float = 0.05

    # persistence
    PERSISTENCE_K: int = 5
    PR_THROTTLE: float = 0.50
    FRIC_THROTTLE: float = 0.70

    # ZeRO Tier-0
    ZERO_K0: int = 3
    ZERO_R0: int = 2
    ZERO_LOW_TH: float = 1.0 / 3.0
    ZERO_HIGH_TH: float = 2.0 / 3.0
    ZERO_SCALE: float = 0.85

    # Too-Good alarm
    TG_K: int = 3
    TG_LIFT_TH: float = 1.80
    TG_TURNEFF_TH: float = 0.01
    TG_ANNCOSTDRAG_TH: float = 0.0005
    TG_DES_MIN: float = 0.002

    # LIVE proxy health thresholds
    SIG_VAR_MIN: float = 1e-4
    SIG_VAR_WIN: int = 26
    MAX_KILL_RATE: float = 0.50
    MAX_TRADE_RATE: float = 0.80

    # project paths
    PART2_TAPE_PRIMARY: str = str(ROOT / "artifacts_part2_v77" / "predictions" / "v77_final_consensus_tape.csv")
    PART2_TAPE_FALLBACK: str = str(ROOT / "v77_final_consensus_tape.csv")
    PRED_LOG_PATH: str = str(ROOT / "artifacts_part3" / "prediction_log.csv")
    OUT_TAPE_PATH: str = str(ROOT / "v141_final_production_tape.csv")
    OUT_GOV_PATH: str = str(ROOT / "v141_final_production_governance.csv")
    LOCAL_TZ: str = "America/New_York"


CFG = V14Config()


# -----------------------------
# 2) Low-level helpers
# -----------------------------
def _safe_num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# -----------------------------
# 3) Part-2 tape discovery
# -----------------------------
def get_certified_tape_v14(cfg: V14Config, path_override: str | None = None):
    required_base = ["Date", "p0", "excess_vol10"]
    prob_cols_any = [["p_final_cal"], ["p_final_raw"], ["p_final"]]

    def _passes(head: pd.DataFrame) -> bool:
        if not all(c in head.columns for c in required_base):
            return False
        if not any(all(c in head.columns for c in cols) for cols in prob_cols_any):
            return False
        if ("w_exec" in head.columns) or ("cost_exec" in head.columns):
            return False
        return True

    if path_override is not None:
        if not os.path.exists(path_override):
            raise FileNotFoundError(f"override path not found: {path_override}")
        head = pd.read_csv(path_override, nrows=1)
        if not _passes(head):
            raise ValueError("override tape failed contract/provenance checks")
        print(f"✅ CANONICAL TAPE DISCOVERED (override): {path_override}")
        return pd.read_csv(path_override), path_override

    priority_paths = [cfg.PART2_TAPE_PRIMARY, cfg.PART2_TAPE_FALLBACK]

    for path in priority_paths:
        if os.path.exists(path):
            try:
                head = pd.read_csv(path, nrows=1)
                if _passes(head):
                    print(f"✅ CANONICAL TAPE DISCOVERED: {path}")
                    return pd.read_csv(path), path
            except Exception:
                continue

    raise FileNotFoundError("CRITICAL: No certified Part 2 tape found. Provenance chain broken.")


# -----------------------------
# 4) Standardize tape
# -----------------------------
def standardize_tape_live_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    for c in ["p0", "excess_vol10"]:
        if c not in df.columns:
            raise ValueError(f"missing required live column: {c}")
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
        raise ValueError("No usable probability column found (p_final_cal/p_final/p_final_raw).")

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

    if "is_live" in df.columns:
        df["is_live"] = pd.to_numeric(df["is_live"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_live"] = 0

    opt_num = [
        "px_voo_t", "px_ief_t",
        "fwd_voo_hat", "fwd_ief_hat",
        "fwd_voo_hat_final", "fwd_ief_hat_final",
        "px_voo_call_7d", "px_ief_call_7d",
    ]
    for c in opt_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["y_voo", "fwd_voo", "fwd_ief", "bench_ret"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# -----------------------------
# 5) Decision-row selection
# -----------------------------
def _pick_decision_row(df: pd.DataFrame, cfg: V14Config) -> pd.Series:
    if (df["is_live"] == 1).any():
        row = df.loc[df["is_live"] == 1].iloc[-1]
    else:
        row = df.iloc[-1]

    today_local = pd.Timestamp.now(tz=cfg.LOCAL_TZ).normalize()
    row_local = pd.Timestamp(row["Date"]).tz_localize(None).normalize()
    age_days = int((today_local.tz_localize(None) - row_local).days)

    if cfg.LIVE_MODE and (age_days > cfg.MAX_STALENESS_DAYS):
        raise RuntimeError(
            f"STALE TAPE REFUSED (LIVE_MODE=True): last Date={row_local.date()} "
            f"age={age_days}d > MAX_STALENESS_DAYS={cfg.MAX_STALENESS_DAYS}"
        )
    if (not cfg.LIVE_MODE) and (age_days > cfg.MAX_STALENESS_DAYS):
        print(
            f"[WARN] Stale tape: last Date={row_local.date()} age={age_days}d "
            f"(MAX_STALENESS_DAYS={cfg.MAX_STALENESS_DAYS})"
        )
    return row


# -----------------------------
# 6) Price call
# -----------------------------
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


# -----------------------------
# 7) Execution engine
# -----------------------------
def run_execution_v14(df_in: pd.DataFrame, cfg: V14Config) -> pd.DataFrame:
    df = df_in.copy()
    p = df["p_final_use"].values.astype(float)
    p0 = df["p0"].values.astype(float)

    p0_floor = np.zeros(len(df), dtype=float)
    p0_floor[0] = 0.2027
    for i in range(1, len(df)):
        p0_floor[i] = np.nanmedian(p0[:i])

    p0_eff = np.maximum(p0, p0_floor)

    vol = df["excess_vol10"].values.astype(float)
    vol_rank = np.array([(vol[:i + 1] <= vol[i]).mean() for i in range(len(vol))], dtype=float)

    w = np.zeros(len(df), dtype=float)
    costs = np.zeros(len(df), dtype=float)

    kill_flag = np.zeros(len(df), dtype=int)
    trade_flag = np.zeros(len(df), dtype=int)
    trade_size = np.zeros(len(df), dtype=float)
    target_w_raw = np.zeros(len(df), dtype=float)
    target_w_postkill = np.zeros(len(df), dtype=float)

    tw0 = np.clip(1.0 - cfg.K_DEF * ((p[0] - p0_eff[0]) / (p0_eff[0] + 1e-12)), 0.0, 1.0)
    target_w_raw[0] = tw0
    k0 = (p[0] > cfg.KILL_MULT * p0_eff[0]) or (p[0] > cfg.KILL_ABS)
    kill_flag[0] = int(k0)
    tw0_pk = 0.0 if k0 else tw0
    target_w_postkill[0] = tw0_pk

    w[0] = tw0_pk
    cur_w, cur_p = w[0], p[0]

    for i in range(1, len(df)):
        tw = np.clip(1.0 - cfg.K_DEF * ((p[i] - p0_eff[i]) / (p0_eff[i] + 1e-12)), 0.0, 1.0)
        target_w_raw[i] = tw

        k = (p[i] > cfg.KILL_MULT * p0_eff[i]) or (p[i] > cfg.KILL_ABS)
        kill_flag[i] = int(k)

        tw_pk = 0.0 if k else tw
        target_w_postkill[i] = tw_pk

        do_trade = bool(k) or (abs(tw_pk - cur_w) > cfg.EPS_W and abs(p[i] - cur_p) > cfg.EPS_P)
        if do_trade:
            trade = abs(tw_pk - cur_w)
            slip = cfg.SLIP_STRESS_BPS if (i >= cfg.MIN_STRESS_HISTORY and vol_rank[i] >= cfg.STRESS_VOL_Q) else cfg.SLIP_BASE_BPS
            costs[i] = (slip / 10000.0) * trade * cfg.LEGS
            trade_flag[i] = 1
            trade_size[i] = trade
            cur_w, cur_p = tw_pk, p[i]

        w[i] = cur_w

    df["p0_floor_val"] = p0_floor
    df["p0_eff"] = p0_eff
    df["vol_rank"] = vol_rank

    df["w_exec"] = w
    df["cost_exec"] = costs
    df["kill_flag"] = kill_flag
    df["trade_flag"] = trade_flag
    df["trade_size"] = trade_size
    df["target_w_raw"] = target_w_raw
    df["target_w_postkill"] = target_w_postkill
    return df


# -----------------------------
# 8) Alpha plumbing
# -----------------------------
def alpha_eff_base(df: pd.DataFrame, cfg: V14Config) -> np.ndarray:
    a = cfg.ALPHA_STAR * df["alpha_scale"].values.astype(float)
    return np.clip(a, cfg.ALPHA_FLOOR, cfg.ALPHA_CAP)


# -----------------------------
# 9) LIVE proxy governance
# -----------------------------
def compute_live_proxy_health(df: pd.DataFrame, cfg: V14Config) -> pd.DataFrame:
    out = df.copy()
    a_base = alpha_eff_base(out, cfg)

    dw = out["w_exec"].diff().abs().fillna(0.0).values.astype(float)
    turn_eff = cfg.BASE_VOO_W * a_base * dw
    out["turn_eff_inst"] = turn_eff
    out["turn_eff_roll"] = pd.Series(turn_eff).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    ann_cost_drag_inst = (cfg.BASE_VOO_W * a_base * out["cost_exec"].values.astype(float)) * (252.0 / cfg.H_REB)
    out["ann_cost_drag_inst"] = ann_cost_drag_inst
    out["ann_cost_drag_roll"] = pd.Series(ann_cost_drag_inst).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    pv = pd.Series(out["p_final_use"].values.astype(float)).rolling(cfg.SIG_VAR_WIN, min_periods=cfg.SIG_VAR_WIN).var()
    out["p_final_var_roll"] = pv.values

    out["kill_rate_roll"] = pd.Series(out["kill_flag"]).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values
    out["trade_rate_roll"] = pd.Series(out["trade_flag"]).rolling(cfg.ROLL_WINDOW, min_periods=1).mean().values

    fric_ok = (out["turn_eff_roll"].values < cfg.MAX_EFF_PERIOD_TURNOVER_AVG) & (out["ann_cost_drag_roll"].values < cfg.MAX_ANN_COST_DRAG)
    sig_ok = np.where(np.isnan(out["p_final_var_roll"].values), True, out["p_final_var_roll"].values >= cfg.SIG_VAR_MIN)
    regime_ok = (out["kill_rate_roll"].values <= cfg.MAX_KILL_RATE) & (out["trade_rate_roll"].values <= cfg.MAX_TRADE_RATE)

    out["GovHealthScore_live"] = (fric_ok.astype(float) + sig_ok.astype(float) + regime_ok.astype(float)) / 3.0
    out["live_fric_fail_now"] = ((~fric_ok).astype(int)).astype(int)
    return out


# -----------------------------
# 10) REALIZED governance helpers
# -----------------------------
def aligned_es_improve(bench_r, port_r, base_r, q=10):
    qv = np.percentile(bench_r, q)
    mask = bench_r <= qv
    if not mask.any():
        return 0.0, 0
    return float(np.mean(port_r[mask]) - np.mean(base_r[mask])), int(mask.sum())


def get_integrated_returns(alpha, df_in: pd.DataFrame, cfg: V14Config) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    w_exec = df_in["w_exec"].values.astype(float)
    w_v = cfg.BASE_VOO_W * ((1.0 - alpha) + alpha * w_exec)
    r_port = w_v * df_in["fwd_voo"].values.astype(float) + (1.0 - w_v) * df_in["fwd_ief"].values.astype(float)
    return r_port - (alpha * cfg.BASE_VOO_W * df_in["cost_exec"].values.astype(float))


def realized_data_available(df: pd.DataFrame) -> bool:
    need = ["y_voo", "fwd_voo", "fwd_ief", "bench_ret"]
    if not all(c in df.columns for c in need):
        return False
    return df[need].notna().any().all()


# -----------------------------
# 11) Governance state machines
# -----------------------------
def build_v14_outputs(df_exec: pd.DataFrame, cfg: V14Config):
    df = compute_live_proxy_health(df_exec.copy(), cfg)
    df["alpha_eff_base"] = alpha_eff_base(df, cfg)

    n = len(df)
    W = cfg.ROLL_WINDOW
    has_realized = realized_data_available(df)

    df["GovStatus"] = "RUN"
    df["GovMode"] = "ACTIVE"
    df["GovReason"] = "OK"

    df["PR_Fails_Streak"] = 0
    df["FRIC_Fails_Streak"] = 0
    df["STALE_Streak"] = 0
    df["TooGood_Streak"] = 0

    df["PR_Soft"] = 0
    df["FRIC_Soft"] = 0
    df["STALE_Mode"] = 0

    df["Lift_End"] = np.nan
    df["dES_P_End"] = np.nan
    df["dES_S_End"] = np.nan
    df["N_P_End"] = np.nan
    df["N_S_End"] = np.nan
    df["GovHealthScore_realized"] = np.nan

    df["ZeRO_Active"] = 0
    df["ZeRO_Scale"] = 1.0
    df["ZeRO_LowStreak"] = 0
    df["ZeRO_HighStreak"] = 0
    df["TooGood_Alarm"] = 0

    df["gov_scale"] = 1.0

    pr_fails = 0
    fric_fails = 0
    stale_fails = 0
    tg_streak = 0

    z_active = False
    z_low = 0
    z_high = 0

    for t in range(n):
        fric_fail = bool(df["live_fric_fail_now"].iloc[t])
        fric_fails = fric_fails + 1 if fric_fail else 0
        df.at[t, "FRIC_Fails_Streak"] = fric_fails

        if has_realized and (t >= W - 1):
            sl = df.iloc[t - W + 1: t + 1]
            y = sl["y_voo"].values.astype(int)
            p_fin = sl["p_final_use"].values.astype(float)
            prev = y.mean()

            if len(np.unique(y)) > 1:
                pr_auc = average_precision_score(y, p_fin)
                lift = pr_auc / prev if prev > 0 else 1.0
                pr_suff = True
            else:
                lift = np.nan
                pr_suff = False

            df.at[t, "Lift_End"] = lift

            a_base_win = df["alpha_eff_base"].iloc[t - W + 1: t + 1].values.astype(float)
            port_r = get_integrated_returns(a_base_win, sl, cfg)

            b = sl["bench_ret"].values.astype(float)
            base_r = sl["bench_ret"].values.astype(float)
            imp_p, n_p = aligned_es_improve(b, port_r, base_r, 10)

            base_port = cfg.BASE_VOO_W * sl["fwd_voo"].values.astype(float) + (1.0 - cfg.BASE_VOO_W) * sl["fwd_ief"].values.astype(float)
            imp_s, n_s = aligned_es_improve(b, port_r, base_port, 10)

            df.at[t, "dES_P_End"] = imp_p
            df.at[t, "dES_S_End"] = imp_s
            df.at[t, "N_P_End"] = n_p
            df.at[t, "N_S_End"] = n_s

            pr_fail = (lift < (1.0 - cfg.PR_LIFT_EPS)) if pr_suff else False
            pr_fails = pr_fails + 1 if pr_fail else 0
            df.at[t, "PR_Fails_Streak"] = pr_fails

            st_now = (abs(imp_p) < cfg.STALE_EPS) and (abs(imp_s) < cfg.STALE_EPS)
            stale_fails = stale_fails + 1 if st_now else 0
            df.at[t, "STALE_Streak"] = stale_fails

            hc = []
            hc.append(1.0 if (not np.isnan(lift) and lift > 1.0) else 0.0)
            hc.append(1.0 if (imp_p > 0.0) else 0.0)
            hc.append(1.0 if (df["turn_eff_roll"].iloc[t] < cfg.MAX_EFF_PERIOD_TURNOVER_AVG) else 0.0)
            df.at[t, "GovHealthScore_realized"] = float(np.mean(hc))

            tg_now = (
                (not np.isnan(lift)) and (lift >= cfg.TG_LIFT_TH) and
                (df["GovHealthScore_realized"].iloc[t] >= 1.0 - 1e-12) and
                (df["turn_eff_roll"].iloc[t] <= cfg.TG_TURNEFF_TH) and
                (df["ann_cost_drag_roll"].iloc[t] <= cfg.TG_ANNCOSTDRAG_TH) and
                (imp_p >= cfg.TG_DES_MIN)
            )
            tg_streak = tg_streak + 1 if tg_now else 0
            df.at[t, "TooGood_Streak"] = tg_streak
            df.at[t, "TooGood_Alarm"] = 1 if (tg_streak >= cfg.TG_K) else 0

            stale_mode = (stale_fails >= cfg.PERSISTENCE_K)
            df.at[t, "STALE_Mode"] = int(stale_mode)

            pr_soft = (pr_fails >= cfg.PERSISTENCE_K) and (not stale_mode)
            fric_soft = (fric_fails >= cfg.PERSISTENCE_K)

            df.at[t, "PR_Soft"] = int(pr_soft)
            df.at[t, "FRIC_Soft"] = int(fric_soft)

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

            df.at[t, "gov_scale"] = scale
            df.at[t, "GovReason"] = "OK" if len(reasons) == 0 else "|".join(reasons)

        else:
            fric_soft = (fric_fails >= cfg.PERSISTENCE_K)
            df.at[t, "FRIC_Soft"] = int(fric_soft)
            df.at[t, "gov_scale"] = (cfg.FRIC_THROTTLE if fric_soft else 1.0)
            df.at[t, "GovMode"] = "ACTIVE"
            df.at[t, "GovReason"] = ("FRIC_THROTTLE" if fric_soft else "OK")

        ghs = df["GovHealthScore_realized"].iloc[t]
        if np.isnan(ghs):
            ghs = float(df["GovHealthScore_live"].iloc[t])

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

        df.at[t, "ZeRO_Active"] = int(z_active)
        df.at[t, "ZeRO_Scale"] = (cfg.ZERO_SCALE if z_active else 1.0)
        df.at[t, "ZeRO_LowStreak"] = z_low
        df.at[t, "ZeRO_HighStreak"] = z_high

    df["alpha_eff_preZero"] = np.clip(
        df["alpha_eff_base"].values.astype(float) * df["gov_scale"].values.astype(float),
        cfg.ALPHA_FLOOR, cfg.ALPHA_CAP
    )
    df["alpha_eff_final"] = np.clip(
        df["alpha_eff_preZero"].values.astype(float) * df["ZeRO_Scale"].values.astype(float),
        cfg.ALPHA_FLOOR, cfg.ALPHA_CAP
    )

    gov_rows = []
    for t in range(W - 1, n):
        gov_rows.append({
            "Date": df["Date"].iloc[t],
            "GovMode": df["GovMode"].iloc[t],
            "GovReason": df["GovReason"].iloc[t],
            "gov_scale": float(df["gov_scale"].iloc[t]),
            "alpha_eff_final": float(df["alpha_eff_final"].iloc[t]),
            "GovHealthScore_live": float(df["GovHealthScore_live"].iloc[t]),
            "GovHealthScore_realized": df["GovHealthScore_realized"].iloc[t],
            "ZeRO_Active_End": int(df["ZeRO_Active"].iloc[t]),
            "ZeRO_Scale_End": float(df["ZeRO_Scale"].iloc[t]),
            "FRIC_Fails_Streak_End": int(df["FRIC_Fails_Streak"].iloc[t]),
            "PR_Fails_Streak_End": int(df["PR_Fails_Streak"].iloc[t]),
            "STALE_Streak_End": int(df["STALE_Streak"].iloc[t]),
            "FRIC_Soft_End": int(df["FRIC_Soft"].iloc[t]),
            "PR_Soft_End": int(df["PR_Soft"].iloc[t]),
            "STALE_Mode_End": int(df["STALE_Mode"].iloc[t]),
            "Lift_End": df["Lift_End"].iloc[t],
            "dES_P_End": df["dES_P_End"].iloc[t],
            "AnnCostDrag_Roll": float(df["ann_cost_drag_roll"].iloc[t]),
            "TurnEff_Roll": float(df["turn_eff_roll"].iloc[t]),
            "TooGood_Alarm_End": int(df["TooGood_Alarm"].iloc[t]),
            "TooGood_Streak_End": int(df["TooGood_Streak"].iloc[t]),
        })
    gov_df = pd.DataFrame(gov_rows)

    return df, gov_df


# -----------------------------
# 12) Prediction log (UPSERT)
# -----------------------------
def _idx_of_date(df: pd.DataFrame, dt) -> int | None:
    m = (df["Date"].dt.normalize() == pd.Timestamp(dt).normalize())
    if not m.any():
        return None
    return int(np.where(m.values)[0][-1])


def append_and_update_prediction_log(df0: pd.DataFrame, decision_row: pd.Series, cfg: V14Config):
    pred_log_path = cfg.PRED_LOG_PATH
    _ensure_dir(pred_log_path)

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

    if os.path.exists(pred_log_path):
        log = pd.read_csv(pred_log_path)
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

    m = (log["decision_date"] == decision_date)
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
        h = int(log.loc[i, "h_reb"])
        idx0 = _idx_of_date(df0, d0)
        if idx0 is None:
            continue

        idxT = idx0 + h
        if idxT >= len(df0):
            continue

        target_date = pd.Timestamp(df0.loc[idxT, "Date"]).normalize()
        px_voo_real = float(df0.loc[idxT, "px_voo_t"]) if "px_voo_t" in df0.columns else np.nan
        px_ief_real = float(df0.loc[idxT, "px_ief_t"]) if "px_ief_t" in df0.columns else np.nan

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
    log.to_csv(pred_log_path, index=False)

    realized = log.dropna(subset=["px_voo_realized", "px_ief_realized"])
    if len(realized) > 0:
        voo_mae = float(realized["voo_abs_err"].mean())
        ief_mae = float(realized["ief_abs_err"].mean())
        voo_mape = float(100.0 * realized["voo_ape"].mean())
        ief_mape = float(100.0 * realized["ief_ape"].mean())
        print(f"[PredLog] realized rows={len(realized)} | VOO MAE={voo_mae:.4f} (MAPE={voo_mape:.2f}%) | IEF MAE={ief_mae:.4f} (MAPE={ief_mape:.2f}%)")
    else:
        print("[PredLog] realized rows=0 (not enough tape history yet).")

    print(f"[PredLog] path: {pred_log_path}")


# -----------------------------
# 13) Main production run
# -----------------------------
def main(cfg: V14Config, tape_override: str | None = None):
    raw, tape_source = get_certified_tape_v14(cfg, tape_override)
    df0 = standardize_tape_live_safe(raw)

    decision_row = _pick_decision_row(df0, cfg)

    px_voo_call_7d, voo_call_src = price_call_from_row(
        decision_row, "px_voo_t", "fwd_voo_hat_final", "fwd_voo_hat", "px_voo_call_7d"
    )
    px_ief_call_7d, ief_call_src = price_call_from_row(
        decision_row, "px_ief_t", "fwd_ief_hat_final", "fwd_ief_hat", "px_ief_call_7d"
    )
    print(f"Decision-time 7D price call: VOO={px_voo_call_7d:.4f} ({voo_call_src}) | IEF={px_ief_call_7d:.4f} ({ief_call_src})")

    print("Part 3 tape_source:", tape_source)
    print(
        "Part 3 last Date:", df0["Date"].max(),
        "| is_live tail:",
        int(df0.sort_values("Date").tail(1)["is_live"].values[0]) if "is_live" in df0.columns else "NA"
    )

    df1 = run_execution_v14(df0, cfg)
    df_final, gov_df = build_v14_outputs(df1, cfg)

    append_and_update_prediction_log(df0, decision_row, cfg)

    last = df0.sort_values("Date").tail(1).iloc[0]
    if "is_live" in df0.columns and int(last["is_live"]) != 1:
        print("⚠️ WARNING: last row is not marked is_live=1 (decision-time append may have failed).")

    from datetime import datetime, UTC
    run_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    df_final["part3_version"] = "V14.1"
    df_final["tape_source"] = tape_source
    df_final["run_utc"] = run_ts

    gov_df["part3_version"] = "V14.1"
    gov_df["tape_source"] = tape_source
    gov_df["run_utc"] = run_ts

    _ensure_dir(cfg.OUT_TAPE_PATH)
    _ensure_dir(cfg.OUT_GOV_PATH)

    df_final.to_csv(cfg.OUT_TAPE_PATH, index=False)
    gov_df.to_csv(cfg.OUT_GOV_PATH, index=False)

    print(f"\n✅ GOLD MASTER V14.1 CERTIFIED AND ARCHIVED. Source: {tape_source}")
    print(f"   Output: {cfg.OUT_TAPE_PATH}, {cfg.OUT_GOV_PATH}")
    print(f"   ZeRO: K0={cfg.ZERO_K0}, R0={cfg.ZERO_R0}, Scale={cfg.ZERO_SCALE}")
    print(f"   TooGood (REALIZED-only): K={cfg.TG_K}, Lift>={cfg.TG_LIFT_TH}, dES_P>={cfg.TG_DES_MIN}")
    print("   Live-safe: execution + friction proxy always; PR/TAIL/TooGood activate when realized data arrives.")

    return {
        "tape_source": tape_source,
        "out_tape_path": cfg.OUT_TAPE_PATH,
        "out_gov_path": cfg.OUT_GOV_PATH,
        "pred_log_path": cfg.PRED_LOG_PATH,
        "decision_date": str(pd.Timestamp(decision_row['Date']).date()),
        "px_voo_call_7d": float(px_voo_call_7d) if np.isfinite(px_voo_call_7d) else np.nan,
        "px_ief_call_7d": float(px_ief_call_7d) if np.isfinite(px_ief_call_7d) else np.nan,
    }


def cli() -> int:
    cfg = V14Config()
    summary = main(cfg)
    print("\nPart 3 summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
