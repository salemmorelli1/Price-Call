
#!/usr/bin/env python3
from __future__ import annotations


# =============================================================================
# PROJECT: VOO vs IEF Tail-Risk — Hardened Cross-Sectional Alpha Sleeve
# VERSION: Part 2A.2.1
#
# SECTION 1 CONTENTS:
# - imports
# - config
# - low-level helpers
# - model builders
# - panel-design helpers
# - ticker eligibility helpers
# - hardened selection helpers
# - date summary / governance helpers
#
# REQUIRES upstream artifact:
#   ./artifacts_part1/alpha_panel.parquet
# =============================================================================

import os
import warnings
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 0) Config
# ============================================================

@dataclass
class Part2A21Config:
    PART1_DIR: str = "./artifacts_part1"
    PRED_DIR: str = "./artifacts_part2a_alpha/predictions"

    # Core geometry
    H: int = 7
    REFIT_FREQ: int = 20
    TRAIN_WINDOW_Y: int = 4
    TRADING_DAYS_PER_YEAR: int = 252
    PURGE: int = 7
    HO_START_DATE: str = "2020-01-01"

    # Data contract
    ALPHA_PANEL_FILE: str = "alpha_panel.parquet"
    benchmark_ticker: str = "VOO"
    alpha_universe: tuple = (
        "XLK", "XLF", "XLI", "XLY", "XLP",
        "XLV", "XLE", "XLU", "XLB", "XLC",
        "SMH", "IWM", "MDY", "EFA", "EEM",
    )

    # Guardrails
    MIN_TRAIN_ROWS: int = 200
    MIN_REBAL_COVERAGE: float = 0.98
    SEED: int = 42

    # Stage 0: broad risk model
    RISK_RIDGE_ALPHA: float = 5.0
    RISK_COLS: tuple = (
        "voo_vol10",
        "excess_vol10",
        "vix_mom5",
        "alpha_credit_spread",
        "alpha_credit_accel",
        "alpha_vix_term",
    )

    # Stage 1: alpha model
    USE_TICKER_DUMMIES: bool = True

    # Probability calibration
    ECE_BINS: int = 10
    ROLL_DIAG: int = 52
    CAL_MIN_SAMPLES: int = 60
    CAL_CLIP: float = 1e-6
    CAL_MIN_POS: int = 10
    CAL_MIN_NEG: int = 10

    # Calibration gate
    CAL_GATE_WINDOW: int = 52
    CAL_GATE_MIN_SAMPLES: int = 52
    CAL_GATE_MIN_POS: int = 8
    CAL_GATE_MIN_NEG: int = 8
    CAL_GATE_MIN_BRIER_IMPROVE: float = 0.001
    CAL_GATE_MAX_ECE_WORSEN: float = 0.000
    CAL_GATE_PERSIST_K: int = 2

    # Score blend
    MU_WEIGHT: float = 0.75
    PROB_WEIGHT: float = 0.25
    DISAGREE_DAMP: float = 0.40

    # Hardened selection
    TOP_K: int = 2
    GROSS_ALPHA_BUDGET: float = 0.08
    PER_NAME_CAP: float = 0.04
    MIN_SCORE_LONG: float = 0.15
    MIN_SCORE_GAP_TOP12: float = 0.05
    MIN_MU_Z: float = 0.20
    MIN_PROB_EDGE: float = 0.05

    # Dynamic ticker eligibility filter
    ELIGIBILITY_LOOKBACK_DATES: int = 126
    ELIGIBILITY_MIN_OBS: int = 12
    ELIGIBILITY_MIN_HIT_RATE: float = 0.50
    ELIGIBILITY_MIN_MEAN_ALPHA: float = 0.0000
    ELIGIBILITY_MIN_SCORE: float = 0.00
    ELIGIBILITY_MIN_NAMES: int = 4
    ELIGIBILITY_USE_SELECTED_ONLY: bool = True

    # Governance / scaling
    ALPHA_THROTTLE: float = 0.50
    SLIP_BPS: float = 5.0

    # Alpha-specific drift governance
    ALPHA_DRIFT_IC_MIN: float = -0.02
    ALPHA_DRIFT_RET_MIN: float = -0.00025
    ALPHA_DRIFT_BREADTH_MIN: float = 0.50
    ALPHA_DRIFT_PERSIST_K: int = 6
    ALPHA_DRIFT_MIN_REALIZED: int = 26

    # Model hyperparams
    CLS_MAX_DEPTH: int = 2
    CLS_N_ESTIMATORS: int = 100
    CLS_LEARNING_RATE: float = 0.05

    REG_MAX_DEPTH: int = 2
    REG_N_ESTIMATORS: int = 250
    REG_LEARNING_RATE: float = 0.03
    REG_SUBSAMPLE: float = 0.9
    REG_COLSAMPLE: float = 0.9

    # Falsification
    DO_FALSIFICATION: bool = True
    SHUFFLE_BLOCK: int = 14
    SHUFFLE_B: int = 200
    MAX_SHUFFLE_AUC: float = 0.515


CFG = Part2A21Config()


# ============================================================
# 1) Low-level helpers
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p, clip=1e-6):
    p = np.clip(p, clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def _safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[m], y_pred[m])))


def _safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(mean_absolute_error(y_true[m], y_pred[m]))


def _annualized_ir(x, H):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.nanstd(x, ddof=1) <= 0:
        return np.nan
    return float(np.nanmean(x) / np.nanstd(x, ddof=1) * np.sqrt(252.0 / H))


def ece_score(y_true, p, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p).astype(float)
    n = len(p)
    if n == 0:
        return np.nan

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.any():
            acc = y_true[m].mean()
            conf = p[m].mean()
            ece += (m.sum() / n) * abs(acc - conf)

    return float(ece)


def fit_mapper_v76(z_val, y_val, p0_vec):
    """
    Mapper:
        p_raw = lam * sigmoid(sign*z/T + b) + (1-lam) * p0
    """
    def objective(params, z, y, p0):
        T, b, lam = params
        p_final = lam * _sigmoid(z / T + b) + (1.0 - lam) * p0
        return np.mean((p_final - y) ** 2)

    res_pos = minimize(
        objective,
        x0=[1.0, 0.0, 0.5],
        args=(z_val, y_val, p0_vec),
        bounds=[(0.1, 20), (-5, 5), (0.0, 1.0)],
    )
    res_neg = minimize(
        objective,
        x0=[1.0, 0.0, 0.5],
        args=(-z_val, y_val, p0_vec),
        bounds=[(0.1, 20), (-5, 5), (0.0, 1.0)],
    )

    return (-1, res_neg.x) if res_neg.fun < res_pos.fun else (1, res_pos.x)


def fit_platt_1d(x, y, max_iter=200, clip=1e-6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n == 0 or np.unique(y).size < 2:
        p = float(np.mean(y)) if n else 0.5
        return 0.0, float(_logit(p, clip=clip))

    a = 1.0
    b = float(_logit(np.mean(y), clip=clip))

    for _ in range(max_iter):
        z = a * x + b
        p = _sigmoid(z)
        w = np.maximum(p * (1.0 - p), 1e-8)

        g_a = np.sum((p - y) * x)
        g_b = np.sum(p - y)

        h_aa = np.sum(w * x * x)
        h_ab = np.sum(w * x)
        h_bb = np.sum(w)

        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-12:
            break

        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (-h_ab * g_a + h_aa * g_b) / det

        a_new = a - step_a
        b_new = b - step_b

        if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return float(a), float(b)


def apply_platt(a, b, x):
    return _sigmoid(a * x + b)


def block_shuffle_labels(y, block_size, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    blocks = [y[i:i + block_size] for i in range(0, len(y), block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:len(y)]


def _fit_gamma_mae(y_true, y_hat):
    y_true = np.asarray(y_true, float)
    y_hat = np.asarray(y_hat, float)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    if m.sum() < 30:
        return 0.0

    gammas = np.linspace(0.0, 1.0, 51)
    maes = [np.mean(np.abs(y_true[m] - g * y_hat[m])) for g in gammas]
    return float(gammas[int(np.argmin(maes))])


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

# ============================================================
# 2) Model builders
# ============================================================

def _make_classifier(cfg: Part2A21Config):
    return xgb.XGBClassifier(
        max_depth=int(cfg.CLS_MAX_DEPTH),
        n_estimators=int(cfg.CLS_N_ESTIMATORS),
        learning_rate=float(cfg.CLS_LEARNING_RATE),
        n_jobs=-1,
        random_state=int(cfg.SEED),
    )


def _make_alpha_regressor(cfg: Part2A21Config):
    return xgb.XGBRegressor(
        max_depth=int(cfg.REG_MAX_DEPTH),
        n_estimators=int(cfg.REG_N_ESTIMATORS),
        learning_rate=float(cfg.REG_LEARNING_RATE),
        subsample=float(cfg.REG_SUBSAMPLE),
        colsample_bytree=float(cfg.REG_COLSAMPLE),
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=int(cfg.SEED),
    )


def _make_risk_model(cfg: Part2A21Config):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(cfg.RISK_RIDGE_ALPHA))
    )


# ============================================================
# 3) Panel-design helpers
# ============================================================

def _make_panel_design(df: pd.DataFrame, feature_cols, use_ticker_dummies=True):
    X = df[list(feature_cols)].copy().reset_index(drop=True)

    if use_ticker_dummies:
        d = pd.get_dummies(
            df["Ticker"].astype(str),
            prefix="tk",
            dtype=float
        ).reset_index(drop=True)
        X = pd.concat([X, d], axis=1)

    return X


def _align_design_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    cols = list(X_train.columns)
    X_test2 = X_test.copy()

    for c in cols:
        if c not in X_test2.columns:
            X_test2[c] = 0.0

    extra = [c for c in X_test2.columns if c not in cols]
    if extra:
        X_test2 = X_test2.drop(columns=extra)

    return X_train[cols], X_test2[cols]


# ============================================================
# 4) Ticker eligibility helpers
# ============================================================

def compute_ticker_eligibility(history_df: pd.DataFrame, cfg: Part2A21Config):
    """
    Determine which tickers remain eligible based on trailing realized behavior.

    Expected columns:
      Date, Ticker, alpha_score, alpha_target_realized, selected
    """
    base = pd.DataFrame({"Ticker": list(cfg.ALPHA_UNIVERSE)})

    if len(history_df) == 0:
        out = base.copy()
        out["n_obs"] = 0
        out["hit_rate"] = np.nan
        out["mean_alpha_target"] = np.nan
        out["mean_score"] = np.nan
        out["eligible"] = 0
        return out.sort_values("Ticker").reset_index(drop=True)

    h = history_df.copy().sort_values(["Date", "Ticker"])
    unique_dates = np.array(sorted(h["Date"].dropna().unique()))
    if len(unique_dates) == 0:
        out = base.copy()
        out["n_obs"] = 0
        out["hit_rate"] = np.nan
        out["mean_alpha_target"] = np.nan
        out["mean_score"] = np.nan
        out["eligible"] = 0
        return out.sort_values("Ticker").reset_index(drop=True)

    keep_dates = unique_dates[-cfg.ELIGIBILITY_LOOKBACK_DATES:]
    h = h[h["Date"].isin(keep_dates)].copy()

    if cfg.ELIGIBILITY_USE_SELECTED_ONLY and "selected" in h.columns:
        h = h[h["selected"] == 1].copy()

    h = h.dropna(subset=["alpha_target_realized", "alpha_score"]).copy()

    if len(h) == 0:
        out = base.copy()
        out["n_obs"] = 0
        out["hit_rate"] = np.nan
        out["mean_alpha_target"] = np.nan
        out["mean_score"] = np.nan
        out["eligible"] = 0
        return out.sort_values("Ticker").reset_index(drop=True)

    grp = h.groupby("Ticker", sort=True).agg(
        n_obs=("alpha_target_realized", "size"),
        hit_rate=("alpha_target_realized", lambda x: float((x > 0).mean())),
        mean_alpha_target=("alpha_target_realized", "mean"),
        mean_score=("alpha_score", "mean"),
    ).reset_index()

    out = base.merge(grp, on="Ticker", how="left")
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out["hit_rate"] = pd.to_numeric(out["hit_rate"], errors="coerce")
    out["mean_alpha_target"] = pd.to_numeric(out["mean_alpha_target"], errors="coerce")
    out["mean_score"] = pd.to_numeric(out["mean_score"], errors="coerce")

    out["eligible"] = (
        (out["n_obs"] >= cfg.ELIGIBILITY_MIN_OBS) &
        (out["hit_rate"] >= cfg.ELIGIBILITY_MIN_HIT_RATE) &
        (out["mean_alpha_target"] >= cfg.ELIGIBILITY_MIN_MEAN_ALPHA) &
        (out["mean_score"] >= cfg.ELIGIBILITY_MIN_SCORE)
    ).astype(int)

    return out.sort_values("Ticker").reset_index(drop=True)

def eligible_ticker_set(elig_df: pd.DataFrame, cfg: Part2A21Config):
    """
    Return the eligible ticker set, with fallback if too few names survive.
    """
    if len(elig_df) == 0:
        return set(cfg.ALPHA_UNIVERSE)

    keep = set(elig_df.loc[elig_df["eligible"] == 1, "Ticker"].astype(str))

    if len(keep) < cfg.ELIGIBILITY_MIN_NAMES:
        ranked = elig_df.copy()
        ranked["mean_alpha_target_rank"] = pd.to_numeric(ranked["mean_alpha_target"], errors="coerce").fillna(-1e9)
        ranked["hit_rate_rank"] = pd.to_numeric(ranked["hit_rate"], errors="coerce").fillna(-1e9)
        ranked["mean_score_rank"] = pd.to_numeric(ranked["mean_score"], errors="coerce").fillna(-1e9)
        ranked["n_obs_rank"] = pd.to_numeric(ranked["n_obs"], errors="coerce").fillna(0)

        ranked = ranked.sort_values(
            ["eligible", "mean_alpha_target_rank", "hit_rate_rank", "mean_score_rank", "n_obs_rank", "Ticker"],
            ascending=[False, False, False, False, False, True],
        )

        keep = set(ranked.head(cfg.ELIGIBILITY_MIN_NAMES)["Ticker"].astype(str))

    if len(keep) == 0:
        keep = set(cfg.ALPHA_UNIVERSE)

    return keep

# ============================================================
# 5) Hardened selection helpers
# ============================================================
def build_hardened_cross_sectional_weights(day_df: pd.DataFrame, eligible_names, cfg: Part2A21Config):
    """
    Long-only hardened selector for a single rebalance date.
    """
    d = day_df.copy().sort_values("alpha_score", ascending=False)
    d["eligible_name"] = d["Ticker"].astype(str).isin(set(eligible_names)).astype(int)
    d["selected"] = 0
    d["weight_raw"] = 0.0

    if "score_prob" not in d.columns:
        d["score_prob"] = np.nan
    if "score_mu" not in d.columns:
        d["score_mu"] = np.nan

    cand = d[
        (d["eligible_name"] == 1) &
        (d["alpha_score"] >= cfg.MIN_SCORE_LONG) &
        (np.abs(d["score_mu"]) >= cfg.MIN_MU_Z) &
        (np.abs(d["score_prob"]) >= cfg.MIN_PROB_EDGE)
    ].copy()

    if len(cand) == 0:
        fallback = d[
            (d["eligible_name"] == 1) &
            np.isfinite(d["alpha_score"])
        ].copy().sort_values("alpha_score", ascending=False)

        if len(fallback) == 0:
            return d

        if float(fallback["alpha_score"].iloc[0]) <= 0.0:
            return d

        cand = fallback.head(1).copy()
    else:
        cand = cand.sort_values("alpha_score", ascending=False).head(cfg.TOP_K).copy()

        if len(cand) >= 2:
            top_gap = float(cand["alpha_score"].iloc[0] - cand["alpha_score"].iloc[1])
            if top_gap < cfg.MIN_SCORE_GAP_TOP12:
                cand = cand.iloc[:1].copy()

    if len(cand) == 0:
        return d

    wt = min(cfg.PER_NAME_CAP, cfg.GROSS_ALPHA_BUDGET / len(cand))
    d.loc[cand.index, "selected"] = 1
    d.loc[cand.index, "weight_raw"] = wt

    return d

def build_cross_sectional_positions_hardened(
    pos_df: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    cfg: Part2A21Config
):
    """
    Build alpha scores and hardened top-K positions, with dynamic ticker filtering.

    Expected columns:
      mu_alpha_hat_final, p_alpha_final, alpha_sigma_train
    """
    out = pos_df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    sigma = np.maximum(np.abs(out["alpha_sigma_train"].values.astype(float)), 1e-6)
    score_mu = np.tanh(out["mu_alpha_hat_final"].values.astype(float) / sigma)
    score_prob = 2.0 * out["p_alpha_final"].values.astype(float) - 1.0

    agree = (
        (np.sign(score_mu) == np.sign(score_prob)) |
        (np.abs(score_mu) < 1e-8) |
        (np.abs(score_prob) < 1e-8)
    )

    alpha_score_raw = (
        float(cfg.MU_WEIGHT) * score_mu +
        float(cfg.PROB_WEIGHT) * score_prob
    )
    alpha_score = np.where(agree, alpha_score_raw, float(cfg.DISAGREE_DAMP) * alpha_score_raw)

    out["score_mu"] = score_mu
    out["score_prob"] = score_prob
    out["score_agree"] = agree.astype(int)
    out["alpha_score_raw"] = alpha_score_raw
    out["alpha_score"] = alpha_score

    elig_set = eligible_ticker_set(eligibility_df, cfg)

    blocks = []
    for dt, g in out.groupby("Date", sort=True):
        g2 = build_hardened_cross_sectional_weights(g, elig_set, cfg)
        blocks.append(g2)

    out = pd.concat(blocks, axis=0).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out


# ============================================================
# 6) Date summary / governance helpers
# ============================================================

def summarize_positions_by_date_safe(pos_df: pd.DataFrame, cfg: Part2A21Config, weight_col="weight_raw"):
    rows = []
    prev_weights = {}

    for dt, g in pos_df.groupby("Date", sort=True):
        g = g.copy()
        sel = g[g["selected"] == 1].copy()

        rank_ic = np.nan
        greal = g.dropna(subset=["alpha_target_realized"])
        if len(greal) >= 4 and greal["alpha_score"].nunique() > 1 and greal["alpha_target_realized"].nunique() > 1:
            rank_ic = greal["alpha_score"].corr(greal["alpha_target_realized"], method="spearman")

        breadth = int(len(sel))
        eligible_breadth = int(g["eligible_name"].sum()) if "eligible_name" in g.columns else np.nan
        gross_used = float(sel[weight_col].sum()) if breadth > 0 else 0.0

        if breadth > 0 and sel["rel_ret"].notna().any():
            topk_rel_ret = float((sel[weight_col] * sel["rel_ret"]).sum())
        else:
            topk_rel_ret = np.nan

        if breadth > 0 and sel["alpha_target_realized"].notna().any():
            selection_hit = float((sel.loc[sel["alpha_target_realized"].notna(), "alpha_target_realized"] > 0).mean())
        else:
            selection_hit = np.nan

        curr_weights = dict(zip(sel["Ticker"], sel[weight_col]))
        union = set(prev_weights) | set(curr_weights)
        turnover = sum(abs(curr_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in union)

        if np.isfinite(topk_rel_ret):
            topk_rel_ret_net = topk_rel_ret - (cfg.SLIP_BPS / 10000.0) * turnover * 2.0
        else:
            topk_rel_ret_net = np.nan

        rows.append({
            "Date": dt,
            "rank_ic": rank_ic,
            "topk_rel_ret": topk_rel_ret,
            "topk_rel_ret_net": topk_rel_ret_net,
            "selection_hit_rate": selection_hit,
            "breadth_selected": breadth,
            "eligible_breadth": eligible_breadth,
            "gross_alpha_budget_used": gross_used,
            "turnover": turnover,
        })

        prev_weights = curr_weights

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def alpha_drift_stream(summary_df: pd.DataFrame, cfg: Part2A21Config):
    df = summary_df.copy().sort_values("Date").reset_index(drop=True)

    n = len(df)
    alarm = np.zeros(n, dtype=int)
    streak = 0

    rank_ic = pd.to_numeric(df["rank_ic"], errors="coerce")
    topk_net = pd.to_numeric(df["topk_rel_ret_net"], errors="coerce")
    breadth = pd.to_numeric(df["breadth_selected"], errors="coerce")

    df["realized_flag"] = (rank_ic.notna() & topk_net.notna()).astype(int)
    df["realized_dates_cum"] = df["realized_flag"].cumsum()

    df["rank_ic_roll"] = rank_ic.rolling(cfg.ROLL_DIAG, min_periods=10).mean()
    df["topk_rel_ret_net_roll"] = topk_net.rolling(cfg.ROLL_DIAG, min_periods=10).mean()

    # Only evaluate breadth on dates where there was actually a selected basket
    breadth_active = breadth.where(breadth > 0, np.nan)
    df["breadth_roll"] = breadth_active.rolling(cfg.ROLL_DIAG, min_periods=10).mean()

    for i in range(n):
        if df.loc[i, "realized_dates_cum"] < cfg.ALPHA_DRIFT_MIN_REALIZED:
            streak = 0
            alarm[i] = 0
            continue

        fail_count = 0

        ic = df.loc[i, "rank_ic_roll"]
        rr = df.loc[i, "topk_rel_ret_net_roll"]
        br = df.loc[i, "breadth_roll"]

        if np.isfinite(ic) and ic < cfg.ALPHA_DRIFT_IC_MIN:
            fail_count += 1
        if np.isfinite(rr) and rr < cfg.ALPHA_DRIFT_RET_MIN:
            fail_count += 1
        if np.isfinite(br) and br < cfg.ALPHA_DRIFT_BREADTH_MIN:
            fail_count += 1

        # Require at least 2 of the 3 drift dimensions to fail
        fail = (fail_count >= 2)

        streak = streak + 1 if fail else 0
        if streak >= cfg.ALPHA_DRIFT_PERSIST_K:
            alarm[i] = 1

    df["alpha_drift_alarm"] = alarm
    df["alpha_overlay_scale"] = np.where(df["alpha_drift_alarm"] == 1, cfg.ALPHA_THROTTLE, 1.0)
    df["alpha_governance_tier"] = np.where(df["alpha_drift_alarm"] == 1, "DRIFT_THROTTLE", "NORMAL")

    return df

# @title PART 2A.2.1 — Hardened Cross-Sectional Alpha Sleeve | SECTION 2

# =============================================================================
# SECTION 2 CONTENTS:
# - feature-contract helpers
# - pooled panel calibration helpers
# - walk-forward engine
# - live-date scoring helper
#
# DEPENDS ON:
# - Section 1 already executed
# =============================================================================


# ============================================================
# 7) Feature-contract helpers
# ============================================================

def infer_panel_feature_cols(panel: pd.DataFrame, cfg: Part2A21Config):
    """
    Infer safe, live-available numeric feature columns from alpha_panel.
    Excludes future/target columns and identifier columns.
    """
    exclude = {
        "Date", "Ticker",
        "px_t",
        "fwd_ret", "benchmark_fwd_ret", "rel_ret",
        "risk_mu_hat",
        "alpha_target_realized", "y_alpha_realized", "y_alpha_avail",
        "z_alpha_raw", "p0_alpha",
        "p_alpha_raw", "p_alpha_cal_candidate", "p_alpha_final",
        "mu_alpha_hat_raw", "mu_alpha_hat_final",
        "alpha_sigma_train",
        "cal_a", "cal_b", "cal_n",
        "calibration_gate_on",
        "weight_raw", "weight",
        "selected", "eligible_name", "eligible_set_size",
        "score_mu", "score_prob", "score_agree",
        "alpha_score_raw", "alpha_score",
        "alpha_overlay_scale", "alpha_drift_alarm", "alpha_governance_tier",
        "is_live",
    }

    num_cols = []
    for c in panel.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            num_cols.append(c)

    risk_cols = [c for c in cfg.RISK_COLS if c in num_cols]
    other_cols = [c for c in num_cols if c not in risk_cols]

    feature_cols = risk_cols + other_cols

    if len(risk_cols) == 0:
        raise RuntimeError("No configured risk columns were found in alpha_panel.")
    if len(feature_cols) == 0:
        raise RuntimeError("No usable numeric feature columns were found in alpha_panel.")

    return feature_cols, risk_cols


def compute_panel_p0_state(y_train: pd.Series, seed_value=0.5):
    """
    Simple pooled base-rate anchor for the panel alpha sign task.
    """
    yv = pd.to_numeric(y_train, errors="coerce").dropna()
    if len(yv) == 0:
        return float(seed_value)
    v = float(yv.mean())
    return v if np.isfinite(v) else float(seed_value)


# ============================================================
# 8) Pooled panel calibration helpers
# ============================================================

def fit_panel_calibration_by_date(pos_df: pd.DataFrame, cfg: Part2A21Config):
    """
    Causal pooled Platt calibration by date.
    Uses only labels revealed from earlier dates:
        y_alpha_avail[ticker, t] = y_alpha_realized[ticker, t-H]
    """
    df = pos_df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    df["y_alpha_avail"] = (
        df.groupby("Ticker", sort=False)["y_alpha_realized"]
          .shift(int(cfg.H))
    )

    df["p_alpha_cal_candidate"] = np.nan
    df["cal_a"] = np.nan
    df["cal_b"] = np.nan
    df["cal_n"] = 0

    unique_dates = np.array(sorted(df["Date"].dropna().unique()))

    for dt in unique_dates:
        now_mask = (df["Date"] == dt)

        hist = df[
            (df["Date"] < dt) &
            np.isfinite(df["y_alpha_avail"].values.astype(float)) &
            np.isfinite(df["p_alpha_raw"].values.astype(float))
        ].copy()

        n_obs = int(len(hist))
        df.loc[now_mask, "cal_n"] = n_obs

        if n_obs < cfg.CAL_MIN_SAMPLES:
            df.loc[now_mask, "p_alpha_cal_candidate"] = df.loc[now_mask, "p_alpha_raw"].values
            continue

        y_fit = hist["y_alpha_avail"].values.astype(int)
        pos = int(np.sum(y_fit == 1))
        neg = int(np.sum(y_fit == 0))

        if pos < cfg.CAL_MIN_POS or neg < cfg.CAL_MIN_NEG:
            df.loc[now_mask, "p_alpha_cal_candidate"] = df.loc[now_mask, "p_alpha_raw"].values
            continue

        x_fit = _logit(hist["p_alpha_raw"].values.astype(float), clip=cfg.CAL_CLIP)
        a, b0 = fit_platt_1d(x_fit, y_fit, clip=cfg.CAL_CLIP)

        x_now = _logit(df.loc[now_mask, "p_alpha_raw"].values.astype(float), clip=cfg.CAL_CLIP)
        p_now = apply_platt(a, b0, x_now)

        df.loc[now_mask, "cal_a"] = a
        df.loc[now_mask, "cal_b"] = b0
        df.loc[now_mask, "p_alpha_cal_candidate"] = np.clip(p_now, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP)

    # if any date still missing, fall back to raw
    miss = ~np.isfinite(df["p_alpha_cal_candidate"].values.astype(float))
    if miss.any():
        df.loc[miss, "p_alpha_cal_candidate"] = df.loc[miss, "p_alpha_raw"].values

    return df


def apply_panel_calibration_gate_by_date(pos_df: pd.DataFrame, cfg: Part2A21Config):
    """
    Causal gate deciding whether calibrated probabilities should replace raw.
    Evaluated by date over trailing pooled revealed rows.
    """
    df = pos_df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    df["calibration_gate_on"] = 0
    df["cal_gate_brier_raw"] = np.nan
    df["cal_gate_brier_cal"] = np.nan
    df["cal_gate_ece_raw"] = np.nan
    df["cal_gate_ece_cal"] = np.nan

    unique_dates = np.array(sorted(df["Date"].dropna().unique()))
    win_streak = 0

    for dt in unique_dates:
        now_mask = (df["Date"] == dt)

        hist_dates = unique_dates[unique_dates < dt]
        if len(hist_dates) > cfg.CAL_GATE_WINDOW:
            hist_dates = hist_dates[-cfg.CAL_GATE_WINDOW:]

        hist = df[
            df["Date"].isin(hist_dates) &
            np.isfinite(df["y_alpha_avail"].values.astype(float)) &
            np.isfinite(df["p_alpha_raw"].values.astype(float)) &
            np.isfinite(df["p_alpha_cal_candidate"].values.astype(float))
        ].copy()

        if len(hist) < cfg.CAL_GATE_MIN_SAMPLES:
            df.loc[now_mask, "p_alpha_final"] = df.loc[now_mask, "p_alpha_raw"].values
            continue

        y_hist = hist["y_alpha_avail"].values.astype(int)
        pos = int(np.sum(y_hist == 1))
        neg = int(np.sum(y_hist == 0))

        if pos < cfg.CAL_GATE_MIN_POS or neg < cfg.CAL_GATE_MIN_NEG:
            df.loc[now_mask, "p_alpha_final"] = df.loc[now_mask, "p_alpha_raw"].values
            continue

        p_raw = hist["p_alpha_raw"].values.astype(float)
        p_cal = hist["p_alpha_cal_candidate"].values.astype(float)

        brier_raw = float(np.mean((p_raw - y_hist) ** 2))
        brier_cal = float(np.mean((p_cal - y_hist) ** 2))
        ece_raw = ece_score(y_hist, p_raw, n_bins=cfg.ECE_BINS)
        ece_cal = ece_score(y_hist, p_cal, n_bins=cfg.ECE_BINS)

        df.loc[now_mask, "cal_gate_brier_raw"] = brier_raw
        df.loc[now_mask, "cal_gate_brier_cal"] = brier_cal
        df.loc[now_mask, "cal_gate_ece_raw"] = ece_raw
        df.loc[now_mask, "cal_gate_ece_cal"] = ece_cal

        cond = (
            np.isfinite(brier_raw) and np.isfinite(brier_cal) and
            np.isfinite(ece_raw) and np.isfinite(ece_cal) and
            (brier_cal <= brier_raw - cfg.CAL_GATE_MIN_BRIER_IMPROVE) and
            (ece_cal <= ece_raw + cfg.CAL_GATE_MAX_ECE_WORSEN)
        )

        win_streak = win_streak + 1 if cond else 0
        gate_on = int(win_streak >= cfg.CAL_GATE_PERSIST_K)

        df.loc[now_mask, "calibration_gate_on"] = gate_on
        if gate_on == 1:
            df.loc[now_mask, "p_alpha_final"] = df.loc[now_mask, "p_alpha_cal_candidate"].values
        else:
            df.loc[now_mask, "p_alpha_final"] = df.loc[now_mask, "p_alpha_raw"].values

    # final fallback
    miss = ~np.isfinite(df["p_alpha_final"].values.astype(float))
    if miss.any():
        df.loc[miss, "p_alpha_final"] = df.loc[miss, "p_alpha_raw"].values

    return df


def fit_panel_gamma(pos_df: pd.DataFrame, cfg: Part2A21Config):
    """
    Fit a single live-safe shrinkage gamma for alpha magnitude forecasts
    using matured historical rows only.
    """
    df = pos_df.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    gamma_alpha = np.nan
    mature = df.dropna(subset=["alpha_target_realized", "mu_alpha_hat_raw"]).copy()

    if len(mature) >= 60:
        mature_dates = np.array(sorted(mature["Date"].dropna().unique()))
        if len(mature_dates) > 52:
            keep_dates = mature_dates[-52:]
            mature = mature[mature["Date"].isin(keep_dates)].copy()

        gamma_alpha = _fit_gamma_mae(
            mature["alpha_target_realized"].values.astype(float),
            mature["mu_alpha_hat_raw"].values.astype(float)
        )

    if np.isfinite(gamma_alpha):
        df["gamma_alpha"] = gamma_alpha
        df["mu_alpha_hat_final"] = gamma_alpha * df["mu_alpha_hat_raw"].values.astype(float)
    else:
        gamma_alpha = 0.0
        df["gamma_alpha"] = gamma_alpha
        df["mu_alpha_hat_final"] = 0.0

    return df, float(gamma_alpha)


# ============================================================
# 9) Walk-forward engine
# ============================================================

def run_panel_walk_forward(panel: pd.DataFrame, cfg: Part2A21Config):
    """
    Walk-forward pooled panel engine.

    Stage 0:
      rel_ret ~ broad-risk model

    Stage 1:
      alpha_target = rel_ret - risk_mu_hat
      y_alpha = 1(alpha_target > 0)

    Predicts all dates from holdout onward in chunked walk-forward fashion.
    """
    panel = panel.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_cols, risk_cols = infer_panel_feature_cols(panel, cfg)
    unique_dates = np.array(sorted(panel["Date"].dropna().unique()))

    ho_start = pd.Timestamp(cfg.HO_START_DATE).normalize()
    holdout_dates = unique_dates[unique_dates >= ho_start]
    if len(holdout_dates) == 0:
        raise RuntimeError("No holdout dates found. Check HO_START_DATE against alpha_panel.")

    train_window_dates = int(cfg.TRAIN_WINDOW_Y * cfg.TRADING_DAYS_PER_YEAR)
    anchors = holdout_dates[::cfg.REFIT_FREQ]

    out = panel.copy()
    out["risk_mu_hat"] = np.nan
    out["alpha_target_realized"] = np.nan
    out["y_alpha_realized"] = np.nan
    out["z_alpha_raw"] = np.nan
    out["p0_alpha"] = np.nan
    out["p_alpha_raw"] = np.nan
    out["mu_alpha_hat_raw"] = np.nan
    out["alpha_sigma_train"] = np.nan

    last_fit = {
        "feature_cols": feature_cols,
        "risk_cols": risk_cols,
        "X_fit_cols": None,
        "risk_model": None,
        "clf_alpha": None,
        "reg_alpha": None,
        "mapper_state": None,
        "p0_state": np.nan,
        "gamma_alpha": 0.0,
        "alpha_sigma": np.nan,
        "gate_last": 0,
        "cal_a": np.nan,
        "cal_b": np.nan,
    }

    for d_anchor in anchors:
        anchor_idx = int(np.where(unique_dates == d_anchor)[0][0])

        train_end_idx = anchor_idx - int(cfg.H) - int(cfg.PURGE)
        if train_end_idx <= 0:
            continue

        train_start_idx = max(0, train_end_idx - train_window_dates)

        train_dates = unique_dates[train_start_idx:train_end_idx]
        if len(train_dates) == 0:
            continue

        df_tr = panel[panel["Date"].isin(train_dates)].copy()
        df_tr = df_tr.dropna(subset=risk_cols + ["rel_ret"]).copy()

        if len(df_tr) < cfg.MIN_TRAIN_ROWS:
            continue

        # -------------------------
        # Stage 0: broad risk model
        # -------------------------
        risk_model = _make_risk_model(cfg)
        risk_model.fit(df_tr[risk_cols], df_tr["rel_ret"].values.astype(float))

        df_tr["risk_mu_hat_train"] = risk_model.predict(df_tr[risk_cols])
        df_tr["alpha_target_train"] = df_tr["rel_ret"] - df_tr["risk_mu_hat_train"]
        df_tr["y_alpha_train"] = (df_tr["alpha_target_train"] > 0.0).astype(int)

        p0_state = compute_panel_p0_state(df_tr["y_alpha_train"], seed_value=0.5)

        # split by date, not by row
        tr_dates_sorted = np.array(sorted(df_tr["Date"].dropna().unique()))
        split_idx = max(1, int(0.8 * len(tr_dates_sorted)))
        fit_dates = tr_dates_sorted[:split_idx]
        cal_dates = tr_dates_sorted[split_idx:]

        df_fit = df_tr[df_tr["Date"].isin(fit_dates)].copy()
        df_cal = df_tr[df_tr["Date"].isin(cal_dates)].copy()

        if len(df_fit) < max(50, int(cfg.MIN_TRAIN_ROWS * 0.60)) or len(df_cal) < 20:
            continue

        # -------------------------
        # Stage 1a: alpha sign classifier
        # -------------------------
        X_fit = _make_panel_design(df_fit, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
        X_cal = _make_panel_design(df_cal, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
        X_fit, X_cal = _align_design_columns(X_fit, X_cal)

        clf_alpha = _make_classifier(cfg)
        clf_alpha.fit(X_fit, df_fit["y_alpha_train"].values.astype(int))

        z_cal = clf_alpha.predict(X_cal, output_margin=True)
        y_cal = df_cal["y_alpha_train"].values.astype(int)
        p0_vec = np.full(len(df_cal), float(p0_state), dtype=float)

        sign_, (T_, b_, lam_) = fit_mapper_v76(z_cal, y_cal, p0_vec)
        mapper_state = {
            "sign": float(sign_),
            "T": float(T_),
            "b": float(b_),
            "lam": float(lam_),
        }

        # -------------------------
        # Stage 1b: alpha magnitude regressor
        # -------------------------
        X_tr_all = _make_panel_design(df_tr, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
        X_tr_all, _ = _align_design_columns(X_tr_all, X_tr_all.copy())

        reg_alpha = _make_alpha_regressor(cfg)
        reg_alpha.fit(X_tr_all, df_tr["alpha_target_train"].values.astype(float))

        alpha_sigma = float(np.nanstd(df_tr["alpha_target_train"].values.astype(float), ddof=1))
        if (not np.isfinite(alpha_sigma)) or (alpha_sigma <= 1e-8):
            alpha_sigma = 1e-4

        # -------------------------
        # Score next chunk
        # -------------------------
        next_anchor_idx = min(anchor_idx + int(cfg.REFIT_FREQ), len(unique_dates))
        score_dates = unique_dates[anchor_idx:next_anchor_idx]

        df_sc = panel[panel["Date"].isin(score_dates)].copy()
        if len(df_sc) == 0:
            continue

        risk_hat = risk_model.predict(df_sc[risk_cols])
        alpha_target_realized = np.where(
            np.isfinite(df_sc["rel_ret"].values.astype(float)),
            df_sc["rel_ret"].values.astype(float) - risk_hat,
            np.nan
        )
        y_alpha_realized = np.where(
            np.isfinite(alpha_target_realized),
            (alpha_target_realized > 0.0).astype(float),
            np.nan
        )

        X_sc = _make_panel_design(df_sc, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
        X_tr_for_align = _make_panel_design(df_fit, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
        X_tr_for_align, X_sc = _align_design_columns(X_tr_for_align, X_sc)

        z_raw = clf_alpha.predict(X_sc, output_margin=True)
        p_raw = (
            mapper_state["lam"] * _sigmoid((mapper_state["sign"] * z_raw) / (mapper_state["T"] + 1e-12) + mapper_state["b"])
            + (1.0 - mapper_state["lam"]) * float(p0_state)
        )
        p_raw = np.clip(p_raw, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP)

        mu_alpha_hat_raw = reg_alpha.predict(X_sc)

        loc = out["Date"].isin(score_dates)
        out.loc[loc, "risk_mu_hat"] = risk_hat
        out.loc[loc, "alpha_target_realized"] = alpha_target_realized
        out.loc[loc, "y_alpha_realized"] = y_alpha_realized
        out.loc[loc, "z_alpha_raw"] = z_raw
        out.loc[loc, "p0_alpha"] = float(p0_state)
        out.loc[loc, "p_alpha_raw"] = p_raw
        out.loc[loc, "mu_alpha_hat_raw"] = mu_alpha_hat_raw
        out.loc[loc, "alpha_sigma_train"] = float(alpha_sigma)

        last_fit = {
            "feature_cols": feature_cols,
            "risk_cols": risk_cols,
            "X_fit_cols": list(X_tr_for_align.columns),
            "risk_model": risk_model,
            "clf_alpha": clf_alpha,
            "reg_alpha": reg_alpha,
            "mapper_state": mapper_state,
            "p0_state": float(p0_state),
            "gamma_alpha": 0.0,
            "alpha_sigma": float(alpha_sigma),
            "gate_last": 0,
            "cal_a": np.nan,
            "cal_b": np.nan,
        }

    holdout_mask = out["Date"].isin(holdout_dates)
    scored_mask = np.isfinite(out.loc[holdout_mask, "p_alpha_raw"].values.astype(float))
    cover = float(np.mean(scored_mask)) if scored_mask.size > 0 else 0.0
    if cover < cfg.MIN_REBAL_COVERAGE:
        raise RuntimeError(
            f"CRITICAL: alpha panel raw-probability coverage too low: "
            f"{cover:.2%} < {cfg.MIN_REBAL_COVERAGE:.2%}"
        )

    # keep only holdout dates
    out = out[holdout_mask].copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # pooled causal calibration + gate
    out = fit_panel_calibration_by_date(out, cfg)
    out = apply_panel_calibration_gate_by_date(out, cfg)

    # shrink magnitude forecasts
    out, gamma_alpha = fit_panel_gamma(out, cfg)

    # update last-fit live scoring contract
    last_fit["gamma_alpha"] = float(gamma_alpha)

    if "cal_a" in out.columns and out["cal_a"].notna().any():
        last_fit["cal_a"] = float(out["cal_a"].dropna().iloc[-1])
    if "cal_b" in out.columns and out["cal_b"].notna().any():
        last_fit["cal_b"] = float(out["cal_b"].dropna().iloc[-1])
    if "calibration_gate_on" in out.columns and out["calibration_gate_on"].notna().any():
        last_fit["gate_last"] = int(pd.to_numeric(out["calibration_gate_on"], errors="coerce").fillna(0).iloc[-1])

    return out, last_fit


# ============================================================
# 9b) Live-date scoring helper
# ============================================================

def score_live_cross_section(
    panel_live: pd.DataFrame,
    feature_cols,
    risk_cols,
    X_fit_cols,
    risk_model,
    clf_alpha,
    reg_alpha,
    mapper_state,
    p0_state,
    gamma_alpha,
    alpha_sigma,
    gate_last,
    cal_a_last,
    cal_b_last,
    cfg: Part2A21Config,
):
    """
    Score the latest live date using the most recent fitted objects.
    """
    if len(panel_live) == 0:
        return panel_live.copy()

    out = panel_live.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # stage 0
    risk_hat = risk_model.predict(out[risk_cols])

    # stage 1 design
    X_live = _make_panel_design(out, feature_cols, use_ticker_dummies=cfg.USE_TICKER_DUMMIES)
    for c in X_fit_cols:
        if c not in X_live.columns:
            X_live[c] = 0.0
    extra = [c for c in X_live.columns if c not in X_fit_cols]
    if extra:
        X_live = X_live.drop(columns=extra)
    X_live = X_live[list(X_fit_cols)]

    z_raw = clf_alpha.predict(X_live, output_margin=True)
    p_raw = (
        mapper_state["lam"] * _sigmoid((mapper_state["sign"] * z_raw) / (mapper_state["T"] + 1e-12) + mapper_state["b"])
        + (1.0 - mapper_state["lam"]) * float(p0_state)
    )
    p_raw = np.clip(p_raw, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP)

    mu_alpha_raw = reg_alpha.predict(X_live)

    if np.isfinite(_safe_float(cal_a_last)) and np.isfinite(_safe_float(cal_b_last)):
        x_live = _logit(p_raw, clip=cfg.CAL_CLIP)
        p_cal = apply_platt(float(cal_a_last), float(cal_b_last), x_live)
        p_cal = np.clip(p_cal, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP)
    else:
        p_cal = p_raw.copy()

    if np.isfinite(_safe_float(gamma_alpha)):
        mu_alpha_final = float(gamma_alpha) * mu_alpha_raw
    else:
        mu_alpha_final = np.zeros_like(mu_alpha_raw, dtype=float)

    out["risk_mu_hat"] = risk_hat
    out["z_alpha_raw"] = z_raw
    out["p0_alpha"] = float(p0_state)
    out["p_alpha_raw"] = p_raw
    out["p_alpha_cal_candidate"] = p_cal
    out["p_alpha_final"] = np.where(int(gate_last) == 1, p_cal, p_raw)
    out["mu_alpha_hat_raw"] = mu_alpha_raw
    out["mu_alpha_hat_final"] = mu_alpha_final
    out["alpha_sigma_train"] = float(alpha_sigma) if np.isfinite(_safe_float(alpha_sigma)) else 1e-4
    out["gamma_alpha"] = float(gamma_alpha) if np.isfinite(_safe_float(gamma_alpha)) else 0.0
    out["cal_a"] = float(cal_a_last) if np.isfinite(_safe_float(cal_a_last)) else np.nan
    out["cal_b"] = float(cal_b_last) if np.isfinite(_safe_float(cal_b_last)) else np.nan
    out["cal_n"] = 0
    out["calibration_gate_on"] = int(gate_last)

    return out

# @title PART 2A.2.1 — Hardened Cross-Sectional Alpha Sleeve | SECTION 3

# =============================================================================
# SECTION 3 CONTENTS:
# - dynamic eligibility application
# - main()
# - live-date refresh / replacement
# - hardened selection
# - governance
# - audit
# - falsification
# - outputs
#
# DEPENDS ON:
# - Section 1 already executed
# - Section 2 already executed
# =============================================================================


# ============================================================
# 10) Dynamic hardened selection
# ============================================================

def apply_dynamic_hardened_selection(
    pos_df_raw: pd.DataFrame,
    cfg: Part2A21Config
):
    """
    Apply hardened selection date-by-date using only prior realized history
    to determine which tickers remain eligible.

    Returns:
      pos_df_selected
      eligibility_history
    """
    df = pos_df_raw.copy().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    sigma = np.maximum(np.abs(df["alpha_sigma_train"].values.astype(float)), 1e-6)
    score_mu = np.tanh(df["mu_alpha_hat_final"].values.astype(float) / sigma)
    score_prob = 2.0 * df["p_alpha_final"].values.astype(float) - 1.0

    agree = (
        (np.sign(score_mu) == np.sign(score_prob)) |
        (np.abs(score_mu) < 1e-8) |
        (np.abs(score_prob) < 1e-8)
    )

    alpha_score_raw = (
        float(cfg.MU_WEIGHT) * score_mu +
        float(cfg.PROB_WEIGHT) * score_prob
    )
    alpha_score = np.where(agree, alpha_score_raw, float(cfg.DISAGREE_DAMP) * alpha_score_raw)

    df["score_mu"] = score_mu
    df["score_prob"] = score_prob
    df["score_agree"] = agree.astype(int)
    df["alpha_score_raw"] = alpha_score_raw
    df["alpha_score"] = alpha_score

    blocks = []
    elig_blocks = []
    hist_df = None

    for dt, g in df.groupby("Date", sort=True):
        g = g.copy()

        if hist_df is None or len(hist_df) == 0:
            elig_set = set(cfg.ALPHA_UNIVERSE)
            elig_df = pd.DataFrame({
                "Ticker": list(cfg.ALPHA_UNIVERSE),
                "n_obs": 0,
                "hit_rate": np.nan,
                "mean_alpha_target": np.nan,
                "mean_score": np.nan,
                "eligible": 1,
            })
        else:
            hist_real = hist_df.dropna(subset=["alpha_target_realized"]).copy()
            elig_df = compute_ticker_eligibility(hist_real, cfg)
            elig_set = eligible_ticker_set(elig_df, cfg)

        g2 = build_hardened_cross_sectional_weights(g, elig_set, cfg)
        g2["eligible_set_size"] = int(len(elig_set))
        blocks.append(g2)

        snap = elig_df.copy()
        snap["asof_date"] = pd.Timestamp(dt).normalize()
        snap["in_elig_set"] = snap["Ticker"].astype(str).isin(elig_set).astype(int)
        elig_blocks.append(snap)

        if hist_df is None:
            hist_df = g2.copy()
        else:
            hist_df = pd.concat([hist_df, g2], axis=0, ignore_index=True)

    pos_df = pd.concat(blocks, axis=0).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    elig_hist = pd.concat(elig_blocks, axis=0).sort_values(["asof_date", "Ticker"]).reset_index(drop=True)

    return pos_df, elig_hist


# ============================================================
# 11) Main
# ============================================================

def main(cfg: Part2A21Config):
    os.makedirs(cfg.PRED_DIR, exist_ok=True)

    # -----------------------------
    # Load alpha panel
    # -----------------------------
    panel_path = os.path.join(cfg.PART1_DIR, cfg.ALPHA_PANEL_FILE)
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Missing required alpha panel: {panel_path}. "
            f"Patch and rerun Part 1 first."
        )

    panel = pd.read_parquet(panel_path)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.normalize()
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    panel = panel[panel["Ticker"].isin(cfg.ALPHA_UNIVERSE)].copy()
    if len(panel) == 0:
        raise RuntimeError("Filtered alpha panel is empty after applying ALPHA_UNIVERSE.")

    required_cols = ["Date", "Ticker", "px_t", "fwd_ret", "benchmark_fwd_ret", "rel_ret"] + list(cfg.RISK_COLS)
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise RuntimeError(f"alpha_panel missing required columns: {missing}")

    meta_path = os.path.join(cfg.PART1_DIR, "part1_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, "r"))
        H_meta = int(meta.get("horizon", cfg.H))
        if H_meta != int(cfg.H):
            print(f"[CONTRACT] Overriding cfg.H={cfg.H} -> {H_meta} from part1_meta.json")
            cfg.H = H_meta

    # -----------------------------
    # Walk-forward model engine
    # -----------------------------
    pos_df_raw, fit_info = run_panel_walk_forward(panel, cfg)

    # -----------------------------
    # Refresh / replace live date rows
    # -----------------------------
    live_date = pd.to_datetime(panel["Date"].max()).normalize()
    panel_live = panel[panel["Date"] == live_date].copy()

    if len(panel_live) == 0:
        raise RuntimeError("No live-date rows found in alpha panel.")

    live_scored = score_live_cross_section(
        panel_live=panel_live,
        feature_cols=fit_info["feature_cols"],
        risk_cols=fit_info["risk_cols"],
        X_fit_cols=fit_info["X_fit_cols"],
        risk_model=fit_info["risk_model"],
        clf_alpha=fit_info["clf_alpha"],
        reg_alpha=fit_info["reg_alpha"],
        mapper_state=fit_info["mapper_state"],
        p0_state=fit_info["p0_state"],
        gamma_alpha=fit_info["gamma_alpha"],
        alpha_sigma=fit_info["alpha_sigma"],
        gate_last=fit_info["gate_last"],
        cal_a_last=fit_info["cal_a"],
        cal_b_last=fit_info["cal_b"],
        cfg=cfg,
    )

    live_scored["alpha_target_realized"] = np.nan
    live_scored["y_alpha_realized"] = np.nan
    live_scored["y_alpha_avail"] = np.nan

    pos_df_raw = pos_df_raw[pos_df_raw["Date"] != live_date].copy()
    pos_df_raw = pd.concat([pos_df_raw, live_scored], axis=0, ignore_index=True)
    pos_df_raw = pos_df_raw.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # -----------------------------
    # Dynamic eligibility + hardened selection
    # -----------------------------
    pos_df, elig_hist = apply_dynamic_hardened_selection(pos_df_raw, cfg)

    sel_by_date = pos_df.groupby("Date")["selected"].sum()
    print(f"Dates with >=1 selected name: {int((sel_by_date > 0).sum())} of {sel_by_date.size}")
    print(f"Mean selected names per date: {sel_by_date.mean()}")
    print(f"Max selected names on any date: {sel_by_date.max()}")

#    sel_by_date = pos_df.groupby("Date")["selected"].sum()
#    print("Dates with >=1 selected name:", int((sel_by_date > 0).sum()), "of", int(sel_by_date.size))
#    print("Mean selected names per date:", float(sel_by_date.mean()))
#    print("Max selected names on any date:", int(sel_by_date.max()))

    # raw summary tape (pre-governance scaling)
    summary_raw = summarize_positions_by_date_safe(pos_df, cfg, weight_col="weight_raw")

    # date-level alpha governance
    summary_gov = alpha_drift_stream(summary_raw, cfg)

    # merge governance scale back to positions
    pos_df = pos_df.merge(
        summary_gov[["Date", "alpha_overlay_scale", "alpha_drift_alarm", "alpha_governance_tier"]],
        on="Date",
        how="left",
    )

    pos_df["weight"] = pos_df["weight_raw"] * pos_df["alpha_overlay_scale"]
    pos_df["is_live"] = (pd.to_datetime(pos_df["Date"]).dt.normalize() == live_date).astype(int)

    # final summary tape (post-governance scaling)
    summary_final = summarize_positions_by_date_safe(pos_df, cfg, weight_col="weight")
    summary_final = summary_final.merge(
        summary_gov[[
            "Date",
            "rank_ic_roll",
            "topk_rel_ret_net_roll",
            "breadth_roll",
            "alpha_drift_alarm",
            "alpha_overlay_scale",
            "alpha_governance_tier",
        ]],
        on="Date",
        how="left",
    )
    summary_final["is_live"] = (pd.to_datetime(summary_final["Date"]).dt.normalize() == live_date).astype(int)

    # -----------------------------
    # Audits
    # -----------------------------
    pos_real = pos_df.dropna(subset=["y_alpha_realized"]).copy()
    y_cls = pos_real["y_alpha_realized"].values.astype(int)
    prev = float(y_cls.mean()) if len(y_cls) else np.nan

    def _summ(y_true_local, p_pred):
        y_true_local = np.asarray(y_true_local).astype(int)
        p_pred = np.asarray(p_pred).astype(float)
        if len(y_true_local) == 0 or len(np.unique(y_true_local)) < 2:
            return {"auc": np.nan, "pr": np.nan, "lift": np.nan, "brier": np.nan, "ece": np.nan}
        auc = roc_auc_score(y_true_local, p_pred)
        pr = average_precision_score(y_true_local, p_pred)
        lift = pr / prev if prev > 0 else np.nan
        brier = float(np.mean((p_pred - y_true_local) ** 2))
        ece = ece_score(y_true_local, p_pred, n_bins=cfg.ECE_BINS)
        return {"auc": auc, "pr": pr, "lift": lift, "brier": brier, "ece": ece}

    raw_stats = _summ(y_cls, pos_real["p_alpha_raw"].values)
    cand_stats = _summ(y_cls, pos_real["p_alpha_cal_candidate"].values)
    final_stats = _summ(y_cls, pos_real["p_alpha_final"].values)

    # stage 0 broad-risk fit
    risk_rmse = _safe_rmse(pos_real["rel_ret"].values, pos_real["risk_mu_hat"].values)
    risk_mae = _safe_mae(pos_real["rel_ret"].values, pos_real["risk_mu_hat"].values)
    rel_mae_zero = _safe_mae(pos_real["rel_ret"].values, np.zeros(len(pos_real)))

    # stage 1 alpha fit
    alpha_rmse = _safe_rmse(pos_real["alpha_target_realized"].values, pos_real["mu_alpha_hat_final"].values)
    alpha_mae = _safe_mae(pos_real["alpha_target_realized"].values, pos_real["mu_alpha_hat_final"].values)
    alpha_mae_zero = _safe_mae(pos_real["alpha_target_realized"].values, np.zeros(len(pos_real)))

    # date-level alpha outcomes
    summary_real = summary_final.dropna(subset=["rank_ic", "topk_rel_ret_net"]).copy()

    mean_ic = float(summary_real["rank_ic"].mean()) if len(summary_real) else np.nan
    mean_topk = float(summary_real["topk_rel_ret_net"].mean()) if len(summary_real) else np.nan
    ir_topk = _annualized_ir(summary_real["topk_rel_ret_net"].values, cfg.H)
    mean_hit = float(summary_real["selection_hit_rate"].mean()) if len(summary_real) else np.nan
    mean_breadth = float(summary_real["breadth_selected"].mean()) if len(summary_real) else np.nan
    mean_eligible_breadth = float(summary_real["eligible_breadth"].mean()) if "eligible_breadth" in summary_real.columns else np.nan
    mean_budget = float(summary_real["gross_alpha_budget_used"].mean()) if len(summary_real) else np.nan

    gate_rate = float(pos_df["calibration_gate_on"].mean()) if "calibration_gate_on" in pos_df.columns else np.nan
    drift_rate = float(summary_final["alpha_drift_alarm"].mean()) if "alpha_drift_alarm" in summary_final.columns else np.nan

    # latest eligibility snapshot
    latest_elig = elig_hist[elig_hist["asof_date"] == live_date].copy()
    live_eligible_names = sorted(latest_elig.loc[latest_elig["in_elig_set"] == 1, "Ticker"].astype(str).tolist())
    live_eligible_count = int(len(live_eligible_names))

    print("\n" + "=" * 108)
    print("🏛️  PART 2A.2.1 AUDIT (Hardened Cross-Sectional Alpha Sleeve | filtered + stricter selection)")
    print("=" * 108)
    print(
        f"Panel rows: {len(pos_df)} | Realized panel rows: {len(pos_real)} | "
        f"Dates: {summary_final['Date'].nunique()} | Realized dates: {len(summary_real)} | "
        f"Alpha-up base rate: {prev:.4f}"
    )
    print(
        f"RAW:    AUC {raw_stats['auc']:.4f} | PR {raw_stats['pr']:.4f} | "
        f"Lift {raw_stats['lift']:.3f} | Brier {raw_stats['brier']:.6f} | ECE {raw_stats['ece']:.6f}"
    )
    print(
        f"CAND:   AUC {cand_stats['auc']:.4f} | PR {cand_stats['pr']:.4f} | "
        f"Lift {cand_stats['lift']:.3f} | Brier {cand_stats['brier']:.6f} | ECE {cand_stats['ece']:.6f}"
    )
    print(
        f"FINAL:  AUC {final_stats['auc']:.4f} | PR {final_stats['pr']:.4f} | "
        f"Lift {final_stats['lift']:.3f} | Brier {final_stats['brier']:.6f} | ECE {final_stats['ece']:.6f}"
    )
    print(
        f"Calibration gate ON rate: {gate_rate:.2%} | "
        f"Alpha drift alarm rate: {drift_rate:.2%}"
    )
    print(
        f"Stage 0 broad risk: RMSE {risk_rmse:.6f} | MAE {risk_mae:.6f} | "
        f"Zero-rel-ret MAE {rel_mae_zero:.6f}"
    )
    print(
        f"Stage 1 alpha residual: RMSE {alpha_rmse:.6f} | MAE {alpha_mae:.6f} | "
        f"Zero-alpha MAE {alpha_mae_zero:.6f}"
    )
    print(
        f"Date-level alpha: mean RankIC {mean_ic:.4f} | "
        f"mean TopK rel-ret net {mean_topk:.6f} | IR {ir_topk:.3f}"
    )
    print(
        f"Selection: hit rate {mean_hit:.2%} | mean breadth {mean_breadth:.2f} | "
        f"mean eligible breadth {mean_eligible_breadth:.2f} | mean budget used {mean_budget:.4f}"
    )
    print(f"Live eligible names ({live_eligible_count}): {live_eligible_names}")

    print("\nAlpha Governance Distribution:")
    print(summary_final["alpha_governance_tier"].value_counts(normalize=True).round(4))

    # -----------------------------
    # Falsification
    # -----------------------------
    fals_pass = True
    auc_shuf_med = np.nan

    if cfg.DO_FALSIFICATION and len(pos_real) and len(np.unique(pos_real["y_alpha_realized"].astype(int))) > 1:
        y_true_audit = pos_real["y_alpha_realized"].values.astype(int)
        p_pred_audit = pos_real["p_alpha_final"].values

        rng = np.random.default_rng(cfg.SEED)
        aucs = []
        for _ in range(cfg.SHUFFLE_B):
            y_shuf = block_shuffle_labels(
                y_true_audit,
                cfg.SHUFFLE_BLOCK,
                seed=int(rng.integers(1, 1_000_000))
            )
            if len(np.unique(y_shuf)) > 1:
                aucs.append(roc_auc_score(y_shuf, p_pred_audit))

        auc_shuf_med = float(np.median(aucs)) if len(aucs) else np.nan
        print(f"Shuffle-AUC (block={cfg.SHUFFLE_BLOCK}, B={cfg.SHUFFLE_B}) median: {auc_shuf_med:.4f}")
        fals_pass = (np.isfinite(auc_shuf_med) and auc_shuf_med <= cfg.MAX_SHUFFLE_AUC)

    MIN_REALIZED_DATES_PASS = 26  # about half a year of weekly observations

    final_pass = bool(
        len(summary_real) >= MIN_REALIZED_DATES_PASS and
        np.isfinite(mean_ic) and (mean_ic > 0.0) and
        np.isfinite(mean_topk) and (mean_topk > 0.0) and
        np.isfinite(ir_topk) and (ir_topk > 0.0) and
        np.isfinite(mean_breadth) and (mean_breadth >= 1.0) and
        fals_pass
    )

    print(f"Realized dates used for pass decision: {len(summary_real)}")
    print(f"\n✅ FINAL VERDICT (Part 2A.2.1): {'PASS' if final_pass else 'FAIL'}")
    print("Live date:", live_date)

    # -----------------------------
    # Outputs
    # -----------------------------
    positions_out = os.path.join(cfg.PRED_DIR, "part2a21_alpha_positions.csv")
    summary_tape_out = os.path.join(cfg.PRED_DIR, "part2a21_alpha_summary_tape.csv")
    summary_json_out = os.path.join(cfg.PRED_DIR, "part2a21_alpha_summary.json")
    eligibility_out = os.path.join(cfg.PRED_DIR, "part2a21_alpha_eligibility.csv")

    pos_df_out = pos_df.copy()
    pos_df_out["Date"] = pd.to_datetime(pos_df_out["Date"]).dt.normalize()
    pos_df_out.to_csv(positions_out, index=False)

    summary_final_out = summary_final.copy()
    summary_final_out["Date"] = pd.to_datetime(summary_final_out["Date"]).dt.normalize()
    summary_final_out.to_csv(summary_tape_out, index=False)

    elig_out = elig_hist.copy()
    elig_out["asof_date"] = pd.to_datetime(elig_out["asof_date"]).dt.normalize()
    elig_out.to_csv(eligibility_out, index=False)

    summary_payload = {
        "part": "part2a21_alpha",
        "version": "PART_2A21_HARDENED_CROSS_SECTIONAL_ALPHA_V1",
        "horizon": int(cfg.H),
        "holdout_start": cfg.HO_START_DATE,
        "panel_rows": int(len(pos_df)),
        "realized_panel_rows": int(len(pos_real)),
        "dates": int(summary_final["Date"].nunique()),
        "realized_dates": int(len(summary_real)),
        "classification_raw": raw_stats,
        "classification_cal_candidate": cand_stats,
        "classification_final_used": final_stats,
        "calibration_gate_on_rate": float(gate_rate) if np.isfinite(gate_rate) else np.nan,
        "alpha_drift_alarm_rate": float(drift_rate) if np.isfinite(drift_rate) else np.nan,
        "risk_rmse": float(risk_rmse) if np.isfinite(risk_rmse) else np.nan,
        "risk_mae": float(risk_mae) if np.isfinite(risk_mae) else np.nan,
        "rel_mae_zero": float(rel_mae_zero) if np.isfinite(rel_mae_zero) else np.nan,
        "alpha_rmse": float(alpha_rmse) if np.isfinite(alpha_rmse) else np.nan,
        "alpha_mae": float(alpha_mae) if np.isfinite(alpha_mae) else np.nan,
        "alpha_mae_zero": float(alpha_mae_zero) if np.isfinite(alpha_mae_zero) else np.nan,
        "mean_rank_ic": float(mean_ic) if np.isfinite(mean_ic) else np.nan,
        "mean_topk_rel_ret_net": float(mean_topk) if np.isfinite(mean_topk) else np.nan,
        "ir_topk_rel_ret_net": float(ir_topk) if np.isfinite(ir_topk) else np.nan,
        "mean_selection_hit_rate": float(mean_hit) if np.isfinite(mean_hit) else np.nan,
        "mean_breadth_selected": float(mean_breadth) if np.isfinite(mean_breadth) else np.nan,
        "mean_eligible_breadth": float(mean_eligible_breadth) if np.isfinite(mean_eligible_breadth) else np.nan,
        "mean_budget_used": float(mean_budget) if np.isfinite(mean_budget) else np.nan,
        "live_eligible_names": live_eligible_names,
        "shuffle_auc_median": float(auc_shuf_med) if np.isfinite(auc_shuf_med) else np.nan,
        "final_pass": bool(final_pass),
        "positions_out": positions_out,
        "summary_tape_out": summary_tape_out,
        "eligibility_out": eligibility_out,
    }

    with open(summary_json_out, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n✅ PART 2A.2.1 POSITION TAPE WRITTEN: {positions_out}")
    print(f"✅ PART 2A.2.1 SUMMARY TAPE WRITTEN:  {summary_tape_out}")
    print(f"✅ PART 2A.2.1 ELIGIBILITY WRITTEN:   {eligibility_out}")
    print(f"✅ PART 2A.2.1 SUMMARY JSON WRITTEN:  {summary_json_out}")

    # sanity
    pos_chk = pd.read_csv(positions_out)
    pos_chk["Date"] = pd.to_datetime(pos_chk["Date"])
    assert pos_chk["Date"].max() == live_date
    assert int(pos_chk.loc[pos_chk["Date"] == live_date, "is_live"].max()) == 1

    sum_chk = pd.read_csv(summary_tape_out)
    sum_chk["Date"] = pd.to_datetime(sum_chk["Date"])
    assert sum_chk["Date"].max() == live_date
    assert int(sum_chk.loc[sum_chk["Date"] == live_date, "is_live"].max()) == 1

    print("Sanity check passed | live positions + live summary row present.")

needed = [
    "TOP_K", "GROSS_ALPHA_BUDGET", "PER_NAME_CAP",
    "MIN_SCORE_LONG", "MIN_SCORE_GAP_TOP12",
    "MIN_MU_Z", "MIN_PROB_EDGE"
]
print({k: hasattr(CFG, k) for k in needed})
print({k: getattr(CFG, k, None) for k in needed})


if __name__ == "__main__":
    main(CFG)
