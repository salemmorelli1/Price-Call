#!/usr/bin/env python3
from __future__ import annotations
import os
import warnings
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# 0) Config
# ============================================================

@dataclass
class Part22Config:
    PART1_DIR: str = "./artifacts_part1"
    PRED_DIR: str = "./artifacts_part2_v77/predictions"

    # Core geometry
    H: int = 7
    REFIT_FREQ: int = 20
    TRAIN_WINDOW_Y: int = 4
    TRADING_DAYS_PER_YEAR: int = 252
    PURGE: int = 7

    # Bootstrap gate
    B_SAMPLES: int = 5000
    GATE_BLOCKS: tuple = (7, 14)
    TAIL_Q: float = 5.0
    MIN_TAIL_PER_BOOT: int = 5
    SEED: int = 42

    # Holdout anchor
    HO_START_DATE: str = "2020-01-01"

    # Guardrails
    MIN_TRAIN_ROWS: int = 200
    MIN_REBAL_COVERAGE: float = 0.98

    # Calibration module
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

    # Drift alarm
    DRIFT_ECE_MAX: float = 0.14
    DRIFT_BRIER_MAX: float = 0.19
    DRIFT_REL_ECE_MULT: float = 1.45
    DRIFT_REL_BRIER_MULT: float = 1.20
    DRIFT_PERSIST_K: int = 7

    # Falsification
    DO_FALSIFICATION: bool = True
    SHUFFLE_BLOCK: int = 14
    SHUFFLE_B: int = 200
    MAX_SHUFFLE_AUC: float = 0.515

    # Governance throttle
    ALPHA_THROTTLE: float = 0.5

    # Benchmark / cost model
    BENCHMARK_WEIGHT_VOO: float = 0.60
    BENCHMARK_WEIGHT_IEF: float = 0.40
    SLIP_BPS: float = 5.0

    # --------------------------------------------------------
    # NEW: one-sided downside mapping
    # --------------------------------------------------------
    DOWNSIDE_K: float = 1.15   # stronger reaction when tail risk rises
    UPSIDE_K: float = 0.20     # much smaller reaction when tail risk falls

    # High-risk state definition
    HIGH_RISK_ABS_P: float = 0.25
    HIGH_RISK_REL_MULT: float = 1.50
    HIGH_RISK_EXTRA_DOWNSIDE: float = 0.50  # extra multiplier on downside tilt

    # Asymmetric active caps
    BASE_MAX_UNDERWEIGHT: float = 0.20
    HIGH_RISK_MAX_UNDERWEIGHT: float = 0.30
    BASE_MAX_OVERWEIGHT: float = 0.08
    HIGH_RISK_MAX_OVERWEIGHT: float = 0.03

    # Hard absolute portfolio bounds
    MIN_W_VOO: float = 0.25
    MAX_W_VOO: float = 0.75

    # Regression / price-call artifacts
    DO_REGRESSION: bool = True
    REG_TRAIN_FILE: str = "regression_train.parquet"
    PRICE_LIVE_FILE: str = "price_calls_live_snapshot.parquet"
    FACTOR_FILE: str = "factor_returns.parquet"
    BENCHMARK_FILE: str = "benchmark_returns.parquet"

    # Regressor hyperparams
    REG_MAX_DEPTH: int = 2
    REG_N_ESTIMATORS: int = 200
    REG_LEARNING_RATE: float = 0.05
    REG_SUBSAMPLE: float = 0.9
    REG_COLSAMPLE: float = 0.9

    # Optional regression diagnostics
    PRINT_REG_AUDIT: bool = True


CFG = Part22Config()


# ============================================================
# 1) Helpers
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


def fit_mapper_v76(z_val, y_val, p0_vec):
    """
    Mapper:
        p_final_raw = lam * sigmoid(sign*z/T + b) + (1-lam) * p0
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


def compute_p0_state_from_revealed_labels(y_series: pd.Series, train_end_idx: int, H: int, seed_value: float) -> float:
    last_revealed = (train_end_idx - 1) - H
    if last_revealed <= 0:
        return float(seed_value)
    v = y_series.iloc[: last_revealed + 1].mean()
    return float(v) if np.isfinite(v) else float(seed_value)


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


# ============================================================
# 2) Non-circular MBB with tail-count guard
# ============================================================

def _mbb_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n - block_size + 1))
        idx.extend(range(start, start + block_size))
    return np.asarray(idx[:n], dtype=int)


def bootstrap_delta_cvar_mbb(bench_r, strat_r, B=5000, block_size=10, q=5.0, min_tail=5, seed=42):
    n = len(bench_r)
    if n == 0:
        return np.array([np.nan, np.nan, np.nan])

    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(B):
        idx = _mbb_indices(n, block_size, rng)
        b_r, s_r = bench_r[idx], strat_r[idx]
        qv = np.percentile(b_r, q)
        m = b_r <= qv
        if m.sum() >= min_tail:
            delta.append(float(s_r[m].mean() - b_r[m].mean()))

    if len(delta) < max(50, int(0.05 * B)):
        return np.array([np.nan, np.nan, np.nan])

    return np.percentile(delta, [5, 50, 95])


# ============================================================
# 3) Falsification
# ============================================================

def block_shuffle_labels(y, block_size, seed=42):
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    blocks = [y[i:i + block_size] for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


# ============================================================
# 4) Platt calibrator
# ============================================================

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


# ============================================================
# 4b) Regression utilities
# ============================================================

def _make_regressor(cfg: Part22Config):
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


def _fit_gamma_mae(y_true, y_hat):
    y_true = np.asarray(y_true, float)
    y_hat = np.asarray(y_hat, float)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    if m.sum() < 30:
        return 0.0
    gammas = np.linspace(0.0, 1.0, 51)
    maes = [np.mean(np.abs(y_true[m] - g * y_hat[m])) for g in gammas]
    return float(gammas[int(np.argmin(maes))])


# ============================================================
# 5) Drift alarm
# ============================================================

def drift_alarm_stream(ece_roll, brier_roll, cfg: Part22Config):
    ece_roll = np.asarray(ece_roll, dtype=float)
    brier_roll = np.asarray(brier_roll, dtype=float)
    n = len(ece_roll)

    base_n = min(n, max(cfg.ROLL_DIAG * 2, 20))
    base_ece = np.nanmedian(ece_roll[:base_n]) if base_n > 0 else np.nan
    base_brier = np.nanmedian(brier_roll[:base_n]) if base_n > 0 else np.nan

    alarm = np.zeros(n, dtype=int)
    streak = 0

    for i in range(n):
        e = ece_roll[i]
        b = brier_roll[i]

        cond_abs = (np.isfinite(e) and e > cfg.DRIFT_ECE_MAX) or (np.isfinite(b) and b > cfg.DRIFT_BRIER_MAX)

        cond_rel = False
        if np.isfinite(base_ece) and np.isfinite(e):
            cond_rel = cond_rel or (e > cfg.DRIFT_REL_ECE_MULT * base_ece)
        if np.isfinite(base_brier) and np.isfinite(b):
            cond_rel = cond_rel or (b > cfg.DRIFT_REL_BRIER_MULT * base_brier)

        if cond_abs or cond_rel:
            streak += 1
        else:
            streak = 0

        if streak >= cfg.DRIFT_PERSIST_K:
            alarm[i] = 1

    return alarm, float(base_ece), float(base_brier)


# ============================================================
# 6) Strategy / audit helpers
# ============================================================

def _annualized_ir(x, H):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or np.nanstd(x, ddof=1) <= 0:
        return np.nan
    return float(np.nanmean(x) / np.nanstd(x, ddof=1) * np.sqrt(252.0 / H))


def _downside_capture(strategy_ret, benchmark_ret):
    """
    Mean(strategy | bench<0) / mean(benchmark | bench<0)
    Both means are usually negative; ratio < 1 means shallower downside than benchmark.
    """
    s = np.asarray(strategy_ret, dtype=float)
    b = np.asarray(benchmark_ret, dtype=float)
    m = np.isfinite(s) & np.isfinite(b) & (b < 0)
    if m.sum() == 0:
        return np.nan
    denom = np.mean(b[m])
    if denom == 0:
        return np.nan
    return float(np.mean(s[m]) / denom)


def _summary_probs(y_true, p_pred, cfg: Part22Config):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "pr": np.nan, "lift": np.nan, "brier": np.nan, "ece": np.nan}

    prev = float(y_true.mean())
    auc = roc_auc_score(y_true, p_pred)
    pr = average_precision_score(y_true, p_pred)
    lift = (pr / prev) if prev > 0 else np.nan
    brier = float(np.mean((p_pred - y_true) ** 2))
    ece = ece_score(y_true, p_pred, n_bins=cfg.ECE_BINS)

    return {"auc": auc, "pr": pr, "lift": lift, "brier": brier, "ece": ece}


def _calibration_gate_stream(m: pd.DataFrame, cfg: Part22Config) -> pd.DataFrame:
    """
    Causal gate:
    Use candidate calibration only when trailing revealed-label diagnostics help.
    """
    m = m.copy()

    y_av = m["y_avail"].values.astype(float)
    p_raw = m["p_final_raw"].values.astype(float)
    p_cal_cand = m["p_final_cal_candidate"].values.astype(float)

    gate_on = np.zeros(len(m), dtype=int)
    gate_n = np.zeros(len(m), dtype=int)
    gate_brier_raw = np.full(len(m), np.nan, dtype=float)
    gate_brier_cal = np.full(len(m), np.nan, dtype=float)
    gate_ece_raw = np.full(len(m), np.nan, dtype=float)
    gate_ece_cal = np.full(len(m), np.nan, dtype=float)

    win_streak = 0

    for i in range(len(m)):
        hist_idx = np.where(
            np.isfinite(y_av[:i]) &
            np.isfinite(p_raw[:i]) &
            np.isfinite(p_cal_cand[:i])
        )[0]

        if len(hist_idx) == 0:
            gate_on[i] = 0
            continue

        hist_idx = hist_idx[-cfg.CAL_GATE_WINDOW:]
        y_hist = y_av[hist_idx].astype(int)
        pr_hist = p_raw[hist_idx]
        pc_hist = p_cal_cand[hist_idx]

        gate_n[i] = int(len(hist_idx))
        pos = int(np.sum(y_hist == 1))
        neg = int(np.sum(y_hist == 0))

        if (
            len(hist_idx) < cfg.CAL_GATE_MIN_SAMPLES or
            pos < cfg.CAL_GATE_MIN_POS or
            neg < cfg.CAL_GATE_MIN_NEG
        ):
            gate_on[i] = 0
            continue

        brier_raw = float(np.mean((pr_hist - y_hist) ** 2))
        brier_cal = float(np.mean((pc_hist - y_hist) ** 2))
        ece_raw = ece_score(y_hist, pr_hist, n_bins=cfg.ECE_BINS)
        ece_cal = ece_score(y_hist, pc_hist, n_bins=cfg.ECE_BINS)

        gate_brier_raw[i] = brier_raw
        gate_brier_cal[i] = brier_cal
        gate_ece_raw[i] = ece_raw
        gate_ece_cal[i] = ece_cal

        cond = (
            np.isfinite(brier_raw) and np.isfinite(brier_cal) and
            np.isfinite(ece_raw) and np.isfinite(ece_cal) and
            (brier_cal <= brier_raw - cfg.CAL_GATE_MIN_BRIER_IMPROVE) and
            (ece_cal <= ece_raw + cfg.CAL_GATE_MAX_ECE_WORSEN)
        )

        if cond:
            win_streak += 1
        else:
            win_streak = 0

        gate_on[i] = 1 if win_streak >= cfg.CAL_GATE_PERSIST_K else 0

    m["calibration_gate_on"] = gate_on
    m["calibration_gate_n"] = gate_n
    m["cal_gate_brier_raw"] = gate_brier_raw
    m["cal_gate_brier_cal"] = gate_brier_cal
    m["cal_gate_ece_raw"] = gate_ece_raw
    m["cal_gate_ece_cal"] = gate_ece_cal

    m["p_final_cal"] = np.where(
        m["calibration_gate_on"].values.astype(int) == 1,
        m["p_final_cal_candidate"].values,
        m["p_final_raw"].values,
    )

    return m


def _strategy_from_edge(m: pd.DataFrame, cfg: Part22Config) -> pd.DataFrame:
    """
    Part 2.2 allocation rule:
    - downside-dominant: higher tail risk drives stronger underweights
    - upside relief: lower tail risk only drives smaller overweights
    - high-risk states allow more underweighting and tighter overweights
    """
    m = m.copy()

    bench_w = float(cfg.BENCHMARK_WEIGHT_VOO)

    m["edge_score"] = m["p_final_cal"] - m["p0"]
    m["downside_edge"] = np.maximum(m["edge_score"], 0.0)
    m["upside_relief"] = np.maximum(-m["edge_score"], 0.0)

    # High-risk state flag
    high_risk_thr = np.maximum(
        float(cfg.HIGH_RISK_ABS_P),
        float(cfg.HIGH_RISK_REL_MULT) * m["p0"].values
    )
    m["high_risk_state"] = (m["p_final_cal"].values >= high_risk_thr).astype(int)

    # One-sided mapping
    downside_mult = 1.0 + float(cfg.HIGH_RISK_EXTRA_DOWNSIDE) * m["high_risk_state"].values

    m["downside_tilt_raw"] = -(
        float(cfg.DOWNSIDE_K) *
        downside_mult *
        m["downside_edge"].values / (m["p0"].values + 1e-9)
    )

    m["upside_tilt_raw"] = (
        float(cfg.UPSIDE_K) *
        m["upside_relief"].values / (1.0 - m["p0"].values + 1e-9)
    )

    m["active_weight_raw"] = m["downside_tilt_raw"] + m["upside_tilt_raw"]

    # Dynamic asymmetric caps
    max_under = np.where(
        m["high_risk_state"].values.astype(int) == 1,
        float(cfg.HIGH_RISK_MAX_UNDERWEIGHT),
        float(cfg.BASE_MAX_UNDERWEIGHT),
    )
    max_over = np.where(
        m["high_risk_state"].values.astype(int) == 1,
        float(cfg.HIGH_RISK_MAX_OVERWEIGHT),
        float(cfg.BASE_MAX_OVERWEIGHT),
    )

    m["max_underweight_cap"] = max_under
    m["max_overweight_cap"] = max_over

    m["active_weight_capped"] = np.minimum(
        np.maximum(m["active_weight_raw"].values, -max_under),
        max_over
    )

    # Governance scales the capped deviation, not the raw one
    m["w_benchmark_voo"] = bench_w
    m["w_strategy_voo_uncapped"] = (
        m["w_benchmark_voo"] + m["alpha_scale"].values * m["active_weight_capped"].values
    )

    m["w_strategy_voo"] = np.clip(
        m["w_strategy_voo_uncapped"].values,
        float(cfg.MIN_W_VOO),
        float(cfg.MAX_W_VOO),
    )

    m["strategy_ret_gross"] = (
        m["w_strategy_voo"].values * m["fwd_voo"].values +
        (1.0 - m["w_strategy_voo"].values) * m["fwd_ief"].values
    )

    if "bench_60_40" in m.columns:
        m["benchmark_ret"] = m["bench_60_40"].values
    else:
        m["benchmark_ret"] = (
            float(cfg.BENCHMARK_WEIGHT_VOO) * m["fwd_voo"].values +
            float(cfg.BENCHMARK_WEIGHT_IEF) * m["fwd_ief"].values
        )

    m["active_ret_gross"] = m["strategy_ret_gross"].values - m["benchmark_ret"].values

    m["turnover"] = pd.Series(m["w_strategy_voo"], index=m.index).diff().abs().fillna(0.0).values
    m["cost_model"] = (float(cfg.SLIP_BPS) / 10000.0) * m["turnover"].values * 2.0

    m["strategy_ret_net"] = m["strategy_ret_gross"].values - m["cost_model"].values
    m["active_ret_net"] = m["strategy_ret_net"].values - m["benchmark_ret"].values

    return m


# ============================================================
# 7) Main
# ============================================================

def main(cfg: Part22Config):
    os.makedirs(cfg.PRED_DIR, exist_ok=True)

    # -----------------------------
    # Load Part 1 artifacts
    # -----------------------------
    X = pd.read_parquet(os.path.join(cfg.PART1_DIR, "X_features.parquet"))
    y = pd.read_parquet(os.path.join(cfg.PART1_DIR, "y_labels_revealed.parquet"))
    factors = pd.read_parquet(os.path.join(cfg.PART1_DIR, cfg.FACTOR_FILE))
    benchmarks = pd.read_parquet(os.path.join(cfg.PART1_DIR, cfg.BENCHMARK_FILE))

    X.index = pd.to_datetime(X.index).tz_localize(None).normalize()
    y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
    factors.index = pd.to_datetime(factors.index).tz_localize(None).normalize()
    benchmarks.index = pd.to_datetime(benchmarks.index).tz_localize(None).normalize()

    # -----------------------------
    # Contract: read Part 1 meta
    # -----------------------------
    meta_path = os.path.join(cfg.PART1_DIR, "part1_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, "r"))
        H_meta = int(meta.get("horizon", cfg.H))
        if H_meta != int(cfg.H):
            print(f"[CONTRACT] Overriding cfg.H={cfg.H} -> {H_meta} from part1_meta.json")
            cfg.H = H_meta

        tail_label_name = meta.get("tail_label_name", "y_voo")
        feat_cols_meta = meta.get("feature_cols", None)
    else:
        meta = {}
        tail_label_name = "y_voo"
        feat_cols_meta = None

    if feat_cols_meta is not None:
        missing = [c for c in feat_cols_meta if c not in X.columns]
        extra = [c for c in X.columns if c not in feat_cols_meta]
        if missing:
            raise RuntimeError(f"[CONTRACT] Missing features vs meta: {missing}")
        if extra:
            print(f"[CONTRACT] Extra features vs meta (allowed but tracked): {extra}")

    # -----------------------------
    # Label selection
    # -----------------------------
    y_col = tail_label_name if tail_label_name in y.columns else ("y_voo" if "y_voo" in y.columns else None)
    if y_col is None:
        raise RuntimeError(
            f"[CONTRACT] No label column found. meta tail_label_name={tail_label_name}, y.columns={list(y.columns)}"
        )

    if y_col != "y_voo":
        y = y.rename(columns={y_col: "y_voo"})
        print(f"[CONTRACT] Using label '{y_col}' as y_voo (normalized).")

    # -----------------------------
    # Optional regression artifacts
    # -----------------------------
    reg_train = None
    px_live = None

    if cfg.DO_REGRESSION:
        reg_path = os.path.join(cfg.PART1_DIR, cfg.REG_TRAIN_FILE)
        px_path = os.path.join(cfg.PART1_DIR, cfg.PRICE_LIVE_FILE)

        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"Regression mode requires {reg_path}")
        if not os.path.exists(px_path):
            raise FileNotFoundError(f"Regression mode requires {px_path}")

        reg_train = pd.read_parquet(reg_path)
        px_live = pd.read_parquet(px_path)

        reg_train.index = pd.to_datetime(reg_train.index).tz_localize(None).normalize()
        px_live.index = pd.to_datetime(px_live.index).tz_localize(None).normalize()

    feat_cols = list(X.columns)

    # Main row-space dataframe
    df = (
        X.join(y, how="left")
         .join(factors, how="left")
         .join(benchmarks, how="left")
         .sort_index()
    )

    # -----------------------------
    # Holdout anchor
    # -----------------------------
    ho_start = pd.Timestamp(cfg.HO_START_DATE)
    ho_idx = np.where(df.index >= ho_start)[0]
    if len(ho_idx) == 0:
        raise ValueError("No holdout rows found. Check HO_START_DATE and data index.")

    ho0 = int(ho_idx[0])
    seed_end = max(0, ho0 - cfg.H)
    seed_value = float(df.iloc[:seed_end]["y_voo"].mean()) if seed_end > 0 else float(df["y_voo"].mean())

    # -----------------------------
    # Storage
    # -----------------------------
    fwd_voo_hat = pd.Series(index=df.index, dtype=float)
    fwd_ief_hat = pd.Series(index=df.index, dtype=float)
    fwd_spd_hat = pd.Series(index=df.index, dtype=float)

    reg_voo = reg_ief = reg_spd = None

    z_raw = pd.Series(index=df.index, dtype=float)
    p0_state_row = pd.Series(index=df.index, dtype=float)
    calib_states = []

    # -----------------------------
    # Classifier
    # -----------------------------
    clf = xgb.XGBClassifier(
        max_depth=2,
        n_estimators=100,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=cfg.SEED,
    )

    train_window_rows = int(cfg.TRAIN_WINDOW_Y * cfg.TRADING_DAYS_PER_YEAR)
    anchors = ho_idx[::cfg.REFIT_FREQ]

    # -----------------------------
    # Walk-forward in row space
    # -----------------------------
    for d_idx in anchors:
        d_idx = int(d_idx)

        train_end = d_idx - cfg.H - cfg.PURGE
        if train_end <= 0:
            continue

        train_start = max(0, train_end - train_window_rows)
        df_tr = df.iloc[train_start:train_end].copy()

        if len(df_tr) < cfg.MIN_TRAIN_ROWS:
            continue

        p0_state = compute_p0_state_from_revealed_labels(
            df["y_voo"], train_end_idx=train_end, H=cfg.H, seed_value=seed_value
        )

        y_avail = df_tr["y_voo"].shift(cfg.H)
        p0_vec = y_avail.expanding(min_periods=1).mean().fillna(p0_state)
        df_tr["p0_causal_vec"] = p0_vec

        split = int(len(df_tr) * 0.8)

        clf.fit(df_tr.iloc[:split][feat_cols], df_tr.iloc[:split]["y_voo"].values.astype(int))

        z_cal = clf.predict(df_tr.iloc[split:][feat_cols], output_margin=True)
        y_cal = df_tr.iloc[split:]["y_voo"].values.astype(int)
        p0_cal = df_tr.iloc[split:]["p0_causal_vec"].values.astype(float)

        s, (T, b, lam) = fit_mapper_v76(z_cal, y_cal, p0_cal)
        calib_states.append({"Date": df.index[d_idx], "sign": s, "T": T, "b": b, "lam": lam})

        # Regression fits
        if cfg.DO_REGRESSION and reg_train is not None:
            d0 = df.index[train_start]
            d1 = df.index[train_end - 1]

            rt = reg_train.loc[(reg_train.index >= d0) & (reg_train.index <= d1)].copy()
            if len(rt) >= cfg.MIN_TRAIN_ROWS:
                Xrt = rt[feat_cols].copy()

                reg_voo = _make_regressor(cfg)
                reg_ief = _make_regressor(cfg)
                reg_spd = _make_regressor(cfg)

                reg_voo.fit(Xrt, rt["fwd_voo"].values.astype(float))
                reg_ief.fit(Xrt, rt["fwd_ief"].values.astype(float))
                reg_spd.fit(Xrt, rt["fwd_spread"].values.astype(float))

                test_end = min(len(df), d_idx + cfg.REFIT_FREQ)
                Xch = df.iloc[d_idx:test_end][feat_cols].copy()

                fwd_voo_hat.iloc[d_idx:test_end] = reg_voo.predict(Xch)
                fwd_ief_hat.iloc[d_idx:test_end] = reg_ief.predict(Xch)
                fwd_spd_hat.iloc[d_idx:test_end] = reg_spd.predict(Xch)

        test_end = min(len(df), d_idx + cfg.REFIT_FREQ)
        chunk = df.iloc[d_idx:test_end]
        z_raw.iloc[d_idx:test_end] = clf.predict(chunk[feat_cols], output_margin=True)
        p0_state_row.iloc[d_idx:test_end] = p0_state

    if len(calib_states) == 0:
        raise RuntimeError("No calibration states produced. Check training window and data coverage.")

    calib_df = pd.DataFrame(calib_states).set_index("Date").sort_index()

    # -----------------------------
    # Rebalance grid
    # -----------------------------
    rebalance_idx = np.arange(ho_idx[0], len(df), cfg.H, dtype=int)
    rebalance_dates = df.index[rebalance_idx]

    m = df.loc[rebalance_dates].copy().sort_index()
    m["z_raw"] = z_raw.loc[rebalance_dates].values
    m["p0"] = p0_state_row.loc[rebalance_dates].values

    # Add regression outputs
    if cfg.DO_REGRESSION and px_live is not None:
        m["fwd_voo_hat"] = fwd_voo_hat.loc[rebalance_dates].values
        m["fwd_ief_hat"] = fwd_ief_hat.loc[rebalance_dates].values
        m["fwd_spread_hat"] = fwd_spd_hat.loc[rebalance_dates].values

        m["fwd_spread_hat_from_legs"] = m["fwd_voo_hat"] - m["fwd_ief_hat"]
        m["spread_model_gap"] = m["fwd_spread_hat"] - m["fwd_spread_hat_from_legs"]

        px_m = px_live.reindex(m.index)
        m["px_voo_t"] = px_m["px_voo_t"].values if "px_voo_t" in px_m.columns else np.nan
        m["px_ief_t"] = px_m["px_ief_t"].values if "px_ief_t" in px_m.columns else np.nan

    # -----------------------------
    # Forecast shrinkage
    # -----------------------------
    if cfg.DO_REGRESSION and "fwd_voo_hat" in m.columns:
        gamma_voo = np.nan
        gamma_ief = np.nan
        Wg = 52

        m_mature = m.dropna(subset=["fwd_voo", "fwd_voo_hat", "fwd_ief", "fwd_ief_hat"]).copy()

        if len(m_mature) >= 60:
            tail = m_mature.tail(Wg)
            gamma_voo = _fit_gamma_mae(tail["fwd_voo"].values, tail["fwd_voo_hat"].values)
            gamma_ief = _fit_gamma_mae(tail["fwd_ief"].values, tail["fwd_ief_hat"].values)

        m["gamma_voo"] = gamma_voo
        m["gamma_ief"] = gamma_ief

        m["fwd_voo_hat_final"] = gamma_voo * m["fwd_voo_hat"] if np.isfinite(gamma_voo) else 0.0
        m["fwd_ief_hat_final"] = gamma_ief * m["fwd_ief_hat"] if np.isfinite(gamma_ief) else 0.0

        m["px_voo_call_7d"] = m["px_voo_t"] * np.exp(m["fwd_voo_hat_final"])
        m["px_ief_call_7d"] = m["px_ief_t"] * np.exp(m["fwd_ief_hat_final"])

        m["px_voo_real_7d"] = np.where(
            np.isfinite(m.get("px_voo_t", np.nan)) & np.isfinite(m.get("fwd_voo", np.nan)),
            m["px_voo_t"] * np.exp(m["fwd_voo"]),
            np.nan,
        )
        m["px_ief_real_7d"] = np.where(
            np.isfinite(m.get("px_ief_t", np.nan)) & np.isfinite(m.get("fwd_ief", np.nan)),
            m["px_ief_t"] * np.exp(m["fwd_ief"]),
            np.nan,
        )

        m["err_px_voo"] = m["px_voo_call_7d"] - m["px_voo_real_7d"]
        m["err_px_ief"] = m["px_ief_call_7d"] - m["px_ief_real_7d"]

        m["err_r_voo"] = m["fwd_voo_hat_final"] - m["fwd_voo"]
        m["err_r_ief"] = m["fwd_ief_hat_final"] - m["fwd_ief"]

        m["hit_sign_voo"] = np.where(
            np.isfinite(m["fwd_voo_hat_final"].values) & np.isfinite(m["fwd_voo"].values),
            (np.sign(m["fwd_voo_hat_final"].values) == np.sign(m["fwd_voo"].values)).astype(float),
            np.nan,
        )

    # -----------------------------
    # Coverage guard
    # -----------------------------
    cover = np.isfinite(m["z_raw"].values).mean()
    if cover < cfg.MIN_REBAL_COVERAGE:
        raise RuntimeError(f"CRITICAL: z_raw coverage too low: {cover:.2%} < {cfg.MIN_REBAL_COVERAGE:.2%}")

    # -----------------------------
    # Apply calibration state by asof
    # -----------------------------
    m = pd.merge_asof(
        m.reset_index().rename(columns={"index": "Date"}),
        calib_df.reset_index().rename(columns={"Date": "CalibDate"}),
        left_on="Date",
        right_on="CalibDate",
        direction="backward",
    ).set_index("Date")

    # -----------------------------
    # Raw probabilities
    # -----------------------------
    m["p_final_raw"] = (
        m["lam"] * _sigmoid((m["sign"] * m["z_raw"]) / m["T"] + m["b"])
        + (1.0 - m["lam"]) * m["p0"]
    )
    m = m.dropna(subset=["p_final_raw", "p0"])

    # -----------------------------
    # Calibration candidate
    # IMPORTANT:
    # rebalance tape reveal lag is shift(1), not shift(H)
    # -----------------------------
    m["y_avail"] = m["y_voo"].shift(1)

    x_all = _logit(m["p_final_raw"].values, clip=cfg.CAL_CLIP)
    y_av = m["y_avail"].values

    p_cal_cand = np.full(len(m), np.nan, dtype=float)
    cal_a = np.full(len(m), np.nan, dtype=float)
    cal_b = np.full(len(m), np.nan, dtype=float)
    cal_n = np.zeros(len(m), dtype=int)

    for i in range(len(m)):
        mask = np.isfinite(y_av[:i])
        n_obs = int(mask.sum())
        cal_n[i] = n_obs

        if n_obs < cfg.CAL_MIN_SAMPLES:
            p_cal_cand[i] = float(m["p_final_raw"].iloc[i])
            continue

        y_fit = y_av[:i][mask].astype(int)
        x_fit = x_all[:i][mask]

        pos = int(np.sum(y_fit == 1))
        neg = int(np.sum(y_fit == 0))

        if pos < cfg.CAL_MIN_POS or neg < cfg.CAL_MIN_NEG:
            p_cal_cand[i] = float(m["p_final_raw"].iloc[i])
            continue

        a, b0 = fit_platt_1d(x_fit, y_fit, clip=cfg.CAL_CLIP)
        cal_a[i], cal_b[i] = a, b0
        p_cal_cand[i] = float(apply_platt(a, b0, x_all[i]))

    m["p_final_cal_candidate"] = np.clip(p_cal_cand, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP)
    m["cal_a"], m["cal_b"], m["cal_n"] = cal_a, cal_b, cal_n

    # -----------------------------
    # Calibration gate
    # -----------------------------
    m = _calibration_gate_stream(m, cfg)

    # -----------------------------
    # Rolling diagnostics on FINAL used probabilities
    # -----------------------------
    y_true = m["y_voo"].values.astype(float)
    p_used = m["p_final_cal"].values.astype(float)
    y_av2 = m["y_avail"].values.astype(float)

    ece_roll = np.full(len(m), np.nan, dtype=float)
    brier_roll = np.full(len(m), np.nan, dtype=float)
    ece_av_roll = np.full(len(m), np.nan, dtype=float)
    brier_av_roll = np.full(len(m), np.nan, dtype=float)

    W = cfg.ROLL_DIAG
    for i in range(len(m)):
        j0 = max(0, i - W + 1)
        sl = slice(j0, i + 1)

        mask_r = np.isfinite(y_true[sl])
        if mask_r.any():
            ece_roll[i] = ece_score(y_true[sl][mask_r].astype(int), p_used[sl][mask_r], n_bins=cfg.ECE_BINS)
            brier_roll[i] = float(np.mean((p_used[sl][mask_r] - y_true[sl][mask_r]) ** 2))

        mask_a = np.isfinite(y_av2[sl])
        if mask_a.any():
            yv = y_av2[sl][mask_a].astype(int)
            pv = p_used[sl][mask_a]
            ece_av_roll[i] = ece_score(yv, pv, n_bins=cfg.ECE_BINS)
            brier_av_roll[i] = float(np.mean((pv - yv) ** 2))

    m["ece_roll"], m["brier_roll"] = ece_roll, brier_roll
    m["ece_avail_roll"], m["brier_avail_roll"] = ece_av_roll, brier_av_roll

    use_e = np.where(np.isfinite(m["ece_avail_roll"].values), m["ece_avail_roll"].values, m["ece_roll"].values)
    use_b = np.where(np.isfinite(m["brier_avail_roll"].values), m["brier_avail_roll"].values, m["brier_roll"].values)

    alarm, base_ece, base_brier = drift_alarm_stream(use_e, use_b, cfg)
    m["drift_alarm"] = alarm
    m["drift_base_ece"] = base_ece
    m["drift_base_brier"] = base_brier

    # -----------------------------
    # Two-tier governance
    # -----------------------------
    m["alpha_scale"] = np.where(m["drift_alarm"] == 1, cfg.ALPHA_THROTTLE, 1.0)
    m["governance_tier"] = np.where(m["drift_alarm"] == 1, "DRIFT_THROTTLE", "NORMAL")

    # -----------------------------
    # Benchmark-relative strategy fields
    # -----------------------------
    m = _strategy_from_edge(m, cfg)

    # -----------------------------
    # Classification audit
    # -----------------------------
    m_real = m.dropna(subset=["y_voo"]).copy()
    y_cls = m_real["y_voo"].values.astype(int)
    prev = float(y_cls.mean()) if len(y_cls) else np.nan

    raw_stats = _summary_probs(y_cls, m_real["p_final_raw"].values, cfg)
    cand_stats = _summary_probs(y_cls, m_real["p_final_cal_candidate"].values, cfg)
    final_stats = _summary_probs(y_cls, m_real["p_final_cal"].values, cfg)

    # -----------------------------
    # Benchmark-relative audit
    # -----------------------------
    audit_cols = [
        "strategy_ret_gross",
        "strategy_ret_net",
        "benchmark_ret",
        "active_ret_gross",
        "active_ret_net",
    ]
    m_audit = m.dropna(subset=["fwd_voo", "fwd_ief"]).copy()
    m_audit = m_audit.dropna(subset=audit_cols)

    active_ir = _annualized_ir(m_audit["active_ret_net"].values, cfg.H)
    strat_ir = _annualized_ir(m_audit["strategy_ret_net"].values, cfg.H)
    active_mean = float(np.nanmean(m_audit["active_ret_net"].values)) if len(m_audit) else np.nan
    active_hit = float(np.nanmean((m_audit["active_ret_net"].values > 0).astype(float))) if len(m_audit) else np.nan

    gate_rate = float(m["calibration_gate_on"].mean()) if "calibration_gate_on" in m.columns else np.nan
    avg_abs_active_raw = float(np.nanmean(np.abs(m_audit["active_weight_raw"].values))) if len(m_audit) else np.nan
    avg_abs_active_capped = float(np.nanmean(np.abs(m_audit["active_weight_capped"].values))) if len(m_audit) else np.nan
    high_risk_rate = float(m["high_risk_state"].mean()) if "high_risk_state" in m.columns else np.nan

    m_down = m_audit.loc[m_audit["benchmark_ret"] < 0].copy()
    neg_bench_active_mean = float(np.nanmean(m_down["active_ret_net"].values)) if len(m_down) else np.nan
    neg_bench_hit = float(np.nanmean((m_down["active_ret_net"].values > 0).astype(float))) if len(m_down) else np.nan
    downside_capture = _downside_capture(m_audit["strategy_ret_net"].values, m_audit["benchmark_ret"].values)

    print("\n" + "=" * 96)
    print("🏛️  PART 2.2 AUDIT (One-Sided Downside Mapping + Conservative High-Risk States)")
    print("=" * 96)
    print(f"Rows (rebalance): {len(m)} | Realized rows: {len(m_real)} | Audit rows: {len(m_audit)} | Base rate: {prev:.4f}")
    print(f"RAW:    AUC {raw_stats['auc']:.4f} | PR {raw_stats['pr']:.4f} | Lift {raw_stats['lift']:.3f} | Brier {raw_stats['brier']:.6f} | ECE {raw_stats['ece']:.6f}")
    print(f"CAND:   AUC {cand_stats['auc']:.4f} | PR {cand_stats['pr']:.4f} | Lift {cand_stats['lift']:.3f} | Brier {cand_stats['brier']:.6f} | ECE {cand_stats['ece']:.6f}")
    print(f"FINAL:  AUC {final_stats['auc']:.4f} | PR {final_stats['pr']:.4f} | Lift {final_stats['lift']:.3f} | Brier {final_stats['brier']:.6f} | ECE {final_stats['ece']:.6f}")
    print(f"Calibration gate ON rate: {gate_rate:.2%}")
    print(f"High-risk state rate: {high_risk_rate:.2%}")
    print(f"Drift baseline: ECE {base_ece:.6f} | Brier {base_brier:.6f} | Alarm rate: {m['drift_alarm'].mean():.2%}")
    print(f"Net active return: mean {active_mean:.6f} | hit rate {active_hit:.2%} | IR {active_ir:.3f}")
    print(f"Net strategy IR: {strat_ir:.3f}")
    print(f"Negative benchmark states: active mean {neg_bench_active_mean:.6f} | hit rate {neg_bench_hit:.2%} | downside capture {downside_capture:.3f}")
    print(f"Avg |active tilt| raw {avg_abs_active_raw:.4f} | capped {avg_abs_active_capped:.4f}")

    print("\nGovernance Tier Distribution:")
    print(m["governance_tier"].value_counts(normalize=True).round(4))

    # -----------------------------
    # Consensus gate: strategy vs benchmark in benchmark tail
    # -----------------------------
    consensus_pass = True
    for bsz in cfg.GATE_BLOCKS:
        ci = bootstrap_delta_cvar_mbb(
            m_audit["benchmark_ret"].values,
            m_audit["strategy_ret_net"].values,
            B=cfg.B_SAMPLES,
            block_size=int(bsz),
            q=cfg.TAIL_Q,
            min_tail=cfg.MIN_TAIL_PER_BOOT,
            seed=cfg.SEED,
        )
        if not np.isfinite(ci[0]):
            consensus_pass = False
            print(f"Block {bsz:2}: ΔCVaR(strategy - benchmark) low: NaN | FAIL (insufficient tail support)")
        else:
            block_pass = (ci[0] >= -0.0001)
            consensus_pass = consensus_pass and block_pass
            print(f"Block {bsz:2}: ΔCVaR(strategy - benchmark) low {ci[0]:.4%} | med {ci[1]:.4%} | high {ci[2]:.4%} | {'PASS' if block_pass else 'FAIL'}")

    # -----------------------------
    # Falsification
    # -----------------------------
    fals_pass = True
    auc_shuf_med = np.nan

    if cfg.DO_FALSIFICATION and len(m_real) and len(np.unique(m_real["y_voo"].astype(int))) > 1:
        y_true_audit = m_real["y_voo"].values.astype(int)
        p_pred_audit = m_real["p_final_cal"].values

        rng = np.random.default_rng(cfg.SEED)
        aucs = []
        for _ in range(cfg.SHUFFLE_B):
            y_shuf = block_shuffle_labels(y_true_audit, cfg.SHUFFLE_BLOCK, seed=int(rng.integers(1, 1_000_000)))
            if len(np.unique(y_shuf)) > 1:
                aucs.append(roc_auc_score(y_shuf, p_pred_audit))

        auc_shuf_med = float(np.median(aucs)) if len(aucs) else np.nan
        print(f"Shuffle-AUC (block={cfg.SHUFFLE_BLOCK}, B={cfg.SHUFFLE_B}) median: {auc_shuf_med:.4f}")
        fals_pass = (np.isfinite(auc_shuf_med) and auc_shuf_med <= cfg.MAX_SHUFFLE_AUC)

    final_pass = bool(consensus_pass and fals_pass)
    print(f"\n✅ FINAL VERDICT (Part 2.2): {'PASS' if final_pass else 'FAIL'}")

    # -----------------------------
    # Regression audit
    # -----------------------------
    if cfg.DO_REGRESSION and cfg.PRINT_REG_AUDIT:
        need = ["fwd_voo", "fwd_ief", "fwd_voo_hat_final", "fwd_ief_hat_final"]
        have_need = all(c in m.columns for c in need)

        if not have_need:
            print("\n" + "-" * 96)
            print("📈 REGRESSION AUDIT")
            print("-" * 96)
            print(f"Skipped: missing columns {need}")
        else:
            m_reg = m.dropna(subset=need).copy()
            if len(m_reg) == 0:
                print("\n" + "-" * 96)
                print("📈 REGRESSION AUDIT")
                print("-" * 96)
                print("Skipped: no realized rows available.")
            else:
                rmse_voo = _safe_rmse(m_reg["fwd_voo"].values, m_reg["fwd_voo_hat_final"].values)
                rmse_ief = _safe_rmse(m_reg["fwd_ief"].values, m_reg["fwd_ief_hat_final"].values)
                mae_voo = _safe_mae(m_reg["fwd_voo"].values, m_reg["fwd_voo_hat_final"].values)
                mae_ief = _safe_mae(m_reg["fwd_ief"].values, m_reg["fwd_ief_hat_final"].values)

                mae_r_voo_naive = _safe_mae(m_reg["fwd_voo"].values, np.zeros(len(m_reg)))
                mae_r_ief_naive = _safe_mae(m_reg["fwd_ief"].values, np.zeros(len(m_reg)))

                if "px_voo_t" in m_reg.columns and "px_voo_call_7d" in m_reg.columns:
                    px_real = m_reg["px_voo_t"].values * np.exp(m_reg["fwd_voo"].values)
                    mae_px_model = _safe_mae(px_real, m_reg["px_voo_call_7d"].values)
                    mae_px_naive = _safe_mae(px_real, m_reg["px_voo_t"].values)
                else:
                    mae_px_model = np.nan
                    mae_px_naive = np.nan

                print("\n" + "-" * 96)
                print("📈 REGRESSION AUDIT (monitor-only)")
                print("-" * 96)
                print(f"Rows used: {len(m_reg)}")
                print(f"RMSE: fwd_voo {rmse_voo:.6f} | fwd_ief {rmse_ief:.6f}")
                print(f" MAE: fwd_voo {mae_voo:.6f} | fwd_ief {mae_ief:.6f}")
                print(f"Naive MAE (returns): VOO {mae_r_voo_naive:.6f} | IEF {mae_r_ief_naive:.6f}")
                print(f"Price MAE: model {mae_px_model:.6f} | naive(P_t) {mae_px_naive:.6f}")

    # -----------------------------
    # LIVE ROW APPEND / UPDATE
    # -----------------------------
    m["is_live"] = 0

    try:
        live_date = pd.to_datetime(X.index.max()).tz_localize(None).normalize()
        print("X last date:", live_date)

        base = df.loc[live_date].copy()
        X_live_row = X.loc[[live_date], feat_cols]
        z_live = float(clf.predict(X_live_row, output_margin=True)[0])

        live_pos = int(np.where(df.index == live_date)[0][0])
        train_end_idx_live = live_pos + 1

        p0_live = float(
            compute_p0_state_from_revealed_labels(
                df["y_voo"], train_end_idx=train_end_idx_live, H=cfg.H, seed_value=seed_value
            )
        )

        calib_asof = calib_df.loc[:live_date].iloc[-1]
        s_live = float(calib_asof["sign"])
        T_live = float(calib_asof["T"])
        b_live = float(calib_asof["b"])
        lam_live = float(calib_asof["lam"])

        p_raw_live = float(
            lam_live * _sigmoid((s_live * z_live) / (T_live + 1e-12) + b_live)
            + (1.0 - lam_live) * p0_live
        )
        p_raw_live = float(np.clip(p_raw_live, cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP))

        a_last = m["cal_a"].dropna().iloc[-1] if m["cal_a"].notna().any() else np.nan
        b_last = m["cal_b"].dropna().iloc[-1] if m["cal_b"].notna().any() else np.nan
        gate_last = int(m["calibration_gate_on"].iloc[-1]) if "calibration_gate_on" in m.columns else 0

        if np.isfinite(a_last) and np.isfinite(b_last):
            x_live = float(_logit(p_raw_live, clip=cfg.CAL_CLIP))
            p_cal_live_cand = float(np.clip(apply_platt(float(a_last), float(b_last), x_live), cfg.CAL_CLIP, 1.0 - cfg.CAL_CLIP))
        else:
            p_cal_live_cand = p_raw_live

        p_final_live = p_cal_live_cand if gate_last == 1 else p_raw_live

        if live_date in m.index:
            m.loc[live_date, "is_live"] = 1
            m.loc[live_date, "p0"] = p0_live
            m.loc[live_date, "z_raw"] = z_live
            m.loc[live_date, "sign"] = s_live
            m.loc[live_date, "T"] = T_live
            m.loc[live_date, "b"] = b_live
            m.loc[live_date, "lam"] = lam_live
            m.loc[live_date, "p_final_raw"] = p_raw_live
            m.loc[live_date, "p_final_cal_candidate"] = p_cal_live_cand
            m.loc[live_date, "p_final_cal"] = p_final_live
            m.loc[live_date, "calibration_gate_on"] = gate_last
        else:
            live_row = pd.Series({c: np.nan for c in m.columns}, name=live_date)

            for c in m.columns:
                if c in base.index and pd.notna(base[c]):
                    live_row[c] = base[c]

            live_row["z_raw"] = z_live
            live_row["p0"] = p0_live
            live_row["sign"] = s_live
            live_row["T"] = T_live
            live_row["b"] = b_live
            live_row["lam"] = lam_live
            live_row["p_final_raw"] = p_raw_live
            live_row["p_final_cal_candidate"] = p_cal_live_cand
            live_row["p_final_cal"] = p_final_live
            live_row["cal_a"] = a_last
            live_row["cal_b"] = b_last
            live_row["cal_n"] = m["cal_n"].dropna().iloc[-1] if m["cal_n"].notna().any() else 0

            live_row["calibration_gate_on"] = gate_last
            live_row["calibration_gate_n"] = m["calibration_gate_n"].dropna().iloc[-1] if "calibration_gate_n" in m.columns and m["calibration_gate_n"].notna().any() else 0
            live_row["cal_gate_brier_raw"] = m["cal_gate_brier_raw"].dropna().iloc[-1] if "cal_gate_brier_raw" in m.columns and m["cal_gate_brier_raw"].notna().any() else np.nan
            live_row["cal_gate_brier_cal"] = m["cal_gate_brier_cal"].dropna().iloc[-1] if "cal_gate_brier_cal" in m.columns and m["cal_gate_brier_cal"].notna().any() else np.nan
            live_row["cal_gate_ece_raw"] = m["cal_gate_ece_raw"].dropna().iloc[-1] if "cal_gate_ece_raw" in m.columns and m["cal_gate_ece_raw"].notna().any() else np.nan
            live_row["cal_gate_ece_cal"] = m["cal_gate_ece_cal"].dropna().iloc[-1] if "cal_gate_ece_cal" in m.columns and m["cal_gate_ece_cal"].notna().any() else np.nan

            last = m.sort_index().iloc[-1]
            live_row["drift_alarm"] = int(last.get("drift_alarm", 0))
            live_row["drift_base_ece"] = float(last.get("drift_base_ece", np.nan))
            live_row["drift_base_brier"] = float(last.get("drift_base_brier", np.nan))
            live_row["alpha_scale"] = float(last.get("alpha_scale", 1.0))
            live_row["governance_tier"] = last.get("governance_tier", "NORMAL")

            if cfg.DO_REGRESSION and (reg_voo is not None) and (reg_ief is not None) and (reg_spd is not None):
                f_voo = float(reg_voo.predict(X_live_row)[0])
                f_ief = float(reg_ief.predict(X_live_row)[0])
                f_spd = float(reg_spd.predict(X_live_row)[0])

                live_row["fwd_voo_hat"] = f_voo
                live_row["fwd_ief_hat"] = f_ief
                live_row["fwd_spread_hat"] = f_spd
                live_row["fwd_spread_hat_from_legs"] = f_voo - f_ief
                live_row["spread_model_gap"] = f_spd - (f_voo - f_ief)

                if px_live is not None and live_date in px_live.index:
                    pxv = px_live.loc[live_date]
                    px_voo_t = float(pxv.get("px_voo_t", np.nan))
                    px_ief_t = float(pxv.get("px_ief_t", np.nan))
                else:
                    px_voo_t = np.nan
                    px_ief_t = np.nan

                g_voo = float(m["gamma_voo"].dropna().iloc[-1]) if ("gamma_voo" in m.columns and m["gamma_voo"].notna().any()) else 0.0
                g_ief = float(m["gamma_ief"].dropna().iloc[-1]) if ("gamma_ief" in m.columns and m["gamma_ief"].notna().any()) else 0.0

                live_row["px_voo_t"] = px_voo_t
                live_row["px_ief_t"] = px_ief_t
                live_row["gamma_voo"] = g_voo
                live_row["gamma_ief"] = g_ief
                live_row["fwd_voo_hat_final"] = g_voo * f_voo
                live_row["fwd_ief_hat_final"] = g_ief * f_ief
                live_row["px_voo_call_7d"] = float(px_voo_t * np.exp(g_voo * f_voo)) if np.isfinite(px_voo_t) else np.nan
                live_row["px_ief_call_7d"] = float(px_ief_t * np.exp(g_ief * f_ief)) if np.isfinite(px_ief_t) else np.nan

            live_row["y_avail"] = np.nan
            live_row["is_live"] = 1

            m = pd.concat([m, live_row.to_frame().T], axis=0).sort_index()

        # Recompute strategy fields so live row gets weights / state diagnostics
        m = _strategy_from_edge(m, cfg)
        m.loc[live_date, "is_live"] = 1

    except Exception as e:
        print(f"[WARN] Live-row append/update skipped due to: {e}")

    # -----------------------------
    # Persist
    # -----------------------------
    out_path = os.path.join(cfg.PRED_DIR, "v77_final_consensus_tape.csv")

    m_out = m.copy().reset_index().rename(columns={"index": "Date"})
    m_out["Date"] = pd.to_datetime(m_out["Date"]).dt.normalize()
    m_out.to_csv(out_path, index=False)

    summary = {
        "part": "part2",
        "version": "PART_2_2_ONE_SIDED_DOWNSIDE",
        "horizon": int(cfg.H),
        "holdout_start": cfg.HO_START_DATE,
        "rows_rebalance": int(len(m)),
        "rows_realized": int(len(m_real)),
        "rows_audit": int(len(m_audit)),
        "classification_raw": raw_stats,
        "classification_cal_candidate": cand_stats,
        "classification_final_used": final_stats,
        "calibration_gate_on_rate": float(gate_rate) if np.isfinite(gate_rate) else np.nan,
        "high_risk_state_rate": float(high_risk_rate) if np.isfinite(high_risk_rate) else np.nan,
        "drift_base_ece": float(base_ece),
        "drift_base_brier": float(base_brier),
        "drift_alarm_rate": float(m["drift_alarm"].fillna(0).mean()) if "drift_alarm" in m.columns else np.nan,
        "active_ret_net_mean": float(active_mean) if np.isfinite(active_mean) else np.nan,
        "active_ret_net_ir": float(active_ir) if np.isfinite(active_ir) else np.nan,
        "strategy_ret_net_ir": float(strat_ir) if np.isfinite(strat_ir) else np.nan,
        "negative_benchmark_active_mean": float(neg_bench_active_mean) if np.isfinite(neg_bench_active_mean) else np.nan,
        "negative_benchmark_hit_rate": float(neg_bench_hit) if np.isfinite(neg_bench_hit) else np.nan,
        "downside_capture_ratio": float(downside_capture) if np.isfinite(downside_capture) else np.nan,
        "avg_abs_active_weight_raw": float(avg_abs_active_raw) if np.isfinite(avg_abs_active_raw) else np.nan,
        "avg_abs_active_weight_capped": float(avg_abs_active_capped) if np.isfinite(avg_abs_active_capped) else np.nan,
        "shuffle_auc_median": float(auc_shuf_med) if np.isfinite(auc_shuf_med) else np.nan,
        "final_pass": bool(final_pass),
        "out_path": out_path,
    }

    with open(os.path.join(cfg.PRED_DIR, "part2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ PART 2.2 TAPE WRITTEN: {out_path}")

    t = pd.read_csv(out_path)
    t["Date"] = pd.to_datetime(t["Date"])

    assert t["Date"].max() == pd.to_datetime(X.index.max()).normalize()
    assert int(t.sort_values("Date").iloc[-1]["is_live"]) == 1

    last_row = t.sort_values("Date").tail(1)
    is_live_last = last_row["is_live"].values[0] if "is_live" in last_row.columns else None
    print("Tape last date:", t["Date"].max(), "| is_live at end:", [is_live_last])

def cli() -> int:
    cfg = V77Config()
    summary = main(cfg)
    print("\nPart 2 summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
