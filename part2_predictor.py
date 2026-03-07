#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

# -----------------------------
# Project root
# -----------------------------
if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    ROOT = Path.cwd()


# -----------------------------
# 0) Config
# -----------------------------
@dataclass
class V77Config:
    PART1_DIR: str = str(ROOT / "artifacts_part1")
    PRED_DIR: str = str(ROOT / "artifacts_part2_v77" / "predictions")

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

    # Strategy transform used in audit
    K_DEFENSIVE: float = 1.0

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

    # Two-tier governance
    ALPHA_THROTTLE: float = 0.5

    # Regression / Price-Call
    DO_REGRESSION: bool = True
    REG_TRAIN_FILE: str = "regression_train.parquet"
    PRICE_LIVE_FILE: str = "price_calls_live_snapshot.parquet"

    # Regressor hyperparams
    REG_MAX_DEPTH: int = 2
    REG_N_ESTIMATORS: int = 200
    REG_LEARNING_RATE: float = 0.05
    REG_SUBSAMPLE: float = 0.9
    REG_COLSAMPLE: float = 0.9

    # Optional regression diagnostics
    PRINT_REG_AUDIT: bool = True


# -----------------------------
# 1) Helpers
# -----------------------------
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
    Mapper: p_final_raw = lam * sigmoid(sign*z/T + b) + (1-lam) * p0
    Fit by MSE on a time-split calibration segment.
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
    """
    Stateful p0 at refit time, using only labels revealed by train_end.
    Revealed at time t are y(t-H).
    """
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


# -----------------------------
# 2) Non-circular MBB with tail-count guard
# -----------------------------
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


# -----------------------------
# 3) Falsification (block shuffle)
# -----------------------------
def block_shuffle_labels(y, block_size, seed=42):
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    blocks = [y[i:i + block_size] for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


# -----------------------------
# 4) Platt calibrator (1D logistic)
# -----------------------------
def fit_platt_1d(x, y, max_iter=200):
    """
    Fit: P(y=1|x) = sigmoid(a*x + b) with Newton steps.
    Returns (a, b). Falls back safely if degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n == 0 or np.unique(y).size < 2:
        p = float(np.mean(y)) if n else 0.5
        return 0.0, float(_logit(p, clip=1e-6))

    a = 1.0
    b = float(_logit(np.mean(y), clip=1e-6))

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


# -----------------------------
# 4b) Regression utilities
# -----------------------------
def _make_regressor(cfg: V77Config):
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


# -----------------------------
# 4c) Forecast shrinkage
# -----------------------------
def _fit_gamma_mae(y_true, y_hat):
    y_true = np.asarray(y_true, float)
    y_hat = np.asarray(y_hat, float)
    m = np.isfinite(y_true) & np.isfinite(y_hat)
    if m.sum() < 30:
        return 0.0
    gammas = np.linspace(0.0, 1.0, 51)
    maes = [np.mean(np.abs(y_true[m] - g * y_hat[m])) for g in gammas]
    return float(gammas[int(np.argmin(maes))])


# -----------------------------
# 5) Drift alarm
# -----------------------------
def drift_alarm_stream(ece_roll, brier_roll, cfg: V77Config):
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


# -----------------------------
# 6) Main
# -----------------------------
def main(cfg: V77Config):
    os.makedirs(cfg.PRED_DIR, exist_ok=True)

    # Load Part 1 artifacts
    X = pd.read_parquet(os.path.join(cfg.PART1_DIR, "X_features.parquet"))
    y = pd.read_parquet(os.path.join(cfg.PART1_DIR, "y_labels_revealed.parquet"))

    X.index = pd.to_datetime(X.index).tz_localize(None).normalize()
    y.index = pd.to_datetime(y.index).tz_localize(None).normalize()

    # Contract: read Part 1 meta
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
            print(f"[CONTRACT] Extra features vs meta (allowed but track): {extra}")

    feat_cols = list(X.columns)

    y_col = tail_label_name if tail_label_name in y.columns else ("y_voo" if "y_voo" in y.columns else None)
    if y_col is None:
        raise RuntimeError(f"[CONTRACT] No label column found. meta tail_label_name={tail_label_name}, y.columns={list(y.columns)}")

    if y_col != "y_voo":
        y = y.rename(columns={y_col: "y_voo"})
        print(f"[CONTRACT] Using label '{y_col}' as y_voo (normalized).")

    # Optional regression artifacts
    reg_train = None
    px_live = None
    if cfg.DO_REGRESSION:
        reg_path = os.path.join(cfg.PART1_DIR, cfg.REG_TRAIN_FILE)
        px_path = os.path.join(cfg.PART1_DIR, cfg.PRICE_LIVE_FILE)

        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"V77 requires {reg_path}.")
        if not os.path.exists(px_path):
            raise FileNotFoundError(f"V77 requires {px_path}.")

        reg_train = pd.read_parquet(reg_path)
        px_live = pd.read_parquet(px_path)

        reg_train.index = pd.to_datetime(reg_train.index).tz_localize(None).normalize()
        px_live.index = pd.to_datetime(px_live.index).tz_localize(None).normalize()

    feat_cols = list(X.columns)
    df = X.join(y, how="left").sort_index()

    ho_start = pd.Timestamp(cfg.HO_START_DATE)
    ho_idx = np.where((df.index >= ho_start))[0]
    if len(ho_idx) == 0:
        raise ValueError("No holdout rows found. Check HO_START_DATE and data index.")

    seed_end = max(0, int(ho_idx[0]) - cfg.H)
    seed_value = float(df.iloc[:seed_end]["y_voo"].mean()) if seed_end > 0 else float(df["y_voo"].mean())

    # Regression storage
    fwd_voo_hat = pd.Series(index=df.index, dtype=float)
    fwd_ief_hat = pd.Series(index=df.index, dtype=float)
    fwd_spd_hat = pd.Series(index=df.index, dtype=float)

    reg_voo = reg_ief = reg_spd = None

    z_raw = pd.Series(index=df.index, dtype=float)
    p0_state_row = pd.Series(index=df.index, dtype=float)
    calib_states = []

    clf = xgb.XGBClassifier(
        max_depth=2,
        n_estimators=100,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=cfg.SEED,
    )

    train_window_rows = int(cfg.TRAIN_WINDOW_Y * cfg.TRADING_DAYS_PER_YEAR)
    anchors = ho_idx[::cfg.REFIT_FREQ]

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

        if cfg.DO_REGRESSION and reg_train is not None:
            d0 = df.index[train_start]
            d1 = df.index[train_end - 1]

            rt = reg_train.loc[(reg_train.index >= d0) & (reg_train.index <= d1)].copy()
            if len(rt) >= cfg.MIN_TRAIN_ROWS:
                Xrt = rt[feat_cols].copy()

                y_voo_rt = rt["fwd_voo"].values.astype(float)
                y_ief_rt = rt["fwd_ief"].values.astype(float)
                y_spd_rt = rt["fwd_spread"].values.astype(float)

                reg_voo = _make_regressor(cfg)
                reg_ief = _make_regressor(cfg)
                reg_spd = _make_regressor(cfg)

                reg_voo.fit(Xrt, y_voo_rt)
                reg_ief.fit(Xrt, y_ief_rt)
                reg_spd.fit(Xrt, y_spd_rt)

                test_end = min(len(df), d_idx + cfg.REFIT_FREQ)
                chunk = df.iloc[d_idx:test_end]
                Xch = chunk[feat_cols].copy()

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

    rebalance_idx = np.arange(ho_idx[0], len(df), cfg.H, dtype=int)
    rebalance_dates = df.index[rebalance_idx]

    m = df.loc[rebalance_dates].copy().sort_index()
    m["z_raw"] = z_raw.loc[rebalance_dates].values
    m["p0"] = p0_state_row.loc[rebalance_dates].values

    if cfg.DO_REGRESSION and px_live is not None:
        m["fwd_voo_hat"] = fwd_voo_hat.loc[rebalance_dates].values
        m["fwd_ief_hat"] = fwd_ief_hat.loc[rebalance_dates].values
        m["fwd_spread_hat"] = fwd_spd_hat.loc[rebalance_dates].values

        m["fwd_spread_hat_from_legs"] = m["fwd_voo_hat"] - m["fwd_ief_hat"]
        m["spread_model_gap"] = m["fwd_spread_hat"] - m["fwd_spread_hat_from_legs"]

        px_m = px_live.reindex(m.index)
        m["px_voo_t"] = px_m["px_voo_t"].values if "px_voo_t" in px_m.columns else np.nan
        m["px_ief_t"] = px_m["px_ief_t"].values if "px_ief_t" in px_m.columns else np.nan

    # Forecast shrinkage
    if cfg.DO_REGRESSION and "fwd_voo_hat" in m.columns:
        Wg = 52
        gamma_voo = np.nan
        gamma_ief = np.nan

        m_mature = m.dropna(subset=["fwd_voo", "fwd_voo_hat", "fwd_ief", "fwd_ief_hat"]).copy()

        if len(m_mature) >= 60:
            tail = m_mature.tail(Wg)
            gamma_voo = _fit_gamma_mae(tail["fwd_voo"].values, tail["fwd_voo_hat"].values)
            gamma_ief = _fit_gamma_mae(tail["fwd_ief"].values, tail["fwd_ief_hat"].values)

        m["gamma_voo"] = gamma_voo
        m["gamma_ief"] = gamma_ief

        if np.isfinite(gamma_voo):
            m["fwd_voo_hat_final"] = gamma_voo * m["fwd_voo_hat"]
        else:
            m["fwd_voo_hat_final"] = 0.0

        if np.isfinite(gamma_ief):
            m["fwd_ief_hat_final"] = gamma_ief * m["fwd_ief_hat"]
        else:
            m["fwd_ief_hat_final"] = 0.0

        m["px_voo_call_7d"] = m["px_voo_t"] * np.exp(m["fwd_voo_hat_final"])
        m["px_ief_call_7d"] = m["px_ief_t"] * np.exp(m["fwd_ief_hat_final"])

        if "px_voo_t" in m.columns and "fwd_voo" in m.columns:
            m["px_voo_real_7d"] = m["px_voo_t"] * np.exp(m["fwd_voo"])
        else:
            m["px_voo_real_7d"] = np.nan

        if "px_ief_t" in m.columns and "fwd_ief" in m.columns:
            m["px_ief_real_7d"] = m["px_ief_t"] * np.exp(m["fwd_ief"])
        else:
            m["px_ief_real_7d"] = np.nan

        m["err_px_voo"] = m["px_voo_call_7d"] - m["px_voo_real_7d"] if "px_voo_call_7d" in m.columns else np.nan
        m["err_px_ief"] = m["px_ief_call_7d"] - m["px_ief_real_7d"] if "px_ief_call_7d" in m.columns else np.nan

        m["err_r_voo"] = m["fwd_voo_hat_final"] - m["fwd_voo"]
        m["err_r_ief"] = m["fwd_ief_hat_final"] - m["fwd_ief"]
        m["hit_sign_voo"] = np.where(
            np.isfinite(m["fwd_voo_hat_final"].values) & np.isfinite(m["fwd_voo"].values),
            (np.sign(m["fwd_voo_hat_final"].values) == np.sign(m["fwd_voo"].values)).astype(float),
            np.nan,
        )

    cover = np.isfinite(m["z_raw"].values).mean()
    if cover < cfg.MIN_REBAL_COVERAGE:
        raise RuntimeError(f"CRITICAL: z_raw coverage too low: {cover:.2%} < {cfg.MIN_REBAL_COVERAGE:.2%}")

    m = pd.merge_asof(
        m.reset_index().rename(columns={"index": "Date"}),
        calib_df.reset_index().rename(columns={"Date": "CalibDate"}),
        left_on="Date",
        right_on="CalibDate",
        direction="backward",
    ).set_index("Date")

    # Raw probabilities
    m["p_final_raw"] = (
        m["lam"] * _sigmoid((m["sign"] * m["z_raw"]) / m["T"] + m["b"])
        + (1.0 - m["lam"]) * m["p0"]
    )
    m = m.dropna(subset=["p_final_raw", "p0"])

    # Calibration overlay
    m["y_avail"] = m["y_voo"].shift(cfg.H)

    x_all = _logit(m["p_final_raw"].values, clip=cfg.CAL_CLIP)
    y_av = m["y_avail"].values

    p_cal = np.full(len(m), np.nan, dtype=float)
    cal_a = np.full(len(m), np.nan, dtype=float)
    cal_b = np.full(len(m), np.nan, dtype=float)
    cal_n = np.zeros(len(m), dtype=int)

    for i in range(len(m)):
        mask = np.isfinite(y_av[:i])
        n_obs = int(mask.sum())
        cal_n[i] = n_obs

        if n_obs < cfg.CAL_MIN_SAMPLES:
            p_cal[i] = float(m["p_final_raw"].iloc[i])
            continue

        a, b0 = fit_platt_1d(x_all[:i][mask], y_av[:i][mask])
        cal_a[i], cal_b[i] = a, b0
        p_cal[i] = float(apply_platt(a, b0, x_all[i]))

    m["p_final_cal"] = np.clip(p_cal, 0.0, 1.0)
    m["cal_a"], m["cal_b"], m["cal_n"] = cal_a, cal_b, cal_n

    # Rolling diagnostics
    y_true = m["y_voo"].values.astype(float)
    p_used = m["p_final_cal"].values
    y_av2 = m["y_avail"].values

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

    # V77 governance
    m["alpha_scale"] = np.where(m["drift_alarm"] == 1, cfg.ALPHA_THROTTLE, 1.0)
    m["governance_tier"] = np.where(m["drift_alarm"] == 1, "DRIFT_THROTTLE", "NORMAL")

    # Audit prints
    m_real = m.dropna(subset=["y_voo"]).copy()
    y_true_stats = m_real["y_voo"].values.astype(int)
    prev = float(y_true_stats.mean()) if len(y_true_stats) else np.nan

    def _summ(p_pred):
        if len(np.unique(y_true_stats)) < 2:
            return {"auc": np.nan, "pr": np.nan, "lift": np.nan, "brier": np.nan, "ece": np.nan}
        auc = roc_auc_score(y_true_stats, p_pred)
        pr = average_precision_score(y_true_stats, p_pred)
        lift = (pr / prev) if prev > 0 else np.nan
        brier = float(np.mean((p_pred - y_true_stats) ** 2))
        ece = ece_score(y_true_stats, p_pred, n_bins=cfg.ECE_BINS)
        return {"auc": auc, "pr": pr, "lift": lift, "brier": brier, "ece": ece}

    raw_stats = _summ(m_real["p_final_raw"].values)
    cal_stats = _summ(m_real["p_final_cal"].values)

    print("\n" + "=" * 78)
    print("🏛️  FINAL PRODUCTION AUDIT (V77 | RAW vs CAL + Tier-1 Throttle)")
    print("=" * 78)
    print(f"Rows (rebalance): {len(m)} | Base rate (prev): {prev:.4f}")
    print(f"RAW: AUC {raw_stats['auc']:.4f} | PR {raw_stats['pr']:.4f} | Lift {raw_stats['lift']:.3f} | Brier {raw_stats['brier']:.6f} | ECE {raw_stats['ece']:.6f}")
    print(f"CAL: AUC {cal_stats['auc']:.4f} | PR {cal_stats['pr']:.4f} | Lift {cal_stats['lift']:.3f} | Brier {cal_stats['brier']:.6f} | ECE {cal_stats['ece']:.6f}")
    print(f"Drift baseline: ECE {base_ece:.6f} | Brier {base_brier:.6f} | Alarm rate: {m['drift_alarm'].mean():.2%}")
    print("\nGovernance Tier Distribution:")
    print(m["governance_tier"].value_counts(normalize=True).round(4))

    # Consensus + falsification
    m_gate = m.dropna(subset=["p_final_raw", "p0"]).copy()
    m_audit = m_gate.dropna(subset=["excess_ret"]).copy()

    m_audit["bench_ret"] = m_audit["excess_ret"]
    m_audit["w_always_raw"] = (
        1.0 - (cfg.K_DEFENSIVE * (m_audit["p_final_raw"] - m_audit["p0"]) / (m_audit["p0"] + 1e-9))
    ).clip(0, 1)
    m_audit["strat_ret_always_raw"] = m_audit["w_always_raw"] * m_audit["bench_ret"]

    y_true_audit = m_audit["y_voo"].values.astype(int)
    p_pred_audit = m_audit["p_final_raw"].values

    consensus_pass = True
    for bsz in cfg.GATE_BLOCKS:
        ci = bootstrap_delta_cvar_mbb(
            m_audit["bench_ret"].values,
            m_audit["strat_ret_always_raw"].values,
            B=cfg.B_SAMPLES,
            block_size=int(bsz),
            q=cfg.TAIL_Q,
            min_tail=cfg.MIN_TAIL_PER_BOOT,
            seed=cfg.SEED,
        )
        if not np.isfinite(ci[0]):
            consensus_pass = False
            print(f"Block {bsz:2}: ΔCVaR CI Low: NaN | FAIL (insufficient tail support)")
        else:
            block_pass = (ci[0] >= -0.0001)
            consensus_pass = consensus_pass and block_pass
            print(f"Block {bsz:2}: ΔCVaR CI Low: {ci[0]:.4%} | Med: {ci[1]:.4%} | High: {ci[2]:.4%} | {'PASS' if block_pass else 'FAIL'}")

    fals_pass = True
    if cfg.DO_FALSIFICATION and len(np.unique(y_true_audit)) > 1:
        rng = np.random.default_rng(cfg.SEED)
        aucs = []
        for _ in range(cfg.SHUFFLE_B):
            y_shuf = block_shuffle_labels(y_true_audit, cfg.SHUFFLE_BLOCK, seed=int(rng.integers(1, 1_000_000)))
            if len(np.unique(y_shuf)) > 1:
                aucs.append(roc_auc_score(y_shuf, p_pred_audit))
        auc_shuf_med = float(np.median(aucs)) if len(aucs) else np.nan
        print(f"Shuffle-AUC (block={cfg.SHUFFLE_BLOCK}, B={cfg.SHUFFLE_B}) median: {auc_shuf_med:.4f}")
        fals_pass = (np.isfinite(auc_shuf_med) and auc_shuf_med <= cfg.MAX_SHUFFLE_AUC)

    final_pass = (consensus_pass and fals_pass)
    print(f"\n✅ FINAL VERDICT (V77): {'PASS' if final_pass else 'FAIL'}")

    # Regression audit
    if getattr(cfg, "DO_REGRESSION", False) and getattr(cfg, "PRINT_REG_AUDIT", True):
        need = ["fwd_voo", "fwd_ief", "fwd_voo_hat", "fwd_ief_hat"]
        have_need = all(c in m_gate.columns for c in need)

        have_spread_hat = ("fwd_spread_hat" in m_gate.columns)
        have_excess_ret = ("excess_ret" in m_gate.columns)

        if not have_need:
            print("\n" + "-" * 78)
            print("📈 V77 REGRESSION AUDIT (monitor-only)")
            print("-" * 78)
            print("Skipped: regression columns not found yet. Need:", need)
        else:
            m_reg = m_gate.dropna(subset=need).copy()
            if len(m_reg) == 0:
                print("\n" + "-" * 78)
                print("📈 V77 REGRESSION AUDIT (monitor-only)")
                print("-" * 78)
                print("Skipped: no realized rows available after dropna on regression targets + hats.")
            else:
                rmse_voo = _safe_rmse(m_reg["fwd_voo"].values, m_reg["fwd_voo_hat"].values)
                rmse_ief = _safe_rmse(m_reg["fwd_ief"].values, m_reg["fwd_ief_hat"].values)
                mae_voo = _safe_mae(m_reg["fwd_voo"].values, m_reg["fwd_voo_hat"].values)
                mae_ief = _safe_mae(m_reg["fwd_ief"].values, m_reg["fwd_ief_hat"].values)

                rmse_spd = np.nan
                mae_spd = np.nan

                mae_r_voo_naive = _safe_mae(m_reg["fwd_voo"].values, np.zeros(len(m_reg)))
                mae_r_ief_naive = _safe_mae(m_reg["fwd_ief"].values, np.zeros(len(m_reg)))

                px_real = m_reg["px_voo_t"].values * np.exp(m_reg["fwd_voo"].values) if "px_voo_t" in m_reg.columns else None
                if px_real is not None and "px_voo_call_7d" in m_reg.columns:
                    mae_px_model = _safe_mae(px_real, m_reg["px_voo_call_7d"].values)
                    mae_px_naive = _safe_mae(px_real, m_reg["px_voo_t"].values)
                else:
                    mae_px_model = np.nan
                    mae_px_naive = np.nan

                print(f"Baseline MAE (returns): VOO {mae_r_voo_naive:.6f} | IEF {mae_r_ief_naive:.6f}")
                print(f"Baseline MAE (price):   model {mae_px_model:.6f} | naive(P_t) {mae_px_naive:.6f}")

                if have_spread_hat and have_excess_ret:
                    m_sp = m_reg.dropna(subset=["excess_ret", "fwd_spread_hat"]).copy()
                    if len(m_sp):
                        rmse_spd = _safe_rmse(m_sp["excess_ret"].values, m_sp["fwd_spread_hat"].values)
                        mae_spd = _safe_mae(m_sp["excess_ret"].values, m_sp["fwd_spread_hat"].values)

                gap_med = np.nan
                if have_spread_hat:
                    m_gap = m_reg.dropna(subset=["fwd_spread_hat"]).copy()
                    if len(m_gap):
                        gap = m_gap["fwd_spread_hat"].values.astype(float) - (
                            m_gap["fwd_voo_hat"].values.astype(float) - m_gap["fwd_ief_hat"].values.astype(float)
                        )
                        gap_med = float(np.nanmedian(gap)) if np.isfinite(gap).any() else np.nan

                print("\n" + "-" * 78)
                print("📈 V77 REGRESSION AUDIT (monitor-only; does not affect PASS)")
                print("-" * 78)
                print(f"Rows used: {len(m_reg)}")
                print(f"RMSE: fwd_voo {rmse_voo:.6f} | fwd_ief {rmse_ief:.6f} | spread(excess_ret) {rmse_spd:.6f}")
                print(f" MAE: fwd_voo {mae_voo:.6f} | fwd_ief {mae_ief:.6f} | spread(excess_ret) {mae_spd:.6f}")
                print(f"Spread consistency: median(gap = direct - (voo_hat-ief_hat)) = {gap_med:.6e}")

        # Live row append
        try:
            live_date = pd.to_datetime(X.index.max()).tz_localize(None).normalize()
            pos = df.index.get_loc(live_date)
            revealed_pos = pos - cfg.H

            print("X last date:", live_date)
            if revealed_pos >= 0:
                print("Expected y_revealed last date (row-based):", df.index[revealed_pos])
            else:
                print("Expected y_revealed last date: <none yet, not enough history>")
            if "y_voo" in y.columns:
                print("y_revealed last date (file max):", y.index.max())

            if "is_live" not in m.columns:
                m["is_live"] = 0
            m["is_live"] = m["is_live"].fillna(0).astype(int)

            if live_date not in m.index:
                base = df.loc[live_date].copy()

                X_live_row = X.loc[[live_date], feat_cols]
                z_live = float(clf.predict(X_live_row, output_margin=True)[0])

                live_pos = int(np.where(df.index == live_date)[0][0])
                train_end_idx_live = live_pos + 1
                p0_live = float(compute_p0_state_from_revealed_labels(
                    df["y_voo"], train_end_idx=train_end_idx_live, H=cfg.H, seed_value=seed_value
                ))

                calib_asof = calib_df.loc[:live_date].iloc[-1]
                s_live = float(calib_asof["sign"])
                T_live = float(calib_asof["T"])
                b_live = float(calib_asof["b"])
                lam_live = float(calib_asof["lam"])

                p_raw_live = float(
                    lam_live * _sigmoid((s_live * z_live) / (T_live + 1e-12) + b_live)
                    + (1.0 - lam_live) * p0_live
                )
                p_raw_live = float(np.clip(p_raw_live, 0.0, 1.0))

                a_last = m["cal_a"].dropna().iloc[-1] if ("cal_a" in m.columns and m["cal_a"].notna().any()) else np.nan
                b_last = m["cal_b"].dropna().iloc[-1] if ("cal_b" in m.columns and m["cal_b"].notna().any()) else np.nan

                if np.isfinite(a_last) and np.isfinite(b_last):
                    x_live = float(_logit(p_raw_live, clip=cfg.CAL_CLIP))
                    p_cal_live = float(apply_platt(float(a_last), float(b_last), x_live))
                    p_cal_live = float(np.clip(p_cal_live, 0.0, 1.0))
                else:
                    p_cal_live = p_raw_live

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
                live_row["p_final_cal"] = p_cal_live

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

                    live_row["px_voo_t"] = px_voo_t
                    live_row["px_ief_t"] = px_ief_t

                    g_voo = float(m["gamma_voo"].dropna().iloc[-1]) if ("gamma_voo" in m.columns and m["gamma_voo"].notna().any()) else 0.0
                    g_ief = float(m["gamma_ief"].dropna().iloc[-1]) if ("gamma_ief" in m.columns and m["gamma_ief"].notna().any()) else 0.0

                    f_voo_final = g_voo * f_voo
                    f_ief_final = g_ief * f_ief

                    live_row["gamma_voo"] = g_voo
                    live_row["gamma_ief"] = g_ief
                    live_row["fwd_voo_hat_final"] = f_voo_final
                    live_row["fwd_ief_hat_final"] = f_ief_final

                    live_row["px_voo_call_7d_raw"] = float(px_voo_t * np.exp(f_voo)) if np.isfinite(px_voo_t) else np.nan
                    live_row["px_ief_call_7d_raw"] = float(px_ief_t * np.exp(f_ief)) if np.isfinite(px_ief_t) else np.nan

                    live_row["px_voo_call_7d"] = float(px_voo_t * np.exp(f_voo_final)) if np.isfinite(px_voo_t) else np.nan
                    live_row["px_ief_call_7d"] = float(px_ief_t * np.exp(f_ief_final)) if np.isfinite(px_ief_t) else np.nan

                last = m.sort_index().iloc[-1]
                live_row["drift_alarm"] = int(last.get("drift_alarm", 0))
                live_row["alpha_scale"] = float(last.get("alpha_scale", 1.0))
                live_row["governance_tier"] = last.get("governance_tier", "NORMAL")

                live_row["is_live"] = 1
                m = pd.concat([m, live_row.to_frame().T], axis=0).sort_index()

        except Exception as e:
            print(f"[WARN] Live-row append skipped due to: {e}")

    out_path = os.path.join(cfg.PRED_DIR, "v77_final_consensus_tape.csv")

    m_out = m.copy()
    m_out = m_out.reset_index().rename(columns={"index": "Date"})
    m_out["Date"] = pd.to_datetime(m_out["Date"]).dt.normalize()
    m_out.to_csv(out_path, index=False)

    print(f"\n✅ V77 TAPE WRITTEN: {out_path}")

    t = pd.read_csv(out_path)
    t["Date"] = pd.to_datetime(t["Date"])
    assert t["Date"].max() == pd.to_datetime(X.index.max()).normalize()
    assert int(t.sort_values("Date").iloc[-1]["is_live"]) == 1

    last_row = t.sort_values("Date").tail(1)
    is_live_last = last_row["is_live"].values[0] if "is_live" in last_row.columns else None
    print("Tape last date:", t["Date"].max(), "| is_live at end:", [is_live_last])

    return {
        "out_path": out_path,
        "rows_rebalance": int(len(m)),
        "drift_alarm_rate": float(m["drift_alarm"].mean()),
        "final_verdict_pass": bool(final_pass),
        "last_date": str(t["Date"].max().date()),
        "last_is_live": int(is_live_last),
    }


def cli() -> int:
    cfg = V77Config()
    summary = main(cfg)
    print("\nPart 2 summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
