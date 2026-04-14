#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @title Part 1 — Feature Builder (fixed live locked-14 contract)

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
import os
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Part1Config:
    start: str = "2010-01-01"
    end: str = date.today().strftime("%Y-%m-%d")
    horizon: int = 1                    # CHANGE: 7-day → 1-day weekday forecast
    tail_threshold: float = float(-0.015 / np.sqrt(7.0))  # severity-matched daily tail threshold

    main_tickers: Tuple[str, ...] = ("VOO", "IEF", "JNK", "RSP", "QQQ")
    vix_ticker: str = "^VIX"
    vix3m_ticker: str = "^VIX3M"

    part0_dir: str = _DRIVE_ROOT + "/artifacts_part0"
    out_dir: str = _DRIVE_ROOT + "/artifacts_part1"

    min_reg_rows: int = 500             # CHANGE: more rows needed for stable daily models
    max_stale_run: int = 3
    max_missing_frac: float = 0.02
    allow_ffill_limit: int = 2


REQ_TICKERS = ("VOO", "IEF", "JNK", "RSP", "QQQ", "^VIX", "^VIX3M")


def _max_consecutive_equal(x: pd.Series) -> int:
    v = x.dropna().values
    if len(v) == 0:
        return 0
    run = best = 1
    for i in range(1, len(v)):
        run = run + 1 if v[i] == v[i - 1] else 1
        best = max(best, run)
    return int(best)


def _rolling_z(s: pd.Series, window: int = 21) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (s - mu) / sd


def _downside_vol(r: pd.Series, window: int = 10) -> pd.Series:
    downside = np.minimum(pd.to_numeric(r, errors="coerce"), 0.0) ** 2
    return np.sqrt(pd.Series(downside, index=r.index).rolling(window).mean()) * np.sqrt(252.0)


def _ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        out = out.dropna(subset=["Date"]).set_index("Date")
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None).normalize()
        out = out[~out.index.isna()]
    else:
        raise ValueError("Price matrix must have a Date column or DatetimeIndex.")
    out = out.sort_index()
    return out


def _load_from_part0(cfg: Part1Config) -> Optional[pd.DataFrame]:
    path = os.path.join(cfg.part0_dir, "close_prices.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = _ensure_date_index(df)
    missing = [c for c in REQ_TICKERS if c not in df.columns]
    if missing:
        return None
    return df.loc[:, list(REQ_TICKERS)].copy()


def _download_prices(cfg: Part1Config) -> pd.DataFrame:
    tickers = list(cfg.main_tickers) + [cfg.vix_ticker, cfg.vix3m_ticker]
    raw = yf.download(tickers, start=cfg.start, end=cfg.end, progress=False, auto_adjust=True)
    data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    data = _ensure_date_index(data)
    return data.loc[:, list(REQ_TICKERS)].copy()


def _load_prices(cfg: Part1Config) -> pd.DataFrame:
    data = _load_from_part0(cfg)
    if data is not None:
        return data
    return _download_prices(cfg)


def _write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_part1_v20(cfg: Part1Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    H = int(cfg.horizon)
    print(f"-> Building Artifacts (V20_P1_DAILY) | H={H} | Tail thr on spread: {cfg.tail_threshold:.2%}")

    data = _load_prices(cfg)

    miss_frac = data.isna().mean().to_dict()
    bad = {k: v for k, v in miss_frac.items() if v > cfg.max_missing_frac}
    if bad:
        print(f"⚠️ Missingness warning (> {cfg.max_missing_frac:.0%}): {bad}")

    data = data.ffill(limit=cfg.allow_ffill_limit)
    data = data.dropna(subset=list(REQ_TICKERS)).copy()

    stale_report = {t: _max_consecutive_equal(data[t]) for t in REQ_TICKERS}
    stale_tickers = {t: r for t, r in stale_report.items() if r > cfg.max_stale_run}
    drop_mask = pd.Series(False, index=data.index)

    if stale_tickers:
        print(f"⚠️ Staleness warning: {stale_tickers}")
        for t in REQ_TICKERS:
            r0 = (np.log(data[t]).diff() == 0.0)
            drop_mask |= (r0.rolling(cfg.max_stale_run + 1).sum() >= cfg.max_stale_run + 1)
        data = data.loc[~drop_mask].copy()

    logp = np.log(data)
    voo_r1 = logp["VOO"].diff()
    ief_r1 = logp["IEF"].diff()
    jnk_r1 = logp["JNK"].diff()
    rsp_r1 = logp["RSP"].diff()
    qqq_r1 = logp["QQQ"].diff()
    vix_r1 = logp[cfg.vix_ticker].diff()
    vix3m_r1 = logp[cfg.vix3m_ticker].diff()

    spread_r1 = voo_r1 - ief_r1
    credit_spread = np.log(data["JNK"] / data["IEF"])
    credit_spread_r1 = credit_spread.diff()
    breadth = np.log(data["RSP"] / data["VOO"])
    tech_relative = np.log(data["QQQ"] / data["VOO"])
    vix_term = data[cfg.vix_ticker] / (data[cfg.vix3m_ticker] + 1e-9)

    vix_z21 = _rolling_z(data[cfg.vix_ticker], 21)
    credit_spread_z21 = _rolling_z(credit_spread, 21)
    breadth_z21 = _rolling_z(breadth, 21)
    tech_relative_z21 = _rolling_z(tech_relative, 21)

    # Auxiliary regime inputs kept outside the 14-feature contract but written
    # for Part 2 compatibility and Part 2A downstream use.
    vix_term_z21 = _rolling_z(vix_term, 21)
    spread_ret21 = spread_r1.rolling(21).sum()
    voo_downside_vol10 = _downside_vol(voo_r1, 10)

    excess_vol10 = spread_r1.rolling(10).std() * np.sqrt(252.0)
    excess_vol10_z21 = _rolling_z(excess_vol10, 21)
    voo_downside_vol10_z21 = _rolling_z(voo_downside_vol10, 21)

    stress_score_raw = (
        0.30 * vix_z21
        + 0.18 * vix_term_z21
        + 0.18 * credit_spread_z21
        + 0.16 * excess_vol10_z21
        + 0.10 * voo_downside_vol10_z21
        - 0.04 * breadth_z21
        - 0.04 * tech_relative_z21
    )
    stress_score_change5 = stress_score_raw.diff(5)

    # Fixed live locked-14 feature schema
    X = pd.DataFrame(index=data.index)
    X["voo_vol10"] = voo_r1.rolling(10).std() * np.sqrt(252.0)
    X["excess_vol10"] = excess_vol10
    X["vix_mom5"] = data[cfg.vix_ticker].diff(5)
    X["alpha_credit_spread"] = credit_spread
    X["alpha_credit_accel"] = credit_spread.diff().diff()
    X["alpha_vix_term"] = vix_term
    X["alpha_breadth"] = breadth
    X["alpha_tech_relative"] = tech_relative
    X["stress_score_raw"] = stress_score_raw
    X["stress_score_change5"] = stress_score_change5
    X["vix_z21"] = vix_z21
    X["credit_spread_z21"] = credit_spread_z21
    X["breadth_z21"] = breadth_z21
    X["tech_relative_z21"] = tech_relative_z21

    X_live = X.dropna().copy()
    # NOTE: 14-feature contract is unchanged — same features, different horizon
    if len(X_live.columns) != 14:
        raise RuntimeError(f"Part 1 locked contract expects 14 features, found {len(X_live.columns)}.")

    X_live.to_parquet(os.path.join(cfg.out_dir, "X_features.parquet"))

    last_feature_date = pd.Timestamp(X_live.index.max()).normalize()
    with open(os.path.join(cfg.out_dir, "asof_date.txt"), "w", encoding="utf-8") as f:
        f.write(last_feature_date.strftime("%Y-%m-%d"))
    print(f"AS-OF DATE WRITTEN: {last_feature_date.strftime('%Y-%m-%d')}")

    px_voo = data["VOO"].astype(float)
    px_ief = data["IEF"].astype(float)

    fwd_voo = np.log(px_voo).shift(-H) - np.log(px_voo)
    fwd_ief = np.log(px_ief).shift(-H) - np.log(px_ief)
    excess_ret = fwd_voo - fwd_ief

    y_rel_tail = (excess_ret < cfg.tail_threshold).astype(float)
    y_rel_tail[excess_ret.isna()] = np.nan

    y_labels = pd.DataFrame(
        {
            "fwd_voo": fwd_voo,
            "fwd_ief": fwd_ief,
            "excess_ret": excess_ret,
            "y_voo": y_rel_tail,
            "y_rel_tail_voo_vs_ief": y_rel_tail,
        },
        index=data.index,
    )

    y_revealed = y_labels.dropna(subset=["y_rel_tail_voo_vs_ief"]).copy()
    y_revealed[["y_rel_tail_voo_vs_ief", "excess_ret", "fwd_voo", "fwd_ief"]].to_parquet(
        os.path.join(cfg.out_dir, "y_labels_revealed.parquet")
    )
    y_revealed[["y_rel_tail_voo_vs_ief", "excess_ret", "fwd_voo", "fwd_ief"]].to_parquet(
        os.path.join(cfg.out_dir, "y_labels_revealed_aligned.parquet")
    )
    y_labels.to_parquet(os.path.join(cfg.out_dir, "y_labels_full.parquet"))

    px_voo_fwd = px_voo * np.exp(fwd_voo)
    px_ief_fwd = px_ief * np.exp(fwd_ief)

    y_reg = pd.DataFrame(
        {
            "px_voo_t": px_voo,
            "px_ief_t": px_ief,
            "fwd_voo": fwd_voo,
            "fwd_ief": fwd_ief,
            "fwd_spread": excess_ret,
            "px_voo_fwd": px_voo_fwd,
            "px_ief_fwd": px_ief_fwd,
        },
        index=data.index,
    )

    y_reg_revealed = y_reg.dropna(subset=["fwd_voo", "fwd_ief", "px_voo_fwd", "px_ief_fwd"]).copy()
    if len(data) >= (H + 1) and len(y_reg_revealed):
        max_ok_date = data.index[-(H + 1)]
        if y_reg_revealed.index.max() > max_ok_date:
            raise RuntimeError(f"LEAKAGE GUARD: y_reg_revealed extends past {max_ok_date.date()}")

    y_reg_revealed.to_parquet(os.path.join(cfg.out_dir, "y_reg_revealed.parquet"))
    y_reg.to_parquet(os.path.join(cfg.out_dir, "y_reg_full.parquet"))

    reg_train = X_live.join(y_reg_revealed, how="inner")
    if len(reg_train) < cfg.min_reg_rows:
        raise RuntimeError(f"Too few regression training rows: {len(reg_train)}")
    if reg_train[X_live.columns].isna().any().any():
        raise RuntimeError("NaNs in regression training features.")
    reg_train.to_parquet(os.path.join(cfg.out_dir, "regression_train.parquet"))

    pd.DataFrame({"px_voo_t": px_voo, "px_ief_t": px_ief}, index=data.index).loc[X_live.index].to_parquet(
        os.path.join(cfg.out_dir, "price_calls_live_snapshot.parquet")
    )
    data[["VOO"]].rename(columns={"VOO": "px"}).loc[X_live.index].to_parquet(
        os.path.join(cfg.out_dir, "target_prices_snapshot.parquet")
    )

    # Optional compatibility artifacts consumed by Part 2 and Part 2A
    factor_returns = pd.DataFrame(
        {
            "voo_r1": voo_r1,
            "ief_r1": ief_r1,
            "spread_r1": spread_r1,
            "jnk_r1": jnk_r1,
            "rsp_r1": rsp_r1,
            "qqq_r1": qqq_r1,
            "vix_r1": vix_r1,
            "vix3m_r1": vix3m_r1,
            "credit_spread_r1": credit_spread_r1,
            "vix_term_z21": vix_term_z21,
            "spread_ret21": spread_ret21,
            "voo_downside_vol10": voo_downside_vol10,
        },
        index=data.index,
    ).dropna(how="all")
    factor_returns.to_parquet(os.path.join(cfg.out_dir, "factor_returns.parquet"))

    benchmark_returns = pd.DataFrame(
        {
            "bench_voo": voo_r1,
            "bench_ief": ief_r1,
            "bench_60_40": 0.60 * voo_r1 + 0.40 * ief_r1,
            "bench_excess_voo_minus_ief": spread_r1,
        },
        index=data.index,
    ).dropna(how="all")
    benchmark_returns.to_parquet(os.path.join(cfg.out_dir, "benchmark_returns.parquet"))

    diag = {
        "date_min_data": str(data.index.min().date()),
        "date_max_data": str(data.index.max().date()),
        "n_data": int(len(data)),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
        "n_rows_lost_join": int(len(X_live) - len(reg_train)),
        "missing_frac": miss_frac,
        "max_equal_close_run": stale_report,
        "stale_dropped_rows": int(drop_mask.sum()),
    }
    _write_json(os.path.join(cfg.out_dir, "part1_diagnostics.json"), diag)

    meta = {
        "part": "part1",
        "version": "V20_P1_DAILY",
        "asof_date": last_feature_date.strftime("%Y-%m-%d"),
        "horizon": H,
        "tail_threshold": float(cfg.tail_threshold),
        "tail_label_name": "y_rel_tail_voo_vs_ief",
        "feature_cols": list(X_live.columns),
        "reg_target_cols": list(y_reg_revealed.columns),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
    }
    _write_json(os.path.join(cfg.out_dir, "part1_meta.json"), meta)

    tail_rate = float(y_revealed["y_rel_tail_voo_vs_ief"].mean()) if len(y_revealed) else float("nan")
    print(f"✅ V20_P1_DAILY COMPLETE | Tail base rate (revealed): {tail_rate:.2%}")
    print("Wrote: X_features.parquet, y_labels_revealed.parquet, y_labels_full.parquet,")
    print("       y_reg_revealed.parquet, y_reg_full.parquet, regression_train.parquet,")
    print("       price_calls_live_snapshot.parquet, target_prices_snapshot.parquet,")
    print("       factor_returns.parquet, benchmark_returns.parquet,")
    print("       asof_date.txt, part1_meta.json, part1_diagnostics.json")

    return {
        "out_dir": cfg.out_dir,
        "asof_date": last_feature_date.strftime("%Y-%m-%d"),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
        "tail_rate": tail_rate,
        "feature_count": int(len(X_live.columns)),
    }


def main():
    cfg = Part1Config()
    summary = build_part1_v20(cfg)
    print("\nPart 1 summary:", summary)


if __name__ == "__main__":
    main()


