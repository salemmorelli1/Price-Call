#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Part1Config:
    start: str = "2010-01-01"
    end: str = date.today().strftime("%Y-%m-%d")
    horizon: int = 7
    tail_threshold: float = -0.015  # threshold on (fwd_voo - fwd_ief)

    main_tickers: Tuple[str, ...] = ("VOO", "IEF", "JNK", "RSP", "QQQ")
    vix_ticker: str = "^VIX"
    vix3m_ticker: str = "^VIX3M"

    benchmark_ticker: str = "VOO"
    alpha_universe: Tuple[str, ...] = (
        "XLK", "XLF", "XLI", "XLY", "XLP",
        "XLV", "XLE", "XLU", "XLB", "XLC",
        "SMH", "IWM", "MDY", "EFA", "EEM",
    )

    out_dir: str = "./artifacts_part1"
    min_reg_rows: int = 200

    # data quality guards
    max_stale_run: int = 3
    max_missing_frac: float = 0.02
    allow_ffill_limit: int = 2


def _max_consecutive_equal(x: pd.Series) -> int:
    """Max run length of identical consecutive values (ignoring NaNs)."""
    v = x.dropna().values
    if len(v) == 0:
        return 0

    run = 1
    best = 1
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return int(best)


def _safe_logret(px: pd.Series) -> pd.Series:
    px = px.astype(float)
    return np.log(px).diff()


def build_alpha_panel_from_prices(
    close_all: pd.DataFrame,
    X_live: pd.DataFrame,
    cfg: Part1Config,
    H: int,
) -> pd.DataFrame:
    """
    Build cross-sectional alpha panel:
      Date, Ticker, px_t, fwd_ret, benchmark_fwd_ret, rel_ret
    plus date-level live features from X_live.

    Important:
    - Keeps unrevealed/latest rows (fwd_ret/rel_ret may be NaN there)
    - Joins only dates where X_live exists
    """
    if cfg.benchmark_ticker not in close_all.columns:
        raise RuntimeError(
            f"Benchmark ticker '{cfg.benchmark_ticker}' missing from downloaded price matrix."
        )

    px_bench = close_all[cfg.benchmark_ticker].astype(float)
    bench_fwd = np.log(px_bench).shift(-H) - np.log(px_bench)

    feat_df = X_live.reset_index().rename(columns={"index": "Date"})
    feat_df["Date"] = pd.to_datetime(feat_df["Date"]).dt.normalize()

    blocks = []
    missing_tickers = []

    for tk in cfg.alpha_universe:
        if tk not in close_all.columns:
            missing_tickers.append(tk)
            continue

        px = close_all[tk].astype(float)

        block = pd.DataFrame(
            {
                "px_t": px,
                "fwd_ret": np.log(px).shift(-H) - np.log(px),
                "benchmark_fwd_ret": bench_fwd,
            },
            index=close_all.index,
        )
        block["rel_ret"] = block["fwd_ret"] - block["benchmark_fwd_ret"]
        block["Ticker"] = tk

        block = block.reset_index().rename(columns={"index": "Date"})
        block["Date"] = pd.to_datetime(block["Date"]).dt.normalize()
        blocks.append(block)

    if missing_tickers:
        print(f"⚠️ Alpha panel missing downloaded tickers: {missing_tickers}")

    if len(blocks) == 0:
        raise RuntimeError("No alpha-universe rows were built for alpha_panel.parquet.")

    alpha_panel = pd.concat(blocks, axis=0, ignore_index=True)

    # Join only dates where live features exist
    alpha_panel = alpha_panel.merge(feat_df, on="Date", how="inner")

    # Keep rows with actual price present; allow unrevealed forward returns
    alpha_panel = alpha_panel.dropna(subset=["px_t"]).copy()

    alpha_panel["is_revealed"] = np.isfinite(alpha_panel["rel_ret"]).astype(int)
    alpha_panel = alpha_panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return alpha_panel


def build_part1_v19(cfg: Part1Config) -> Dict[str, object]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    H = int(cfg.horizon)
    tickers = sorted(
        set(list(cfg.main_tickers) + [cfg.vix_ticker, cfg.vix3m_ticker] + list(cfg.alpha_universe))
    )

    print(f"-> Building Artifacts (V19) | H={H} | Tail thr on spread: {cfg.tail_threshold:.2%}")

    raw = yf.download(
        tickers,
        start=cfg.start,
        end=cfg.end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )

    # Robust close extraction
    data = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()

    # -----------------------------
    # 0) Data hygiene
    # -----------------------------
    miss_frac = data.isna().mean().to_dict()
    bad = {k: v for k, v in miss_frac.items() if v > cfg.max_missing_frac}
    if bad:
        print(f"⚠️ Missingness after cleaning warning (> {cfg.max_missing_frac:.0%}): {bad}")

    data = data.ffill(limit=cfg.allow_ffill_limit)

    req = ["VOO", "IEF", "JNK", "RSP", "QQQ", cfg.vix_ticker, cfg.vix3m_ticker]

    missing_required = [c for c in req if c not in data.columns]
    if missing_required:
        raise RuntimeError(f"Missing required downloaded columns: {missing_required}")

    all_nan_required = [c for c in req if data[c].isna().all()]
    if all_nan_required:
        raise RuntimeError(
            f"Required tickers downloaded as entirely missing/all-NaN: {all_nan_required}"
        )

    data = data.dropna(subset=req)

    if data.empty:
        raise RuntimeError(
            "No rows remain after dropping rows with missing required tickers. "
            f"Missingness summary: {miss_frac}"
        )

    stale_report: Dict[str, int] = {}
    for t in req:
        stale_report[t] = _max_consecutive_equal(data[t])

    stale_tickers = {t: r for t, r in stale_report.items() if r > cfg.max_stale_run}
    drop_mask = pd.Series(False, index=data.index)

    if stale_tickers:
        print(f"⚠️ Staleness warning (max equal-close run > {cfg.max_stale_run}): {stale_tickers}")
        for t in req:
            r0 = (_safe_logret(data[t]) == 0.0)
            w = cfg.max_stale_run + 1
            drop_mask |= (r0.rolling(w).sum() >= w)
        data = data.loc[~drop_mask].copy()

    # -----------------------------
    # 1) Features (live-available)
    # -----------------------------
    X = pd.DataFrame(index=data.index)

    voo_r1 = _safe_logret(data["VOO"])
    ief_r1 = _safe_logret(data["IEF"])
    spread_r1 = voo_r1 - ief_r1

    X["voo_vol10"] = voo_r1.rolling(10).std() * np.sqrt(252)
    X["excess_vol10"] = spread_r1.rolling(10).std() * np.sqrt(252)
    X["vix_mom5"] = data[cfg.vix_ticker].diff(5)

    credit_spread = np.log(data["JNK"] / data["IEF"])
    X["alpha_credit_spread"] = credit_spread
    X["alpha_credit_accel"] = credit_spread.diff().diff()
    X["alpha_vix_term"] = data[cfg.vix_ticker] / (data[cfg.vix3m_ticker] + 1e-9)
    X["alpha_breadth"] = np.log(data["RSP"] / data["VOO"])
    X["alpha_tech_relative"] = np.log(data["QQQ"] / data["VOO"])

    X_live = X.dropna().copy()

    if X_live.empty:
        raise RuntimeError(
            "Part 1 produced no valid feature rows after cleaning. "
            "Most likely cause: one or more required Yahoo tickers failed to download "
            f"or were entirely missing. Missingness summary: {miss_frac}"
        )

    X_live.to_parquet(os.path.join(cfg.out_dir, "X_features.parquet"))

    last_feature_date = pd.Timestamp(X_live.index.max()).normalize()
    with open(os.path.join(cfg.out_dir, "asof_date.txt"), "w") as f:
        f.write(last_feature_date.strftime("%Y-%m-%d"))
    print(f"AS-OF DATE WRITTEN: {last_feature_date.strftime('%Y-%m-%d')}")

    # -----------------------------
    # 2) Forward returns + tail label
    # -----------------------------
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
    y_labels.to_parquet(os.path.join(cfg.out_dir, "y_labels_full.parquet"))

    # -----------------------------
    # 2A) Daily factor / benchmark artifacts for Part 2
    # -----------------------------
    voo_ret_1d = _safe_logret(data["VOO"])
    ief_ret_1d = _safe_logret(data["IEF"])
    jnk_ret_1d = _safe_logret(data["JNK"])
    rsp_ret_1d = _safe_logret(data["RSP"])
    qqq_ret_1d = _safe_logret(data["QQQ"])

    factor_returns = pd.DataFrame(
        {
            "voo_ret_1d": voo_ret_1d,
            "ief_ret_1d": ief_ret_1d,
            "jnk_ret_1d": jnk_ret_1d,
            "rsp_ret_1d": rsp_ret_1d,
            "qqq_ret_1d": qqq_ret_1d,
            "spread_ret_1d": voo_ret_1d - ief_ret_1d,
        },
        index=data.index,
    ).loc[X_live.index].copy()
    factor_returns.to_parquet(os.path.join(cfg.out_dir, "factor_returns.parquet"))

    benchmark_returns = pd.DataFrame(
        {
            "bench_60_40": 0.60 * voo_ret_1d + 0.40 * ief_ret_1d,
        },
        index=data.index,
    ).loc[X_live.index].copy()
    benchmark_returns.to_parquet(os.path.join(cfg.out_dir, "benchmark_returns.parquet"))

    # -----------------------------
    # 2B) Alpha panel for Part 2A21
    # -----------------------------
    alpha_panel = build_alpha_panel_from_prices(data, X_live, cfg, H)
    alpha_panel.to_parquet(os.path.join(cfg.out_dir, "alpha_panel.parquet"))

    # -----------------------------
    # 3) Regression targets
    # -----------------------------
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
            raise RuntimeError(
                f"LEAKAGE GUARD: y_reg_revealed extends too far. "
                f"max_yreg={y_reg_revealed.index.max().date()} > max_ok={max_ok_date.date()}"
            )

    y_reg_revealed.to_parquet(os.path.join(cfg.out_dir, "y_reg_revealed.parquet"))
    y_reg.to_parquet(os.path.join(cfg.out_dir, "y_reg_full.parquet"))

    # -----------------------------
    # 4) Regression training join
    # -----------------------------
    reg_train = X_live.join(y_reg_revealed, how="inner")

    if len(reg_train) < cfg.min_reg_rows:
        raise RuntimeError(f"Too few regression training rows after join: {len(reg_train)}")

    if reg_train[X_live.columns].isna().any().any():
        raise RuntimeError("NaNs detected in regression training features after join.")

    reg_train.to_parquet(os.path.join(cfg.out_dir, "regression_train.parquet"))

    # -----------------------------
    # 5) Diagnostics
    # -----------------------------
    diag = {
        "date_min_data": str(data.index.min().date()),
        "date_max_data": str(data.index.max().date()),
        "date_min_X_live": str(X_live.index.min().date()) if len(X_live) else None,
        "date_max_X_live": str(X_live.index.max().date()) if len(X_live) else None,
        "date_min_y_reg_revealed": str(y_reg_revealed.index.min().date()) if len(y_reg_revealed) else None,
        "date_max_y_reg_revealed": str(y_reg_revealed.index.max().date()) if len(y_reg_revealed) else None,
        "n_data": int(len(data)),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
        "n_alpha_panel": int(len(alpha_panel)),
        "n_rows_lost_join": int(len(X_live) - len(reg_train)),
        "missing_frac": miss_frac,
        "max_equal_close_run": stale_report,
        "stale_dropped_rows": int(drop_mask.sum()),
    }
    with open(os.path.join(cfg.out_dir, "part1_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # -----------------------------
    # 6) Live snapshots
    # -----------------------------
    price_calls_live_snapshot = pd.DataFrame(
        {"px_voo_t": px_voo, "px_ief_t": px_ief},
        index=data.index,
    ).loc[X_live.index].copy()
    price_calls_live_snapshot.to_parquet(os.path.join(cfg.out_dir, "price_calls_live_snapshot.parquet"))

    data[["VOO"]].rename(columns={"VOO": "px"}).loc[X_live.index].to_parquet(
        os.path.join(cfg.out_dir, "target_prices_snapshot.parquet")
    )

    # -----------------------------
    # 7) Metadata
    # -----------------------------
    meta = {
        "part": "part1",
        "version": "V19",
        "asof_date": last_feature_date.strftime("%Y-%m-%d"),
        "horizon": H,
        "tail_threshold": float(cfg.tail_threshold),
        "tail_label_name": "y_rel_tail_voo_vs_ief",
        "feature_cols": list(X_live.columns),
        "reg_target_cols": list(y_reg_revealed.columns),
        "benchmark_ticker": cfg.benchmark_ticker,
        "alpha_universe": list(cfg.alpha_universe),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
        "n_alpha_panel": int(len(alpha_panel)),
    }
    with open(os.path.join(cfg.out_dir, "part1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    tail_rate = float(y_revealed["y_rel_tail_voo_vs_ief"].mean()) if len(y_revealed) else np.nan
    print(f"✅ V19 COMPLETE | Relative Tail Base Rate (revealed): {tail_rate:.2%}")
    print(
        "Wrote: X_features.parquet, y_labels_revealed.parquet, y_labels_full.parquet, "
        "factor_returns.parquet, benchmark_returns.parquet, alpha_panel.parquet, "
        "y_reg_revealed.parquet, y_reg_full.parquet, regression_train.parquet, "
        "price_calls_live_snapshot.parquet, target_prices_snapshot.parquet, asof_date.txt, "
        "part1_meta.json, part1_diagnostics.json"
    )

    return {
        "out_dir": cfg.out_dir,
        "asof_date": last_feature_date.strftime("%Y-%m-%d"),
        "n_X_live": int(len(X_live)),
        "n_y_reg_revealed": int(len(y_reg_revealed)),
        "n_reg_train": int(len(reg_train)),
        "n_alpha_panel": int(len(alpha_panel)),
        "tail_rate": tail_rate,
    }


def main() -> int:
    cfg = Part1Config()
    summary = build_part1_v19(cfg)
    print("\nPart 1 summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
