#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from fredapi import Fred
    HAVE_FRED = True
except Exception:
    Fred = None
    HAVE_FRED = False

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False


@dataclass(frozen=True)
class Part0Config:
    version: str = "V1_COLAB_FINAL"
    start: str = "2005-01-01"
    end: str = date.today().strftime("%Y-%m-%d")
    horizon: int = 7

    root_env_var: str = "PRICECALL_ROOT"
    default_drive_root: str = "/content/drive/MyDrive/PriceCallProject"
    out_dir_name: str = "artifacts_part0"
    db_filename: str = "market_data.duckdb"

    equity_tickers: Tuple[str, ...] = (
        "VOO", "IEF", "TLT", "SHY",
        "GLD", "UUP",
        "JNK", "HYG", "LQD",
        "RSP", "QQQ", "IWM", "MDY",
        "XLK", "XLF", "XLI", "XLY", "XLP",
        "XLV", "XLE", "XLU", "XLB", "XLC",
        "SMH", "EFA", "EEM",
        "VNQ", "DBC",
    )
    vix_tickers: Tuple[str, ...] = ("^VIX", "^VIX3M", "^SKEW", "^MOVE")
    benchmark_ticker: str = "VOO"

    fred_series: Dict[str, str] = field(default_factory=lambda: {
        "DFF": "fed_funds_rate",
        "DGS2": "yield_2y",
        "DGS10": "yield_10y",
        "DGS30": "yield_30y",
        "T10Y2Y": "curve_2s10s",
        "T10Y3M": "curve_3m10y",
        "BAMLH0A0HYM2": "hy_spread",
        "BAMLC0A0CM": "ig_spread",
        "TEDRATE": "ted_spread",
        "VIXCLS": "vix_fred",
        "DCOILWTICO": "wti_oil",
        "DTWEXBGS": "dollar_index",
        "UMCSENT": "consumer_sentiment",
        "USREC": "recession_flag",
    })
    fred_api_key: str = "09c48c7ed1bb6d3e9811c8e85bd5c48d"

    allow_ffill_limit: int = 2
    max_pre_clean_warn_frac: float = 0.10
    min_history_years: float = 5.0
    core_tickers: Tuple[str, ...] = ("VOO", "IEF")


CFG = Part0Config()


def _resolve_project_root(cfg: Part0Config) -> Path:
    candidates = []

    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(Path(cfg.default_drive_root))

    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    candidates.append(Path.cwd())

    seen = set()
    cleaned = []
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        key = str(rp)
        if key not in seen:
            seen.add(key)
            cleaned.append(rp)

    for p in cleaned:
        if p.exists():
            return p

    return Path.cwd().resolve()


def _out_dir(cfg: Part0Config) -> Path:
    return _resolve_project_root(cfg) / cfg.out_dir_name


def _db_path(cfg: Part0Config) -> Path:
    return _out_dir(cfg) / cfg.db_filename


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=True).encode()).hexdigest()[:16]


def _standardize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    try:
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
    except Exception:
        pass
    out.index = out.index.normalize()
    out = out[~out.index.isna()].sort_index()
    out.index.name = "Date"
    return out


def _business_day_calendar(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def _max_consecutive_equal(x: pd.Series) -> int:
    arr = pd.Series(x).dropna().values
    if len(arr) == 0:
        return 0
    best = run = 1
    for i in range(1, len(arr)):
        run = run + 1 if arr[i] == arr[i - 1] else 1
        best = max(best, run)
    return int(best)


def download_market_data(cfg: Part0Config):
    tickers = list(dict.fromkeys(cfg.equity_tickers + cfg.vix_tickers))
    raw = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Part 0 failed to download market data.")

    bidx = _business_day_calendar(cfg.start, cfg.end)
    close = pd.DataFrame(index=bidx)
    volume = pd.DataFrame(index=bidx)
    quality: Dict[str, Dict[str, object]] = {}

    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].copy()
                c = sub["Close"] if "Close" in sub.columns else None
                v = sub["Volume"] if "Volume" in sub.columns else None
            else:
                c = raw["Close"] if "Close" in raw.columns else None
                v = raw["Volume"] if "Volume" in raw.columns else None
            if c is None:
                continue

            c = pd.Series(c).astype(float)
            c.index = pd.to_datetime(c.index)
            c = c.reindex(bidx)
            close[t] = c

            if v is not None:
                vv = pd.Series(v).astype(float)
                vv.index = pd.to_datetime(vv.index)
                volume[t] = vv.reindex(bidx)

            first_valid = c.dropna().index.min()
            years_history = 0.0
            if pd.notna(first_valid):
                years_history = (bidx.max() - first_valid).days / 365.25

            quality[t] = {
                "missing_pre_clean": float(c.isna().mean()),
                "max_equal_close_run": _max_consecutive_equal(c),
                "first_valid_date": str(first_valid.date()) if pd.notna(first_valid) else None,
                "years_history": round(years_history, 2),
                "usable_for_model": bool(years_history >= cfg.min_history_years),
            }
        except Exception as e:
            quality[t] = {"error": str(e)}

    close = _standardize_index(close)
    volume = _standardize_index(volume)

    core = [t for t in cfg.core_tickers if t in close.columns]
    if close.empty or len(core) != len(cfg.core_tickers):
        raise RuntimeError(
            f"Part 0 requires core tickers {cfg.core_tickers}, found columns={list(close.columns)}"
        )

    pre_bad = {
        k: v["missing_pre_clean"]
        for k, v in quality.items()
        if isinstance(v, dict) and v.get("missing_pre_clean", 0) > cfg.max_pre_clean_warn_frac
    }
    if pre_bad:
        print(f"[Part 0] Pre-clean missingness warning (> {cfg.max_pre_clean_warn_frac:.0%}): {pre_bad}")

    core_first_valid = {t: close[t].dropna().index.min() for t in core}
    bad_first_valid = {t: v for t, v in core_first_valid.items() if pd.isna(v)}
    if bad_first_valid:
        raise RuntimeError(f"Part 0 core tickers never become valid: {bad_first_valid}")

    common_start = max(core_first_valid.values())
    close = close.loc[close.index >= common_start].copy()
    volume = volume.loc[volume.index >= common_start].copy()

    close = close.ffill(limit=cfg.allow_ffill_limit)
    volume = volume.ffill(limit=cfg.allow_ffill_limit)

    post_missing = close[core].isna().mean().to_dict()
    bad_post = {k: float(v) for k, v in post_missing.items() if float(v) > 0.0}
    if bad_post:
        raise RuntimeError(
            "Part 0 core tickers still have NaN after core-history trim and cleaning. "
            f"common_start={common_start.date()} | Post-clean missingness: {bad_post}"
        )

    print(
        f"[Part 0] Market data: {close.shape[0]} days × {close.shape[1]} tickers "
        f"| core_history_start={common_start.date()}"
    )
    return close, volume, quality


def download_fred_data(cfg: Part0Config) -> pd.DataFrame:
    api_key = cfg.fred_api_key or os.environ.get("FRED_API_KEY", "")
    if not HAVE_FRED or not api_key:
        print("[Part 0] Skipping FRED download (fredapi not installed or FRED_API_KEY missing).")
        return pd.DataFrame(index=_business_day_calendar(cfg.start, cfg.end))

    fred = Fred(api_key=api_key)
    bidx = _business_day_calendar(cfg.start, cfg.end)
    cols = []
    for series_id, col_name in cfg.fred_series.items():
        try:
            s = fred.get_series(series_id, observation_start=cfg.start, observation_end=cfg.end)
            s = pd.Series(s.values, index=pd.to_datetime(s.index), name=col_name)
            s.index = s.index.normalize()
            s = s.reindex(bidx).ffill(limit=5)
            cols.append(s)
            print(f"  FRED {series_id:18s} -> {col_name:22s} | {int(s.notna().sum())} obs")
        except Exception as e:
            print(f"  FRED {series_id} FAILED: {e}")

    if not cols:
        return pd.DataFrame(index=bidx)
    macro = pd.concat(cols, axis=1)
    macro.index.name = "Date"
    print(f"[Part 0] FRED macro: {macro.shape[0]} days × {macro.shape[1]} series")
    return macro


def compute_market_features(close: pd.DataFrame, macro: pd.DataFrame, cfg: Part0Config) -> pd.DataFrame:
    X = pd.DataFrame(index=close.index)

    def _log_ret(ticker: str, n: int = 1) -> pd.Series:
        if ticker not in close.columns:
            return pd.Series(np.nan, index=close.index)
        return np.log(close[ticker]).diff(n)

    def _vol(ticker: str, window: int, ann: bool = True) -> pd.Series:
        r = _log_ret(ticker)
        v = r.rolling(window).std()
        return v * np.sqrt(252) if ann else v

    X["voo_vol5"] = _vol("VOO", 5)
    X["voo_vol10"] = _vol("VOO", 10)
    X["voo_vol21"] = _vol("VOO", 21)
    X["ief_vol10"] = _vol("IEF", 10)
    X["spread_vol10"] = (_log_ret("VOO") - _log_ret("IEF")).rolling(10).std() * np.sqrt(252)

    X["vix_level"] = close.get("^VIX", pd.Series(np.nan, index=close.index))
    X["vix_z21"] = (X["vix_level"] - X["vix_level"].rolling(21).mean()) / (X["vix_level"].rolling(21).std() + 1e-9)

    if {"^VIX", "^VIX3M"} <= set(close.columns):
        X["vix_term_ratio"] = close["^VIX"] / (close["^VIX3M"] + 1e-9)
        X["vix_term_z21"] = (X["vix_term_ratio"] - X["vix_term_ratio"].rolling(21).mean()) / (X["vix_term_ratio"].rolling(21).std() + 1e-9)
    else:
        X["vix_term_ratio"] = np.nan
        X["vix_term_z21"] = np.nan

    X["vix_rv_gap"] = X["vix_level"] / (100.0 * X["voo_vol21"] + 1e-9)

    if "^SKEW" in close.columns:
        X["skew_index"] = close["^SKEW"]
        X["skew_z21"] = (X["skew_index"] - X["skew_index"].rolling(21).mean()) / (X["skew_index"].rolling(21).std() + 1e-9)
    else:
        X["skew_index"] = np.nan
        X["skew_z21"] = np.nan

    sectors = [t for t in ["XLK","XLF","XLI","XLY","XLP","XLV","XLE","XLU","XLB","XLC"] if t in close.columns]
    if len(sectors) >= 5:
        sec_rets = pd.concat([_log_ret(t) for t in sectors], axis=1)
        X["sector_dispersion"] = sec_rets.std(axis=1).rolling(5).mean()
    else:
        X["sector_dispersion"] = np.nan

    X["voo_mom5"] = _log_ret("VOO", 5)
    X["voo_mom21"] = _log_ret("VOO", 21)
    X["voo_mom63"] = _log_ret("VOO", 63)
    X["ief_mom5"] = _log_ret("IEF", 5)
    X["ief_mom21"] = _log_ret("IEF", 21)
    X["spread_mom5"] = X["voo_mom5"] - X["ief_mom5"]
    X["spread_mom21"] = X["voo_mom21"] - X["ief_mom21"]
    X["voo_trend_up21"] = (_log_ret("VOO") > 0).rolling(21).mean()
    voo_log = np.log(close["VOO"])
    X["voo_z63"] = (voo_log - voo_log.rolling(63).mean()) / (voo_log.rolling(63).std() + 1e-9)

    if "GLD" in close.columns and "VOO" in close.columns:
        rel = np.log(close["GLD"] / close["VOO"])
        X["gld_voo_ratio_z21"] = (rel - rel.rolling(21).mean()) / (rel.rolling(21).std() + 1e-9)

    if "UUP" in close.columns:
        X["dollar_mom21"] = _log_ret("UUP", 21)
    elif "DBC" in close.columns:
        X["commodities_mom21"] = _log_ret("DBC", 21)

    jnk = close.get("JNK", pd.Series(np.nan, index=close.index))
    lqd = close.get("LQD", pd.Series(np.nan, index=close.index))
    ief = close.get("IEF", pd.Series(np.nan, index=close.index))
    X["hy_ig_spread"] = np.log(jnk / lqd)
    X["hy_ig_z21"] = (X["hy_ig_spread"] - X["hy_ig_spread"].rolling(21).mean()) / (X["hy_ig_spread"].rolling(21).std() + 1e-9)
    X["hy_ig_momentum5"] = X["hy_ig_spread"].diff(5)
    X["jnk_ief_spread"] = np.log(jnk / ief)
    X["credit_accel"] = X["jnk_ief_spread"].diff().diff()

    if "TLT" in close.columns and "SHY" in close.columns:
        dur = np.log(close["TLT"] / close["SHY"])
        X["duration_spread_proxy"] = dur
        X["duration_spread_z21"] = (dur - dur.rolling(21).mean()) / (dur.rolling(21).std() + 1e-9)
        X["duration_spread_mom5"] = dur.diff(5)
    else:
        X["duration_spread_proxy"] = np.nan
        X["duration_spread_z21"] = np.nan
        X["duration_spread_mom5"] = np.nan

    if "TLT" in close.columns:
        X["tlt_vol21"] = _vol("TLT", 21)
        X["tlt_mom21"] = _log_ret("TLT", 21)

    def _safe_log_ratio(a: str, b: str) -> pd.Series:
        if a not in close.columns or b not in close.columns:
            return pd.Series(np.nan, index=close.index)
        return np.log(close[a] / close[b])

    X["breadth_rsp_voo"] = _safe_log_ratio("RSP", "VOO")
    X["tech_relative"] = _safe_log_ratio("QQQ", "VOO")
    X["smallcap_rel"] = _safe_log_ratio("IWM", "VOO")
    X["intl_rel"] = _safe_log_ratio("EFA", "VOO")
    for col in ["breadth_rsp_voo", "tech_relative", "smallcap_rel", "intl_rel"]:
        X[f"{col}_z21"] = (X[col] - X[col].rolling(21).mean()) / (X[col].rolling(21).std() + 1e-9)

    X["vix_mom5"] = close.get("^VIX", pd.Series(np.nan, index=close.index)).diff(5)

    if macro is not None and not macro.empty:
        mm = macro.reindex(close.index).ffill(limit=5)
        if "curve_2s10s" in mm.columns:
            X["yield_curve_2s10s"] = mm["curve_2s10s"]
            X["yield_curve_2s10s_chg5"] = mm["curve_2s10s"].diff(5)
            X["curve_inverted"] = (mm["curve_2s10s"] < 0).astype(float)
        if {"yield_10y", "yield_2y"} <= set(mm.columns):
            X["yield_10y"] = mm["yield_10y"]
            X["yield_2y"] = mm["yield_2y"]
            X["yield_10y_chg21"] = mm["yield_10y"].diff(21)
        if "hy_spread" in mm.columns:
            X["hy_spread_fred"] = mm["hy_spread"]
            X["hy_spread_z21"] = (mm["hy_spread"] - mm["hy_spread"].rolling(21).mean()) / (mm["hy_spread"].rolling(21).std() + 1e-9)
        if "dollar_index" in mm.columns:
            X["dollar_index"] = mm["dollar_index"]
            X["dollar_mom21"] = mm["dollar_index"].diff(21)
        if "recession_flag" in mm.columns:
            X["in_recession"] = mm["recession_flag"].ffill()
        if "consumer_sentiment" in mm.columns:
            X["consumer_sentiment_z21"] = (mm["consumer_sentiment"] - mm["consumer_sentiment"].rolling(21).mean()) / (mm["consumer_sentiment"].rolling(21).std() + 1e-9)

    idx = pd.to_datetime(X.index)
    X["dow_monday"] = (idx.weekday == 0).astype(float)
    X["dow_friday"] = (idx.weekday == 4).astype(float)
    X["month_end"] = idx.is_month_end.astype(float)
    X["quarter_end"] = idx.is_quarter_end.astype(float)

    X.index.name = "Date"
    print(f"[Part 0] Feature matrix: {X.shape[0]} days × {X.shape[1]} features")
    return X


def compute_labels(close: pd.DataFrame, cfg: Part0Config) -> pd.DataFrame:
    H = cfg.horizon
    voo = close["VOO"].astype(float)
    ief = close["IEF"].astype(float)

    labels = pd.DataFrame(index=close.index)
    labels.index.name = "Date"

    for h in [5, 7, 10, 21]:
        fwd_voo = np.log(voo).shift(-h) - np.log(voo)
        fwd_ief = np.log(ief).shift(-h) - np.log(ief)
        excess = fwd_voo - fwd_ief
        labels[f"fwd_voo_{h}d"] = fwd_voo
        labels[f"fwd_ief_{h}d"] = fwd_ief
        labels[f"excess_ret_{h}d"] = excess
        labels[f"y_tail_{h}d"] = np.where(np.isfinite(excess), (excess < -0.015).astype(float), np.nan)
        labels[f"excess_rank_{h}d"] = excess.rolling(252, min_periods=63).rank(pct=True)

    labels["y_rel_tail_voo_vs_ief"] = labels[f"y_tail_{H}d"]
    labels["fwd_voo"] = labels[f"fwd_voo_{H}d"]
    labels["fwd_ief"] = labels[f"fwd_ief_{H}d"]
    labels["excess_ret"] = labels[f"excess_ret_{H}d"]
    labels["px_voo_t"] = voo
    labels["px_ief_t"] = ief
    labels["px_voo_fwd"] = voo * np.exp(labels[f"fwd_voo_{H}d"])
    labels["px_ief_fwd"] = ief * np.exp(labels[f"fwd_ief_{H}d"])
    return labels


def save_outputs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    features: pd.DataFrame,
    macro: pd.DataFrame,
    labels: pd.DataFrame,
    quality: Dict,
    cfg: Part0Config,
) -> None:
    out_dir = _out_dir(cfg)
    db_path = _db_path(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    close.to_parquet(out_dir / "close_prices.parquet")
    volume.to_parquet(out_dir / "volume_data.parquet")
    features.to_parquet(out_dir / "features_full.parquet")
    if macro is not None and not macro.empty:
        macro.to_parquet(out_dir / "macro_data.parquet")
    labels.dropna(subset=["y_rel_tail_voo_vs_ief"]).to_parquet(out_dir / "y_labels_revealed.parquet")
    labels.to_parquet(out_dir / "y_labels_full.parquet")

    if HAVE_DUCKDB:
        con = duckdb.connect(str(db_path))
        try:
            for name, df in [
                ("close_prices", close.reset_index()),
                ("volume", volume.reset_index()),
                ("features_full", features.reset_index()),
                ("y_labels_full", labels.reset_index()),
            ]:
                con.execute(f"DROP TABLE IF EXISTS {name}")
                con.register("tmp_df", df)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
                con.unregister("tmp_df")
            if macro is not None and not macro.empty:
                con.execute("DROP TABLE IF EXISTS macro_data")
                con.register("tmp_df", macro.reset_index())
                con.execute("CREATE TABLE macro_data AS SELECT * FROM tmp_df")
                con.unregister("tmp_df")
        finally:
            con.close()
        print(f"[Part 0] Saved DuckDB + parquet compatibility artifacts to {out_dir}")
    else:
        print(f"[Part 0] Saved parquet compatibility artifacts to {out_dir}")

    meta = {
        "version": cfg.version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(_resolve_project_root(cfg)),
        "out_dir": str(out_dir),
        "date_range": {"start": cfg.start, "end": cfg.end},
        "horizon": cfg.horizon,
        "n_market_tickers": int(len(close.columns)),
        "n_features": int(len(features.columns)),
        "n_macro_series": int(len(macro.columns)) if macro is not None and not macro.empty else 0,
        "fred_enabled": bool(macro is not None and not macro.empty),
        "data_quality": quality,
        "features_checksum": _sha256_df(features),
        "close_checksum": _sha256_df(close),
        "tail_rate": float(labels["y_rel_tail_voo_vs_ief"].mean()),
    }
    with open(out_dir / "part0_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)


def main() -> int:
    cfg = CFG
    out_dir = _out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PART 0 — Data Infrastructure v1 (Colab-compatible)")
    print("=" * 70)
    print(f"[Part 0] Project root: {_resolve_project_root(cfg)}")
    print(f"[Part 0] Output dir:    {out_dir}")

    close, volume, quality = download_market_data(cfg)
    macro = download_fred_data(cfg)
    features = compute_market_features(close, macro, cfg)
    labels = compute_labels(close, cfg)
    save_outputs(close, volume, features, macro, labels, quality, cfg)

    core_cols = [c for c in ["voo_vol5", "voo_vol10", "voo_vol21", "ief_vol10", "spread_vol10"] if c in features.columns]
    full_rows_core = int(features[core_cols].notna().all(axis=1).sum()) if core_cols else 0
    usable_feature_frac = float(features.notna().mean().mean())
    tail_rate = float(labels["y_rel_tail_voo_vs_ief"].mean())

    fred_text = (
        "YES (" + str(len(macro.columns)) + " series)"
        if macro is not None and not macro.empty
        else "NO (set FRED_API_KEY)"
    )

    print("\n✅ PART 0 COMPLETE")
    print(f"   Market tickers:  {len(close.columns)}")
    print(f"   Features:        {len(features.columns)}")
    print(f"   FRED macro:      {fred_text}")
    print(f"   Core full rows:  {full_rows_core}")
    print(f"   Avg fill rate:   {usable_feature_frac:.2%}")
    print(f"   Tail base rate:  {tail_rate:.2%}")
    print(
        "   Wrote:           close_prices.parquet, volume_data.parquet, features_full.parquet, "
        "y_labels_revealed.parquet, y_labels_full.parquet, part0_meta.json"
        + (", market_data.duckdb" if HAVE_DUCKDB else "")
    )
    return 0


if __name__ == "__main__":
    main()

