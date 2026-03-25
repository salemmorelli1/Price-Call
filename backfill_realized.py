#!/usr/bin/env python3
from __future__ import annotations

"""
backfill_realized.py
==============================================================================
Backfill realized prices and forecast error metrics into prediction_log.csv.

This version is workflow-safe and supports both artifact families:
- ./artifacts_part3/prediction_log.csv
- ./artifacts_part3_v3b/prediction_log.csv
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


def resolve_root() -> Path:
    if "__file__" in globals():
        try:
            return Path(__file__).resolve().parent
        except Exception:
            pass
    return Path.cwd().resolve()


ROOT = resolve_root()


def discover_pred_log() -> str:
    candidates = [
        ROOT / "artifacts_part3" / "prediction_log.csv",
        ROOT / "artifacts_part3_v3b" / "prediction_log.csv",
        Path("./artifacts_part3/prediction_log.csv"),
        Path("./artifacts_part3_v3b/prediction_log.csv"),
    ]
    existing = [p.resolve() for p in candidates if p.exists()]
    if existing:
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(existing[0])
    return str((ROOT / "artifacts_part3" / "prediction_log.csv").resolve())


@dataclass
class BackfillConfig:
    pred_log_path: str = field(default_factory=discover_pred_log)
    tickers: tuple[str, str] = ("VOO", "IEF")
    price_start_buffer_days: int = 14
    price_end_buffer_days: int = 3


CFG = BackfillConfig()


def safe_num(x: object) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def ensure_columns(log: pd.DataFrame) -> pd.DataFrame:
    required = {
        "decision_date": pd.NaT,
        "h_reb": 7,
        "p_final_cal": np.nan,
        "p0": np.nan,
        "alpha_scale": np.nan,
        "governance_tier": np.nan,
        "px_voo_t": np.nan,
        "px_ief_t": np.nan,
        "px_voo_call_7d": np.nan,
        "px_ief_call_7d": np.nan,
        "voo_call_src": np.nan,
        "ief_call_src": np.nan,
        "target_date": pd.NaT,
        "px_voo_realized": np.nan,
        "px_ief_realized": np.nan,
        "voo_err": np.nan,
        "ief_err": np.nan,
        "voo_abs_err": np.nan,
        "ief_abs_err": np.nan,
        "voo_ape": np.nan,
        "ief_ape": np.nan,
        "spread_err": np.nan,
        "hit_direction": np.nan,
    }
    for col, default in required.items():
        if col not in log.columns:
            log[col] = default
    return log


def load_log(cfg: BackfillConfig) -> pd.DataFrame:
    path = Path(cfg.pred_log_path)
    if not path.exists():
        raise FileNotFoundError(f"prediction_log.csv not found: {path}")

    log = pd.read_csv(path)
    log = ensure_columns(log)
    log["decision_date"] = pd.to_datetime(log["decision_date"], errors="coerce").dt.normalize()
    log["target_date"] = pd.to_datetime(log["target_date"], errors="coerce").dt.normalize()
    return log


def extract_close_panel(raw: pd.DataFrame, tickers: tuple[str, str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("Downloaded price data is missing 'Close' level.")
        px = raw["Close"].copy()
    else:
        px = raw.copy()

    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()

    for t in tickers:
        if t not in px.columns:
            raise RuntimeError(f"Downloaded price data is missing ticker: {t}")

    px = px[list(tickers)].dropna().sort_index()
    if px.empty:
        raise RuntimeError("Downloaded close panel is empty after cleaning.")
    return px


def _download_one_ticker(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if isinstance(raw, pd.DataFrame):
        if "Close" in raw.columns:
            s = pd.to_numeric(raw["Close"], errors="coerce")
        elif raw.shape[1] == 1:
            s = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        else:
            raise RuntimeError(f"Single-ticker download for {ticker} is missing a usable close column.")
    else:
        s = pd.to_numeric(raw, errors="coerce")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s.dropna().sort_index()


def download_close_panel(cfg: BackfillConfig, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    end_plus = end_date + pd.Timedelta(days=cfg.price_end_buffer_days)

    try:
        raw = yf.download(
            list(cfg.tickers),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_plus.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        px = extract_close_panel(raw, cfg.tickers)
        if not px.empty:
            return px
    except Exception as e:
        print(f"[WARN] Batch price download failed: {e}")

    # Fallback: download tickers one-by-one to avoid yfinance cache/SQLite lock issues.
    series_list = []
    for ticker in cfg.tickers:
        try:
            s = _download_one_ticker(ticker, start_date, end_plus)
            if s.empty:
                print(f"[WARN] Single-ticker download returned empty for {ticker}.")
            series_list.append(s)
        except Exception as e:
            print(f"[WARN] Single-ticker download failed for {ticker}: {e}")

    if not series_list:
        raise RuntimeError("All yfinance downloads failed.")

    px = pd.concat(series_list, axis=1)
    for ticker in cfg.tickers:
        if ticker not in px.columns:
            px[ticker] = np.nan
    px = px[list(cfg.tickers)].dropna().sort_index()
    if px.empty:
        raise RuntimeError("Downloaded close panel is empty after cleaning.")
    return px


def idx_of_date(index: pd.Index, dt: pd.Timestamp) -> Optional[int]:
    matches = np.where(index == pd.Timestamp(dt).normalize())[0]
    if len(matches) == 0:
        return None
    return int(matches[-1])


def backfill_log(log: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    updated_rows = 0
    px_index = prices.index

    for i in range(len(log)):
        d0 = log.loc[i, "decision_date"]
        if pd.isna(d0):
            continue

        already_done = pd.notna(log.loc[i, "px_voo_realized"]) and pd.notna(log.loc[i, "px_ief_realized"])
        if already_done:
            continue

        h = int(safe_num(log.loc[i, "h_reb"])) if pd.notna(log.loc[i, "h_reb"]) else 7
        idx0 = idx_of_date(px_index, pd.Timestamp(d0))
        if idx0 is None:
            continue

        idxT = idx0 + h
        if idxT >= len(px_index):
            continue

        target_date = px_index[idxT]
        px_voo_real = float(prices.loc[target_date, "VOO"])
        px_ief_real = float(prices.loc[target_date, "IEF"])

        log.loc[i, "target_date"] = target_date
        log.loc[i, "px_voo_realized"] = px_voo_real
        log.loc[i, "px_ief_realized"] = px_ief_real

        pv = safe_num(log.loc[i, "px_voo_call_7d"])
        pi = safe_num(log.loc[i, "px_ief_call_7d"])

        if np.isfinite(pv):
            err = px_voo_real - pv
            log.loc[i, "voo_err"] = err
            log.loc[i, "voo_abs_err"] = abs(err)
            log.loc[i, "voo_ape"] = abs(err) / (abs(px_voo_real) + 1e-12)

        if np.isfinite(pi):
            err = px_ief_real - pi
            log.loc[i, "ief_err"] = err
            log.loc[i, "ief_abs_err"] = abs(err)
            log.loc[i, "ief_ape"] = abs(err) / (abs(px_ief_real) + 1e-12)

        if np.isfinite(pv) and np.isfinite(pi):
            pred_spread = pv - pi
            real_spread = px_voo_real - px_ief_real
            log.loc[i, "spread_err"] = real_spread - pred_spread
            log.loc[i, "hit_direction"] = int(np.sign(real_spread) == np.sign(pred_spread))

        updated_rows += 1

    return log, updated_rows


def print_summary(log: pd.DataFrame) -> None:
    realized = log.dropna(subset=["px_voo_realized", "px_ief_realized"]).copy()
    if realized.empty:
        print("No matured rows available yet.")
        return

    voo_mae = float(realized["voo_abs_err"].mean()) if "voo_abs_err" in realized else np.nan
    ief_mae = float(realized["ief_abs_err"].mean()) if "ief_abs_err" in realized else np.nan
    voo_mape = float(100.0 * realized["voo_ape"].mean()) if "voo_ape" in realized else np.nan
    ief_mape = float(100.0 * realized["ief_ape"].mean()) if "ief_ape" in realized else np.nan

    hit_rate = np.nan
    if "hit_direction" in realized.columns and realized["hit_direction"].notna().any():
        hit_rate = float(realized["hit_direction"].dropna().mean())

    print(f"Realized rows: {len(realized)}")
    print(f"VOO MAE:  {voo_mae:.4f} | VOO MAPE:  {voo_mape:.2f}%")
    print(f"IEF MAE:  {ief_mae:.4f} | IEF MAPE:  {ief_mape:.2f}%")
    if np.isfinite(hit_rate):
        print(f"Direction hit rate: {100.0 * hit_rate:.2f}%")


def print_path_status(cfg: BackfillConfig) -> None:
    print(f"ROOT:     {ROOT}")
    print(f"PRED_LOG: {cfg.pred_log_path} | exists={Path(cfg.pred_log_path).exists()}")


def main(force: bool = False, show_paths: bool = True, pred_log_path: Optional[str] = None) -> int:
    cfg = BackfillConfig(pred_log_path=pred_log_path or CFG.pred_log_path)

    if show_paths:
        print_path_status(cfg)

    if not Path(cfg.pred_log_path).exists():
        print("prediction_log.csv not found yet. Nothing to backfill.")
        return 0

    log = load_log(cfg)
    if log.empty:
        print("prediction_log.csv is empty. Nothing to backfill.")
        return 0

    valid_decisions = log["decision_date"].dropna()
    if valid_decisions.empty:
        print("prediction_log.csv has no valid decision_date values.")
        return 0

    start_date = valid_decisions.min() - pd.Timedelta(days=cfg.price_start_buffer_days)
    end_date = pd.Timestamp.today().normalize()
    prices = download_close_panel(cfg, start_date, end_date)

    before = int(log["px_voo_realized"].notna().sum()) if "px_voo_realized" in log.columns else 0
    log, updated_rows = backfill_log(log, prices)
    after = int(log["px_voo_realized"].notna().sum()) if "px_voo_realized" in log.columns else before

    log = log.sort_values("decision_date").reset_index(drop=True)
    Path(cfg.pred_log_path).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(cfg.pred_log_path, index=False)

    print(f"Backfill updated {updated_rows} row(s).")
    print(f"Realized rows before: {before} | after: {after}")
    print(f"Saved: {cfg.pred_log_path}")
    print_summary(log)
    return 0


def cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill realized prices and error metrics in prediction_log.csv.")
    parser.add_argument("--force", action="store_true", help="Reserved for future use.")
    parser.add_argument("--no-paths", action="store_true", help="Suppress path-status printing.")
    parser.add_argument("--pred-log-path", type=str, default=None, help="Override prediction_log.csv path.")

    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        args, _ = parser.parse_known_args(args=[] if argv is None else list(argv))
    else:
        args = parser.parse_args(argv)

    return main(force=args.force, show_paths=not args.no_paths, pred_log_path=args.pred_log_path)


if __name__ == "__main__":
    rc = cli()
    if "ipykernel" not in sys.modules and "google.colab" not in sys.modules:
        raise SystemExit(rc)



