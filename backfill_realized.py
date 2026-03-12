#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Environment / project root
# -----------------------------
def resolve_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


ROOT = resolve_root()


# -----------------------------
# Config
# -----------------------------
@dataclass
class BackfillConfig:
    pred_log_path: str = str(ROOT / "artifacts_part3" / "prediction_log.csv")
    tickers: tuple[str, str] = ("VOO", "IEF")
    price_start_buffer_days: int = 14
    price_end_buffer_days: int = 2


CFG = BackfillConfig()


# -----------------------------
# Helpers
# -----------------------------
def safe_num(x):
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


def download_close_panel(cfg: BackfillConfig, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download(
        list(cfg.tickers),
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=cfg.price_end_buffer_days)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    return extract_close_panel(raw, cfg.tickers)


def idx_of_date(index: pd.Index, dt: pd.Timestamp) -> int | None:
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
            # not matured yet
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


# -----------------------------
# Main job
# -----------------------------
def main(force: bool = False, show_paths: bool = True) -> int:
    if show_paths:
        print_path_status(CFG)

    if not Path(CFG.pred_log_path).exists():
        print("prediction_log.csv not found yet. Nothing to backfill.")
        return 0

    log = load_log(CFG)

    if log.empty:
        print("prediction_log.csv is empty. Nothing to backfill.")
        return 0

    valid_decisions = log["decision_date"].dropna()
    if valid_decisions.empty:
        print("prediction_log.csv has no valid decision_date values.")
        return 0

    start_date = valid_decisions.min() - pd.Timedelta(days=CFG.price_start_buffer_days)
    end_date = pd.Timestamp.today().normalize()

    prices = download_close_panel(CFG, start_date, end_date)

    before = int(log["px_voo_realized"].notna().sum()) if "px_voo_realized" in log.columns else 0
    log, updated_rows = backfill_log(log, prices)
    after = int(log["px_voo_realized"].notna().sum()) if "px_voo_realized" in log.columns else before

    log = log.sort_values("decision_date").reset_index(drop=True)
    Path(CFG.pred_log_path).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(CFG.pred_log_path, index=False)

    print(f"Backfill updated {updated_rows} row(s).")
    print(f"Realized rows before: {before} | after: {after}")
    print(f"Saved: {CFG.pred_log_path}")
    print_summary(log)
    return 0


# -----------------------------
# CLI wrapper
# -----------------------------
def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill realized prices and error metrics in prediction_log.csv.")
    parser.add_argument("--force", action="store_true", help="Reserved for future use.")
    parser.add_argument("--no-paths", action="store_true", help="Suppress path-status printing.")

    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        args, _ = parser.parse_known_args(args=[] if argv is None else argv)
    else:
        args = parser.parse_args(argv)

    return main(force=args.force, show_paths=not args.no_paths)


if __name__ == "__main__":
    rc = cli()
    if "ipykernel" not in sys.modules and "google.colab" not in sys.modules:
        raise SystemExit(rc)
