#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical realized backfill script for the current daily PriceCall stack.

Behavior
--------
- Mounts Google Drive when running in Colab.
- Audits the canonical Part 3 / Part 7 / Part 8 / Part 9 / Part 10 artifacts.
- Backfills matured rows directly into artifacts_part3/prediction_log.csv.
- Uses the daily H=1 model as the authoritative interpretation.
- Prefers px_*_call_1d columns and falls back to _7d aliases only for compatibility.
"""
# @title File B Overwrite

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "yfinance is required for backfill_realized.py. Install it with `%pip install yfinance`."
    ) from exc


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def maybe_mount_drive() -> bool:
    try:
        from google.colab import drive  # type: ignore

        mount_root = Path("/content/drive")
        if not (mount_root / "MyDrive").exists():
            drive.mount(str(mount_root), force_remount=False)
        else:
            print("Drive already mounted.")
        return True
    except Exception:
        return False


IN_COLAB = maybe_mount_drive()


def resolve_project_dir() -> Path:
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    drive_root = Path("/content/drive/MyDrive/PriceCallProject")
    if IN_COLAB:
        return drive_root

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = resolve_project_dir()
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PRICECALL_ROOT", str(PROJECT_DIR))
os.environ.setdefault("PRICECALL_STRICT_DRIVE_ONLY", "1")
os.environ.setdefault("PRICECALL_ALPHA_FAMILY", "part2a21")

PREDLOG_PATH = PROJECT_DIR / "artifacts_part3" / "prediction_log.csv"

CANONICAL_ARTIFACTS: Dict[str, str] = {
    "PART3_TAPE": "artifacts_part3_v1/v1_final_production_tape.csv",
    "PART3_GOV": "artifacts_part3_v1/v1_final_production_governance.csv",
    "PART3_ALLOC": "artifacts_part3_v1/v1_fusion_allocations.csv",
    "PART3_SUMMARY": "artifacts_part3_v1/part3_summary.json",
    "PREDLOG": "artifacts_part3/prediction_log.csv",
    "PART7": "artifacts_part7/portfolio_weights_tape.csv",
    "PART8_META": "artifacts_part8/part8_meta.json",
    "PART9": "artifacts_part9/live_attribution_report.json",
    "PART10_STATE": "artifacts_part10_bot/portfolio_state.json",
    "PART10_REPORT": "artifacts_part10_bot/performance_report.json",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def audit_paths(project_dir: Path) -> List[Tuple[str, Path, bool]]:
    rows: List[Tuple[str, Path, bool]] = []
    for label, rel in CANONICAL_ARTIFACTS.items():
        path = project_dir / rel
        rows.append((label, path, path.exists()))
    return rows


def _pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in names:
        c = lower_map.get(name.lower())
        if c is not None:
            return c
    return None


def _to_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt.dt.normalize()


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _resolve_call_value(row: pd.Series, asset: str) -> float:
    asset = asset.lower()
    if asset == "voo":
        candidates = [
            "px_voo_call_1d",
            "px_voo_call_7d",   # backward-compat alias
            "voo_call_1d",
            "voo_call_7d",
        ]
    elif asset == "ief":
        candidates = [
            "px_ief_call_1d",
            "px_ief_call_7d",   # backward-compat alias
            "ief_call_1d",
            "ief_call_7d",
        ]
    else:
        return np.nan

    for c in candidates:
        if c in row.index:
            v = _safe_float(row.get(c, np.nan))
            if np.isfinite(v):
                return v
    return np.nan


def _download_close_history(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_str = start.strftime("%Y-%m-%d")
    end_str = (end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    data = yf.download(
        ["VOO", "IEF"],
        start=start_str,
        end=end_str,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        else:
            close = data.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        close = data.copy()

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[[c for c in ["VOO", "IEF"] if c in close.columns]].copy()
    close = close.dropna(how="any")

    if close.empty or not {"VOO", "IEF"}.issubset(close.columns):
        raise RuntimeError("Unable to download usable VOO/IEF close history for backfill.")

    return close


def _resolve_target_trading_date(
    decision_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    h_reb: int,
    explicit_target_date: Optional[pd.Timestamp],
) -> Optional[pd.Timestamp]:
    if explicit_target_date is not None and not pd.isna(explicit_target_date):
        pos = trading_dates.searchsorted(explicit_target_date)
        if pos < len(trading_dates):
            return pd.Timestamp(trading_dates[pos]).normalize()
        return None

    pos = trading_dates.searchsorted(decision_date)
    if pos >= len(trading_dates):
        return None

    target_pos = int(pos + max(int(h_reb), 1))
    if target_pos >= len(trading_dates):
        return None

    return pd.Timestamp(trading_dates[target_pos]).normalize()


def _compute_direction_hit(row: pd.Series) -> float:
    px_voo_t = _safe_float(row.get("px_voo_t", np.nan))
    px_ief_t = _safe_float(row.get("px_ief_t", np.nan))
    px_voo_call = _resolve_call_value(row, "voo")
    px_ief_call = _resolve_call_value(row, "ief")
    px_voo_real = _safe_float(row.get("px_voo_realized", row.get("voo_realized", np.nan)))
    px_ief_real = _safe_float(row.get("px_ief_realized", row.get("ief_realized", np.nan)))

    if not all(np.isfinite(v) for v in [px_voo_t, px_ief_t, px_voo_call, px_ief_call, px_voo_real, px_ief_real]):
        return np.nan
    if px_voo_t == 0 or px_ief_t == 0:
        return np.nan

    pred_spread = (px_voo_call / px_voo_t - 1.0) - (px_ief_call / px_ief_t - 1.0)
    real_spread = (px_voo_real / px_voo_t - 1.0) - (px_ief_real / px_ief_t - 1.0)
    return float(int(np.sign(pred_spread) == np.sign(real_spread)))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    print(f"ROOT: {PROJECT_DIR}")
    print(f"IN_COLAB: {IN_COLAB}")
    print(f"Prediction log exists: {PREDLOG_PATH.exists()}")

    print("\n=== CANONICAL BACKFILL AUDIT ===")
    for label, path, exists in audit_paths(PROJECT_DIR):
        print(f"{label}: {path} | exists = {exists}")

    if not PREDLOG_PATH.exists():
        print("\n[ERROR] artifacts_part3/prediction_log.csv is missing.")
        print("Run File A first so Part 3 writes the canonical prediction log.")
        return 1

    df = pd.read_csv(PREDLOG_PATH)
    if df.empty:
        print("\n[WARN] prediction_log.csv is empty. Nothing to backfill.")
        return 0

    decision_col = _pick_col(df, ["decision_date", "Date"])
    if decision_col is None:
        print("\n[ERROR] prediction log is missing decision_date / Date.")
        return 1
    df[decision_col] = _to_datetime_series(df[decision_col])

    target_col = _pick_col(df, ["target_date"])
    if target_col is not None:
        df[target_col] = _to_datetime_series(df[target_col])

    numeric_cols = [
        "px_voo_realized", "px_ief_realized",
        "voo_realized", "ief_realized",
        "voo_err", "ief_err",
        "voo_abs_err", "ief_abs_err",
        "voo_ape", "ief_ape",
        "spread_err", "hit_direction",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "realized_target_date" not in df.columns:
        df["realized_target_date"] = pd.Series([None] * len(df), dtype="object")
    else:
        df["realized_target_date"] = df["realized_target_date"].astype("object")
        
    start = df[decision_col].dropna().min() - pd.Timedelta(days=20)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    close = _download_close_history(start, end)
    trading_dates = pd.DatetimeIndex(close.index).sort_values()
    latest_trading_date = pd.Timestamp(trading_dates.max()).normalize()

    matured_rows = 0
    updated_rows = 0

    for idx, row in df.iterrows():
        decision_date = row[decision_col]
        if pd.isna(decision_date):
            continue

        # Daily canonical default is H=1.
        h_reb_raw = row.get("h_reb", 1)
        h_reb = int(round(_safe_float(h_reb_raw))) if pd.notna(h_reb_raw) else 1
        if h_reb <= 0:
            h_reb = 1

        explicit_target = row[target_col] if target_col is not None else pd.NaT

        target_trading_date = _resolve_target_trading_date(
            decision_date=pd.Timestamp(decision_date).normalize(),
            trading_dates=trading_dates,
            h_reb=h_reb,
            explicit_target_date=None if pd.isna(explicit_target) else pd.Timestamp(explicit_target).normalize(),
        )

        if target_trading_date is None:
            continue
        if target_trading_date > latest_trading_date:
            continue

        matured_rows += 1

        px_voo_realized = float(close.loc[target_trading_date, "VOO"])
        px_ief_realized = float(close.loc[target_trading_date, "IEF"])

        already_done = (
            np.isfinite(_safe_float(row.get("px_voo_realized", np.nan)))
            and np.isfinite(_safe_float(row.get("px_ief_realized", np.nan)))
        )

        df.at[idx, "realized_target_date"] = target_trading_date.strftime("%Y-%m-%d")
        df.at[idx, "px_voo_realized"] = px_voo_realized
        df.at[idx, "px_ief_realized"] = px_ief_realized
        df.at[idx, "voo_realized"] = px_voo_realized
        df.at[idx, "ief_realized"] = px_ief_realized

        px_voo_call = _resolve_call_value(row, "voo")
        px_ief_call = _resolve_call_value(row, "ief")
        px_voo_t = _safe_float(row.get("px_voo_t", np.nan))
        px_ief_t = _safe_float(row.get("px_ief_t", np.nan))

        if np.isfinite(px_voo_call):
            voo_err = px_voo_realized - px_voo_call
            df.at[idx, "voo_err"] = voo_err
            df.at[idx, "voo_abs_err"] = abs(voo_err)
            if px_voo_call != 0:
                df.at[idx, "voo_ape"] = abs(voo_err) / abs(px_voo_call)

        if np.isfinite(px_ief_call):
            ief_err = px_ief_realized - px_ief_call
            df.at[idx, "ief_err"] = ief_err
            df.at[idx, "ief_abs_err"] = abs(ief_err)
            if px_ief_call != 0:
                df.at[idx, "ief_ape"] = abs(ief_err) / abs(px_ief_call)

        if all(
            np.isfinite(v)
            for v in [px_voo_t, px_ief_t, px_voo_realized, px_ief_realized, px_voo_call, px_ief_call]
        ) and px_voo_t != 0 and px_ief_t != 0:
            real_spread = (px_voo_realized / px_voo_t - 1.0) - (px_ief_realized / px_ief_t - 1.0)
            pred_spread = (px_voo_call / px_voo_t - 1.0) - (px_ief_call / px_ief_t - 1.0)
            df.at[idx, "spread_err"] = real_spread - pred_spread
            df.at[idx, "hit_direction"] = float(int(np.sign(real_spread) == np.sign(pred_spread)))
        else:
            df.at[idx, "hit_direction"] = _compute_direction_hit(df.loc[idx])

        if not already_done:
            updated_rows += 1

    df.to_csv(PREDLOG_PATH, index=False)

    realized_mask = (
        pd.to_numeric(df["px_voo_realized"], errors="coerce").notna()
        & pd.to_numeric(df["px_ief_realized"], errors="coerce").notna()
    )
    realized_count = int(realized_mask.sum())

    live_realized_count = realized_count
    if "horizon_legacy" in df.columns:
        legacy_mask = pd.to_numeric(df["horizon_legacy"], errors="coerce").fillna(0).astype(int) == 1
        live_realized_count = int((realized_mask & ~legacy_mask).sum())

    print("\n=== BACKFILL SUMMARY ===")
    print(f"Prediction log: {PREDLOG_PATH}")
    print(f"Trading calendar last date: {latest_trading_date.date()}")
    print(f"Matured rows identified: {matured_rows}")
    print(f"Rows newly updated: {updated_rows}")
    print(f"Rows with realized prices now present: {realized_count}")
    print(f"Rows with realized prices now present (non-legacy H=1 live rows): {live_realized_count}")
    return 0

if __name__ == "__main__":
    main()



