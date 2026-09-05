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

from artifact_integrity import current_evidence_mask, write_json_strict
from market_calendar import latest_completed_xnys_session

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "yfinance is required for backfill_realized.py. Install it with `%pip install yfinance`."
    ) from exc

# FIX (Quant-Guild Part 32 Audit — Backfill OperationalError):
# yfinance stores timezone data in a SQLite database under the system user-cache
# directory (~/.cache/py-yfinance/tkr-tz.db by default). On GitHub Actions the
# same path is accessed by every concurrent job step, causing SQLite
# "database is locked" errors that abort the backfill before any price data
# is downloaded. The fix redirects ALL yfinance caches (tz, cookie, ISIN) to a
# per-process /tmp directory. Each process gets its own directory (keyed by PID),
# so concurrent jobs never contend on the same file. The temp directory is created
# here unconditionally; yfinance will populate it lazily on first use.
import os as _os, tempfile as _tempfile
_yf_cache_dir = _os.path.join(_tempfile.gettempdir(), f"yf_cache_backfill_{_os.getpid()}")
_os.makedirs(_yf_cache_dir, exist_ok=True)
try:
    yf.set_tz_cache_location(_yf_cache_dir)
except Exception as _yf_cache_exc:
    print(f"[backfill] WARNING: could not set yfinance cache location: {_yf_cache_exc}")
    print("[backfill] Proceeding; SQLite lock errors may still occur.")


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
    # FIX (Quant-Guild Part 32 Audit — Backfill OperationalError):
    # Added retry loop (3 attempts, 5-second sleep between attempts) to handle:
    #   (a) transient yfinance SQLite lock errors that survive the cache relocation fix,
    #   (b) transient network failures that return an empty DataFrame on first attempt.
    # On each retry the per-process cache directory is re-pointed to a fresh temp path
    # so a corrupted or locked db from a prior attempt cannot block subsequent attempts.
    import time as _time

    start_str = start.strftime("%Y-%m-%d")
    # yfinance's end is exclusive. Never request or accept a bar after the
    # exchange calendar's latest completed session.
    end_str = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    _max_attempts = 3
    _last_exc: Optional[Exception] = None

    for _attempt in range(1, _max_attempts + 1):
        try:
            # On retries, redirect cache to a fresh unique directory so any
            # corrupted or locked SQLite file from the previous attempt is bypassed.
            if _attempt > 1:
                import os as _os2, tempfile as _tf2
                _retry_cache = _os2.path.join(_tf2.gettempdir(), f"yf_cache_retry_{_os2.getpid()}_{_attempt}")
                _os2.makedirs(_retry_cache, exist_ok=True)
                try:
                    yf.set_tz_cache_location(_retry_cache)
                except Exception:
                    pass
                _sleep_secs = 5 * _attempt
                print(f"[backfill] Download attempt {_attempt}/{_max_attempts} "
                      f"(sleeping {_sleep_secs}s after prior failure) ...")
                _time.sleep(_sleep_secs)

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
            close = close.loc[close.index <= end]

            if close.empty or not {"VOO", "IEF"}.issubset(close.columns):
                raise RuntimeError("yfinance returned empty or incomplete data for VOO/IEF.")

            return close

        except Exception as _exc:
            _last_exc = _exc
            print(f"[backfill] Download attempt {_attempt}/{_max_attempts} failed: {_exc}")

    raise RuntimeError(
        f"Unable to download usable VOO/IEF close history for backfill "
        f"after {_max_attempts} attempts. Last error: {_last_exc}"
    )


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
        print("\n[ERROR] prediction_log.csv is empty; backfill cannot be verified.")
        return 1

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
    end = latest_completed_xnys_session()
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

    eligible_mask = current_evidence_mask(df)
    live_realized_count = int(current_evidence_mask(df, require_realized=True).sum())
    current_eligible_predictions = int(eligible_mask.sum())

    print("\n=== BACKFILL SUMMARY ===")
    print(f"Prediction log: {PREDLOG_PATH}")
    print(f"Trading calendar last date: {latest_trading_date.date()}")
    print(f"Matured rows identified: {matured_rows}")
    print(f"Rows newly updated: {updated_rows}")
    print(f"Rows with realized prices now present: {realized_count}")
    print(f"Rows with eligible realized prices in the current protocol: {live_realized_count}")

    # FIX (F1, Quant-Guild Part 54 Audit): Patch part3_summary.json after the
    # predlog is written with realized prices.
    #
    # ROOT CAUSE OF PERSISTENT live_realized_dates=0 (fifth consecutive session):
    # Part 3 has four phases that try to count realized rows at progressively later
    # points within the pipeline run. All four phases read the same file because
    # Part 3 itself writes predlog_out with 0 realized rows (the CI restore gave it
    # a stale predlog), and the backfill runs AFTER the pipeline commits. Phase 4
    # (added S53) re-reads predlog_out AFTER the summary write — but predlog_out
    # was WRITTEN BY Part3 with 0 realized rows, so Phase 4 also reads 0.
    #
    # The correct architectural fix is to move the summary patch OUTSIDE Part 3
    # and into the backfill script, which is the only process that:
    #   (a) runs AFTER realized prices are confirmed available, and
    #   (b) has already written the amended predlog to disk.
    #
    # After this block runs, part3_summary.json will reflect the true live
    # realized count regardless of the race between the Tuesday pipeline and
    # the daily backfill. The patch is:
    #   - Exact: may increase or decrease after a protocol/eligibility change
    #   - Idempotent: no-op if the summary already shows the exact eligible count
    #   - Failure-closed: any write/regeneration failure aborts publication
    #   - Scope-limited: ONLY live_realized_dates, prediction_log_realized_rows,
    #     prediction_log_realized_pct are patched; no other fields are touched
    #
    # This fix renders Phases 1-4 inside Part 3 permanently safe: if the race
    # fires and Part 3 writes live_realized_dates=0, the backfill corrects it
    # within the same CI job (before the commit step commits the summary). [FIX F1/S54]
    _summary_path = PROJECT_DIR / "artifacts_part3_v1" / "part3_summary.json"
    if _summary_path.exists():
        try:
            import json as _json_bf
            with open(_summary_path, "r", encoding="utf-8") as _sf:
                _summary = _json_bf.load(_sf)
            _current_live = int(_summary.get("live_realized_dates", 0) or 0)
            if live_realized_count != _current_live:
                _total_predlog_rows = max(current_eligible_predictions, 1)
                _summary["live_realized_dates"] = live_realized_count
                _summary["prediction_log_realized_rows"] = live_realized_count
                _summary["prediction_log_realized_pct"] = round(
                    live_realized_count / _total_predlog_rows, 4
                )
                write_json_strict(_summary_path, _summary)
                print(
                    f"[backfill] FIX F1/S54: part3_summary.json patched — "
                    f"eligible live_realized_dates {_current_live} → {live_realized_count}"
                )
            else:
                print(
                    f"[backfill] FIX F1/S54: part3_summary.json already shows "
                    f"eligible live_realized_dates={_current_live} — no patch needed."
                )
        except Exception as _summary_patch_exc:
            raise RuntimeError(
                "Part 3 summary regeneration failed after backfill; "
                f"refusing a green workflow: {_summary_patch_exc}"
            ) from _summary_patch_exc
    else:
        raise FileNotFoundError(
            f"Part 3 summary is missing after backfill: {_summary_path}"
        )

    # FIX (F3, Quant-Guild S58 Audit): After patching part3_summary.json, also
    # regenerate live_attribution_report.json by invoking Part 9's generate_live_report.
    #
    # ROOT CAUSE: Part 9 runs at 9:35 AM ET during the pipeline and reads the predlog
    # BEFORE backfill adds realized prices. It writes live_attribution_report.json with
    # n_live_realized=0 (or the count before today's backfill). The F1/S58 fix in Part 9
    # adds a disk re-read at runtime to catch prior-session backfill data. But for the
    # CURRENT session's backfill — where Part 9 already ran and wrote the stale report —
    # Part 9 needs to be called again AFTER backfill completes.
    #
    # This fix calls Part 9's generate_live_report() directly (no subprocess), which is
    # lightweight (just reads predlog + writes JSON). The call is exception-safe:
    # any failure falls through silently, leaving the existing report intact.
    #
    # The Part 9 F1/S58 disk re-read ensures it picks up the just-committed realized prices
    # even on the first call after backfill. This is idempotent if Part 9 was already correct.
    # [FIX F3/S58]
    _part9_report_path = PROJECT_DIR / "artifacts_part9" / "live_attribution_report.json"
    try:
        import importlib.util as _ilu
        import sys as _sys
        _p9_spec = _ilu.spec_from_file_location(
            "part9_live_attribution",
            PROJECT_DIR / "part9_live_attribution.py"
        )
        if _p9_spec is not None and _p9_spec.loader is not None:
            _p9_mod = _ilu.module_from_spec(_p9_spec)
            # dataclasses and other import-time machinery require the module to
            # be present in sys.modules while it executes.
            _sys.modules[_p9_spec.name] = _p9_mod
            _p9_spec.loader.exec_module(_p9_mod)
            _p9_cfg = _p9_mod.Part9Config()
            _p9_report = _p9_mod.generate_live_report(_p9_cfg)
            _part9_report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_strict(_part9_report_path, _p9_report)
            _n_live_new = _p9_report.get("n_live_realized", 0)
            print(
                f"[backfill] FIX F3/S58: live_attribution_report.json regenerated — "
                f"n_live_realized={_n_live_new}.  [S58 F3]"
            )
        else:
            raise RuntimeError("could not load part9_live_attribution.py")
    except Exception as _p9_exc:
        raise RuntimeError(
            f"Part 9 regeneration failed after backfill; refusing a green workflow: {_p9_exc}"
        ) from _p9_exc

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    
