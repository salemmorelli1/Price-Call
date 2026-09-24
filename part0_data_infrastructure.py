#!/usr/bin/env python3
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



import csv
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from market_calendar import completed_xnys_sessions, latest_completed_xnys_session

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
    version: str = "V2_DAILY_CANONICAL"
    start: str = "2005-01-01"
    end: str = date.today().strftime("%Y-%m-%d")
    horizon: int = 1

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
        candidates.append(Path(_DRIVE_ROOT))
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
    """Compatibility name for the completed XNYS session calendar."""
    return completed_xnys_sessions(start, end)


def _max_consecutive_equal(x: pd.Series) -> int:
    arr = pd.Series(x).dropna().values
    if len(arr) == 0:
        return 0
    best = run = 1
    for i in range(1, len(arr)):
        run = run + 1 if arr[i] == arr[i - 1] else 1
        best = max(best, run)
    return int(best)


def _extract_yfinance_field(
    frame: pd.DataFrame,
    ticker: str,
    field: str,
) -> pd.Series | None:
    """Extract one ticker field from either yfinance MultiIndex layout.

    ``yf.download`` has returned both ``(ticker, field)`` and
    ``(field, ticker)`` column orders across versions and call shapes.  The
    bulk request asks for ticker-first columns, while a one-ticker retry can
    still return field-first columns.  Treating only one layout as valid makes
    the recovery path silently ineffective.
    """
    if frame is None or frame.empty:
        return None
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame[field] if field in frame.columns else None

    for level in range(frame.columns.nlevels):
        labels = frame.columns.get_level_values(level).astype(str)
        if ticker not in set(labels):
            continue
        subset = frame.xs(ticker, axis=1, level=level, drop_level=True)
        if isinstance(subset, pd.Series):
            return subset if field in {ticker, subset.name} else None
        if field in subset.columns:
            value = subset[field]
            return value.iloc[:, -1] if isinstance(value, pd.DataFrame) else value

    # Defensive fallback for an unexpected level name/order.
    for column in frame.columns:
        labels = {str(value) for value in column}
        if ticker in labels and field in labels:
            return frame[column]
    return None


def _series_on_calendar(
    values: pd.Series | None,
    ticker: str,
    calendar: pd.DatetimeIndex,
) -> pd.Series | None:
    if values is None:
        return None
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    series.index = pd.to_datetime(series.index, errors="coerce")
    frame = _standardize_index(pd.DataFrame({ticker: series}))
    if frame.empty:
        return pd.Series(index=calendar, dtype=float, name=ticker)
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[ticker].reindex(calendar)


def _recover_verified_historical_core_closes(
    cfg: Part0Config,
    close: pd.DataFrame,
    quality: Dict[str, Dict[str, object]],
) -> None:
    """Replay one historical session's previously verified *observed* closes.

    Yahoo occasionally omits a bar in subsequent downloads even though the
    previous production run and its independent realized-price backfill stored
    that bar.  Accept only the exact prior verified session, with an intact
    published manifest, matching production lineage, agreeing anchor/backfill
    observations, and a matching adjacent-session price scale.  Never recover
    the latest session or synthesize a close.  All other gaps still fail below.
    """
    root = _resolve_project_root(cfg)
    status_path = root / "artifacts_part10_bot" / "pipeline_status.json"
    meta_path = root / "artifacts_part0" / "part0_meta.json"
    log_path = root / "artifacts_part3" / "prediction_log.csv"
    if not all(path.is_file() for path in (status_path, meta_path, log_path)):
        return

    try:
        from artifact_integrity import PROTOCOL_VERSION, verify_run_manifest

        failures = verify_run_manifest(root)
        if failures:
            raise ValueError(f"published manifest is invalid: {failures[0]}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        session = pd.Timestamp(status["market_data_asof"])
        session_date = session.date().isoformat()
        if (
            status.get("result") != "verified"
            or status.get("protocol_version") != PROTOCOL_VERSION
            or status.get("expected_completed_market_session") != session_date
            or meta.get("market_data_asof") != session_date
            or meta.get("market_values_are_raw_observations") is not True
            or session not in close.index
            or session >= close.index.max()
        ):
            raise ValueError("previous verified session or raw-observation provenance is absent")

        previous = close.index[close.index.get_loc(session) - 1]
        if previous >= session:
            raise ValueError("no preceding exchange session for price-scale check")
        last_raw = meta.get("last_raw_observation_by_ticker", {})
        if any(last_raw.get(ticker) != session_date for ticker in cfg.core_tickers):
            raise ValueError("prior snapshot did not observe both core closes on that session")

        with log_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        def _run_number(value: str) -> Decimal:
            number = Decimal(str(value))
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("non-integral run provenance")
            return number

        source_run = _run_number(status["github_run_id"])
        source_attempt = _run_number(status["github_run_attempt"])
        matching = [
            row for row in rows
            if row.get("decision_date") == session_date
            and _run_number(row.get("pipeline_run_id", "")) == source_run
            and _run_number(row.get("pipeline_run_attempt", "")) == source_attempt
            and row.get("model_code_sha") == status.get("source_code_sha")
            and row.get("model_protocol_version") == PROTOCOL_VERSION
        ]
        realized = [
            row for row in rows
            if row.get("decision_date") == previous.date().isoformat()
            and row.get("target_date") == session_date
            and row.get("realized_target_date") == session_date
            and row.get("model_protocol_version") == PROTOCOL_VERSION
        ]
        if len(matching) != 1 or len(realized) != 1:
            raise ValueError("unique matching anchor and realized observations are absent")

        # Validate the complete pair before mutating either ticker's price.
        recovered = {}
        for ticker in cfg.core_tickers:
            if ticker not in close.columns:
                continue
            name = ticker.lower()
            anchor = float(matching[0][f"px_{name}_t"])
            backfilled = float(realized[0][f"px_{name}_realized"])
            prior_anchor = float(realized[0][f"px_{name}_t"])
            prior_download = float(close.at[previous, ticker])
            if not all(np.isfinite(value) and value > 0 for value in (
                anchor, backfilled, prior_anchor, prior_download
            )) or not np.isclose(anchor, backfilled, rtol=1e-7, atol=1e-6):
                raise ValueError(f"{ticker} archived anchor and realized close disagree")
            if not np.isclose(prior_anchor, prior_download, rtol=1e-4, atol=1e-6):
                raise ValueError(f"{ticker} archived and current price scales disagree")
            observed = close.at[session, ticker]
            if pd.notna(observed):
                if not np.isclose(float(observed), anchor, rtol=1e-4, atol=1e-6):
                    raise ValueError(f"{ticker} archived and current closes disagree")
            else:
                recovered[ticker] = anchor

        for ticker, price in recovered.items():
            close.at[session, ticker] = price
            entry = dict(quality.get(ticker, {}))
            entry["verified_archive_recovered_dates"] = [session_date]
            entry["verified_archive_source_run_id"] = str(source_run)
            entry["missing_after_retry"] = float(close[ticker].isna().mean())
            quality[ticker] = entry
            print(f"[Part 0]   {ticker} recovered observed {session_date} close "
                  f"from verified production run {source_run}")
    except (OSError, ValueError, TypeError, KeyError, IndexError, InvalidOperation,
            csv.Error) as exc:
        print(f"[Part 0] Verified historical close recovery unavailable: {exc}")


def download_market_data(cfg: Part0Config):
    tickers = list(dict.fromkeys(cfg.equity_tickers + cfg.vix_tickers))
    bidx = _business_day_calendar(cfg.start, cfg.end)
    if bidx.empty:
        raise RuntimeError("Part 0 has no completed XNYS sessions in the requested range.")
    # yfinance's end boundary is exclusive. Request the calendar day after the
    # final completed session, never the wall-clock configuration date.
    download_end = (bidx.max() + pd.Timedelta(days=1)).date().isoformat()
    raw = yf.download(
        tickers=tickers,
        start=cfg.start,
        end=download_end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Part 0 failed to download market data.")

    close = pd.DataFrame(index=bidx)
    volume = pd.DataFrame(index=bidx)
    quality: Dict[str, Dict[str, object]] = {}

    for t in tickers:
        try:
            c = _extract_yfinance_field(raw, t, "Close")
            v = _extract_yfinance_field(raw, t, "Volume")
            if c is None:
                continue

            c = _series_on_calendar(c, t, bidx)
            if c is None:
                continue
            close[t] = c

            if v is not None:
                vv = _series_on_calendar(v, t, bidx)
                if vv is not None:
                    volume[t] = vv

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

    # Retry transient bulk-download gaps as individual, single-threaded requests.
    # Core prices are still never forward-filled: a retry must return an actual raw
    # observation for every post-inception XNYS session or the hard failure below
    # remains authoritative.
    _core_set = set(cfg.core_tickers)
    _noncore_all_nan = [
        t for t in tickers
        if t in close.columns and close[t].isna().all() and t not in _core_set
    ]
    _core_with_gaps = []
    for _t in cfg.core_tickers:
        if _t not in close.columns or close[_t].isna().all():
            _core_with_gaps.append(_t)
            continue
        _first = close[_t].first_valid_index()
        if _first is not None and close.loc[_first:, _t].isna().any():
            _core_with_gaps.append(_t)
    _retry_tickers = list(dict.fromkeys(_core_with_gaps + _noncore_all_nan))

    if _retry_tickers:
        print(
            f"[Part 0] Retrying {len(_retry_tickers)} incomplete ticker(s) "
            f"individually: {_retry_tickers}"
        )
        for _t in _retry_tickers:
            existing = (
                pd.to_numeric(close[_t], errors="coerce")
                if _t in close.columns
                else pd.Series(index=bidx, dtype=float, name=_t)
            )
            first_valid = existing.first_valid_index()
            gaps = (
                existing.loc[first_valid:].index[existing.loc[first_valid:].isna()]
                if first_valid is not None
                else bidx
            )
            retry_start_date = (
                max(pd.Timestamp(cfg.start), gaps.min() - pd.Timedelta(days=7))
                if len(gaps)
                else pd.Timestamp(cfg.start)
            )
            recovered_total = 0
            attempts_used = 0
            for _attempt in range(1, 4):  # up to 3 retry attempts
                attempts_used = _attempt
                try:
                    if _attempt > 1:
                        time.sleep(1.5 * (_attempt - 1))
                    _r = yf.download(
                        tickers=[_t],
                        start=retry_start_date.date().isoformat(),
                        end=download_end,
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=False,   # single-threaded: no SQLite lock contention
                    )
                    if _r is None or _r.empty:
                        print(f"[Part 0]   {_t} attempt {_attempt}: empty response")
                        continue
                    _c = _series_on_calendar(
                        _extract_yfinance_field(_r, _t, "Close"), _t, bidx
                    )
                    if _c is None or _c.isna().all():
                        print(f"[Part 0]   {_t} attempt {_attempt}: still all-NaN")
                        continue
                    recovered = existing.isna() & _c.notna()
                    recovered_total += int(recovered.sum())
                    existing = existing.combine_first(_c)
                    close[_t] = existing

                    _fv = existing.first_valid_index()
                    remaining = (
                        int(existing.loc[_fv:].isna().sum())
                        if _fv is not None
                        else int(len(existing))
                    )
                    print(
                        f"[Part 0]   {_t} attempt {_attempt}: recovered "
                        f"{int(recovered.sum())} row(s); remaining post-inception gaps={remaining}"
                    )
                    if _t not in _core_set or remaining == 0:
                        break
                except Exception as _retry_e:
                    print(f"[Part 0]   {_t} attempt {_attempt}: {_retry_e}")

            _fv = existing.first_valid_index()
            quality_entry = dict(quality.get(_t, {}))
            quality_entry.update({
                "individual_retry_attempted": True,
                "individual_retry_attempts": attempts_used,
                "individual_retry_recovered_rows": recovered_total,
                "missing_after_retry": float(existing.isna().mean()),
                "first_valid_date": str(_fv.date()) if _fv is not None else None,
                "years_history": (
                    round((bidx.max() - _fv).days / 365.25, 2)
                    if _fv is not None else 0.0
                ),
                "usable_for_model": bool(
                    _fv is not None
                    and (bidx.max() - _fv).days / 365.25 >= cfg.min_history_years
                ),
            })
            quality[_t] = quality_entry

    # A short, paired VOO/IEF request is the last raw-data recovery path.  The
    # realized-price backfill uses this request shape successfully when the
    # long multi-ticker download and single-ticker requests omit recent bars.
    # Use actual observations only; an empty or incomplete response still fails
    # the core-close check below.  Never forward-fill an exchange session.
    core_gaps = []
    for ticker in cfg.core_tickers:
        if ticker not in close.columns or close[ticker].isna().all():
            core_gaps.append(bidx.min())
        else:
            first = close[ticker].first_valid_index()
            missing = close.loc[first:, ticker]
            if missing.isna().any():
                core_gaps.append(missing.index[missing.isna()].min())
    if core_gaps:
        paired_start = max(pd.Timestamp(cfg.start), min(core_gaps) - pd.Timedelta(days=7))
        print(
            f"[Part 0] Retrying incomplete core closes together from "
            f"{paired_start.date()}"
        )
        try:
            # Match the backfill's two-ticker request shape and default column
            # layout; _extract_yfinance_field supports either MultiIndex order.
            paired = yf.download(
                list(cfg.core_tickers),
                start=paired_start.date().isoformat(),
                end=download_end,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            print(f"[Part 0] Paired core retry failed: {exc}")
        else:
            for ticker in cfg.core_tickers:
                candidate = _series_on_calendar(
                    _extract_yfinance_field(paired, ticker, "Close"), ticker, bidx
                )
                if candidate is None:
                    continue
                candidate = candidate.where(np.isfinite(candidate) & (candidate > 0))
                existing = (
                    close[ticker] if ticker in close.columns
                    else pd.Series(index=bidx, dtype=float, name=ticker)
                )
                overlap = existing.notna() & candidate.notna()
                if overlap.any() and not np.allclose(
                    existing.loc[overlap], candidate.loc[overlap], rtol=1e-4, atol=1e-6
                ):
                    raise RuntimeError(
                        f"Part 0 paired core retry disagrees with existing {ticker} "
                        "prices on overlapping sessions."
                    )
                recovered = existing.isna() & candidate.notna()
                close[ticker] = existing.combine_first(candidate)
                paired_volume = _series_on_calendar(
                    _extract_yfinance_field(paired, ticker, "Volume"), ticker, bidx
                )
                if paired_volume is not None:
                    prior_volume = (
                        volume[ticker] if ticker in volume.columns
                        else pd.Series(index=bidx, dtype=float, name=ticker)
                    )
                    volume[ticker] = prior_volume.combine_first(
                        paired_volume.where(np.isfinite(paired_volume) & (paired_volume >= 0))
                    )
                entry = dict(quality.get(ticker, {}))
                entry["paired_retry_recovered_rows"] = int(recovered.sum())
                entry["paired_retry_recovered_dates"] = [
                    day.date().isoformat() for day in bidx[recovered]
                ]
                entry["paired_retry_start"] = paired_start.date().isoformat()
                entry["missing_after_retry"] = float(close[ticker].isna().mean())
                first_valid = close[ticker].first_valid_index()
                years_history = (
                    (bidx.max() - first_valid).days / 365.25
                    if first_valid is not None else 0.0
                )
                entry["first_valid_date"] = (
                    first_valid.date().isoformat() if first_valid is not None else None
                )
                entry["years_history"] = round(years_history, 2)
                entry["usable_for_model"] = bool(
                    first_valid is not None and years_history >= cfg.min_history_years
                )
                quality[ticker] = entry
                print(f"[Part 0]   {ticker} paired retry recovered {int(recovered.sum())} row(s)")

    # VOO did not exist at cfg.start; only gaps *after* its first observation
    # need archive recovery.  Do not inspect the archive on ordinary runs.
    outstanding_core_gaps = [
        ticker for ticker in cfg.core_tickers
        if ticker in close
        and close[ticker].first_valid_index() is not None
        and close.loc[close[ticker].first_valid_index():, ticker].isna().any()
    ]
    if outstanding_core_gaps:
        _recover_verified_historical_core_closes(cfg, close, quality)

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

    # Market closes remain true source observations. Filling is allowed only
    # later, after Part 1 has measured per-ticker freshness on this raw table.
    post_missing = close[core].isna().mean().to_dict()
    bad_post = {k: float(v) for k, v in post_missing.items() if float(v) > 0.0}
    if bad_post:
        missing_dates = {
            ticker: [date.date().isoformat() for date in close.index[close[ticker].isna()][:10]]
            for ticker in bad_post
        }
        raise RuntimeError(
            "Part 0 core tickers still have NaN after raw-data retries and "
            "verified historical-close recovery. "
            f"common_start={common_start.date()} | Post-retry missingness: {bad_post} "
            f"| First missing XNYS dates: {missing_dates}"
        )

    print(
        f"[Part 0] Market data: {close.shape[0]} days × {close.shape[1]} tickers "
        f"| core_history_start={common_start.date()}"
    )
    return close, volume, quality


def download_fred_data(cfg: Part0Config) -> pd.DataFrame:
    # Credentials are accepted from the runner environment only. Keeping a
    # configurable plaintext default would cause it to override GitHub Secrets.
    api_key = os.environ.get("FRED_API_KEY", "").strip()
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


def _fill_vixcls_from_market(macro: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Backfill vix_fred from ^VIX market close when FRED VIXCLS download failed.

    VIXCLS is the CBOE VIX index from FRED. It is identical to ^VIX from yfinance
    up to rounding. When FRED returns None (rate-limit, API outage, or series
    temporarily unavailable), we substitute the yfinance ^VIX close so that
    n_macro_series stays at 14 and any downstream feature that reads vix_fred
    does not receive NaN. The substitution is noted in the returned DataFrame's
    column name (vix_fred is preserved for schema compatibility).
    """
    if "vix_fred" not in macro.columns and "^VIX" in close.columns:
        vix_market = close["^VIX"].copy()
        vix_market.index.name = "Date"
        macro = macro.copy()
        macro["vix_fred"] = vix_market.reindex(macro.index).ffill(limit=5)
        print("[Part 0] vix_fred: FRED VIXCLS unavailable — filled from ^VIX market close (equivalent series).")
    return macro


def _fill_macro_from_last_good(
    macro: pd.DataFrame,
    cfg: Part0Config,
    out_dir: "Path",
) -> pd.DataFrame:
    """Carry forward last-known-good values for FRED series that failed this run.

    When FRED returns a 500 or rate-limit error, the affected column is simply
    absent from the macro DataFrame. This function:
      1. Identifies which expected series are missing from the current run.
      2. Reads the previously saved macro_data.parquet from the Part 0 output dir.
      3. For each missing series present in the prior artifact, reindexes those
         values to the current date range and forward-fills up to 10 business
         days (covers weekends, market holidays, and short FRED outages).
      4. Logs clearly which series came from cache vs live.

    This is the generic complement to _fill_vixcls_from_market. VIXCLS has a
    live market equivalent (^VIX) and is handled there. Series without a live
    market proxy — primarily curve_3m10y (T10Y3M) and hy_spread (BAMLH0A0HYM2)
    — are recovered here via the persisted artifact.

    Design properties:
      • Stable checksum: a successful run writes all 14 series to macro_data.parquet.
        A failed run falls back to those same values, so the feature matrix produced
        is identical to the previous clean run. features_checksum stays constant
        under transient FRED outages.
      • Cold-start safe: if no prior artifact exists, missing series remain absent
        and an error is printed. No crash, no silent NaN injection.
      • Self-healing: as soon as FRED recovers, the live value replaces the cached
        one automatically, and the updated series is written back to the artifact.
      • No side effects on market data or the locked Part 1/Part 2 feature contract.
    """
    expected_cols = set(cfg.fred_series.values())
    # vix_fred is handled by _fill_vixcls_from_market; exclude from this path.
    missing = sorted(c for c in expected_cols if c not in macro.columns and c != "vix_fred")
    if not missing:
        return macro

    prev_path = out_dir / "macro_data.parquet"
    if not prev_path.exists():
        for col in missing:
            sid = next((k for k, v in cfg.fred_series.items() if v == col), col.upper())
            print(
                f"[Part 0] {col}: FRED {sid} unavailable — "
                f"no prior artifact for fallback (cold start, will recover on next successful run)."
            )
        return macro

    try:
        prev = pd.read_parquet(prev_path)
        prev = _standardize_index(prev)
    except Exception as e:
        print(f"[Part 0] Could not load prior macro artifact for fallback: {e}")
        return macro

    macro = macro.copy()
    for col in missing:
        sid = next((k for k, v in cfg.fred_series.items() if v == col), col.upper())
        if col not in prev.columns:
            print(
                f"[Part 0] {col}: FRED {sid} unavailable — "
                f"not present in prior artifact (was never successfully fetched)."
            )
            continue
        carried = prev[col].reindex(macro.index).ffill(limit=10)
        n_filled = int(carried.notna().sum())
        if n_filled == 0:
            print(
                f"[Part 0] {col}: FRED {sid} unavailable — "
                f"prior artifact had no usable values for current date range."
            )
            continue
        macro[col] = carried
        print(
            f"[Part 0] {col}: FRED {sid} unavailable — "
            f"carried forward from last-good artifact ({n_filled} obs)."
        )
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
            # FIX: renamed to avoid silent overwrite of UUP-based dollar_mom21 (L353).
            # These are different signals: UUP = currency ETF; FRED = trade-weighted index.
            X["dollar_mom21_fred"] = mm["dollar_index"].diff(21)
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

    for h in [1, 5, 7, 10, 21]:
        fwd_voo = np.log(voo).shift(-h) - np.log(voo)
        fwd_ief = np.log(ief).shift(-h) - np.log(ief)
        excess = fwd_voo - fwd_ief
        labels[f"fwd_voo_{h}d"] = fwd_voo
        labels[f"fwd_ief_{h}d"] = fwd_ief
        labels[f"excess_ret_{h}d"] = excess
        # FIX (Finding 26, Audit 2026-04-21):
        # The prior formula -0.015 * sqrt(h/7) anchors the threshold to H=7 and
        # scales DOWN. At H=1 this gives -0.00567 — the H=7 value divided by sqrt(7).
        # The correct sqrt-time scaling rule (for independent increments) goes UP from
        # a base daily value: thr(H) = thr(1) * sqrt(H).
        # At H=1: -0.015 (base daily threshold)
        # At H=7: -0.015 * sqrt(7) = -0.03969
        # Note: Part 1 overrides these fixed labels with rolling-quantile labels for
        # the training dataset. These fixed-threshold labels in y_labels_full.parquet
        # and y_labels_revealed.parquet are used for analysis/cross-check only.
        thr_h = float(-0.015 * np.sqrt(h))
        labels[f"y_tail_{h}d"] = np.where(np.isfinite(excess), (excess < thr_h).astype(float), np.nan)
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
    close.notna().astype("uint8").to_parquet(out_dir / "market_observation_mask.parquet")
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
        "date_range": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "requested_end": cfg.end,
        },
        "market_calendar": "XNYS",
        "latest_completed_market_session": str(latest_completed_xnys_session().date()),
        "market_data_asof": str(close.index.max().date()),
        "market_values_are_raw_observations": True,
        "last_raw_observation_by_ticker": {
            ticker: (
                str(close.index[close[ticker].notna()].max().date())
                if close[ticker].notna().any()
                else None
            )
            for ticker in close.columns
        },
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
    macro = _fill_vixcls_from_market(macro, close)
    macro = _fill_macro_from_last_good(macro, cfg, out_dir)
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
        "   Wrote:           close_prices.parquet, market_observation_mask.parquet, volume_data.parquet, features_full.parquet, "
        "y_labels_revealed.parquet, y_labels_full.parquet, part0_meta.json"
        + (", market_data.duckdb" if HAVE_DUCKDB else "")
    )
    return 0


if __name__ == "__main__":
    main()
