#!/usr/bin/env python3
"""Build macro features from releases available by each decision date.

This adapter runs immediately after Part 0.  It deliberately obtains its API
credential only from the environment and contains no embedded credential.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fredapi import Fred

from artifact_integrity import PROTOCOL_VERSION, read_json_strict, sha256_file, write_json_strict
import part0_data_infrastructure as part0


FRED_SERIES = {
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
}


class AlfredCoverageError(RuntimeError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def realtime_windows(
    start: object,
    end: object,
    *,
    years: int = 4,
) -> list[tuple[str, str]]:
    """Create non-overlapping ALFRED real-time windows below FRED's cap.

    The FRED file endpoint rejects responses spanning more than 2,000 vintage
    dates. Four-year windows remain below that ceiling even for daily series.
    """
    if years < 1:
        raise ValueError("years must be positive")
    cursor = pd.Timestamp(start).normalize()
    final = pd.Timestamp(end).normalize()
    if cursor > final:
        raise ValueError("start must not be after end")
    windows: list[tuple[str, str]] = []
    while cursor <= final:
        window_end = min(cursor + pd.DateOffset(years=years) - pd.Timedelta(days=1), final)
        windows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def get_series_releases_chunked(
    fred: Fred,
    series_id: str,
    realtime_start: object,
    realtime_end: object,
) -> pd.DataFrame:
    """Fetch releases while isolating pre-history failures by window.

    ALFRED can reject a window before its archived history begins. Only that
    specific error may precede a usable chunk; other API errors are fatal.
    Missing pre-archive dates remain missing in the returned time series.
    """
    chunks: list[pd.DataFrame] = []
    diagnostics: list[dict[str, str | int]] = []
    first_usable_window_start: str | None = None
    seen_usable = False
    coverage_errors: list[str] = []
    required = {"date", "realtime_start", "value"}
    for start, end in realtime_windows(realtime_start, realtime_end):
        try:
            chunk = pd.DataFrame(
                fred.get_series_all_releases(
                    series_id,
                    realtime_start=start,
                    realtime_end=end,
                )
            )
            missing = sorted(required - set(chunk.columns))
            if chunk.empty:
                diagnostics.append({
                    "start": start,
                    "end": end,
                    "status": "empty",
                    "rows": 0,
                })
                continue
            if missing:
                raise ValueError(f"release response missing {missing}")
            usable = chunk.dropna(subset=["date", "realtime_start", "value"])
            if usable.empty:
                diagnostics.append({
                    "start": start,
                    "end": end,
                    "status": "empty_after_validation",
                    "rows": 0,
                })
                continue
            if first_usable_window_start is None:
                first_usable_window_start = start
            seen_usable = True
            chunks.append(chunk)
            diagnostics.append({
                "start": start,
                "end": end,
                "status": "ok",
                "rows": int(len(chunk)),
            })
        except Exception as exc:
            prearchive = (
                not seen_usable
                and "series does not exist in ALFRED" in str(exc)
            )
            status = "prehistory_error" if prearchive else "coverage_error"
            detail = f"{type(exc).__name__}: {exc}"
            diagnostics.append({
                "start": start,
                "end": end,
                "status": status,
                "rows": 0,
                "error": detail,
            })
            if not prearchive:
                coverage_errors.append(f"{start}..{end}: {detail}")

    if coverage_errors:
        raise AlfredCoverageError(
            f"ALFRED coverage failed for {series_id}: "
            + "; ".join(coverage_errors),
            diagnostics,
        )
    releases = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not required.issubset(releases.columns):
        raise ValueError(f"release response missing {sorted(required - set(releases.columns))}")
    releases = releases.drop_duplicates(["date", "realtime_start"], keep="first")
    if releases.empty:
        raise ValueError(f"ALFRED returned no releases for {series_id}")
    releases.attrs["chunk_diagnostics"] = diagnostics
    releases.attrs["continuous_retrieval_start"] = first_usable_window_start
    releases.attrs["requested_realtime_start"] = str(pd.Timestamp(realtime_start).date())
    releases.attrs["requested_realtime_end"] = str(pd.Timestamp(realtime_end).date())
    return releases


def first_release_series(
    releases: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    name: str,
) -> pd.Series:
    """Use each first retrieved vintage from the next exchange session onward.

    ALFRED supplies release dates without an intraday timestamp. Deferring a
    value to the following session avoids assuming it existed by that day's
    price-close decision cutoff.
    """
    frame = pd.DataFrame(releases).copy()
    required = {"date", "realtime_start", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"release response missing {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["realtime_start"] = pd.to_datetime(
        frame["realtime_start"], errors="coerce"
    ).dt.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "realtime_start", "value"])
    if frame.empty:
        raise ValueError("release response contains no usable observations")
    first = frame.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="first")
    availability = first[["date", "realtime_start"]].max(axis=1)
    series = pd.Series(first["value"].to_numpy(), index=availability, name=name)
    # Several observations can be released together.  At a daily decision
    # frequency, the newest observation available that day is authoritative.
    series = series[~series.index.duplicated(keep="last")].sort_index()
    # A release on a closed day becomes available at the next session; a
    # release on a trading day waits one session because its hour is unknown.
    sessions = pd.DatetimeIndex(calendar).sort_values()
    session_positions = sessions.searchsorted(series.index, side="right")
    usable = session_positions < len(sessions)
    available = pd.Series(
        series.to_numpy()[usable],
        index=sessions[session_positions[usable]],
        name=name,
    )
    available = available[~available.index.duplicated(keep="last")].sort_index()
    return available.reindex(sessions).ffill().reindex(calendar)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def _refresh_duckdb(db_path: Path, macro: pd.DataFrame, features: pd.DataFrame) -> None:
    """Replace Part 0's initial revised tables when the optional DB exists."""
    if not db_path.is_file():
        return
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        for table, frame in (("macro_data", macro), ("features_full", features)):
            con.register("pit_frame", frame.reset_index())
            try:
                con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM pit_frame")
            finally:
                con.unregister("pit_frame")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def rebuild_point_in_time_macro(root: Path) -> dict[str, Any]:
    output = root / "artifacts_part0"
    close_path = output / "close_prices.parquet"
    meta_path = output / "part0_meta.json"
    macro_path = output / "macro_data.parquet"
    if not close_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Part 0 outputs are incomplete; cannot apply point-in-time macro policy")

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the point-in-time macro adapter")

    close = pd.read_parquet(close_path)
    close.index = pd.to_datetime(close.index, errors="coerce").normalize()
    close = close[~close.index.isna()].sort_index()
    calendar = pd.DatetimeIndex(close.index.unique(), name="Date")
    fred = Fred(api_key=api_key)
    realtime_start = calendar.min() - pd.DateOffset(years=2)
    realtime_end = calendar.max()
    columns: list[pd.Series] = []
    modes: dict[str, str] = {}
    errors: dict[str, str] = {}
    chunk_diagnostics: dict[str, Any] = {}
    first_availability: dict[str, str | None] = {}
    available_observations: dict[str, int] = {}
    for series_id, name in FRED_SERIES.items():
        try:
            releases = get_series_releases_chunked(
                fred,
                series_id,
                realtime_start,
                realtime_end,
            )
            chunk_diagnostics[name] = releases.attrs.get("chunk_diagnostics", [])
            series = first_release_series(releases, calendar, name)
            if not series.notna().any():
                raise ValueError(f"ALFRED has no releases by {calendar.max().date()} for {series_id}")
            first_window = pd.Timestamp(releases.attrs["continuous_retrieval_start"])
            modes[name] = (
                "alfred_first_release_partial"
                if first_window > calendar.min()
                else "alfred_first_release_chunked"
            )
        except Exception as exc:
            if isinstance(exc, AlfredCoverageError):
                chunk_diagnostics[name] = exc.diagnostics
            # No revised-history or previous-artifact substitution is safe for
            # a backtest. A missing vintage stays missing and closes the gate.
            series = pd.Series(index=calendar, dtype=float, name=name)
            modes[name] = "unavailable"
            errors[name] = f"{type(exc).__name__}: {exc}"
        first = series.first_valid_index()
        first_availability[name] = first.date().isoformat() if first is not None else None
        available_observations[name] = int(series.notna().sum())
        columns.append(series)

    macro = pd.concat(columns, axis=1).reindex(calendar)
    macro.index.name = "Date"
    complete = bool(modes) and all(
        mode in {"alfred_first_release_chunked", "alfred_first_release_partial"}
        for mode in modes.values()
    )

    # Part 6 consumes features_full.parquet, so rebuilding the macro file alone
    # would not remove revised values from the regime engine.
    features = part0.compute_market_features(close, macro, part0.CFG)
    _atomic_parquet(macro, macro_path)
    features_path = output / "features_full.parquet"
    _atomic_parquet(features, features_path)
    _refresh_duckdb(output / "market_data.duckdb", macro, features)

    meta = read_json_strict(meta_path)
    meta.update({
        "protocol_version": PROTOCOL_VERSION,
        "fred_vintage_policy": (
            "earliest retrieved ALFRED release, available at the next exchange session; "
            "pre-archive dates missing, never revised-filled"
        ),
        "fred_vintage_retrieval": "bounded non-overlapping four-year real-time windows",
        "fred_vintage_mode_by_series": modes,
        "fred_vintage_errors": errors,
        "fred_vintage_chunk_diagnostics": chunk_diagnostics,
        "fred_first_availability_by_series": first_availability,
        "fred_available_sessions_by_series": available_observations,
        "fred_revised_fallback_count": 0,
        "historical_point_in_time_policy": "no-revised-values; missing before archive",
        "historical_point_in_time_complete": complete,
        "point_in_time_adapter": "point_in_time_macro.py",
        "features_checksum": part0._sha256_df(features),
        "macro_data_sha256": sha256_file(macro_path),
        "features_file_sha256": sha256_file(features_path),
    })
    write_json_strict(meta_path, meta)
    print(
        f"[Point-in-time macro] complete={complete} "
        f"full={sum(mode == 'alfred_first_release_chunked' for mode in modes.values())} "
        f"partial={sum(mode == 'alfred_first_release_partial' for mode in modes.values())} "
        f"unavailable={sum(mode == 'unavailable' for mode in modes.values())}"
    )
    return meta


def main() -> int:
    root = Path(os.environ.get("PRICECALL_ROOT", ".")).resolve()
    rebuild_point_in_time_macro(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
