#!/usr/bin/env python3
"""Repair supported market feeds from authoritative, secret-free sources.

The adapter runs immediately after Part 0.  A failed fallback never makes stale
data appear fresh: it records the failure and leaves the original observations
unchanged so Part 1's raw-observation freshness gate remains authoritative.
"""
from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import pandas as pd

from artifact_integrity import read_json_strict, sha256_file, write_json_strict


CBOE_VIX3M_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
)


def _dataframe_checksum(frame: pd.DataFrame) -> str:
    """Match Part 0's compact, content-level dataframe checksum."""
    return hashlib.sha256(frame.to_csv(index=True).encode()).hexdigest()[:16]


def parse_cboe_vix3m_csv(payload: bytes | str) -> pd.Series:
    """Parse Cboe's daily VIX3M history into a normalized close series."""
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if not {"DATE", "CLOSE"}.issubset(frame.columns):
        raise ValueError("Cboe VIX3M response must contain DATE and CLOSE columns")
    dates = pd.to_datetime(frame["DATE"], errors="coerce").dt.tz_localize(None).dt.normalize()
    values = pd.to_numeric(frame["CLOSE"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name="^VIX3M").dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("Cboe VIX3M response contains no usable observations")
    return series


def _download(
    url: str,
    opener: Callable[..., Any] = urlopen,
    timeout: int = 30,
) -> bytes:
    request = Request(url, headers={"User-Agent": "Price-Call/causal-integrity"})
    with opener(request, timeout=timeout) as response:
        return response.read()


def refresh_vix3m(
    root: Path,
    *,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Overlay Cboe VIX3M closes onto Part 0's common market calendar.

    Only dates already present in Part 0's close matrix are updated.  This
    prevents a single auxiliary series from inventing a new decision row before
    the core VOO and IEF observations exist.
    """
    output = root / "artifacts_part0"
    close_path = output / "close_prices.parquet"
    meta_path = output / "part0_meta.json"
    if not close_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Part 0 outputs are incomplete; cannot repair VIX3M")

    meta = read_json_strict(meta_path)
    source: dict[str, Any] = {
        "provider": "Cboe Global Markets",
        "url": CBOE_VIX3M_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
    }
    close_checksum: str | None = None
    try:
        payload = _download(CBOE_VIX3M_URL, opener=opener)
        official = parse_cboe_vix3m_csv(payload)
        close = pd.read_parquet(close_path)
        close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None).normalize()
        close = close[~close.index.isna()].sort_index()
        if "^VIX3M" not in close.columns:
            close["^VIX3M"] = float("nan")
        overlap = close.index.intersection(official.index)
        if overlap.empty:
            raise ValueError("Cboe VIX3M history does not overlap the Part 0 calendar")
        close.loc[overlap, "^VIX3M"] = official.reindex(overlap).to_numpy()
        temporary = close_path.with_suffix(close_path.suffix + ".tmp")
        close.to_parquet(temporary)
        temporary.replace(close_path)
        close_checksum = _dataframe_checksum(close)
        source.update({
            "status": "applied",
            "download_sha256": hashlib.sha256(payload).hexdigest(),
            "rows_downloaded": int(official.size),
            "rows_applied": int(overlap.size),
            "latest_observation": official.index.max().date().isoformat(),
        })
    except Exception as exc:
        source["error"] = f"{type(exc).__name__}: {exc}"

    sources = dict(meta.get("market_data_source_overrides", {}))
    sources["^VIX3M"] = source
    meta.update({
        "market_data_source_overrides": sources,
        "vix3m_authoritative_fallback_applied": source["status"] == "applied",
        "close_prices_sha256": sha256_file(close_path),
    })
    if close_checksum is not None:
        meta["close_checksum"] = close_checksum
    write_json_strict(meta_path, meta)
    print(
        "[Market integrity] Cboe VIX3M fallback "
        f"status={source['status']} latest={source.get('latest_observation', 'unavailable')}"
    )
    return meta


def main() -> int:
    root = Path(os.environ.get("PRICECALL_ROOT", ".")).resolve()
    refresh_vix3m(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
