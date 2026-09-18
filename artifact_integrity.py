#!/usr/bin/env python3
"""Strict JSON and provenance utilities for published Price-Call artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROTOCOL_VERSION = "causal-integrity-v3"
LEGACY_PROTOCOL_VERSION = "legacy-pre-causal-integrity-v3"

REQUIRED_PUBLISHED_FILES = (
    "artifacts_part0/part0_meta.json",
    "artifacts_part1/part1_meta.json",
    "artifacts_part1/part1_diagnostics.json",
    "artifacts_part2_g532/predictions/part2_g532_summary.json",
    "artifacts_part2_g532/predictions/g532_final_consensus_tape.csv",
    "artifacts_part3/prediction_log.csv",
    "artifacts_part3_v1/part3_summary.json",
    "artifacts_part3_v1/v1_final_production_governance.csv",
    "artifacts_part3_v1/v1_final_production_tape.csv",
    "artifacts_part3_v1/v1_fusion_allocations.csv",
    "artifacts_part6/part6_meta.json",
    "artifacts_part7/current_target_weights.json",
    "artifacts_part7/portfolio_weights_tape.csv",
    "artifacts_part8/part8_meta.json",
    "artifacts_part8/execution_instructions.json",
    "artifacts_part8/execution_cost_tape.csv",
    "artifacts_part9/live_attribution_report.json",
    "artifacts_part9/backfill_run_date.txt",
    "artifacts_part9/backfill_status.json",
    "artifacts_part10_bot/portfolio_state.json",
    "artifacts_part10_bot/pipeline_status.json",
    "artifacts_part10_bot/signal_log.csv",
    "artifacts_part10_bot/trade_log.csv",
    "artifacts_part10_bot/performance_report.json",
    "artifacts_part10_bot/pipeline_run_date.txt",
    "artifacts_dashboard/dashboard_snapshot.json",
    "index.html",
)

# These CSVs are append/upsert audit ledgers.  A production or backfill run may
# replace the row for its own completed session or add a newer row, but it must
# never silently discard older rows.
ACCUMULATING_CSV_FILES = (
    "artifacts_part3/prediction_log.csv",
    "artifacts_part3_v1/v1_final_production_governance.csv",
    "artifacts_part8/execution_cost_tape.csv",
    "artifacts_part10_bot/signal_log.csv",
    "artifacts_part10_bot/trade_log.csv",
)

MANIFEST_TEXT_SUFFIXES = frozenset({".csv", ".html", ".json", ".txt"})


def json_safe(value: Any) -> Any:
    """Convert common scientific values into strict, portable JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json_strict(path: str | Path, payload: Any) -> None:
    """Atomically write RFC-compliant JSON (NaN and Infinity are rejected)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temp.replace(target)


def read_json_strict(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_file_record(path: str | Path) -> dict[str, Any]:
    """Fingerprint canonical Git text, independent of checkout line endings."""
    target = Path(path)
    if target.suffix.lower() not in MANIFEST_TEXT_SUFFIXES:
        return {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    content = target.read_bytes().replace(b"\r\n", b"\n")
    return {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def current_evidence_mask(frame: pd.DataFrame, *, require_realized: bool = False) -> pd.Series:
    """Return the single canonical eligibility mask for the current protocol."""
    mask = pd.Series(False, index=frame.index, dtype=bool)
    if frame.empty or "model_protocol_version" not in frame.columns:
        return mask
    eligible = pd.to_numeric(
        frame.get("evidence_eligible", pd.Series(0, index=frame.index)),
        errors="coerce",
    ).fillna(0).astype(int).eq(1)
    mask = frame["model_protocol_version"].fillna(LEGACY_PROTOCOL_VERSION).astype(str).eq(
        PROTOCOL_VERSION
    ) & eligible
    if "horizon_legacy" in frame.columns:
        mask &= pd.to_numeric(frame["horizon_legacy"], errors="coerce").fillna(0).astype(int).eq(0)
    if require_realized:
        voo = next((c for c in ("px_voo_realized", "voo_realized") if c in frame.columns), None)
        ief = next((c for c in ("px_ief_realized", "ief_realized") if c in frame.columns), None)
        if voo is None or ief is None:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask &= pd.to_numeric(frame[voo], errors="coerce").notna()
        mask &= pd.to_numeric(frame[ief], errors="coerce").notna()
    return mask


def validate_json_files(root: str | Path) -> list[str]:
    failures: list[str] = []
    published = [
        "artifacts_part0/part0_meta.json",
        "artifacts_part1/part1_meta.json",
        "artifacts_part1/part1_diagnostics.json",
        "artifacts_part2_g532/predictions/part2_g532_summary.json",
        "artifacts_part3_v1/part3_summary.json",
        "artifacts_part6/part6_meta.json",
        "artifacts_part7/current_target_weights.json",
        "artifacts_part8/part8_meta.json",
        "artifacts_part8/execution_instructions.json",
        "artifacts_part9/live_attribution_report.json",
        "artifacts_part9/backfill_status.json",
        "artifacts_part10_bot/pipeline_status.json",
        "artifacts_dashboard/dashboard_snapshot.json",
    ]
    for path in [Path(root) / rel for rel in published if (Path(root) / rel).is_file()]:
        try:
            read_json_strict(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: {exc}")
    return failures


def validate_required_files(root: str | Path) -> list[str]:
    root_path = Path(root)
    return [rel for rel in REQUIRED_PUBLISHED_FILES if not (root_path / rel).is_file()]


def write_pipeline_status(root: str | Path) -> Path:
    """Record workflow/code identity only after a verified pipeline completes."""
    root_path = Path(root)
    now = datetime.now(timezone.utc)
    et_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    marker = root_path / "artifacts_part10_bot" / "pipeline_run_date.txt"
    pipeline_date = (
        marker.read_text(encoding="utf-8").strip()
        if marker.is_file()
        else os.environ.get("PRICECALL_RUN_DATE_ET", et_date)
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": now.isoformat(),
        "pipeline_run_date": pipeline_date,
        "workflow_completed_et_date": et_date,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_code_sha": os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA"),
        "result": "verified",
    }
    part0_meta = root_path / "artifacts_part0" / "part0_meta.json"
    part1_meta = root_path / "artifacts_part1" / "part1_meta.json"
    if part0_meta.is_file():
        payload["market_data_asof"] = read_json_strict(part0_meta).get("market_data_asof")
    if part1_meta.is_file():
        payload["expected_completed_market_session"] = read_json_strict(part1_meta).get(
            "expected_completed_market_session"
        )
    path = root_path / "artifacts_part10_bot" / "pipeline_status.json"
    write_json_strict(path, payload)
    return path


def write_backfill_status(root: str | Path) -> Path:
    """Record backfill identity only after realized-price regeneration succeeds."""
    root_path = Path(root)
    marker = root_path / "artifacts_part9" / "backfill_run_date.txt"
    if not marker.is_file():
        raise FileNotFoundError("backfill_run_date.txt must be written before status")
    backfill_date = marker.read_text(encoding="utf-8").strip()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backfill_run_date": backfill_date,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_code_sha": os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA"),
        "result": "verified",
    }
    path = root_path / "artifacts_part9" / "backfill_status.json"
    write_json_strict(path, payload)
    return path


def validate_status_markers(root: str | Path) -> list[str]:
    """Require each date marker to agree with its structured status record."""
    root_path = Path(root)
    failures: list[str] = []
    contracts = (
        (
            "artifacts_part10_bot/pipeline_run_date.txt",
            "artifacts_part10_bot/pipeline_status.json",
            "pipeline_run_date",
        ),
        (
            "artifacts_part9/backfill_run_date.txt",
            "artifacts_part9/backfill_status.json",
            "backfill_run_date",
        ),
    )
    for marker_rel, status_rel, date_field in contracts:
        marker_path = root_path / marker_rel
        status_path = root_path / status_rel
        if not marker_path.is_file() or not status_path.is_file():
            continue
        marker = marker_path.read_text(encoding="utf-8").strip()
        try:
            parsed = datetime.strptime(marker, "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError):
            failures.append(f"{marker_rel} is not an ISO date")
            continue
        if parsed != marker:
            failures.append(f"{marker_rel} is not a canonical ISO date")
            continue
        status = read_json_strict(status_path)
        if status.get(date_field) != marker:
            failures.append(f"{status_rel} {date_field} does not match {marker_rel}")
        if status.get("result") != "verified":
            failures.append(f"{status_rel} is not verified")
        for field in ("source_code_sha", "github_run_id", "github_run_attempt"):
            if not status.get(field):
                failures.append(f"{status_rel} lacks {field}")
    return failures


def _canonical_date_text(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed).date().isoformat()


def _identity_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _is_true_flag(value: Any) -> bool:
    """Accept the explicit true encodings produced by CSV round-trips."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _latest_csv_rows(path: Path, date_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or date_column not in frame.columns:
        return pd.DataFrame()
    # pandas 2+ infers one strict format for an entire Series.  Historical
    # ledgers legitimately contain both date-only and midnight-timestamp text,
    # so the default parser can turn valid older rows into NaT.  Explicit mixed
    # parsing keeps those rows visible to lineage validation.
    dates = pd.to_datetime(frame[date_column], errors="coerce", format="mixed")
    raw = frame[date_column]
    invalid = dates.isna() & raw.notna() & raw.astype(str).str.strip().ne("")
    if invalid.any():
        bad_values = raw.loc[invalid].astype(str).unique().tolist()
        raise ValueError(
            f"{path} contains invalid {date_column} values: {bad_values[:5]}"
        )
    if not dates.notna().any():
        return pd.DataFrame()
    return frame.loc[dates == dates.max()].copy()


def ledger_row_counts(root: str | Path) -> dict[str, int]:
    """Read every accumulating CSV and return its data-row count."""
    root_path = Path(root)
    counts: dict[str, int] = {}
    for rel in ACCUMULATING_CSV_FILES:
        path = root_path / rel
        if not path.is_file():
            raise FileNotFoundError(f"accumulating ledger is missing: {rel}")
        try:
            counts[rel] = int(len(pd.read_csv(path)))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError(f"could not count accumulating ledger {rel}: {exc}") from exc
    return counts


def write_ledger_baseline(root: str | Path, baseline_path: str | Path) -> Path:
    """Snapshot pre-run row counts outside the publication tree."""
    target = Path(baseline_path)
    write_json_strict(
        target,
        {
            "protocol_version": PROTOCOL_VERSION,
            "files": ledger_row_counts(root),
        },
    )
    return target


def validate_ledger_preservation(
    root: str | Path,
    baseline_path: str | Path,
) -> list[str]:
    """Reject a run that shrank any accumulating audit ledger."""
    try:
        baseline = read_json_strict(baseline_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"ledger baseline could not be read: {exc}"]
    if baseline.get("protocol_version") != PROTOCOL_VERSION:
        return ["ledger baseline protocol_version does not match the running code"]
    expected = baseline.get("files")
    if not isinstance(expected, dict):
        return ["ledger baseline lacks a files mapping"]
    try:
        current = ledger_row_counts(root)
    except (OSError, ValueError) as exc:
        return [str(exc)]

    failures: list[str] = []
    for rel in ACCUMULATING_CSV_FILES:
        before = expected.get(rel)
        if not isinstance(before, int) or before < 0:
            failures.append(f"ledger baseline lacks a valid row count for {rel}")
            continue
        after = current[rel]
        if after < before:
            failures.append(
                f"accumulating ledger shrank: {rel} rows_before={before} rows_after={after}"
            )
    return failures


def validate_execution_lineage(root: str | Path) -> list[str]:
    """Require Parts 3, 7, 8, and 10 to share one publication identity."""
    root_path = Path(root)
    paths = {
        "pipeline status": root_path / "artifacts_part10_bot" / "pipeline_status.json",
        "Part 3 summary": root_path / "artifacts_part3_v1" / "part3_summary.json",
        "Part 3 allocation": root_path / "artifacts_part3_v1" / "v1_fusion_allocations.csv",
        "Part 7 target": root_path / "artifacts_part7" / "current_target_weights.json",
        "Part 8 instructions": root_path / "artifacts_part8" / "execution_instructions.json",
        "Part 8 metadata": root_path / "artifacts_part8" / "part8_meta.json",
        "Part 8 cost tape": root_path / "artifacts_part8" / "execution_cost_tape.csv",
        "Part 10 signal log": root_path / "artifacts_part10_bot" / "signal_log.csv",
        "Part 10 state": root_path / "artifacts_part10_bot" / "portfolio_state.json",
        "Part 10 performance": root_path / "artifacts_part10_bot" / "performance_report.json",
    }
    missing = [
        f"execution lineage source is missing: {label}"
        for label, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        return missing

    failures: list[str] = []
    try:
        status = read_json_strict(paths["pipeline status"])
        part3 = read_json_strict(paths["Part 3 summary"])
        part7 = read_json_strict(paths["Part 7 target"])
        instructions = read_json_strict(paths["Part 8 instructions"])
        meta = read_json_strict(paths["Part 8 metadata"])
        bot_state = read_json_strict(paths["Part 10 state"])
        bot_performance = read_json_strict(paths["Part 10 performance"])
        allocations = _latest_csv_rows(paths["Part 3 allocation"], "Date")
        costs = _latest_csv_rows(paths["Part 8 cost tape"], "Date")
        bot_signals = _latest_csv_rows(
            paths["Part 10 signal log"], "source_decision_date"
        )
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        return [f"execution lineage source could not be parsed: {exc}"]

    if allocations.empty:
        failures.append("Part 3 allocation has no valid dated row")
    if costs.empty:
        failures.append("Part 8 cost tape has no valid dated row")
    if bot_signals.empty:
        failures.append("Part 10 signal log has no valid dated row")

    expected_date = _canonical_date_text(
        status.get("expected_completed_market_session")
        or status.get("pipeline_run_date")
    )
    if expected_date is None:
        failures.append("pipeline status lacks a valid completed-session date")
    else:
        date_values: dict[str, Any] = {
            "pipeline status pipeline_run_date": status.get("pipeline_run_date"),
            "Part 3 decision_date": part3.get("decision_date"),
            "Part 3 pipeline_run_date": part3.get("pipeline_run_date"),
            "Part 7 Date": part7.get("Date") or part7.get("decision_date"),
            "Part 8 decision_date": instructions.get("decision_date"),
            "Part 8 source_decision_date": instructions.get("source_decision_date"),
            "Part 8 pipeline_run_date": instructions.get("pipeline_run_date"),
            "Part 8 metadata decision_date": meta.get("decision_date"),
            "Part 8 metadata pipeline_run_date": meta.get("pipeline_run_date"),
            "Part 10 state decision_date": bot_state.get("decision_date"),
            "Part 10 state pipeline_run_date": bot_state.get("pipeline_run_date"),
            "Part 10 state price_session": bot_state.get("price_session"),
            "Part 10 performance decision_date": bot_performance.get("decision_date"),
            "Part 10 performance pipeline_run_date": bot_performance.get(
                "pipeline_run_date"
            ),
            "Part 10 performance price_session": bot_performance.get("price_session"),
        }
        if not allocations.empty:
            date_values["Part 3 allocation Date"] = allocations.iloc[0].get("Date")
            date_values["Part 3 allocation source_decision_date"] = allocations.iloc[0].get(
                "source_decision_date"
            )
        if not costs.empty:
            date_values["Part 8 cost-tape Date"] = costs.iloc[0].get("Date")
            date_values["Part 8 cost-tape pipeline_run_date"] = costs.iloc[0].get(
                "pipeline_run_date"
            )
        if not bot_signals.empty:
            for field in (
                "date",
                "source_decision_date",
                "pipeline_run_date",
                "price_session",
            ):
                date_values[f"Part 10 signal {field}"] = bot_signals.iloc[0].get(
                    field
                )
        for label, value in date_values.items():
            actual = _canonical_date_text(value)
            if actual != expected_date:
                failures.append(
                    f"{label} does not match completed session: "
                    f"actual={actual or 'missing'} expected={expected_date}"
                )

    expected_sha = _identity_text(status.get("source_code_sha"))
    sha_values: dict[str, Any] = {
        "Part 3 source_code_sha": part3.get("source_code_sha"),
        "Part 7 model_code_sha": part7.get("model_code_sha"),
        "Part 8 source_code_sha": instructions.get("source_code_sha"),
        "Part 8 metadata source_code_sha": meta.get("source_code_sha"),
        "Part 10 state source_code_sha": bot_state.get("source_code_sha"),
        "Part 10 performance source_code_sha": bot_performance.get("source_code_sha"),
    }
    if not allocations.empty:
        sha_values["Part 3 allocation model_code_sha"] = allocations.iloc[0].get(
            "model_code_sha"
        )
    if not costs.empty:
        sha_values["Part 8 cost-tape source_code_sha"] = costs.iloc[0].get(
            "source_code_sha"
        )
    if not bot_signals.empty:
        sha_values["Part 10 signal model_code_sha"] = bot_signals.iloc[0].get(
            "model_code_sha"
        )
        sha_values["Part 10 signal execution_source_code_sha"] = bot_signals.iloc[
            0
        ].get("execution_source_code_sha")
    for label, value in sha_values.items():
        actual = _identity_text(value)
        if not expected_sha or actual != expected_sha:
            failures.append(
                f"{label} does not match pipeline source SHA: "
                f"actual={actual or 'missing'} expected={expected_sha or 'missing'}"
            )

    expected_run_id = _identity_text(status.get("github_run_id"))
    run_values: dict[str, Any] = {
        "Part 3 pipeline_run_id": part3.get("pipeline_run_id"),
        "Part 8 pipeline_run_id": instructions.get("pipeline_run_id"),
        "Part 8 metadata pipeline_run_id": meta.get("pipeline_run_id"),
        "Part 10 state pipeline_run_id": bot_state.get("pipeline_run_id"),
        "Part 10 performance pipeline_run_id": bot_performance.get("pipeline_run_id"),
    }
    if not allocations.empty:
        run_values["Part 3 allocation pipeline_run_id"] = allocations.iloc[0].get(
            "pipeline_run_id"
        )
    if not costs.empty:
        run_values["Part 8 cost-tape pipeline_run_id"] = costs.iloc[0].get(
            "pipeline_run_id"
        )
    if not bot_signals.empty:
        run_values["Part 10 signal pipeline_run_id"] = bot_signals.iloc[0].get(
            "pipeline_run_id"
        )
    for label, value in run_values.items():
        actual = _identity_text(value)
        if not expected_run_id or actual != expected_run_id:
            failures.append(
                f"{label} does not match pipeline run ID: "
                f"actual={actual or 'missing'} expected={expected_run_id or 'missing'}"
            )

    expected_attempt = _identity_text(status.get("github_run_attempt"))
    attempt_values: dict[str, Any] = {
        "Part 3 pipeline_run_attempt": part3.get("pipeline_run_attempt"),
        "Part 8 pipeline_run_attempt": instructions.get("pipeline_run_attempt"),
        "Part 8 metadata pipeline_run_attempt": meta.get("pipeline_run_attempt"),
        "Part 10 state pipeline_run_attempt": bot_state.get("pipeline_run_attempt"),
        "Part 10 performance pipeline_run_attempt": bot_performance.get(
            "pipeline_run_attempt"
        ),
    }
    if not allocations.empty:
        attempt_values["Part 3 allocation pipeline_run_attempt"] = allocations.iloc[0].get(
            "pipeline_run_attempt"
        )
    if not costs.empty:
        attempt_values["Part 8 cost-tape pipeline_run_attempt"] = costs.iloc[0].get(
            "pipeline_run_attempt"
        )
    if not bot_signals.empty:
        attempt_values["Part 10 signal pipeline_run_attempt"] = bot_signals.iloc[
            0
        ].get("pipeline_run_attempt")
    for label, value in attempt_values.items():
        actual = _identity_text(value)
        if not expected_attempt or actual != expected_attempt:
            failures.append(
                f"{label} does not match pipeline run attempt: "
                f"actual={actual or 'missing'} expected={expected_attempt or 'missing'}"
            )

    protocol_values: dict[str, Any] = {
        "pipeline status": status.get("protocol_version"),
        "Part 3 summary": part3.get("protocol_version"),
        "Part 7 target": part7.get("model_protocol_version"),
        "Part 8 instructions": instructions.get("protocol_version"),
        "Part 8 metadata": meta.get("protocol_version"),
        "Part 10 state": bot_state.get("protocol_version"),
        "Part 10 performance": bot_performance.get("protocol_version"),
    }
    if not allocations.empty:
        protocol_values["Part 3 allocation"] = allocations.iloc[0].get(
            "model_protocol_version"
        )
    if not costs.empty:
        protocol_values["Part 8 cost tape"] = costs.iloc[0].get("protocol_version")
    if not bot_signals.empty:
        protocol_values["Part 10 signal"] = bot_signals.iloc[0].get(
            "model_protocol_version"
        )
    for label, value in protocol_values.items():
        if _identity_text(value) != PROTOCOL_VERSION:
            failures.append(
                f"{label} protocol does not match {PROTOCOL_VERSION}: {value!r}"
            )

    if instructions.get("lineage_verified") is not True:
        failures.append("Part 8 instructions are not marked lineage_verified")
    if meta.get("lineage_verified") is not True:
        failures.append("Part 8 metadata are not marked lineage_verified")
    if bot_state.get("execution_lineage_verified") is not True:
        failures.append("Part 10 state is not marked execution_lineage_verified")
    if bot_performance.get("execution_lineage_verified") is not True:
        failures.append("Part 10 performance is not marked execution_lineage_verified")
    if not bot_signals.empty and not _is_true_flag(
        bot_signals.iloc[0].get("execution_lineage_verified")
    ):
        failures.append("Part 10 signal is not marked execution_lineage_verified")
    if instructions.get("allocation_source") != "v1_fusion_allocations":
        failures.append("Part 8 instructions do not identify the Part 3 fusion allocation")
    if meta.get("allocation_source") != "v1_fusion_allocations":
        failures.append("Part 8 metadata do not identify the Part 3 fusion allocation")
    if meta.get("latest_order_instructions") != instructions:
        failures.append("Part 8 metadata and execution_instructions.json disagree")
    for label, source in (
        ("Part 10 state", bot_state.get("price_source")),
        ("Part 10 performance", bot_performance.get("price_source")),
    ):
        if source != "artifacts_part0/close_prices.parquet":
            failures.append(f"{label} does not identify the verified Part 0 price source")
    if not bot_signals.empty and bot_signals.iloc[0].get(
        "price_source"
    ) != "artifacts_part0/close_prices.parquet":
        failures.append("Part 10 signal does not identify the verified Part 0 price source")

    # Multi-row allocation records must carry one identity across every sleeve.
    # Checking only the first VOO/IEF row would allow a partially overwritten CSV
    # to pass even though the portfolio as a whole had mixed provenance.
    for label, frame, fields in (
        (
            "Part 3 allocation",
            allocations,
            {
                "source_decision_date": _canonical_date_text,
                "model_protocol_version": _identity_text,
                "model_code_sha": _identity_text,
                "pipeline_run_id": _identity_text,
                "pipeline_run_attempt": _identity_text,
            },
        ),
        (
            "Part 8 cost tape latest session",
            costs,
            {
                "pipeline_run_date": _canonical_date_text,
                "protocol_version": _identity_text,
                "source_code_sha": _identity_text,
                "pipeline_run_id": _identity_text,
                "pipeline_run_attempt": _identity_text,
            },
        ),
    ):
        if frame.empty:
            continue
        for field, normalize in fields.items():
            if field not in frame.columns:
                continue
            values = {
                normalize(value) or "<missing>" for value in frame[field].tolist()
            }
            if len(values) != 1:
                failures.append(
                    f"{label} has mixed {field} values: {sorted(values)}"
                )
    return failures


def validate_completed_session_inputs(root: str | Path) -> list[str]:
    """Verify the retained market panel contains true completed XNYS rows."""
    from market_calendar import completed_xnys_sessions, latest_completed_xnys_session

    root_path = Path(root)
    close_path = root_path / "artifacts_part0" / "close_prices.parquet"
    mask_path = root_path / "artifacts_part0" / "market_observation_mask.parquet"
    meta_path = root_path / "artifacts_part0" / "part0_meta.json"
    failures: list[str] = []
    for path in (close_path, mask_path, meta_path):
        if not path.is_file():
            failures.append(f"required run input is missing: {path.relative_to(root_path)}")
    if failures:
        return failures

    close = pd.read_parquet(close_path)
    observed = pd.read_parquet(mask_path)
    close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None).normalize()
    observed.index = pd.to_datetime(observed.index, errors="coerce").tz_localize(None).normalize()
    if close.empty or close.index.isna().any():
        return ["close_prices.parquet has no valid completed-session index"]
    expected = latest_completed_xnys_session()
    sessions = completed_xnys_sessions(close.index.min(), expected)
    invalid_rows = close.index.difference(sessions)
    if len(invalid_rows):
        failures.append(
            "close_prices.parquet contains non-XNYS or uncompleted rows: "
            + ", ".join(str(value.date()) for value in invalid_rows[-5:])
        )
    if close.index.max() != expected:
        failures.append(
            "market panel does not end on the latest completed XNYS session: "
            f"actual={close.index.max().date()} expected={expected.date()}"
        )
    if not observed.index.equals(close.index) or list(observed.columns) != list(close.columns):
        failures.append("market_observation_mask.parquet is not aligned to close_prices.parquet")
    else:
        mask_values = observed.fillna(0).astype(bool)
        if not mask_values.equals(close.notna()):
            failures.append("market observation mask differs from true close availability")
    meta = read_json_strict(meta_path)
    if meta.get("market_calendar") != "XNYS":
        failures.append("Part 0 metadata does not declare the XNYS calendar")
    if meta.get("market_values_are_raw_observations") is not True:
        failures.append("Part 0 metadata does not guarantee raw market observations")
    if meta.get("market_data_asof") != expected.date().isoformat():
        failures.append("Part 0 market_data_asof differs from the completed XNYS session")
    return failures


def build_run_manifest(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    files: dict[str, Any] = {}
    for rel in REQUIRED_PUBLISHED_FILES:
        path = root_path / rel
        files[rel] = (
            manifest_file_record(path)
            if path.is_file() else {"missing": True}
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha": os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "files": files,
    }


def verify_run_manifest(root: str | Path) -> list[str]:
    root_path = Path(root)
    manifest_path = root_path / "artifacts_manifest.json"
    if not manifest_path.is_file():
        return ["artifacts_manifest.json is missing"]
    try:
        manifest = read_json_strict(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"artifacts_manifest.json: {exc}"]
    failures: list[str] = []
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("manifest protocol_version does not match the running code")
    entries = manifest.get("files", {})
    for rel in REQUIRED_PUBLISHED_FILES:
        path = root_path / rel
        entry = entries.get(rel)
        if not isinstance(entry, dict) or entry.get("missing"):
            failures.append(f"manifest lacks a complete entry for {rel}")
            continue
        if not path.is_file():
            failures.append(f"published file is missing after manifest generation: {rel}")
            continue
        actual = manifest_file_record(path)
        if entry.get("bytes") != actual["bytes"]:
            failures.append(f"manifest byte count differs for {rel}")
        if entry.get("sha256") != actual["sha256"]:
            failures.append(f"manifest SHA-256 differs for {rel}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("PRICECALL_ROOT", "."))
    parser.add_argument("--write-pipeline-status", action="store_true")
    parser.add_argument("--write-backfill-status", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--verify-run-inputs", action="store_true")
    ledger_group = parser.add_mutually_exclusive_group()
    ledger_group.add_argument("--write-ledger-baseline")
    ledger_group.add_argument("--verify-ledger-baseline")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.write_ledger_baseline:
        write_ledger_baseline(root, args.write_ledger_baseline)
        return 0
    if args.verify_ledger_baseline:
        ledger_failures = validate_ledger_preservation(
            root, args.verify_ledger_baseline
        )
        if ledger_failures:
            raise SystemExit(
                "Accumulating-ledger preservation failed:\n"
                + "\n".join(ledger_failures)
            )
        return 0
    if args.write_pipeline_status:
        write_pipeline_status(root)
    if args.write_backfill_status:
        write_backfill_status(root)
    if args.status_only:
        if not (args.write_pipeline_status or args.write_backfill_status):
            raise SystemExit("--status-only requires a status-writing flag")
        return 0
    if args.verify_run_inputs:
        input_failures = validate_completed_session_inputs(root)
        if input_failures:
            raise SystemExit(
                "Completed-session input validation failed:\n" + "\n".join(input_failures)
            )
    missing = validate_required_files(root)
    if missing:
        raise SystemExit("Required publication artifacts are missing:\n" + "\n".join(missing))
    failures = validate_json_files(root)
    if failures:
        raise SystemExit("Strict JSON validation failed:\n" + "\n".join(failures))
    status_failures = validate_status_markers(root)
    if status_failures:
        raise SystemExit("Status-marker validation failed:\n" + "\n".join(status_failures))
    lineage_failures = validate_execution_lineage(root)
    if lineage_failures:
        raise SystemExit(
            "Execution-lineage validation failed:\n" + "\n".join(lineage_failures)
        )
    write_json_strict(root / "artifacts_manifest.json", build_run_manifest(root))
    manifest_failures = verify_run_manifest(root)
    if manifest_failures:
        raise SystemExit("Artifact manifest verification failed:\n" + "\n".join(manifest_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
