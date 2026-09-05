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
            {"sha256": sha256_file(path), "bytes": path.stat().st_size}
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
        if entry.get("bytes") != path.stat().st_size:
            failures.append(f"manifest byte count differs for {rel}")
        if entry.get("sha256") != sha256_file(path):
            failures.append(f"manifest SHA-256 differs for {rel}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("PRICECALL_ROOT", "."))
    parser.add_argument("--write-pipeline-status", action="store_true")
    parser.add_argument("--write-backfill-status", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--verify-run-inputs", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
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
    write_json_strict(root / "artifacts_manifest.json", build_run_manifest(root))
    manifest_failures = verify_run_manifest(root)
    if manifest_failures:
        raise SystemExit("Artifact manifest verification failed:\n" + "\n".join(manifest_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
