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
    "artifacts_part3_v1/part3_summary.json",
    "artifacts_part6/part6_meta.json",
    "artifacts_part7/current_target_weights.json",
    "artifacts_part8/part8_meta.json",
    "artifacts_part8/execution_instructions.json",
    "artifacts_part9/live_attribution_report.json",
    "artifacts_part10_bot/portfolio_state.json",
    "artifacts_part10_bot/pipeline_status.json",
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
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_at_utc": now.isoformat(),
        "pipeline_run_date": os.environ.get("PRICECALL_RUN_DATE_ET", et_date),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_code_sha": os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA"),
        "result": "verified",
    }
    path = root_path / "artifacts_part10_bot" / "pipeline_status.json"
    write_json_strict(path, payload)
    return path


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
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.write_pipeline_status:
        write_pipeline_status(root)
    missing = validate_required_files(root)
    if missing:
        raise SystemExit("Required publication artifacts are missing:\n" + "\n".join(missing))
    failures = validate_json_files(root)
    if failures:
        raise SystemExit("Strict JSON validation failed:\n" + "\n".join(failures))
    write_json_strict(root / "artifacts_manifest.json", build_run_manifest(root))
    manifest_failures = verify_run_manifest(root)
    if manifest_failures:
        raise SystemExit("Artifact manifest verification failed:\n" + "\n".join(manifest_failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
