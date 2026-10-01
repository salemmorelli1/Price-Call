"""Input availability is separate from statistical approval and completion."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from artifact_integrity import (
    PROTOCOL_VERSION,
    current_evidence_mask,
    read_json_strict,
    verify_run_manifest,
    write_json_strict,
)


DATA_PENDING_EXIT_CODE = 75


class DataPendingError(RuntimeError):
    """A required provider input is temporarily unavailable after retries."""

    def __init__(self, message: str, *, stage: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


def record_data_pending(error: DataPendingError, root: Path) -> int:
    """Write attempt diagnostics only; leave published ledgers/markers alone."""
    path = Path(os.environ.get(
        "PRICECALL_INPUT_STATUS_PATH",
        str(root / "artifacts_diagnostics/production_input_status.json"),
    ))
    write_json_strict(path, {
        "status": "DATA_PENDING",
        "stage": error.stage,
        "reason": str(error),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_sha": os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "diagnostics": error.diagnostics,
    })
    print(f"[{error.stage}] DATA_PENDING: {error}")
    print("No forecast or production completion marker may be published for this attempt.")
    return DATA_PENDING_EXIT_CODE


def retry_incomplete_macro_session(
    root: Path, session_date: str, *, now: object | None = None,
) -> bool:
    """Recover an old incomplete-macro run only while its target is still open.

    Statistical rejection never requests a rerun. Eligible issued forecasts
    remain immutable. Recovery must issue a new forecast before its next-session
    target closes; it cannot relabel a past diagnostic as prospective evidence.
    """
    import pandas as pd
    from market_calendar import _calendar, next_xnys_session

    meta = read_json_strict(root / "artifacts_part0/part0_meta.json")
    if (meta.get("market_data_asof") != session_date
            or meta.get("historical_point_in_time_complete") is not False):
        return False
    failures = verify_run_manifest(root)
    if failures:
        raise RuntimeError("Cannot recover an unverified publication: " + "; ".join(failures))

    frame = pd.read_csv(root / "artifacts_part3/prediction_log.csv")
    required = {"decision_date", "target_date", "model_protocol_version", "evidence_eligible"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Published prediction ledger lacks recovery provenance")
    rows = frame.loc[
        pd.to_datetime(frame["decision_date"], errors="raise", format="mixed").dt.strftime(
            "%Y-%m-%d"
        ).eq(session_date) & frame["model_protocol_version"].eq(PROTOCOL_VERSION)
    ]
    if len(rows) != 1:
        raise RuntimeError("Recovery requires exactly one current-protocol decision row")
    if current_evidence_mask(rows).any():
        return False
    target = next_xnys_session(session_date)
    if pd.Timestamp(rows.iloc[0]["target_date"]).normalize() != target:
        raise RuntimeError("Published recovery target differs from the frozen H=1 definition")
    issued = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if issued.tzinfo is None:
        raise ValueError("Recovery time must be timezone-aware")
    return bool(issued.tz_convert("UTC") < _calendar().session_close(target))
