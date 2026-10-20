#!/usr/bin/env python3
"""Explain publication blocks without retraining, rewriting ledgers, or approving allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from artifact_integrity import json_safe, read_json_strict
from prepare_pages import _source_failures
from statistical_tests import delong_auc_statistics
from sync_dashboard import build_snapshot


def audit(root: Path, *, as_of: str | None = None) -> dict[str, object]:
    failures = _source_failures(root)
    if failures:
        raise ValueError("Publication integrity failed: " + "; ".join(failures))
    if as_of is not None and pd.Timestamp(as_of).tzinfo is None:
        raise ValueError("Audit --as-of must include an explicit UTC offset")
    snapshot = build_snapshot(root, now=as_of)
    summary = read_json_strict(root / "artifacts_part2_g532/predictions/part2_g532_summary.json")
    prospective = snapshot["prospective_validation"]
    tape = pd.read_csv(root / "artifacts_part2_g532/predictions/g532_final_consensus_tape.csv")
    realized = tape.loc[pd.to_numeric(tape["y_avail"], errors="coerce").eq(1)]
    reference_auc = delong_auc_statistics(
        realized["y_rel_tail_voo_vs_ief"].to_numpy(dtype=float),
        realized["p_final_cal"].to_numpy(dtype=float),
    )
    return json_safe({
        "integrity_status": "PASS",
        "audit_as_of_utc": (pd.Timestamp(as_of).tz_convert("UTC").isoformat()
                            if as_of is not None else pd.Timestamp.now(tz="UTC").isoformat()),
        "governance_status": "VALIDATED" if snapshot["final_pass"] else "BLOCKED",
        "protocol_version": snapshot["protocol_version"],
        "decision_date": snapshot["decision_date"],
        "publish_mode": snapshot["publish_mode"],
        "alpha_state": snapshot["alpha_state"],
        "paper_weights": snapshot["weights"],
        "operator_validation": snapshot["operator_validation"],
        "historical_evidence": {
            "auc": snapshot["metrics"]["backtest_auc"],
            "stored_auc_p_one_sided": snapshot["metrics"]["backtest_auc_p_value"],
            "brier_skill_causal": snapshot["metrics"]["backtest_brier_skill"],
            "auc_p_max": snapshot["operator_validation"]["auc_p_value_max"],
            "brier_skill_min": snapshot["operator_validation"]["brier_skill_min"],
            "reference_auc_uncertainty": reference_auc,
            "forecast_generator_changed": False,
            "reference_use": "read-only review; does not replace frozen forecasting or allocation logic",
            "independent_review_required": True,
        },
        "prospective_validation": prospective,
        "publication": {
            "fresh": snapshot["publication_freshness_ok"],
            "expected_session": snapshot["lineage"]["expected_completed_market_session"],
            "component_dates": snapshot["lineage"]["publication_component_dates"],
            "session_age": snapshot["lineage"]["publication_session_age"],
            "expires_at_utc": snapshot["freshness_expires_at_utc"],
            "execution_lineage_ok": snapshot["execution_lineage_ok"],
        },
        "independent_validation_status": summary.get("independent_validation_status"),
        "allocation_approved": False,
        "source_mutations": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--as-of", help="Replay publication freshness at an explicit ISO timestamp")
    args = parser.parse_args()
    # A statistically blocked model is a valid audit result, not a process crash.
    # Missing or corrupt provenance raises and produces a nonzero process exit.
    print(json.dumps(audit(args.root.resolve(), as_of=args.as_of), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
