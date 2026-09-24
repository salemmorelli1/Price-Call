#!/usr/bin/env python3
"""Read-only, prespecified score for the first 60 eligible v4 paper outcomes.

This report never changes the allocation gate. Inspect it in a separate review
after the first 60 outcomes; subsequent daily monitoring must not repeatedly
retest the same cohort as if it were a fresh confirmatory holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_integrity import PROTOCOL_VERSION, current_evidence_mask, json_safe
from market_calendar import _calendar, next_xnys_session
from part2_predictor import Part2Gen53Config, _delong_auc_ztest, _historical_evidence_gate


FIRST_DECISION_DATE = pd.Timestamp("2026-09-24")
MIN_OUTCOMES = 60
TARGET_DEFINITION = "rowwise_trailing_63_observation_20th_percentile_shifted_1"


def evaluate(frame: pd.DataFrame) -> dict[str, object]:
    """Select prospectively issued v4 rows, with no revised backtest rows."""
    result: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "frozen first 60 eligible realized paper forecasts",
        "probability_source": "p_final_cal",
        "auc_p_max": Part2Gen53Config.HISTORICAL_AUC_P_MAX,
        "brier_skill_min": Part2Gen53Config.HISTORICAL_BRIER_SKILL_MIN,
        "required_outcomes": MIN_OUTCOMES,
        "independent_validation_ok": False,
        "allocation_approved": False,
    }
    candidates = frame[current_evidence_mask(frame)].copy()
    result["eligible_issued"] = int(len(candidates))
    if candidates.empty:
        return {**result, "eligible_realized": 0, "status": "awaiting_prospective_forecasts"}

    required = {
        "decision_date", "target_date", "realized_target_date",
        "forecast_issued_at_utc", "evidence_prospective", "provenance_complete",
        "data_freshness_ok", "target_definition_id", "model_code_sha",
        "pipeline_run_id", "pipeline_run_attempt", "p_final_cal", "base_rate",
        "tail_threshold", "px_voo_t", "px_ief_t", "px_voo_realized",
        "px_ief_realized",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Eligible ledger lacks required columns: {sorted(missing)}")
    decisions = pd.to_datetime(candidates["decision_date"], errors="raise", format="mixed").dt.normalize()
    targets = pd.to_datetime(candidates["target_date"], errors="raise", format="mixed").dt.normalize()
    issuances = pd.to_datetime(candidates["forecast_issued_at_utc"], errors="raise", utc=True)
    if decisions.lt(FIRST_DECISION_DATE).any() or decisions.duplicated().any():
        raise ValueError("Eligible cohort includes an older or duplicate decision date")
    calendar = _calendar()
    for index, decision, target, issued in zip(candidates.index, decisions, targets, issuances):
        if (
            pd.isna(issued)
            or target != next_xnys_session(decision)
            or issued >= calendar.session_close(target)
            or any(
                pd.to_numeric(
                    pd.Series([candidates.at[index, key]]), errors="coerce"
                ).iloc[0] != 1
                for key in ("evidence_prospective", "provenance_complete", "data_freshness_ok")
            )
            or candidates.at[index, "target_definition_id"] != TARGET_DEFINITION
            or len(str(candidates.at[index, "model_code_sha"])) != 40
            or not str(candidates.at[index, "pipeline_run_id"]).strip()
            or not str(candidates.at[index, "pipeline_run_attempt"]).strip()
        ):
            raise ValueError(f"Eligible forecast fails issuance or lineage checks: {decision.date()}")

    prices = candidates[["px_voo_realized", "px_ief_realized"]].apply(
        pd.to_numeric, errors="coerce"
    )
    partially_realized = prices.notna().any(axis=1) & ~prices.notna().all(axis=1)
    if partially_realized.any():
        raise ValueError("Eligible forecast has a partially realized VOO/IEF price pair")
    realized = candidates[prices.notna().all(axis=1)].copy()
    result["eligible_realized"] = int(len(realized))
    if realized.empty:
        return {**result, "status": "awaiting_first_outcome"}
    actual_targets = pd.to_datetime(
        realized["realized_target_date"], errors="raise", format="mixed"
    ).dt.normalize()
    if not actual_targets.eq(targets.loc[realized.index]).all():
        raise ValueError("Realized close is assigned to a different target session")
    realized = realized.assign(_decision=decisions.loc[realized.index]).sort_values("_decision")
    for key in ("px_voo_t", "px_ief_t", "px_voo_realized", "px_ief_realized",
                "p_final_cal", "base_rate", "tail_threshold"):
        realized[key] = pd.to_numeric(realized[key], errors="coerce")
        if not np.isfinite(realized[key].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {key} in the eligible holdout")
    for key in ("px_voo_t", "px_ief_t", "px_voo_realized", "px_ief_realized"):
        if (realized[key] <= 0).any():
            raise ValueError(f"Nonpositive {key} in the eligible holdout")
    for key in ("p_final_cal", "base_rate"):
        if (~realized[key].between(0, 1)).any():
            raise ValueError(f"Invalid {key} probability in the eligible holdout")
    if len(realized) < MIN_OUTCOMES:
        return {**result, "status": "awaiting_60_outcomes"}

    locked = realized.iloc[:MIN_OUTCOMES]
    log_spread = np.log(locked["px_voo_realized"] / locked["px_voo_t"]) - np.log(
        locked["px_ief_realized"] / locked["px_ief_t"]
    )
    outcomes = (log_spread < locked["tail_threshold"]).astype(int).to_numpy()
    predicted = locked["p_final_cal"].to_numpy(dtype=float)
    baseline = locked["base_rate"].to_numpy(dtype=float)
    auc_test = _delong_auc_ztest(outcomes, predicted)
    base_brier = float(np.mean((outcomes - baseline) ** 2))
    model_brier = float(np.mean((outcomes - predicted) ** 2))
    skill = 1.0 - model_brier / base_brier if base_brier > 0 else float("nan")
    passed = _historical_evidence_gate(
        auc_test["auc"], auc_test["p_one_sided"], skill, Part2Gen53Config()
    )
    return {
        **result,
        "locked_first_decision": locked["_decision"].iloc[0].date().isoformat(),
        "locked_last_decision": locked["_decision"].iloc[-1].date().isoformat(),
        "locked_n": MIN_OUTCOMES,
        "n_events": int(sum(outcomes)),
        "n_nonevents": int(MIN_OUTCOMES - sum(outcomes)),
        "auc": auc_test["auc"],
        "auc_p_one_sided": auc_test["p_one_sided"],
        "brier_model": model_brier,
        "brier_causal_baseline": base_brier,
        "brier_skill_causal": skill,
        "prospective_thresholds_pass": passed,
        "status": "thresholds_pass_pending_separate_review" if passed else "thresholds_not_met",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    source = args.root / "artifacts_part3/prediction_log.csv"
    report = evaluate(pd.read_csv(source))
    print(json.dumps(json_safe(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
