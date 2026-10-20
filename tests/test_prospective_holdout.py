"""Prospective scoring must stay separate from historical model selection."""
import numpy as np
import pandas as pd
import pytest

from artifact_integrity import PROTOCOL_VERSION
from evaluate_prospective_cohort import TARGET_DEFINITION, evaluate
from market_calendar import _calendar, next_xnys_session


def _paper_rows(n=61):
    calendar = _calendar()
    dates = calendar.sessions_in_range("2026-09-24", "2027-02-01")[:n]
    rows = []
    for i, decision in enumerate(dates):
        target = next_xnys_session(decision)
        event = i % 5 == 0
        p = float(np.clip(0.18 + 0.36 * event + (i % 7 - 3) * 0.05, 0.01, 0.99))
        if i == 1:
            p = 0.70  # Preserve a genuine AUC variance estimate.
        rows.append({
            "decision_date": decision.date().isoformat(),
            "target_date": target.date().isoformat(),
            "realized_target_date": target.date().isoformat(),
            "forecast_issued_at_utc": (calendar.session_close(decision) +
                                       pd.Timedelta(minutes=25)).isoformat(),
            "model_protocol_version": PROTOCOL_VERSION,
            "evidence_eligible": 1,
            "evidence_prospective": 1,
            "provenance_complete": 1,
            "data_freshness_ok": 1,
            "target_definition_id": TARGET_DEFINITION,
            "model_code_sha": "a" * 40,
            "pipeline_run_id": str(10000 + i),
            "pipeline_run_attempt": "1",
            "p_final_cal": p,
            "base_rate": 0.2,
            "tail_threshold": -0.005,
            "px_voo_t": 100.0,
            "px_ief_t": 100.0,
            "px_voo_outcome_anchor": 100.0,
            "px_ief_outcome_anchor": 100.0,
            "px_voo_realized": 98.0 if event else 102.0,
            "px_ief_realized": 100.0,
        })
    return pd.DataFrame(rows)


def test_first_60_only_score_and_exclude_legacy_and_future_observations():
    rows = _paper_rows()
    legacy = rows.iloc[[0]].copy()
    legacy["model_protocol_version"] = "causal-integrity-v3"
    assert evaluate(pd.concat([legacy, rows.iloc[:59]], ignore_index=True))[
        "status"
    ] == "awaiting_60_outcomes"
    first = evaluate(pd.concat([legacy, rows.iloc[:60]], ignore_index=True))
    assert first["eligible_realized"] == 60
    assert first["locked_n"] == 60
    assert first["prospective_thresholds_pass"] is True
    assert first["allocation_approved"] is False
    assert first["independent_validation_ok"] is False
    more = evaluate(pd.concat([legacy, rows], ignore_index=True))
    assert more["eligible_realized"] == 61
    for key in ("auc", "auc_p_one_sided", "brier_skill_causal", "locked_last_decision"):
        assert first[key] == more[key]
    assert first["cohort_issuance_sha256"] == more["cohort_issuance_sha256"]


def test_csv_roundtrip_with_legacy_nulls_keeps_current_flags(tmp_path):
    rows = _paper_rows(2)
    legacy = rows.iloc[[0]].copy()
    legacy["model_protocol_version"] = "causal-integrity-v3"
    legacy["evidence_prospective"] = np.nan
    path = tmp_path / "prediction_log.csv"
    pd.concat([legacy, rows], ignore_index=True).to_csv(path, index=False)
    report = evaluate(pd.read_csv(path))
    assert report["eligible_realized"] == 2
    assert report["status"] == "awaiting_60_outcomes"


def test_locked_cohort_uses_same_vintage_anchor_and_rejects_missing_lineage():
    rows = _paper_rows(60)
    original = evaluate(rows)
    # The issuance anchor remains fixed, while the realized adjusted return
    # uses the contemporaneous adjusted anchor frozen at realization.
    rows.loc[0, "px_voo_outcome_anchor"] = 95.0
    changed = evaluate(rows)
    assert changed["n_events"] == original["n_events"] - 1
    rows.loc[0, "px_ief_outcome_anchor"] = np.nan
    with pytest.raises(ValueError, match="invalid px_ief_outcome_anchor"):
        evaluate(rows)


def test_unresolved_early_forecast_cannot_be_replaced_by_a_later_outcome():
    rows = _paper_rows(61)
    rows.loc[0, ["px_voo_realized", "px_ief_realized"]] = np.nan
    rows.loc[0, "realized_target_date"] = None
    report = evaluate(rows.sample(frac=1, random_state=7))
    assert report["status"] == "awaiting_locked_outcomes"
    assert report["locked_issued"] == 60
    assert report["locked_realized"] == 59
    assert report["locked_first_decision"] == rows.loc[0, "decision_date"]
    assert "prospective_thresholds_pass" not in report


def test_fractional_eligibility_flag_does_not_admit_a_forecast():
    rows = _paper_rows(1)
    rows["evidence_eligible"] = 1.9
    assert evaluate(rows)["eligible_issued"] == 0


@pytest.mark.parametrize("column,value", [
    ("model_code_sha", "z" * 40), ("pipeline_run_id", "nan"),
    ("pipeline_run_attempt", "0"), ("p_final_cal", float("inf")),
    ("base_rate", 1.01), ("tail_threshold", float("nan")),
    ("px_voo_t", -1.0),
])
def test_pending_issuances_still_require_valid_inputs_and_lineage(column, value):
    rows = _paper_rows(1)
    rows.loc[0, ["px_voo_realized", "px_ief_realized"]] = np.nan
    rows.loc[0, "realized_target_date"] = None
    rows.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate(rows)


def test_filling_locked_outcome_preserves_cohort_identity():
    rows = _paper_rows(61)
    missing = rows.copy()
    missing.loc[0, ["px_voo_realized", "px_ief_realized"]] = np.nan
    missing.loc[0, "realized_target_date"] = None
    pending = evaluate(missing)
    final = evaluate(rows)
    assert pending["cohort_issuance_sha256"] == final["cohort_issuance_sha256"]
    assert final["locked_realized"] == 60
    assert final["allocation_approved"] is False


@pytest.mark.parametrize("change", ["late", "early", "wrong_target", "wrong_realization", "duplicate"])
def test_prospective_holdout_rejects_broken_evidence(change):
    rows = _paper_rows(1)
    if change == "late":
        rows.loc[0, "forecast_issued_at_utc"] = _calendar().session_close(
            pd.Timestamp(rows.loc[0, "target_date"])
        ).isoformat()
    elif change == "early":
        rows.loc[0, "forecast_issued_at_utc"] = _calendar().session_close(
            pd.Timestamp(rows.loc[0, "decision_date"])
        ).isoformat()
    elif change == "wrong_target":
        rows.loc[0, "target_date"] = rows.loc[0, "decision_date"]
    elif change == "wrong_realization":
        rows.loc[0, "realized_target_date"] = rows.loc[0, "decision_date"]
    else:
        rows = pd.concat([rows, rows], ignore_index=True)
    with pytest.raises(ValueError):
        evaluate(rows)
