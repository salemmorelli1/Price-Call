"""Independent dashboard approvals must obey the frozen protocol, not metadata."""
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from artifact_integrity import REQUIRED_PUBLISHED_FILES
import sync_dashboard

ROOT = Path(__file__).resolve().parents[1]
PART2 = "artifacts_part2_g532/predictions/part2_g532_summary.json"
PART3 = "artifacts_part3_v1/part3_summary.json"
PART9 = "artifacts_part9/live_attribution_report.json"


def update(root, relative, **values):
    path = root / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(values)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def passing_sources(tmp_path, monkeypatch):
    for relative in REQUIRED_PUBLISHED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    update(tmp_path, PART2, final_pass=True, publish_mode="NORMAL",
           historical_evidence_ok=True, independent_validation_ok=True,
           macro_point_in_time_ok=True, part1_data_freshness_ok=True,
           historical_auc_p_max=0.10, historical_brier_skill_min=0.005,
           historical_brier_skill_causal=0.02, delong_overall_auc={"auc": 0.60, "p_one_sided": 0.04})
    update(tmp_path, PART3, final_pass=True, publish_mode="NORMAL", deployment_mode="NORMAL")
    update(tmp_path, PART9, classification_stats_backtest={"auc": 0.60})
    expected = json.loads((tmp_path / "artifacts_part10_bot/pipeline_status.json").read_text())["pipeline_run_date"]
    # Isolate the approval contract; existing tests separately cover real
    # calendar/lineage checks and scoring of actual prospective issuances.
    monkeypatch.setattr(sync_dashboard, "latest_completed_xnys_session", lambda *args: pd.Timestamp(expected))
    monkeypatch.setattr(sync_dashboard, "validate_execution_lineage", lambda root: [])
    monkeypatch.setattr(sync_dashboard, "evaluate", lambda frame: {
        "eligible_issued": 60, "eligible_realized": 60, "locked_realized": 60,
        "prospective_thresholds_pass": True, "status": "thresholds_pass_pending_separate_review",
    })
    return tmp_path


def test_consistent_passing_evidence_can_still_display_validation(passing_sources):
    result = sync_dashboard.build_snapshot(passing_sources)
    assert result["final_pass"] is True
    assert all(item["passed"] for item in result["gate_checks"])


def test_saved_thresholds_cannot_relax_the_prespecified_protocol(passing_sources):
    update(passing_sources, PART2, historical_auc_p_max=0.50,
           historical_brier_skill_min=-0.01, historical_brier_skill_causal=0.002,
           delong_overall_auc={"auc": 0.60, "p_one_sided": 0.15})
    result = sync_dashboard.build_snapshot(passing_sources)
    assert result["final_pass"] is False
    assert result["operator_validation"]["auc_p_value_max"] == 0.10
    assert result["operator_validation"]["brier_skill_min"] == 0.005


@pytest.mark.parametrize("relative,values", [
    (PART2, {"final_pass": False}),
    (PART2, {"publish_mode": "FAIL_CLOSED_NEUTRAL"}),
    (PART3, {"publish_mode": "FAIL_CLOSED_NEUTRAL"}),
    (PART3, {"publish_mode": "UNKNOWN"}),
])
def test_a_closed_core_or_governance_mode_cannot_display_an_open_gate(passing_sources, relative, values):
    update(passing_sources, relative, **values)
    result = sync_dashboard.build_snapshot(passing_sources)
    assert result["final_pass"] is False
    assert result["operator_validation"]["status"] == "NOT_VALIDATED"


@pytest.mark.parametrize("auc,p,skill", [
    (True, 0.04, 0.02), (1.2, 0.04, 0.02), (0.60, False, 0.02),
    (0.60, -0.01, 0.02), (0.60, 0.04, True), (0.60, 0.04, 1.2),
])
def test_invalid_numeric_evidence_cannot_pass(passing_sources, auc, p, skill):
    update(passing_sources, PART9, classification_stats_backtest={"auc": auc})
    update(passing_sources, PART2, delong_overall_auc={"auc": auc, "p_one_sided": p},
           historical_brier_skill_causal=skill)
    assert sync_dashboard.build_snapshot(passing_sources)["final_pass"] is False


def test_diagnostics_identify_each_failed_control(passing_sources):
    update(passing_sources, PART2, independent_validation_ok=False,
           historical_brier_skill_causal=0.002, delong_overall_auc={"auc": 0.60, "p_one_sided": 0.15})
    result = sync_dashboard.build_snapshot(passing_sources)
    checks = {item["id"]: item for item in result["gate_checks"]}
    assert checks["historical_auc_p"]["passed"] is False
    assert checks["historical_brier_skill"]["passed"] is False
    assert checks["independent_review"]["passed"] is False
    assert checks["prospective_holdout"]["passed"] is True
    assert result["final_pass"] is False


def test_threshold_metadata_mismatch_is_explicit_and_blocks(passing_sources):
    update(passing_sources, PART2, historical_auc_p_max=0.20)
    result = sync_dashboard.build_snapshot(passing_sources)
    checks = {item["id"]: item for item in result["gate_checks"]}
    assert checks["protocol_thresholds"]["passed"] is False
    assert "threshold" in " ".join(result["operator_validation"]["reasons"])
    assert result["final_pass"] is False


def test_immature_prospective_result_cannot_pass_even_with_a_true_flag(passing_sources, monkeypatch):
    monkeypatch.setattr(sync_dashboard, "evaluate", lambda frame: {
        "eligible_issued": 22, "eligible_realized": 21, "locked_realized": 21,
        "prospective_thresholds_pass": True, "status": "invalid_test_fixture",
    })
    assert sync_dashboard.build_snapshot(passing_sources)["final_pass"] is False


@pytest.mark.parametrize("values", [
    {"independent_validation_ok": False}, {"historical_evidence_ok": False},
    {"macro_point_in_time_ok": False}, {"part1_data_freshness_ok": False},
    {"final_pass": "true"},
])
def test_each_existing_source_control_still_blocks_when_other_evidence_passes(passing_sources, values):
    update(passing_sources, PART2, **values)
    result = sync_dashboard.build_snapshot(passing_sources)
    assert result["final_pass"] is False
    assert any(not item["passed"] for item in result["gate_checks"])


def test_pages_upload_preserves_every_file_including_nojekyll():
    # The inspected Oct 27 GitHub artifact had 12 files but its manifest
    # expected 13: v5's default filter removed .nojekyll after verification.
    text = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    upload = re.search(r"      - name: Upload Pages artifact\n(.*?)(?=\n  deploy:)", text, re.S)
    assert upload is not None
    assert "include-hidden-files: true" in upload.group(1)


def test_pr_pages_validation_cannot_deploy_or_interrupt_production():
    text = (ROOT / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    default_permissions = text.split("permissions:\n", 1)[1].split("env:\n", 1)[0]
    assert "pages: write" not in default_permissions
    assert "id-token: write" not in default_permissions
    assert "pull_request:" in text
    assert "github.event.pull_request.number" in text
    assert "cancel-in-progress: false" in text
    deploy = text.split("  deploy:\n", 1)[1]
    assert "github.event_name != 'pull_request' && github.ref_name == 'main'" in deploy
