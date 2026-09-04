import json
from pathlib import Path

import pytest

from artifact_integrity import (
    PROTOCOL_VERSION,
    REQUIRED_PUBLISHED_FILES,
    build_run_manifest,
    read_json_strict,
    verify_run_manifest,
    write_json_strict,
)


def test_strict_json_converts_nonfinite_values_to_null(tmp_path):
    path = tmp_path / "report.json"
    write_json_strict(path, {"bad": float("nan"), "good": 1.5})
    assert json.loads(path.read_text()) == {"bad": None, "good": 1.5}
    assert read_json_strict(path)["bad"] is None


def test_strict_json_rejects_legacy_nan_token(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"bad": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_strict(path)


def test_backfill_manual_dispatch_cannot_bypass_close_gate():
    text = Path(".github/workflows/daily-backfill.yml").read_text(encoding="utf-8")
    assert "should_run = manual or settled" not in text
    assert "Manual backfill refused" in text


def test_workflows_publish_dashboard_and_manifest():
    for workflow in ["tuesday-pipeline.yml", "daily-backfill.yml"]:
        text = Path(".github/workflows", workflow).read_text(encoding="utf-8")
        assert "sync_dashboard.py" in text
        assert "artifact_integrity.py" in text
        assert "artifacts_manifest.json" in text
        assert "gh workflow run pages.yml" in text
        assert "--strategy-option=ours" not in text


def test_production_gate_has_no_upper_time_window():
    text = Path(".github/workflows/tuesday-pipeline.yml").read_text(encoding="utf-8")
    assert "after_start = now.hour >= 8" in text
    assert "now.hour <=" not in text
    assert "completed != today" in text


def test_dashboard_snapshot_excludes_legacy_evidence(tmp_path):
    from sync_dashboard import build_snapshot

    sources = {
        "artifacts_part2_g532/predictions/part2_g532_summary.json": {
            "publish_mode": "FAIL_CLOSED_NEUTRAL", "final_pass": False,
            "part1_data_freshness_ok": False,
            "macro_point_in_time_ok": False,
            "historical_evidence_ok": False,
            "historical_brier_skill_causal": -0.01,
            "delong_overall_auc": {"auc": 0.51, "p_one_sided": 0.30},
            "distributional_diagnostics": {"conf_coverage": 0.89},
        },
        "artifacts_part3_v1/part3_summary.json": {
            "publish_mode": "FAIL_CLOSED_NEUTRAL", "deployment_mode": "DEFENSE_ONLY",
            "current_alpha_live_status": "SHADOW", "final_pass": False,
        },
        "artifacts_part9/live_attribution_report.json": {
            "evidence_cohort": PROTOCOL_VERSION, "n_live_realized": 0,
            "legacy_realized_rows": 28, "health_status": "IMMATURE",
        },
    }
    for rel, payload in sources.items():
        write_json_strict(tmp_path / rel, payload)
    snapshot = build_snapshot(tmp_path)
    assert snapshot["publish_mode"] == "FAIL_CLOSED_NEUTRAL"
    assert snapshot["data_freshness_ok"] is False
    assert snapshot["evidence"]["eligible_realized"] == 0
    assert snapshot["evidence"]["legacy_realized_excluded"] == 28
    assert snapshot["operator_validation"]["status"] == "NOT_VALIDATED"
    assert snapshot["metrics"]["backtest_auc_p_value"] == 0.30
    assert snapshot["prediction_interval"]["nominal_coverage"] == 0.90
    assert snapshot["prediction_interval"]["empirical_coverage"] == 0.89


def test_dashboard_html_sync_updates_ledger_and_binds_snapshot(tmp_path):
    import pandas as pd

    from sync_dashboard import sync_html

    (tmp_path / "artifacts_part3").mkdir()
    pd.DataFrame([{
        "target_date": "2026-09-03",
        "px_voo_call_1d": 100.0,
        "model_protocol_version": PROTOCOL_VERSION,
        "evidence_eligible": 1,
    }]).to_csv(tmp_path / "artifacts_part3" / "prediction_log.csv", index=False)
    (tmp_path / "index.html").write_text(
        "<html><body><script>\nconst rows=[{\"old\":true}];\n"
        "const botRows=[{\"date\":\"legacy\"}];\n</script></body></html>",
        encoding="utf-8",
    )
    sync_html(tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert '"target_date":"2026-09-03"' in html
    assert 'const botRows=[{"date":"legacy"}]' in html
    assert 'id="pricecall-verified-snapshot"' in html
    assert "AUC p-value" in html
    assert "operator_validation.status" in html


def test_dashboard_html_sync_refuses_empty_prediction_ledger(tmp_path):
    from sync_dashboard import sync_html

    (tmp_path / "index.html").write_text(
        "<html><body><script>\nconst rows=[];\nconst botRows=[];\n</script></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prediction_log"):
        sync_html(tmp_path)


def test_pages_bundle_contains_every_local_reference(tmp_path):
    from prepare_pages import prepare_site

    (tmp_path / "artifacts_dashboard").mkdir()
    (tmp_path / "artifacts_dashboard" / "dashboard_snapshot.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts_part10_bot").mkdir()
    for name in ("signal_log.csv", "portfolio_state.json", "performance_report.json"):
        (tmp_path / "artifacts_part10_bot" / name).write_text("{}", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        "<a href='artifacts_part10_bot/signal_log.csv'>log</a>"
        "<script>fetch('artifacts_dashboard/dashboard_snapshot.json')</script>",
        encoding="utf-8",
    )
    site = tmp_path / "site"
    prepare_site(tmp_path, site)
    assert (site / "artifacts_dashboard" / "dashboard_snapshot.json").is_file()
    assert (site / "artifacts_part10_bot" / "signal_log.csv").is_file()


def test_second_pass_governance_uses_rowwise_base_rate():
    text = Path("part2_predictor.py").read_text(encoding="utf-8")
    assert 'float(out.loc[i, "base_rate"]),' in text
    assert 'float(out.loc[i, "p_final_cal"]),\n            base_rate,' not in text


def test_pages_workflow_uses_verified_builder():
    text = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")
    assert "python prepare_pages.py --root . --site _site" in text
    assert "workflow_dispatch:" in text
    assert "push:" not in text


def test_manifest_detects_a_post_generation_change(tmp_path):
    for rel in REQUIRED_PUBLISHED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    write_json_strict(tmp_path / "artifacts_manifest.json", build_run_manifest(tmp_path))
    assert verify_run_manifest(tmp_path) == []
    (tmp_path / "index.html").write_text("changed", encoding="utf-8")
    assert any("index.html" in failure for failure in verify_run_manifest(tmp_path))


def test_manifest_reports_a_post_generation_deletion(tmp_path):
    for rel in REQUIRED_PUBLISHED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    write_json_strict(tmp_path / "artifacts_manifest.json", build_run_manifest(tmp_path))
    (tmp_path / "index.html").unlink()
    failures = verify_run_manifest(tmp_path)
    assert failures == ["published file is missing after manifest generation: index.html"]


def test_manifest_directly_hashes_mutable_evidence_ledgers():
    required = set(REQUIRED_PUBLISHED_FILES)
    assert "artifacts_part2_g532/predictions/g532_final_consensus_tape.csv" in required
    assert "artifacts_part3/prediction_log.csv" in required
    assert "artifacts_part8/execution_cost_tape.csv" in required
    assert "artifacts_part10_bot/signal_log.csv" in required
    assert "artifacts_part10_bot/trade_log.csv" in required


def test_current_html_has_no_pre_snapshot_performance_claims():
    html = Path("index.html").read_text(encoding="utf-8")
    assert "Committed artifact snapshot · August 25, 2026" not in html
    assert "<strong>25 / 60</strong>live evidence" not in html
    assert "<label>AUC p-value</label><strong>—</strong>" in html
    assert "<span class=\"kicker\">Operator validation</span>" in html
