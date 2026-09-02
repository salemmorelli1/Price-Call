import json
from pathlib import Path

import pytest

from artifact_integrity import PROTOCOL_VERSION, read_json_strict, write_json_strict


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


def test_dashboard_html_sync_refuses_empty_prediction_ledger(tmp_path):
    from sync_dashboard import sync_html

    (tmp_path / "index.html").write_text(
        "<html><body><script>\nconst rows=[];\nconst botRows=[];\n</script></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prediction_log"):
        sync_html(tmp_path)
