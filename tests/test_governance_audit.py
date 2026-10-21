"""Regression checks for the October 20 governance and publication audit."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from artifact_integrity import REQUIRED_PUBLISHED_FILES
from statistical_tests import delong_auc_statistics


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def publication_inputs(tmp_path):
    for rel in (*REQUIRED_PUBLISHED_FILES, "artifacts_manifest.json"):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for folder in ("assets", "report"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    return tmp_path


def fingerprints(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_delong_matches_published_stata_tied_rating_example():
    # Stata roccomp manual, Table on p. 9, modality 1 (58 normal, 54 abnormal).
    # The published AUC/SE are 0.8828/0.0317, not a binomial or Q1/Q2 SE.
    normal = np.repeat(np.arange(6) / 5, [12, 28, 8, 6, 4, 0])
    abnormal = np.repeat(np.arange(6) / 5, [1, 3, 6, 13, 22, 9])
    y = np.r_[np.zeros(len(normal)), np.ones(len(abnormal))]
    p = np.r_[normal, abnormal]
    result = delong_auc_statistics(y, p)
    assert result["auc"] == pytest.approx(0.8828, abs=0.00005)
    assert result["se"] == pytest.approx(0.0317, abs=0.00005)
    reverse = delong_auc_statistics(1 - y, p)
    assert reverse["auc"] == pytest.approx(1 - result["auc"])
    assert reverse["se"] == pytest.approx(result["se"])


def test_delong_matches_independent_dense_reference_on_imbalanced_sample():
    rng = np.random.default_rng(532)
    positive, negative = rng.random(11), rng.random(73)
    comparison = (positive[:, None] > negative).astype(float)
    reference = comparison.mean(axis=1).var(ddof=1) / 11 + comparison.mean(axis=0).var(ddof=1) / 73
    result = delong_auc_statistics(np.r_[np.ones(11), np.zeros(73)], np.r_[positive, negative])
    assert result["auc"] == pytest.approx(comparison.mean())
    assert result["se"] ** 2 == pytest.approx(reference)


@pytest.mark.parametrize("positive,negative", [(4, 60), (60, 4), (5, 5)])
def test_delong_insufficient_sample_cannot_claim_significance(positive, negative):
    y = np.r_[np.ones(positive), np.zeros(negative)]
    result = delong_auc_statistics(y, np.linspace(0, 1, len(y)))
    assert result["inference_eligible"] is False
    assert np.isnan(result["p_one_sided"])


@pytest.mark.parametrize("p", [np.repeat(.2, 60), np.r_[np.ones(12), np.zeros(48)]])
def test_zero_auc_variance_does_not_claim_normal_inference(p):
    result = delong_auc_statistics(np.r_[np.ones(12), np.zeros(48)], p)
    assert result["inference_eligible"] is False
    assert np.isnan(result["p_one_sided"])


def test_corrupt_manifest_source_cannot_be_packaged(publication_inputs):
    from prepare_pages import prepare_site
    root = publication_inputs
    (root / "artifacts_dashboard/dashboard_snapshot.json").write_text("{}", encoding="utf-8")
    output = root / "preview"
    with pytest.raises(ValueError, match="source verification failed"):
        prepare_site(root, output)
    assert not output.exists()


def test_verified_pages_refresh_only_the_bundle_and_detect_tampering(publication_inputs):
    from prepare_pages import prepare_site, validate_site
    root = publication_inputs
    before = fingerprints(root)
    output = root.with_name(root.name + "-publication")
    prepare_site(root, output)
    validate_site(output)
    assert fingerprints(root) == before
    snapshot = json.loads((output / "artifacts_dashboard/dashboard_snapshot.json").read_text())
    assert snapshot["prospective_validation"]["required_outcomes"] == 60
    assert snapshot["final_pass"] is False
    (output / "index.html").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        validate_site(output)


def test_unlisted_pages_file_is_rejected(publication_inputs):
    from prepare_pages import prepare_site, validate_site
    output = publication_inputs.with_name(publication_inputs.name + "-publication")
    prepare_site(publication_inputs, output)
    (output / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="file set differs"):
        validate_site(output)


def test_pages_cannot_overwrite_inputs_or_an_existing_output(publication_inputs):
    from prepare_pages import prepare_site
    root = publication_inputs
    with pytest.raises(ValueError, match="overlap"):
        prepare_site(root, root)
    with pytest.raises(ValueError, match="overlap"):
        prepare_site(root, root / "artifacts_dashboard/new")
    existing = root / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="fresh directory"):
        prepare_site(root, existing)


def test_summary_cannot_validate_contradictory_evidence(publication_inputs, monkeypatch):
    import sync_dashboard
    root = publication_inputs
    for rel in ("artifacts_part2_g532/predictions/part2_g532_summary.json",
                "artifacts_part3_v1/part3_summary.json"):
        path = root / rel
        value = json.loads(path.read_text())
        value.update(final_pass=True, publish_mode="NORMAL")
        path.write_text(json.dumps(value), encoding="utf-8")
    expected = json.loads((root / "artifacts_part10_bot/pipeline_status.json").read_text())["pipeline_run_date"]
    monkeypatch.setattr(sync_dashboard, "latest_completed_xnys_session", lambda: pd.Timestamp(expected))
    monkeypatch.setattr(sync_dashboard, "validate_execution_lineage", lambda root: [])
    snapshot = sync_dashboard.build_snapshot(root)
    assert snapshot["governance_final_pass"] is True
    assert snapshot["final_pass"] is False
    assert snapshot["publish_mode"] == "FAIL_CLOSED_NEUTRAL"
    assert snapshot["operator_validation"]["status"] == "NOT_VALIDATED"


def test_snapshot_expiry_observes_early_close_and_holiday(publication_inputs):
    import sync_dashboard
    from market_calendar import _calendar, next_xnys_session
    root = publication_inputs
    snapshot = sync_dashboard.build_snapshot(root)
    first = min(snapshot["lineage"]["publication_component_dates"].values())
    expected = _calendar().session_close(next_xnys_session(first)) + pd.Timedelta(minutes=20)
    assert pd.Timestamp(snapshot["freshness_expires_at_utc"]) == expected
    # The same exchange helper used by the builder must honor Thanksgiving and
    # the November 27 early close, including the change from EDT to EST.
    next_session = next_xnys_session("2026-11-25")
    assert next_session == pd.Timestamp("2026-11-27")
    assert (_calendar().session_close(next_session) + pd.Timedelta(minutes=20)).isoformat() == "2026-11-27T18:20:00+00:00"


def test_read_only_audit_preserves_every_source_file(publication_inputs):
    from audit_governance import audit
    root = publication_inputs
    before = fingerprints(root)
    report = audit(root)
    assert report["integrity_status"] == "PASS"
    assert report["governance_status"] == "BLOCKED"
    assert report["allocation_approved"] is False
    assert report["historical_evidence"]["reference_auc_uncertainty"]["method"] == "delong_structural_components_v1"
    assert fingerprints(root) == before


def test_audit_can_replay_screenshot_time_without_guessing_timezone(publication_inputs):
    from audit_governance import audit
    from market_calendar import _calendar, next_xnys_session
    report = audit(publication_inputs, as_of="2026-10-20T23:33:32Z")
    assert report["publication"]["expected_session"] == "2026-10-20"
    last = max(report["publication"]["component_dates"].values())
    next_settlement = _calendar().session_close(next_xnys_session(last)) + pd.Timedelta(minutes=20)
    expired = audit(publication_inputs, as_of=next_settlement.isoformat())
    assert expired["publication"]["fresh"] is False
    with pytest.raises(ValueError, match="explicit UTC offset"):
        audit(publication_inputs, as_of="2026-10-20T23:33:32")


def test_actual_browser_binding_closes_an_expired_open_page(publication_inputs):
    from sync_dashboard import _DASHBOARD_BINDING, build_snapshot
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the actual dashboard binding")
    snapshot = build_snapshot(publication_inputs)
    snapshot.update(final_pass=True, data_freshness_ok=True, publication_freshness_ok=True,
                    publish_mode="NORMAL", deployment_mode="NORMAL")
    snapshot["operator_validation"].update(status="VALIDATED", reasons=[])
    script = re.search(r'<script id="pricecall-verified-snapshot">(.*?)</script>', _DASHBOARD_BINDING, re.S)[1]
    javascript = r'''
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const freshness = require(input.helper);
let clock = Date.parse(input.snapshot.freshness_expires_at_utc) - 1;
const timers = [], nodes = {};
const node = id => nodes[id] ||= {textContent:'', classList:{toggle(){}, add(){}}};
const document = {documentElement:{dataset:{}}, querySelectorAll:()=>[],
  querySelector:()=>null, getElementById:node, addEventListener(){}};
class Clock extends Date {static now(){return clock;}}
const context = {document, Date:Clock, PriceCallFreshness:freshness,
  fetch:async()=>({ok:true, json:async()=>input.snapshot}),
  window:{setInterval:f=>timers.push(f), addEventListener(){}}};
(async()=>{
  vm.runInNewContext(input.script, context);
  await new Promise(setImmediate);
  if (node('hero-gate-state').textContent !== 'GATE OPEN') throw Error('initial render');
  clock += 2;
  timers[0]();
  if (node('hero-gate-state').textContent !== 'FAIL-CLOSED') throw Error('expired render');
  if (!input.snapshot.final_pass) throw Error('source snapshot was mutated');
  console.log('PASS');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", javascript], input=json.dumps({
        "snapshot": snapshot, "script": script,
        "helper": str(ROOT / "assets/snapshot_freshness.js"),
    }), text=True, capture_output=True, check=False, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"
