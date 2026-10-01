"""Regressions for the real September 30 close and ALFRED failures."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import point_in_time_macro as pit
import production_readiness as readiness


def releases(start):
    return pd.DataFrame({"date": [start], "realtime_start": [start], "value": [2.0]})


def test_last_alfred_window_valueerror_none_is_retried_without_losing_prior_chunks(monkeypatch):
    calls = []

    class Fred:
        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            calls.append((realtime_start, realtime_end))
            if realtime_start == "2024-09-09" and calls.count(calls[-1]) == 1:
                raise ValueError(None)  # Exact fredapi failure observed in run #370.
            return releases(realtime_start)

    monkeypatch.setattr(pit.time, "sleep", lambda _: None)
    frame = pit.get_series_releases_chunked(Fred(), "T10Y2Y", "2020-09-09", "2026-09-30")
    assert len(calls) == 3
    assert frame["value"].tolist() == [2.0, 2.0]
    assert frame.attrs["chunk_diagnostics"][-1]["attempts"] == 2
    assert frame.attrs["chunk_diagnostics"][-1]["status"] == "ok"


@pytest.mark.parametrize("error,retryable,attempts", [
    (TimeoutError("provider timeout"), True, 3),
    (ValueError(None), True, 3),
    (ValueError("api_key is not registered"), False, 1),
])
def test_request_failures_have_bounded_attempts_and_correct_classification(monkeypatch, error, retryable, attempts):
    calls = []

    class Fred:
        def get_series_all_releases(self, *args, **kwargs):
            calls.append(kwargs)
            raise error

    monkeypatch.setattr(pit.time, "sleep", lambda _: None)
    with pytest.raises(pit.AlfredCoverageError) as failure:
        pit.get_series_releases_chunked(Fred(), "T10Y2Y", "2024-09-09", "2026-09-30")
    assert len(calls) == attempts
    assert failure.value.retryable is retryable
    assert failure.value.diagnostics[0]["attempts"] == attempts
    assert failure.value.diagnostics[0]["status"] == "coverage_error"


def test_schema_failure_is_fatal_without_retry(monkeypatch):
    calls = []

    class Fred:
        def get_series_all_releases(self, *args, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"date": ["2026-09-30"], "value": [3.0]})

    with pytest.raises(pit.AlfredCoverageError) as failure:
        pit.get_series_releases_chunked(Fred(), "T10Y2Y", "2024-09-09", "2026-09-30")
    assert len(calls) == 1
    assert failure.value.retryable is False


@pytest.mark.parametrize("hole_day,pending", [("2026-09-30", True), ("2026-09-29", False)])
def test_current_core_close_delay_is_pending_but_historical_hole_is_fatal(monkeypatch, hole_day, pending):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(["2026-09-28", "2026-09-29", "2026-09-30"], name="Date")
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close", "Volume"]])
    bulk = pd.DataFrame([[100, 10, 90, 9], [101, 10, 91, 9], [102, 10, 92, 9]],
                        index=sessions, columns=columns, dtype=float)
    bulk.loc[hole_day, ("VOO", "Close")] = float("nan")
    monkeypatch.setattr(part0, "_business_day_calendar", lambda *args: sessions)
    monkeypatch.setattr(part0, "latest_completed_xnys_session", lambda: sessions[-1])
    monkeypatch.setattr(part0.yf, "download", lambda *args, **kwargs: bulk.copy())
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    monkeypatch.setattr(part0, "_recover_verified_historical_core_closes", lambda *args: None)
    monkeypatch.setattr(part0, "_recover_verified_adjacent_backfilled_core_closes", lambda *args: None)
    cfg = part0.Part0Config(start="2026-09-28", end="2026-09-30",
                           equity_tickers=("VOO", "IEF"), vix_tickers=(), min_history_years=0.0)
    with pytest.raises(RuntimeError) as failure:
        part0.download_market_data(cfg)
    assert isinstance(failure.value, readiness.DataPendingError) is pending


@pytest.mark.parametrize("module", [pit, pytest.param("part0", id="part0")])
def test_pending_entrypoints_return_75_and_write_only_attempt_diagnostics(monkeypatch, tmp_path, module):
    if module == "part0":
        import part0_data_infrastructure as module
        operation = "download_market_data"
    else:
        operation = "rebuild_point_in_time_macro"
    sentinel = tmp_path / "artifacts_part10_bot/pipeline_run_date.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("2026-09-29\n")
    monkeypatch.setenv("PRICECALL_ROOT", str(tmp_path))
    monkeypatch.delenv("PRICECALL_INPUT_STATUS_PATH", raising=False)

    def pending(*args):
        raise readiness.DataPendingError("provider still unavailable", stage="TEST", diagnostics={"attempts": 3})

    monkeypatch.setattr(module, operation, pending)
    assert module.main() == 75
    payload = json.loads((tmp_path / "artifacts_diagnostics/production_input_status.json").read_text())
    assert payload["status"] == "DATA_PENDING"
    assert sentinel.read_text() == "2026-09-29\n"


@pytest.mark.parametrize("pending_label", ["PART0", "PIT_MACRO"])
def test_runner_stops_before_training_after_pending_inputs(monkeypatch, pending_label):
    import run_tuesday_prediction as runner

    calls = []

    def run(cmd, *args, **kwargs):
        calls.append(Path(cmd[-1]).name)
        return 75 if calls[-1] == runner.CANONICAL_FILES[pending_label] else 0

    monkeypatch.setattr(runner, "run_subprocess", run)
    assert runner.run_direct_pipeline(Path.cwd()) == 75
    assert runner.CANONICAL_FILES["PART6"] not in calls
    assert runner.CANONICAL_FILES["PART3"] not in calls


def recovery_publication(tmp_path, monkeypatch, *, complete=False, eligible=0):
    from artifact_integrity import write_json_strict

    write_json_strict(tmp_path / "artifacts_part0/part0_meta.json", {
        "market_data_asof": "2026-09-30", "historical_point_in_time_complete": complete,
    })
    path = tmp_path / "artifacts_part3/prediction_log.csv"
    path.parent.mkdir()
    pd.DataFrame([{
        "decision_date": "2026-09-30", "target_date": "2026-10-01",
        "model_protocol_version": "causal-integrity-v4", "evidence_eligible": eligible,
    }]).to_csv(path, index=False)
    monkeypatch.setattr(readiness, "verify_run_manifest", lambda root: [])


@pytest.mark.parametrize("complete,eligible,now,retry", [
    (False, 0, "2026-10-01T03:00:00Z", True),
    (False, 0, "2026-10-01T20:00:00Z", False),
    (False, 1, "2026-10-01T03:00:00Z", False),
    (True, 0, "2026-10-01T03:00:00Z", False),  # Statistical rejection is not retraining permission.
])
def test_recovery_never_rewrites_eligible_or_closed_target_forecasts(tmp_path, monkeypatch, complete, eligible, now, retry):
    recovery_publication(tmp_path, monkeypatch, complete=complete, eligible=eligible)
    ledger = tmp_path / "artifacts_part3/prediction_log.csv"
    before = ledger.read_bytes()
    assert readiness.retry_incomplete_macro_session(tmp_path, "2026-09-30", now=now) is retry
    assert ledger.read_bytes() == before


def test_recovery_requires_verified_publication(tmp_path, monkeypatch):
    recovery_publication(tmp_path, monkeypatch)
    monkeypatch.setattr(readiness, "verify_run_manifest", lambda root: ["ledger hash differs"])
    with pytest.raises(RuntimeError, match="unverified publication"):
        readiness.retry_incomplete_macro_session(tmp_path, "2026-09-30", now="2026-10-01T03:00:00Z")


@pytest.mark.parametrize("result,expected_exit,output", [(75, 0, "ready=false"), (1, 1, ""), (0, 0, "ready=true")])
def test_actual_production_shell_handles_pending_and_failure_distinctly(tmp_path, result, expected_exit, output):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("Git Bash or bash is required to execute the workflow shell")
    workflow = Path(".github/workflows/tuesday-pipeline.yml").read_text()
    step = workflow.split("      - name: Run pipeline\n", 1)[1].split("\n      - name:", 1)[0]
    script = "\n".join(line[10:] for line in step.split("        run: |\n", 1)[1].splitlines())
    (tmp_path / "run_tuesday_prediction.py").write_text(f"raise SystemExit({result})\n")
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"stage": "PIT_MACRO", "reason": "incomplete retrieval"}))
    env = dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "output.txt"),
               GITHUB_STEP_SUMMARY=str(tmp_path / "summary.md"), PRICECALL_INPUT_STATUS_PATH=str(status))
    # Use the tested interpreter even when the caller's PATH lacks its venv.
    script = script.replace("python ", f'"{sys.executable}" ')
    run = subprocess.run([bash, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run.returncode == expected_exit, run.stderr
    if output:
        assert output in (tmp_path / "output.txt").read_text()
    if result == 75:
        assert "DATA_PENDING" in (tmp_path / "summary.md").read_text()


def test_all_post_pipeline_mutations_require_ready_inputs():
    workflow = Path(".github/workflows/tuesday-pipeline.yml").read_text()
    for step in (
        "Reconcile prospective outcomes from verified Part 0 closes",
        "Evaluate fixed prospective paper cohort", "Record successful production session",
        "Synchronize dashboard", "Record provenance and validate strict JSON",
        "Commit and push artifacts", "Deploy the committed dashboard",
    ):
        condition = workflow.split(f"      - name: {step}\n", 1)[1].splitlines()[0]
        assert "steps.pipeline.outputs.ready == 'true'" in condition
