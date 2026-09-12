import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from artifact_integrity import PROTOCOL_VERSION, validate_execution_lineage, write_json_strict


SESSION = "2026-09-11"
RUN_ID = "34655470242"
RUN_ATTEMPT = "1"
SOURCE_SHA = "b1727d7df15d0177a189cb98c11a146e53f57937"


def _write_execution_fixture(root: Path) -> None:
    write_json_strict(
        root / "artifacts_part10_bot" / "pipeline_status.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "pipeline_run_date": SESSION,
            "expected_completed_market_session": SESSION,
            "source_code_sha": SOURCE_SHA,
            "github_run_id": RUN_ID,
            "github_run_attempt": RUN_ATTEMPT,
            "result": "verified",
        },
    )
    write_json_strict(
        root / "artifacts_part3_v1" / "part3_summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "decision_date": SESSION,
            "pipeline_run_date": SESSION,
            "source_code_sha": SOURCE_SHA,
            "pipeline_run_id": RUN_ID,
            "pipeline_run_attempt": RUN_ATTEMPT,
            "publish_mode": "FAIL_CLOSED_NEUTRAL",
            "final_pass": 0,
        },
    )
    allocation = pd.DataFrame(
        [
            {"Date": SESSION, "sleeve": "VOO", "weight": 0.6, "is_alpha": 0},
            {"Date": SESSION, "sleeve": "IEF", "weight": 0.4, "is_alpha": 0},
        ]
    )
    allocation["model_protocol_version"] = PROTOCOL_VERSION
    allocation["model_code_sha"] = SOURCE_SHA
    allocation["pipeline_run_id"] = RUN_ID
    allocation["pipeline_run_attempt"] = RUN_ATTEMPT
    allocation["source_decision_date"] = SESSION
    allocation_path = root / "artifacts_part3_v1" / "v1_fusion_allocations.csv"
    allocation_path.parent.mkdir(parents=True, exist_ok=True)
    allocation.to_csv(allocation_path, index=False)
    write_json_strict(
        root / "artifacts_part7" / "current_target_weights.json",
        {
            "Date": SESSION,
            "w_target_voo": 0.6,
            "w_target_ief": 0.4,
            "model_protocol_version": PROTOCOL_VERSION,
            "model_code_sha": SOURCE_SHA,
        },
    )
    instructions = {
        "protocol_version": PROTOCOL_VERSION,
        "lineage_verified": True,
        "decision_date": SESSION,
        "source_decision_date": SESSION,
        "pipeline_run_date": SESSION,
        "source_code_sha": SOURCE_SHA,
        "pipeline_run_id": RUN_ID,
        "pipeline_run_attempt": RUN_ATTEMPT,
        "allocation_source": "v1_fusion_allocations",
        "instructions": [],
        "n_trades": 0,
    }
    write_json_strict(root / "artifacts_part8" / "execution_instructions.json", instructions)
    write_json_strict(
        root / "artifacts_part8" / "part8_meta.json",
        {
            **{key: instructions[key] for key in (
                "protocol_version",
                "lineage_verified",
                "decision_date",
                "pipeline_run_date",
                "source_code_sha",
                "pipeline_run_id",
                "pipeline_run_attempt",
                "allocation_source",
            )},
            "latest_order_instructions": instructions,
        },
    )
    costs = pd.DataFrame([{
        "Date": SESSION,
        "built_at": "2026-09-11T23:15:13+00:00",
        "protocol_version": PROTOCOL_VERSION,
        "lineage_verified": True,
        "pipeline_run_date": SESSION,
        "source_code_sha": SOURCE_SHA,
        "pipeline_run_id": RUN_ID,
        "pipeline_run_attempt": RUN_ATTEMPT,
        "allocation_source": "v1_fusion_allocations",
    }])
    costs.to_csv(root / "artifacts_part8" / "execution_cost_tape.csv", index=False)
    bot_lineage = {
        "protocol_version": PROTOCOL_VERSION,
        "execution_lineage_verified": True,
        "decision_date": SESSION,
        "pipeline_run_date": SESSION,
        "pipeline_run_id": RUN_ID,
        "pipeline_run_attempt": RUN_ATTEMPT,
        "source_code_sha": SOURCE_SHA,
        "price_source": "artifacts_part0/close_prices.parquet",
        "price_session": SESSION,
    }
    write_json_strict(
        root / "artifacts_part10_bot" / "portfolio_state.json", bot_lineage
    )
    write_json_strict(
        root / "artifacts_part10_bot" / "performance_report.json", bot_lineage
    )
    pd.DataFrame(
        [
            {
                "date": SESSION,
                "run_date": SESSION,
                "source_decision_date": SESSION,
                "model_protocol_version": PROTOCOL_VERSION,
                "model_code_sha": SOURCE_SHA,
                "pipeline_run_date": SESSION,
                "pipeline_run_id": RUN_ID,
                "pipeline_run_attempt": RUN_ATTEMPT,
                "execution_source_code_sha": SOURCE_SHA,
                # A legacy-schema migration can round-trip True through a
                # nullable numeric column as 1.0.  That is still an explicit
                # true marker and must pass validation.
                "execution_lineage_verified": 1.0,
                "price_source": "artifacts_part0/close_prices.parquet",
                "price_session": SESSION,
            }
        ]
    ).to_csv(root / "artifacts_part10_bot" / "signal_log.csv", index=False)


def test_direct_and_validator_orders_governance_before_execution():
    from run_tuesday_prediction import DIRECT_PIPELINE_ORDER

    assert DIRECT_PIPELINE_ORDER.index("MARKET_INTEGRITY") < DIRECT_PIPELINE_ORDER.index("PIT_MACRO")
    assert DIRECT_PIPELINE_ORDER.index("PART7") < DIRECT_PIPELINE_ORDER.index("PART3")
    assert DIRECT_PIPELINE_ORDER.index("PART3") < DIRECT_PIPELINE_ORDER.index("PART8")

    source = Path("part5_validator.py").read_text(encoding="utf-8")
    syntax_order = source.split("ordered_scripts = [", 1)[1].split("]", 1)[0]
    runtime_order = source.split("ordered = [", 1)[1].split("]", 1)[0]
    runtime_tail = source.split("ordered_tail = [", 1)[1].split("]", 1)[0]
    assert syntax_order.index("MARKET_INTEGRITY") < syntax_order.index("POINT_IN_TIME_MACRO")
    assert runtime_order.index("MARKET INTEGRITY") < runtime_order.index("POINT-IN-TIME MACRO")
    assert runtime_tail.index("PART 3") < runtime_tail.index("PART 8")

    part7_source = Path("part7_portfolio_construction.py").read_text(encoding="utf-8")
    part10_source = Path("part10_tradingbot.py").read_text(encoding="utf-8")
    assert 'raise SystemExit(main())' in part7_source
    assert 'raise SystemExit(main())' in part10_source


def test_part8_accepts_only_same_run_governed_allocation(tmp_path, monkeypatch):
    from part8_execution_model import Part8Config, load_verified_execution_context

    _write_execution_fixture(tmp_path)
    monkeypatch.setenv("PRICECALL_RUN_DATE_ET", SESSION)
    monkeypatch.setenv("PRICECALL_CODE_SHA", SOURCE_SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN_ID)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", RUN_ATTEMPT)
    cfg = Part8Config(
        part7_dir=str(tmp_path / "artifacts_part7"),
        part3_dir=str(tmp_path / "artifacts_part3_v1"),
        part2_dir=str(tmp_path / "artifacts_part2_g532" / "predictions"),
        part0_dir=str(tmp_path / "artifacts_part0"),
        out_dir=str(tmp_path / "artifacts_part8"),
    )

    latest, _, lineage = load_verified_execution_context(cfg)

    assert latest["source"] == "v1_fusion_allocations"
    assert latest["w_target_voo"] == pytest.approx(0.6)
    assert lineage["lineage_verified"] is True
    assert lineage["pipeline_run_id"] == RUN_ID


def test_part8_rejects_prior_session_allocation(tmp_path, monkeypatch):
    from part8_execution_model import Part8Config, load_verified_execution_context

    _write_execution_fixture(tmp_path)
    path = tmp_path / "artifacts_part3_v1" / "v1_fusion_allocations.csv"
    frame = pd.read_csv(path)
    frame["Date"] = "2026-09-10"
    frame["source_decision_date"] = "2026-09-10"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("PRICECALL_RUN_DATE_ET", SESSION)
    monkeypatch.setenv("PRICECALL_CODE_SHA", SOURCE_SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN_ID)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", RUN_ATTEMPT)
    cfg = Part8Config(
        part7_dir=str(tmp_path / "artifacts_part7"),
        part3_dir=str(tmp_path / "artifacts_part3_v1"),
    )

    with pytest.raises(RuntimeError, match="date mismatch"):
        load_verified_execution_context(cfg)


def test_publication_validator_rejects_mixed_execution_run(tmp_path):
    _write_execution_fixture(tmp_path)
    assert validate_execution_lineage(tmp_path) == []

    path = tmp_path / "artifacts_part8" / "execution_instructions.json"
    instructions = json.loads(path.read_text(encoding="utf-8"))
    instructions["pipeline_run_id"] = "older-run"
    write_json_strict(path, instructions)

    failures = validate_execution_lineage(tmp_path)
    assert any("Part 8 pipeline_run_id" in failure for failure in failures)
    assert any("metadata and execution_instructions" in failure for failure in failures)


def test_publication_validator_rejects_false_numeric_signal_flag(tmp_path):
    _write_execution_fixture(tmp_path)
    path = tmp_path / "artifacts_part10_bot" / "signal_log.csv"
    signals = pd.read_csv(path)
    signals["execution_lineage_verified"] = 0.0
    signals.to_csv(path, index=False)

    failures = validate_execution_lineage(tmp_path)

    assert "Part 10 signal is not marked execution_lineage_verified" in failures


def test_execution_cost_upsert_sorts_and_replaces_same_session(tmp_path):
    from part8_execution_model import _upsert_execution_cost_record

    path = tmp_path / "execution_cost_tape.csv"
    pd.DataFrame([
        {"Date": "2026-09-10", "built_at": "2026-09-10T22:00:00Z", "value": 1},
        {"Date": "2026-08-20", "built_at": "2026-08-20T22:00:00Z", "value": 2},
        {"Date": "2026-09-11", "built_at": "2026-09-11T21:00:00Z", "value": 3},
    ]).to_csv(path, index=False)

    output = _upsert_execution_cost_record(
        path,
        {"Date": "2026-09-11", "built_at": "2026-09-11T23:00:00Z", "value": 4},
    )

    assert pd.to_datetime(output["Date"]).dt.date.astype(str).tolist() == [
        "2026-08-20", "2026-09-10", "2026-09-11"
    ]
    assert output.iloc[-1]["value"] == 4


def test_part10_uses_exact_observed_part0_session(tmp_path, monkeypatch):
    from part10_tradingbot import BotConfig, load_verified_bot_context

    _write_execution_fixture(tmp_path)
    write_json_strict(
        tmp_path / "artifacts_part0" / "part0_meta.json",
        {
            "market_calendar": "XNYS",
            "market_values_are_raw_observations": True,
            "market_data_asof": SESSION,
        },
    )
    close = pd.DataFrame(
        {"VOO": [696.65, 702.56], "IEF": [91.18, 91.01]},
        index=pd.to_datetime(["2026-09-10", SESSION]),
    )
    observed = pd.DataFrame(
        {"VOO": [1, 1], "IEF": [1, 1]}, index=close.index
    )
    close_path = tmp_path / "artifacts_part0" / "close_prices.parquet"
    mask_path = tmp_path / "artifacts_part0" / "market_observation_mask.parquet"

    def fake_read_parquet(path):
        if Path(path) == close_path:
            return close.copy()
        if Path(path) == mask_path:
            return observed.copy()
        raise AssertionError(f"unexpected parquet path: {path}")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setenv("PRICECALL_RUN_DATE_ET", SESSION)
    monkeypatch.setenv("PRICECALL_CODE_SHA", SOURCE_SHA)
    monkeypatch.setenv("GITHUB_RUN_ID", RUN_ID)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", RUN_ATTEMPT)
    cfg = BotConfig(
        part0_close_path=str(close_path),
        part0_observation_mask_path=str(mask_path),
        part0_meta_path=str(tmp_path / "artifacts_part0" / "part0_meta.json"),
        part8_instructions_path=str(
            tmp_path / "artifacts_part8" / "execution_instructions.json"
        ),
    )

    session, prices, lineage = load_verified_bot_context(cfg)

    assert session == SESSION
    assert prices == {"VOO": pytest.approx(702.56), "IEF": pytest.approx(91.01)}
    assert lineage["price_session"] == SESSION
    assert lineage["pipeline_run_id"] == RUN_ID


def test_signal_log_replaces_same_session_revision_and_sorts(tmp_path):
    from part10_tradingbot import SignalLog

    path = tmp_path / "signal_log.csv"
    log = SignalLog(str(path))
    log.append(
        {
            "date": SESSION,
            "run_date": SESSION,
            "source_decision_date": SESSION,
            "pipeline_run_id": "old-run",
            "p_tail": 0.30,
        }
    )
    log.append(
        {
            "date": "2026-09-10",
            "run_date": SESSION,
            "source_decision_date": "2026-09-10",
            "pipeline_run_id": "prior-run",
            "p_tail": 0.25,
        }
    )
    log.append(
        {
            "date": f"{SESSION}T00:00:00",
            "run_date": "2026-09-12",
            "source_decision_date": f"{SESSION}T00:00:00",
            "pipeline_run_id": RUN_ID,
            "p_tail": 0.20,
        }
    )

    output = pd.read_csv(path)
    assert output["source_decision_date"].tolist() == ["2026-09-10", SESSION]
    assert str(output.iloc[-1]["pipeline_run_id"]) == RUN_ID
    assert output.iloc[-1]["p_tail"] == pytest.approx(0.20)


def test_signal_log_schema_migration_canonicalizes_lineage_flag(tmp_path):
    from part10_tradingbot import SignalLog

    path = tmp_path / "signal_log.csv"
    legacy_columns = [
        "date", "run_date", "source_decision_date", "model_protocol_version",
        "model_code_sha", "p_tail", "base_rate", "edge", "target_w_voo",
        "action_reason", "target_source", "dry_run", "accuracy_gate_passed",
        "accuracy_gate_reason", "publish_mode", "final_pass", "raw_val_auc",
        "px_voo", "px_ief", "nav",
    ]
    pd.DataFrame(
        [["2026-09-10", SESSION, "2026-09-10"] + [None] * 17],
        columns=legacy_columns,
    ).to_csv(path, index=False)

    log = SignalLog(str(path))
    log.append(
        {
            "date": SESSION,
            "run_date": "2026-09-12",
            "source_decision_date": SESSION,
            "model_protocol_version": PROTOCOL_VERSION,
            "model_code_sha": SOURCE_SHA,
            "pipeline_run_date": SESSION,
            "pipeline_run_id": RUN_ID,
            "pipeline_run_attempt": RUN_ATTEMPT,
            "execution_source_code_sha": SOURCE_SHA,
            "execution_lineage_verified": True,
            "price_source": "artifacts_part0/close_prices.parquet",
            "price_session": SESSION,
        }
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["source_decision_date"] == SESSION
    assert rows[-1]["execution_lineage_verified"] == "true"
    assert rows[0]["execution_lineage_verified"] == ""


def test_current_sec_section31_fee_is_sell_only(monkeypatch):
    from part8_execution_model import Part8Config, PreTradeAnalyzer

    monkeypatch.delenv("PRICECALL_SEC_FEE_BPS", raising=False)
    cfg = Part8Config()
    analyzer = PreTradeAnalyzer(cfg)
    buy = analyzer.estimate_cost(
        "VOO", 1000.0, 10000.0, direction="buy", use_dynamic_params=False
    )
    sell = analyzer.estimate_cost(
        "VOO", 1000.0, 10000.0, direction="sell", use_dynamic_params=False
    )

    assert cfg.sec_fee_bps == pytest.approx(0.206)
    assert buy["sec_fee_bps"] == 0.0
    assert sell["sec_fee_bps"] == pytest.approx(0.206)
    assert sell["total_bps"] - buy["total_bps"] == pytest.approx(0.206)


def test_dashboard_latest_signal_breaks_same_run_date_tie_by_source_date(tmp_path):
    from sync_dashboard import _latest_record

    path = tmp_path / "signal_log.csv"
    pd.DataFrame([
        {"run_date": SESSION, "source_decision_date": "2026-09-10", "p_tail": 0.30},
        {"run_date": SESSION, "source_decision_date": SESSION, "p_tail": 0.20},
    ]).to_csv(path, index=False)

    latest = _latest_record(path, ("source_decision_date", "date", "run_date"))

    assert latest["source_decision_date"] == SESSION
    assert latest["p_tail"] == pytest.approx(0.20)
