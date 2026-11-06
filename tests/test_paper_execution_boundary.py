"""Replay the paper instruction and accounting boundary with real local files."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import part8_execution_model as execution
import part10_tradingbot as bot
from artifact_integrity import (
    PROTOCOL_VERSION, read_json_strict, validate_execution_lineage, write_json_strict,
)
from test_execution_lineage import (
    RUN_ATTEMPT, RUN_ID, SESSION, SOURCE_SHA, _write_execution_fixture,
)


@pytest.fixture
def paper_run(tmp_path, monkeypatch):
    _write_execution_fixture(tmp_path)
    for key, value in {
        "PRICECALL_ROOT": str(tmp_path),
        "PRICECALL_RUN_DATE_ET": SESSION,
        "PRICECALL_CODE_SHA": SOURCE_SHA,
        "GITHUB_RUN_ID": RUN_ID,
        "GITHUB_RUN_ATTEMPT": RUN_ATTEMPT,
    }.items():
        monkeypatch.setenv(key, value)
    p0 = tmp_path / "artifacts_part0"
    p2 = tmp_path / "artifacts_part2_g532" / "predictions"
    p3 = tmp_path / "artifacts_part3_v1"
    p7 = tmp_path / "artifacts_part7"
    p8 = tmp_path / "artifacts_part8"
    p9 = tmp_path / "artifacts_part9"
    p10 = tmp_path / "artifacts_part10_bot"
    cfg8 = execution.Part8Config(part0_dir=str(p0), part2_dir=str(p2),
                                 part3_dir=str(p3), part7_dir=str(p7), out_dir=str(p8))
    monkeypatch.setattr(execution, "Part8Config", lambda: cfg8)
    monkeypatch.setattr(execution, "_DYNAMIC_PARAMS_CACHE", {})
    monkeypatch.setattr(execution, "_load_live_vix", lambda *args, **kwargs: 18.0)

    def no_network(*args, **kwargs):
        pytest.fail("paper integration must not fetch market prices")

    monkeypatch.setattr(bot.yf, "download", no_network)
    write_json_strict(p0 / "part0_meta.json", {
        "market_calendar": "XNYS", "market_values_are_raw_observations": True,
        "market_data_asof": SESSION,
    })
    idx = pd.to_datetime(["2026-09-10", SESSION])
    pd.DataFrame({"VOO": [696.65, 702.56], "IEF": [91.18, 91.01]}, index=idx).to_parquet(
        p0 / "close_prices.parquet"
    )
    pd.DataFrame({"VOO": [1, 1], "IEF": [1, 1]}, index=idx).to_parquet(
        p0 / "market_observation_mask.parquet"
    )
    pd.DataFrame({"VOO": [5_000_000, 5_000_000], "IEF": [2_000_000, 2_000_000]}, index=idx).to_parquet(
        p0 / "volume_data.parquet"
    )
    write_json_strict(p2 / "part2_g532_summary.json", {
        "publish_mode": "NORMAL", "final_pass": True, "part1_data_freshness_ok": True,
    })
    summary = read_json_strict(p3 / "part3_summary.json")
    write_json_strict(p3 / "part3_summary.json", {**summary, "publish_mode": "NORMAL", "final_pass": 1})
    allocation = pd.read_csv(p3 / "v1_fusion_allocations.csv")
    allocation["weight"] = allocation["sleeve"].map({"VOO": 0.75, "IEF": 0.25})
    allocation.to_csv(p3 / "v1_fusion_allocations.csv", index=False)
    write_json_strict(p7 / "current_target_weights.json", {
        "Date": SESSION, "p_tail_base": 0.16, "base_rate": 0.20,
        "w_target_voo": 0.85, "w_target_ief": 0.15, "publish_mode": "NORMAL",
        "final_pass": 1, "model_protocol_version": PROTOCOL_VERSION, "model_code_sha": SOURCE_SHA,
    })
    write_json_strict(p9 / "live_attribution_report.json", {
        "n_live_realized": 60, "health_status": "ACTIVE", "evidence_cohort": PROTOCOL_VERSION,
        "classification_stats_live": {"t_stat_auc": 2.5, "balanced_accuracy": 0.60,
                                      "brier_skill_score": 0.02, "inference_eligible": True},
        "calibration_live": {"ece": 0.05},
    })
    write_json_strict(p10 / "portfolio_state.json", {
        "cash": 1000.0, "shares_voo": 0.0, "shares_ief": 0.0,
        "trade_count": 0, "initialized": True, "is_stopped": False, "peak_nav": 1000.0,
    })
    return bot.BotConfig(
        root_dir=str(tmp_path),
        part2_summary_path=str(p2 / "part2_g532_summary.json"),
        part2_tape_path=str(p2 / "g532_final_consensus_tape.csv"),
        part0_close_path=str(p0 / "close_prices.parquet"),
        part0_observation_mask_path=str(p0 / "market_observation_mask.parquet"),
        part0_meta_path=str(p0 / "part0_meta.json"),
        part7_current_target_path=str(p7 / "current_target_weights.json"),
        part7_weights_tape_path=str(p7 / "portfolio_weights_tape.csv"),
        part8_instructions_path=str(p8 / "execution_instructions.json"),
        part9_report_path=str(p9 / "live_attribution_report.json"),
        bot_dir=str(p10),
    )


def _set_json(path, **fields):
    write_json_strict(path, {**read_json_strict(path), **fields})


def _state(cfg):
    return read_json_strict(Path(cfg.bot_dir) / "portfolio_state.json")


def _trades(cfg):
    return pd.read_csv(Path(cfg.bot_dir) / "trade_log.csv")


def _signal(cfg):
    frame = pd.read_csv(Path(cfg.bot_dir) / "signal_log.csv")
    return frame.loc[frame["date"].eq(SESSION)].iloc[-1]


@pytest.mark.parametrize("field,value", [
    ("part1_data_freshness_ok", False),
    ("part1_data_freshness_ok", "false"),
    ("final_pass", "false"),
])
def test_part8_does_not_generate_instructions_on_stale_or_truthy_false_input(paper_run, field, value):
    _set_json(paper_run.part2_summary_path, **{field: value})
    assert execution.main() == 0
    instructions = read_json_strict(paper_run.part8_instructions_path)
    assert instructions["status"] == "NO_ACTION_STALE_OR_UNCLEARED"
    assert instructions["instructions"] == []
    assert instructions["n_trades"] == 0


def test_part8_rejects_textual_part3_approval(paper_run):
    _set_json(Path(paper_run.root_dir) / "artifacts_part3_v1/part3_summary.json", final_pass="0")
    assert execution.main() == 0
    assert read_json_strict(paper_run.part8_instructions_path)["n_trades"] == 0


@pytest.mark.parametrize("accuracy_disabled", [False, True])
def test_verified_part8_veto_blocks_even_favorable_part9_or_disabled_accuracy(paper_run, accuracy_disabled):
    cfg = replace(paper_run, require_significance=not accuracy_disabled)
    _set_json(Path(cfg.root_dir) / "artifacts_part3_v1/part3_summary.json",
              publish_mode="FAIL_CLOSED_NEUTRAL", final_pass=0)
    assert execution.main() == 0
    assert read_json_strict(cfg.part8_instructions_path)["n_trades"] == 0

    assert bot.run_daily(cfg) == 0

    assert _trades(cfg).empty
    assert _state(cfg)["cash"] == 1000.0
    assert _state(cfg)["shares_voo"] == _state(cfg)["shares_ief"] == 0.0
    assert bool(_signal(cfg)["dry_run"])


def test_dry_run_stop_loss_does_not_liquidate_or_debit_positions(paper_run):
    cfg = paper_run
    path = Path(cfg.bot_dir) / "portfolio_state.json"
    _set_json(path, cash=0.0, shares_voo=1.0, shares_ief=1.0, trade_count=4)
    _set_json(cfg.part9_report_path, n_live_realized=29, health_status="IMMATURE")
    assert execution.main() == 0

    assert bot.run_daily(cfg) == 0

    state = _state(cfg)
    assert state["cash"] == 0.0
    assert state["shares_voo"] == state["shares_ief"] == 1.0
    assert state["trade_count"] == 4
    assert state["is_stopped"] is False
    assert _trades(cfg).empty


def test_paper_buy_uses_governed_weights_and_spends_cash_including_fees(paper_run):
    cfg = paper_run
    assert execution.main() == 0
    assert read_json_strict(cfg.part8_instructions_path)["n_trades"] == 2

    assert bot.run_daily(cfg) == 0

    trade, = _trades(cfg).to_dict("records")
    state = _state(cfg)
    assert state["w_voo"] == pytest.approx(0.75)
    assert state["cash"] == pytest.approx(0.0)
    assert state["nav"] == pytest.approx(1000.0 / (1.0 + cfg.tc_bps_one_way / 10_000.0), abs=1e-4)
    assert trade["nav_after"] == pytest.approx(trade["nav_before"] - trade["tc_dollars"], abs=1e-4)
    assert trade["mode"] == "paper"
    assert state["mode"] == "PAPER"
    assert state["paper_execution_plan_cleared"] is True
    assert not bool(_signal(cfg)["dry_run"])


@pytest.mark.parametrize("changes", [
    {"status": "UNKNOWN"},
    {"data_freshness_ok": False},
    {"governance_part2_final_pass": "false"},
    {"governance_part3_publish_mode": "FAIL_CLOSED_NEUTRAL"},
    {"n_trades": 0, "instructions": []},
    {"n_trades": True},
    {"governed_target_weights": None},
    {"governed_target_weights": {"VOO": 0.95, "IEF": 0.05}},
    {"governed_target_weights": {"VOO": 0.75, "IEF": 0.35}},
    {"instructions": [{"ticker": "VOO"}, {"ticker": "IEF"}]},
    {"governed_target_weights": {"VOO": 0.70, "IEF": 0.30}},
])
def test_incomplete_or_inconsistent_plan_keeps_actual_paper_state_dry(paper_run, changes):
    cfg = paper_run
    assert execution.main() == 0
    _set_json(cfg.part8_instructions_path, **changes)
    assert bot.run_daily(cfg) == 0
    assert _state(cfg)["mode"] == "DRY_RUN"
    assert _state(cfg)["cash"] == 1000.0
    assert _state(cfg)["paper_execution_plan_cleared"] is False
    assert _trades(cfg).empty


def test_signal_publication_can_veto_an_approved_plan(paper_run):
    cfg = paper_run
    assert execution.main() == 0
    _set_json(cfg.part7_current_target_path, final_pass="false")
    assert bot.run_daily(cfg) == 0
    assert _trades(cfg).empty
    assert _state(cfg)["mode"] == "DRY_RUN"
    assert "Signal publication" in _state(cfg)["paper_execution_reason"]


@pytest.mark.parametrize("changes", [
    {"n_live_realized": 59},
    {"n_live_realized": 60.9},
    {"health_status": "IMMATURE"},
    {"evidence_cohort": "legacy"},
    {"classification_stats_live": {"t_stat_auc": 2.5, "inference_eligible": True}},
    {"calibration_live": {"ece": None}},
    {"calibration_live": {"ece": -0.01}},
])
def test_favorable_plan_does_not_replace_live_accuracy_evidence(paper_run, changes):
    cfg = paper_run
    assert execution.main() == 0
    _set_json(cfg.part9_report_path, **changes)
    assert bot.run_daily(cfg) == 0
    assert _state(cfg)["paper_execution_plan_cleared"] is True
    assert _state(cfg)["mode"] == "DRY_RUN"
    assert not bool(_signal(cfg)["accuracy_gate_passed"])
    assert _trades(cfg).empty


def test_publication_rejects_trade_or_active_signal_in_a_vetoed_session(paper_run):
    cfg = paper_run
    _set_json(Path(cfg.root_dir) / "artifacts_part3_v1/part3_summary.json",
              publish_mode="FAIL_CLOSED_NEUTRAL", final_pass=0)
    assert execution.main() == 0
    assert bot.run_daily(cfg) == 0
    assert validate_execution_lineage(cfg.root_dir) == []

    trade_log = Path(cfg.bot_dir) / "trade_log.csv"
    pd.DataFrame([{"decision_date": SESSION, "nav_after": 1999.94}]).to_csv(trade_log, index=False)
    signals = Path(cfg.bot_dir) / "signal_log.csv"
    frame = pd.read_csv(signals)
    frame["dry_run"] = False
    frame.to_csv(signals, index=False)
    failures = validate_execution_lineage(cfg.root_dir)
    assert "Part 10 recorded a same-session trade despite the Part 8 no-action veto" in failures
    assert "Part 10 signal did not honor the Part 8 no-action veto" in failures


def test_rebalance_accounts_for_sales_and_drifted_holdings():
    cfg = bot.BotConfig()
    portfolio = bot.PaperPortfolio(cfg)
    portfolio.initialize_cash_only()
    portfolio.rebalance(0.75, 100.0, 100.0, SESSION, 0.15, "test initial buy")
    before = portfolio.nav(120.0, 95.0)
    record = portfolio.rebalance(0.45, 120.0, 95.0, "2026-09-14", 0.23, "test rebalance")
    assert record["trade_voo_shares"] < 0 < record["trade_ief_shares"]
    assert portfolio.cash == pytest.approx(0.0, abs=1e-8)
    assert portfolio.weights(120.0, 95.0) == pytest.approx((0.45, 0.55))
    assert portfolio.nav(120.0, 95.0) == pytest.approx(before - record["tc_dollars"], abs=1e-6)


def test_sparse_trade_events_do_not_produce_daily_sharpe():
    tape = pd.DataFrame({
        "decision_date": ["2026-01-05", "2026-01-06", "2026-02-09"],
        "nav_after": [1000.0, 1004.0, 1012.0],
    })
    report = bot.compute_performance(tape, bot.BotConfig())
    assert report["sharpe_ratio"] is None
    assert report["sharpe_status"] == "withheld_no_daily_paper_nav"
    assert report["total_return_pct"] == pytest.approx(1.2)
