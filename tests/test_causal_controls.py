import numpy as np
import pandas as pd
import pytest


def test_causal_base_rate_uses_only_supplied_history():
    from part2_predictor import _causal_base_rate

    train = pd.DataFrame({"y_rel_tail_voo_vs_ief": [0, 0, 1, 0]})
    val = pd.DataFrame({"y_rel_tail_voo_vs_ief": [1, 0]})
    assert np.isclose(_causal_base_rate(train, val), 2 / 6)


def test_row_level_tail_threshold_is_propagated():
    from part2_predictor import _tail_threshold_for_row

    assert np.isclose(_tail_threshold_for_row(pd.Series({"tail_threshold_dynamic": -0.0123})), -0.0123)
    with np.testing.assert_raises_regex(RuntimeError, "tail_threshold_dynamic"):
        _tail_threshold_for_row(pd.Series({}))


def test_historical_evidence_requires_significance_and_material_brier_skill():
    from part2_predictor import Part2Gen53Config, _historical_evidence_gate

    cfg = Part2Gen53Config()
    assert _historical_evidence_gate(0.56, 0.04, 0.02, cfg)
    assert not _historical_evidence_gate(0.56, 0.20, 0.02, cfg)
    assert not _historical_evidence_gate(0.56, 0.04, 0.001, cfg)


def test_execution_weights_are_lagged_one_row():
    from part2_predictor import _lag_execution_weights

    signal = pd.Series([0.60, 0.45, 0.70])
    executed = _lag_execution_weights(signal, 0.60)
    np.testing.assert_allclose(executed.values, [0.60, 0.60, 0.45])


def test_majority_class_forecast_has_balanced_accuracy_half():
    from part9_live_attribution import t_stat_sign_accuracy

    y = np.array([0] * 8 + [1] * 2)
    p = np.repeat(0.10, len(y))
    stats = t_stat_sign_accuracy(y, p, base_rate=0.20)
    assert np.isclose(stats["accuracy"], 0.8)
    assert np.isclose(stats["balanced_accuracy"], 0.5)
    assert np.isclose(stats["matthews_corrcoef"], 0.0)
    assert stats["inference_eligible"] is False
    assert stats["better_than_null_auc_5pct"] is False
    assert stats["worse_than_null_auc_5pct"] is False


def test_rowwise_base_rate_controls_brier_null():
    from part9_live_attribution import t_stat_sign_accuracy

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    p = np.repeat(0.4, len(y))
    causal_base = np.linspace(0.1, 0.5, len(y))
    stats = t_stat_sign_accuracy(y, p, causal_base)
    assert np.isclose(stats["brier_null"], np.mean((y - causal_base) ** 2))


def test_current_protocol_base_rates_must_be_complete_and_bounded():
    from part9_live_attribution import _require_causal_base_rates

    valid = pd.DataFrame({"base_rate": [0.18, 0.21, 0.24]}, index=[4, 5, 6])
    np.testing.assert_allclose(
        _require_causal_base_rates(valid, context="test cohort"),
        [0.18, 0.21, 0.24],
    )

    with pytest.raises(RuntimeError, match="missing.*base_rate"):
        _require_causal_base_rates(pd.DataFrame({"p": [0.2]}), context="test cohort")
    with pytest.raises(RuntimeError, match=r"rows \[5\]"):
        _require_causal_base_rates(
            pd.DataFrame({"base_rate": [0.18, np.nan]}, index=[4, 5]),
            context="test cohort",
        )
    with pytest.raises(RuntimeError, match="invalid.*base_rate"):
        _require_causal_base_rates(
            pd.DataFrame({"base_rate": [1.01]}),
            context="test cohort",
        )


def test_live_report_handles_an_empty_current_evidence_cohort(tmp_path):
    from artifact_integrity import PROTOCOL_VERSION
    from part9_live_attribution import Part9Config, generate_live_report

    prediction_log = tmp_path / "prediction_log.csv"
    pd.DataFrame([{
        "decision_date": "2026-09-21",
        "model_protocol_version": PROTOCOL_VERSION,
        "evidence_eligible": 1,
        "px_voo_realized": np.nan,
        "px_ief_realized": np.nan,
        "px_voo_t": 600.0,
        "px_ief_t": 100.0,
        "p_final_cal": 0.20,
        "base_rate": 0.19,
        "tail_threshold": -0.01,
    }]).to_csv(prediction_log, index=False)
    cfg = Part9Config(
        predlog_path=str(prediction_log),
        part2_tape_path=str(tmp_path / "missing-tape.csv"),
        part6_dir=str(tmp_path / "missing-part6"),
        out_dir=str(tmp_path / "part9"),
        part8_cost_path=str(tmp_path / "missing-costs.csv"),
        part1_dir=str(tmp_path / "missing-part1"),
    )

    report = generate_live_report(cfg)

    assert report["n_live_realized"] == 0
    assert report["health_status"] == "IMMATURE"
    assert "classification_stats_live" not in report


def test_live_tail_labels_match_part1_log_returns(tmp_path, monkeypatch):
    from artifact_integrity import PROTOCOL_VERSION
    import part9_live_attribution as part9

    # Simple-return spread is -0.0100 on the first row, but the Part 1
    # log-return spread is about -0.009852: only the second row is an event.
    prediction_log = tmp_path / "prediction_log.csv"
    pd.DataFrame([
        {
            "decision_date": "2026-09-21",
            "model_protocol_version": PROTOCOL_VERSION,
            "evidence_eligible": 1,
            "px_voo_t": 100.0, "px_ief_t": 100.0,
            "px_voo_realized": 101.0, "px_ief_realized": 102.0,
            "p_final_cal": 0.20, "base_rate": 0.20,
            "tail_threshold": -0.0099,
        },
        {
            "decision_date": "2026-09-22",
            "model_protocol_version": PROTOCOL_VERSION,
            "evidence_eligible": 1,
            "px_voo_t": 100.0, "px_ief_t": 100.0,
            "px_voo_realized": 99.0, "px_ief_realized": 100.0,
            "p_final_cal": 0.80, "base_rate": 0.20,
            "tail_threshold": -0.0099,
        },
    ]).to_csv(prediction_log, index=False)
    original_stats = part9.t_stat_sign_accuracy
    monkeypatch.setattr(
        part9, "t_stat_sign_accuracy",
        lambda *args, **kwargs: original_stats(*args, n_perm=20, **kwargs),
    )
    cfg = part9.Part9Config(
        predlog_path=str(prediction_log),
        part2_tape_path=str(tmp_path / "missing-tape.csv"),
        part6_dir=str(tmp_path / "missing-part6"),
        out_dir=str(tmp_path / "part9"),
        part8_cost_path=str(tmp_path / "missing-costs.csv"),
        part1_dir=str(tmp_path / "missing-part1"),
    )

    report = part9.generate_live_report(cfg)

    assert report["n_live_realized"] == 2
    assert report["classification_stats_live"]["n_positive"] == 1
    assert report["classification_stats_live"]["brier"] == pytest.approx(0.04)
    assert report["classification_stats_live"]["inference_eligible"] is False


def test_live_log_spread_fails_closed_on_invalid_prices():
    from part9_live_attribution import _realized_log_spread

    frame = pd.DataFrame({
        "px_voo_t": [100.0, 0.0], "px_ief_t": [100.0, 100.0],
        "px_voo_realized": [101.0, 101.0], "px_ief_realized": [102.0, 102.0],
    }, index=[7, 8])
    with pytest.raises(RuntimeError, match=r"rows \[8\]"):
        _realized_log_spread(frame, "px_voo_realized", "px_ief_realized")


def test_backfill_spread_diagnostic_uses_log_returns():
    from backfill_realized import _log_return_spread

    spread = _log_return_spread(101.0, 102.0, 100.0, 100.0)
    assert spread == pytest.approx(np.log(1.01) - np.log(1.02))
    assert spread > -0.0099  # simple-return spread would be -0.0100
    assert np.isnan(_log_return_spread(101.0, 102.0, 0.0, 100.0))


def test_regime_platt_auc_validation_failure_excludes_regime(tmp_path, monkeypatch):
    import sklearn.metrics

    import part3_governance

    if not part3_governance.HAVE_PLATT:
        pytest.skip("optional Platt dependencies are unavailable")

    dates = pd.date_range("2025-01-02", periods=120, freq="B")
    labels = (np.arange(len(dates)) % 2).astype(float)
    defense = pd.DataFrame({
        "Date": dates,
        "p_final_cal": np.where(labels == 1.0, 0.65, 0.35),
    })
    label_frame = pd.DataFrame(
        {"y_rel_tail_voo_vs_ief": labels},
        index=dates,
    )
    regime_frame = pd.DataFrame({"regime_label": "risk_on"}, index=dates)
    label_path = tmp_path / "labels.parquet"
    regime_path = tmp_path / "regimes.parquet"
    label_path.touch()
    regime_path.touch()

    def fake_read_parquet(path, *args, **kwargs):
        del args, kwargs
        return label_frame.copy() if path == label_path else regime_frame.copy()

    def fail_auc(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("forced AUC validation failure")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(sklearn.metrics, "roc_auc_score", fail_auc)

    params = part3_governance._fit_regime_platt_scaling(
        defense,
        label_path,
        regime_path,
    )

    assert "risk_on" not in params
    assert "_global" in params


def test_one_positive_never_creates_auc_significance():
    from part9_live_attribution import Part9Config, evaluate_stopping_rules, t_stat_sign_accuracy

    y = np.array([1] + [0] * 27, dtype=float)
    p = np.linspace(0.01, 0.99, len(y))
    stats = t_stat_sign_accuracy(y, p, np.repeat(0.2, len(y)))
    assert stats["n_positive"] == 1
    assert stats["inference_status"] == "insufficient_class_counts"
    assert np.isnan(stats["p_value_auc_better"])
    assert not any(
        stats[key]
        for key in stats
        if key.startswith("better_than_null") or key.startswith("worse_than_null")
    )
    health = evaluate_stopping_rules(stats, {}, pd.DataFrame(), Part9Config(min_live_n=20))
    assert health["status"] == "IMMATURE"
    assert "Class-count gate" in health["reasons"][0]


def test_latest_completed_close_uses_last_available_row():
    from part10_tradingbot import _latest_completed_close

    s = pd.Series([100.0, 101.0, 102.0])
    assert _latest_completed_close(s) == 102.0


def test_current_evidence_counter_excludes_legacy_rows():
    from artifact_integrity import PROTOCOL_VERSION
    from part3_governance import _count_realized_predlog_rows

    frame = pd.DataFrame({
        "px_voo_realized": [100.0, 101.0, 102.0],
        "px_ief_realized": [90.0, 91.0, 92.0],
        "model_protocol_version": ["legacy", PROTOCOL_VERSION, PROTOCOL_VERSION],
        "evidence_eligible": [1, 0, 1],
    })
    assert _count_realized_predlog_rows(frame) == 1


@pytest.mark.parametrize("existing_eligible", [0, 1])
def test_prediction_upsert_replaces_stale_numeric_run_provenance(
    tmp_path, monkeypatch, existing_eligible
):
    from artifact_integrity import PROTOCOL_VERSION
    from part3_governance import _upsert_prediction_log

    # Historical ledger fixtures exercise upsert behavior, not issuance on the
    # production branch. Keep the real production deadline guard in place.
    monkeypatch.setenv("GITHUB_REF_NAME", "codex/test-fixture")
    path = tmp_path / "prediction_log.csv"
    pd.DataFrame([{
        "decision_date": "2026-09-04",
        "model_protocol_version": PROTOCOL_VERSION,
        "pipeline_run_id": 33827957626,
        "model_code_sha": "old",
        "px_voo_realized": 100.0,
        "px_ief_realized": 90.0,
        "evidence_eligible": existing_eligible,
    }]).to_csv(path, index=False)
    monkeypatch.setenv("PRICECALL_CODE_SHA", "new-code-sha")
    monkeypatch.setenv("GITHUB_RUN_ID", "33890824408")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    alpha_status = {
        "latest_state": "SHADOW",
        "alpha_live": 0,
        "current_alpha_live_status": "SHADOW",
        "current_alpha_reason": "not_eligible",
        "current_alpha_eligible": 0,
        "current_alpha_abs": 0.0,
    }
    alpha_sources = {
        "positions": tmp_path / "positions.csv",
        "summary_tape": tmp_path / "summary.csv",
        "eligibility": tmp_path / "eligibility.csv",
        "summary_json": tmp_path / "summary.json",
    }
    if existing_eligible:
        original = path.read_bytes()
    args = (
        path,
        pd.Timestamp("2026-09-04"),
        pd.Timestamp("2026-09-08"),
        101.0,
        91.0,
        "FAIL_CLOSED_NEUTRAL",
        0,
        alpha_status,
        tmp_path / "defense.csv",
        alpha_sources,
        pd.Series({
            "tail_threshold_dynamic": -0.01,
            "px_voo_t": 100.0,
            "px_ief_t": 90.0,
            "p_final_cal": 0.2,
            "base_rate": 0.2,
        }),
        {
            "part1_data_freshness_ok": True,
            "macro_point_in_time_ok": True,
            "tail_event_definition": "rowwise_trailing_63_observation_20th_percentile_shifted_1",
        },
    )
    if existing_eligible:
        with pytest.raises(RuntimeError, match="eligible paper forecast is immutable"):
            _upsert_prediction_log(*args)
        assert path.read_bytes() == original
        return
    frame, _ = _upsert_prediction_log(*args)
    row = frame.loc[frame["model_protocol_version"].eq(PROTOCOL_VERSION)].iloc[-1]
    assert row["pipeline_run_id"] == "33890824408"
    assert row["model_code_sha"] == "new-code-sha"
    assert row["prediction_revision_id"].endswith(":33890824408:2")
    assert row["provenance_complete"] == 1
    assert row["px_voo_realized"] == 100.0


def test_prediction_upsert_appends_new_date_without_mixed_date_types(tmp_path, monkeypatch):
    from artifact_integrity import PROTOCOL_VERSION
    from part3_governance import _upsert_prediction_log

    monkeypatch.setenv("GITHUB_REF_NAME", "codex/test-fixture")
    path = tmp_path / "prediction_log.csv"
    pd.DataFrame([{
        "decision_date": "2026-09-04 00:00:00",
        "target_date": "2026-09-08T00:00:00",
        "model_protocol_version": PROTOCOL_VERSION,
        "evidence_eligible": 0,
    }]).to_csv(path, index=False)
    monkeypatch.setenv("PRICECALL_CODE_SHA", "new-code-sha")
    monkeypatch.setenv("GITHUB_RUN_ID", "34539046830")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    alpha_status = {
        "latest_state": "SHADOW",
        "alpha_live": 0,
        "current_alpha_live_status": "SHADOW",
        "current_alpha_reason": "not_eligible",
        "current_alpha_eligible": 0,
        "current_alpha_abs": 0.0,
    }
    alpha_sources = {
        "positions": tmp_path / "positions.csv",
        "summary_tape": tmp_path / "summary.csv",
        "eligibility": tmp_path / "eligibility.csv",
        "summary_json": tmp_path / "summary.json",
    }

    frame, _ = _upsert_prediction_log(
        path,
        pd.Timestamp("2026-09-10"),
        pd.Timestamp("2026-09-11"),
        101.0,
        91.0,
        "FAIL_CLOSED_NEUTRAL",
        0,
        alpha_status,
        tmp_path / "defense.csv",
        alpha_sources,
        pd.Series({
            "tail_threshold_dynamic": -0.01,
            "px_voo_t": 100.0,
            "px_ief_t": 90.0,
            "p_final_cal": 0.2,
            "base_rate": 0.2,
        }),
        {
            "part1_data_freshness_ok": True,
            "macro_point_in_time_ok": True,
            "tail_event_definition": "rowwise_trailing_63_observation_20th_percentile_shifted_1",
        },
    )

    assert frame["decision_date"].tolist() == ["2026-09-04", "2026-09-10"]
    assert frame["target_date"].tolist() == ["2026-09-08", "2026-09-11"]
    assert frame["decision_date"].map(type).eq(str).all()


def test_missing_alpha_gate_fields_default_closed_and_are_namespaced():
    from part3_governance import _build_governance_df, _load_alpha_status

    status = _load_alpha_status(pd.DataFrame(), {}, live_realized_rows=0)
    assert status["trial_gate_open"] == 0
    assert status["fused_gate_open"] == 0
    assert status["promotion_ready"] == 0
    frame = _build_governance_df(pd.Timestamp("2026-09-04"), {}, status)
    assert "alpha_trial_gate_open" in frame.columns
    assert "alpha_fused_gate_open" in frame.columns
    assert "alpha_promotion_ready" in frame.columns
    assert "promotion_ready" not in frame.columns


def test_shared_evidence_mask_excludes_prior_protocol_and_ineligible_rows():
    from artifact_integrity import PROTOCOL_VERSION, current_evidence_mask

    frame = pd.DataFrame({
        "model_protocol_version": ["causal-integrity-v3", PROTOCOL_VERSION, PROTOCOL_VERSION],
        "evidence_eligible": [1, 0, 1],
        "px_voo_realized": [100.0, 101.0, 102.0],
        "px_ief_realized": [90.0, 91.0, 92.0],
    })
    assert current_evidence_mask(frame).tolist() == [False, False, True]
    assert current_evidence_mask(frame, require_realized=True).sum() == 1


def test_non_vintage_macro_history_forces_fail_closed():
    from part2_predictor import Part2Gen53Config, _should_fail_closed

    summary = {
        "historical_evidence_ok": True,
        "part1_data_freshness_ok": True,
        "macro_point_in_time_ok": False,
        "suspicious_perf_flag": False,
        "drift_alarm_rate": 0.0,
        "calibration_gate_on_rate": 1.0,
        "conditional_active_ir_tmean": np.nan,
    }
    assert _should_fail_closed(summary, Part2Gen53Config()) is True


def test_new_macro_method_stays_paper_only_until_independent_validation():
    from part2_predictor import Part2Gen53Config, _should_fail_closed

    summary = {
        "historical_evidence_ok": True,
        "independent_validation_ok": False,
        "part1_data_freshness_ok": True,
        "macro_point_in_time_ok": True,
        "suspicious_perf_flag": False,
        "drift_alarm_rate": 0.0,
        "calibration_gate_on_rate": 1.0,
        "conditional_active_ir_tmean": np.nan,
    }
    assert _should_fail_closed(summary, Part2Gen53Config()) is True


def test_only_prospective_main_branch_forecasts_are_eligible(monkeypatch):
    from datetime import datetime, timezone

    from part3_governance import _prospective_evidence_status

    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    issued, eligible = _prospective_evidence_status(
        "2026-09-25", now=datetime(2026, 9, 24, 21, tzinfo=timezone.utc)
    )
    assert issued == "2026-09-24T21:00:00+00:00"
    assert eligible is True
    with pytest.raises(RuntimeError, match="target session closed"):
        _prospective_evidence_status(
            "2026-09-25", now=datetime(2026, 9, 25, 21, tzinfo=timezone.utc)
        )
    monkeypatch.setenv("GITHUB_REF_NAME", "codex/pit-macro-replay-v1")
    assert not _prospective_evidence_status(
        "2026-09-25", now=datetime(2026, 9, 24, 21, tzinfo=timezone.utc)
    )[1]
