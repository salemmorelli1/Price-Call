import numpy as np
import pandas as pd


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


def test_shared_evidence_mask_excludes_prior_protocol_and_ineligible_rows():
    from artifact_integrity import PROTOCOL_VERSION, current_evidence_mask

    frame = pd.DataFrame({
        "model_protocol_version": ["causal-integrity-v2", PROTOCOL_VERSION, PROTOCOL_VERSION],
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
