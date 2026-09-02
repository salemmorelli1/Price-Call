import numpy as np
import pandas as pd


def test_causal_base_rate_uses_only_supplied_history():
    from part2_predictor import _causal_base_rate

    train = pd.DataFrame({"y_rel_tail_voo_vs_ief": [0, 0, 1, 0]})
    val = pd.DataFrame({"y_rel_tail_voo_vs_ief": [1, 0]})
    assert np.isclose(_causal_base_rate(train, val), 2 / 6)


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


def test_latest_completed_close_uses_last_available_row():
    from part10_tradingbot import _latest_completed_close

    s = pd.Series([100.0, 101.0, 102.0])
    assert _latest_completed_close(s) == 102.0
