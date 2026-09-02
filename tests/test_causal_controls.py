import numpy as np
import pandas as pd

from part2_predictor import _causal_base_rate, _lag_execution_weights
from part9_live_attribution import t_stat_sign_accuracy
from part10_tradingbot import _latest_completed_close


def test_causal_base_rate_uses_only_supplied_history():
    train = pd.DataFrame({"y_rel_tail_voo_vs_ief": [0, 0, 1, 0]})
    val = pd.DataFrame({"y_rel_tail_voo_vs_ief": [1, 0]})
    assert _causal_base_rate(train, val) == 2 / 6


def test_execution_weights_are_lagged_one_row():
    signal = pd.Series([0.60, 0.45, 0.70])
    executed = _lag_execution_weights(signal, 0.60)
    np.testing.assert_allclose(executed.values, [0.60, 0.60, 0.45])


def test_majority_class_forecast_has_balanced_accuracy_half():
    y = np.array([0] * 8 + [1] * 2)
    p = np.repeat(0.10, len(y))
    stats = t_stat_sign_accuracy(y, p, base_rate=0.20)
    assert stats["accuracy"] == 0.8
    assert stats["balanced_accuracy"] == 0.5
    assert stats["matthews_corrcoef"] == 0.0


def test_latest_completed_close_uses_last_available_row():
    s = pd.Series([100.0, 101.0, 102.0])
    assert _latest_completed_close(s) == 102.0
