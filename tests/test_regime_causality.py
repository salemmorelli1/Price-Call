import numpy as np
import pandas as pd

from part6_regime_engine import RegimeEngine, Part6Config


def test_regime_fit_does_not_backward_fill(monkeypatch):
    cfg = Part6Config(
        n_regimes=2,
        hmm_min_train_rows=3,
        regime_features=("vix_z21", "spread_vol10", "vix_term_ratio"),
        regime_features_nofed=("vix_z21", "spread_vol10", "vix_term_ratio"),
    )
    frame = pd.DataFrame(
        {
            "vix_z21": [None, None, 1.0, 1.1, 1.2],
            "spread_vol10": [0.1, 0.2, 0.3, 0.4, 0.5],
            "vix_term_ratio": [0.9, 0.95, 1.0, 1.05, 1.1],
        },
        index=pd.date_range("2020-01-01", periods=5),
    )
    engine = RegimeEngine(cfg)
    selected = engine._select_features(frame)
    filled = frame[selected].apply(pd.to_numeric, errors="coerce").ffill()
    assert pd.isna(filled.iloc[0]["vix_z21"])
    assert pd.isna(filled.iloc[1]["vix_z21"])


def test_regime_train_end_tracks_actual_refit_window(monkeypatch):
    import part6_regime_engine as module

    class FakeEngine:
        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, frame):
            self.feature_cols = list(frame.columns)

        def predict(self, frame):
            return pd.DataFrame({
                "regime_label": ["calm"] * len(frame),
                "regime_id": [0] * len(frame),
                "regime_prob_calm": [1.0] * len(frame),
                "regime_prob_risk_on": [0.0] * len(frame),
                "regime_prob_high_vol": [0.0] * len(frame),
                "regime_prob_crisis": [0.0] * len(frame),
                "regime_persistence": [1.0] * len(frame),
                "transition_prob_crisis": [0.0] * len(frame),
            }, index=frame.index)

    monkeypatch.setattr(module, "RegimeEngine", FakeEngine)
    idx = pd.date_range("2026-01-01", periods=8, freq="B")
    frame = pd.DataFrame({"a": np.arange(8), "b": np.arange(8), "c": np.arange(8)}, index=idx)
    cfg = Part6Config(hmm_min_train_rows=3, causal_refit_frequency=3, causal_train_window_rows=10)
    result, _, _ = module.build_causal_regime_history(frame, cfg)
    assert result.loc[idx[3], "regime_model_train_end"] == idx[2]
    assert result.loc[idx[4], "regime_model_train_end"] == idx[2]
    assert result.loc[idx[5], "regime_model_train_end"] == idx[2]
    assert result.loc[idx[6], "regime_model_train_end"] == idx[5]
