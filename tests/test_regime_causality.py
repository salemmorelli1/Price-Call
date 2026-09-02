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
