from pathlib import Path

import pandas as pd


def test_cboe_vix3m_parser_normalizes_and_deduplicates():
    from market_data_integrity import parse_cboe_vix3m_csv

    result = parse_cboe_vix3m_csv(
        b"DATE,OPEN,HIGH,LOW,CLOSE\n09/02/2026,18,19,17,17.73\n"
        b"09/03/2026,17,18,16,17.40\n09/03/2026,17,18,16,17.42\n"
    )
    assert result.index.tolist() == [pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03")]
    assert result.iloc[-1] == 17.42


def test_market_integrity_runs_before_point_in_time_features():
    text = Path("run_tuesday_prediction.py").read_text(encoding="utf-8")
    order = text.split("DIRECT_PIPELINE_ORDER: List[str] = [", 1)[1].split("]", 1)[0]
    assert order.index('"MARKET_INTEGRITY"') < order.index('"PIT_MACRO"')


def test_next_xnys_session_skips_labor_day():
    from market_calendar import next_xnys_session

    assert next_xnys_session("2026-09-04") == pd.Timestamp("2026-09-08")
