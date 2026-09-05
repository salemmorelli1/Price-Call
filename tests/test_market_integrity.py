from pathlib import Path

import pandas as pd
import pytest


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


def test_latest_completed_session_observes_close_delay_and_holiday():
    from market_calendar import latest_completed_xnys_session

    assert latest_completed_xnys_session("2026-09-04 11:42-04:00") == pd.Timestamp("2026-09-03")
    assert latest_completed_xnys_session("2026-09-04 16:19-04:00") == pd.Timestamp("2026-09-03")
    assert latest_completed_xnys_session("2026-09-04 16:20-04:00") == pd.Timestamp("2026-09-04")
    assert latest_completed_xnys_session("2026-09-07 18:00-04:00") == pd.Timestamp("2026-09-04")


def test_completed_session_calendar_excludes_holidays_and_unsettled_rows():
    from market_calendar import completed_xnys_sessions

    sessions = completed_xnys_sessions(
        "2026-08-28", "2026-09-08", now="2026-09-08 10:00-04:00"
    )
    assert pd.Timestamp("2026-09-07") not in sessions
    assert pd.Timestamp("2026-09-08") not in sessions
    assert sessions.max() == pd.Timestamp("2026-09-04")


def test_completed_session_calendar_covers_configured_2005_history():
    from market_calendar import completed_xnys_sessions

    sessions = completed_xnys_sessions(
        "2005-01-01", "2005-01-07", now="2026-09-05 02:45-04:00"
    )
    assert sessions.tolist() == [
        pd.Timestamp("2005-01-03"),
        pd.Timestamp("2005-01-04"),
        pd.Timestamp("2005-01-05"),
        pd.Timestamp("2005-01-06"),
        pd.Timestamp("2005-01-07"),
    ]


def test_part0_market_panel_does_not_forward_fill_source_closes():
    text = Path("part0_data_infrastructure.py").read_text(encoding="utf-8")
    market_download = text.split("def download_market_data", 1)[1].split(
        "def download_fred_data", 1
    )[0]
    assert "close = close.ffill" not in market_download
    assert "market_observation_mask.parquet" in text
    assert "fred_api_key:" not in text
    assert 'os.environ.get("FRED_API_KEY", "").strip()' in text


def test_part0_rejects_missing_core_close_instead_of_synthesizing_it(monkeypatch):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(pd.to_datetime(["2026-09-03", "2026-09-04"]), name="Date")
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close", "Volume"]])
    raw = pd.DataFrame(
        [[100.0, 10.0, 90.0, 9.0], [float("nan"), 11.0, 91.0, 10.0]],
        index=sessions,
        columns=columns,
    )
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", fake_download)
    cfg = part0.Part0Config(
        start="2026-09-03",
        end="2026-09-04",
        equity_tickers=("VOO", "IEF"),
        vix_tickers=(),
        core_tickers=("VOO", "IEF"),
        min_history_years=0.0,
    )
    with pytest.raises(RuntimeError, match="core tickers still have NaN"):
        part0.download_market_data(cfg)
    assert calls[0]["end"] == "2026-09-05"


def test_completed_session_input_validator_rejects_non_session_row(tmp_path, monkeypatch):
    from artifact_integrity import validate_completed_session_inputs, write_json_strict
    import market_calendar

    output = tmp_path / "artifacts_part0"
    output.mkdir()
    close_path = output / "close_prices.parquet"
    mask_path = output / "market_observation_mask.parquet"
    close_path.touch()
    mask_path.touch()
    write_json_strict(output / "part0_meta.json", {
        "market_calendar": "XNYS",
        "market_values_are_raw_observations": True,
        "market_data_asof": "2026-09-07",
    })
    close = pd.DataFrame(
        {"VOO": [100.0, 101.0], "IEF": [90.0, 91.0]},
        index=pd.to_datetime(["2026-09-04", "2026-09-07"]),
    )
    observed = close.notna().astype("uint8")
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: observed.copy() if Path(path).name == mask_path.name else close.copy(),
    )
    monkeypatch.setattr(
        market_calendar,
        "latest_completed_xnys_session",
        lambda: pd.Timestamp("2026-09-04"),
    )
    monkeypatch.setattr(
        market_calendar,
        "completed_xnys_sessions",
        lambda start, end: pd.DatetimeIndex([pd.Timestamp("2026-09-04")]),
    )
    failures = validate_completed_session_inputs(tmp_path)
    assert any("non-XNYS or uncompleted" in failure for failure in failures)
