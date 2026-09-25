import csv
import json
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

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", fake_download)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    cfg = part0.Part0Config(
        start="2026-09-03",
        end="2026-09-04",
        equity_tickers=("VOO", "IEF"),
        vix_tickers=(),
        core_tickers=("VOO", "IEF"),
        min_history_years=0.0,
    )
    with pytest.raises(RuntimeError, match="core tickers still have NaN") as error:
        part0.download_market_data(cfg)
    assert calls[0][1]["end"] == "2026-09-05"
    assert len(calls) == 5
    assert all(call[1].get("threads") is False for call in calls[1:4])
    assert calls[-1][0] == (["VOO", "IEF"],)
    assert "2026-09-04" in str(error.value)


def test_part0_recovers_partial_core_gap_from_raw_individual_retry(monkeypatch):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2026-09-03", "2026-09-04"]), name="Date"
    )
    bulk_columns = pd.MultiIndex.from_product(
        [["VOO", "IEF"], ["Close", "Volume"]]
    )
    bulk = pd.DataFrame(
        [[100.0, 10.0, 90.0, 9.0], [float("nan"), 11.0, 91.0, 10.0]],
        index=sessions,
        columns=bulk_columns,
    )
    # Exercise the alternate field-first MultiIndex returned by some one-ticker
    # yfinance calls.
    retry_columns = pd.MultiIndex.from_product([["Close", "Volume"], ["VOO"]])
    retry = pd.DataFrame(
        [[100.0, 10.0], [101.0, 11.0]],
        index=sessions,
        columns=retry_columns,
    )
    responses = iter([bulk, retry])
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", fake_download)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    cfg = part0.Part0Config(
        start="2026-09-03",
        end="2026-09-04",
        equity_tickers=("VOO", "IEF"),
        vix_tickers=(),
        core_tickers=("VOO", "IEF"),
        min_history_years=0.0,
    )

    close, _, quality = part0.download_market_data(cfg)

    assert close.loc[pd.Timestamp("2026-09-04"), "VOO"] == 101.0
    assert not close[["VOO", "IEF"]].isna().any().any()
    assert calls[1]["tickers"] == ["VOO"]
    assert calls[1]["threads"] is False
    assert quality["VOO"]["individual_retry_recovered_rows"] == 1


def test_part0_recovers_core_closes_with_short_paired_request(monkeypatch):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"]), name="Date"
    )
    bulk_columns = pd.MultiIndex.from_product(
        [["VOO", "IEF"], ["Close", "Volume"]]
    )
    bulk = pd.DataFrame(
        [[None, 10.0, 90.0, 9.0], [None] * 4, [None] * 4],
        index=sessions,
        columns=bulk_columns,
    )
    paired_columns = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["VOO", "IEF"]]
    )
    paired = pd.DataFrame(
        [[100.0, 90.0, 10.0, 9.0],
         [101.0, 91.0, 11.0, 10.0],
         [102.0, 92.0, 12.0, 11.0]],
        index=sessions,
        columns=paired_columns,
    )
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return paired if len(calls) == 8 else bulk

    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", fake_download)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    cfg = part0.Part0Config(
        start="2026-09-21",
        end="2026-09-23",
        equity_tickers=("VOO", "IEF"),
        vix_tickers=(),
        core_tickers=("VOO", "IEF"),
        min_history_years=0.0,
    )

    close, volume, quality = part0.download_market_data(cfg)

    assert len(calls) == 8  # bulk, three per ticker, then one paired request
    assert calls[-1][0] == (["VOO", "IEF"],)
    assert calls[-1][1]["start"] == "2026-09-21"
    assert calls[-1][1]["end"] == "2026-09-24"
    assert close.loc[pd.Timestamp("2026-09-23"), ["VOO", "IEF"]].tolist() == [102.0, 92.0]
    assert volume.loc[pd.Timestamp("2026-09-23"), ["VOO", "IEF"]].tolist() == [12.0, 11.0]
    assert quality["VOO"]["paired_retry_recovered_rows"] == 3
    assert quality["IEF"]["paired_retry_recovered_rows"] == 2
    assert quality["VOO"]["paired_retry_recovered_dates"] == [
        "2026-09-21", "2026-09-22", "2026-09-23"
    ]
    assert quality["VOO"]["first_valid_date"] == "2026-09-21"
    assert quality["VOO"]["usable_for_model"] is True


def test_part0_paired_retry_rejects_conflicting_raw_prices(monkeypatch):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2026-09-21", "2026-09-22"]), name="Date"
    )
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close"]])
    bulk = pd.DataFrame([[100.0, 90.0], [None, None]], index=sessions, columns=columns)
    paired = pd.DataFrame([[105.0, 90.0], [101.0, 91.0]], index=sessions, columns=columns)
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return paired if len(calls) == 8 else bulk

    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", fake_download)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    cfg = part0.Part0Config(
        start="2026-09-21",
        end="2026-09-22",
        equity_tickers=("VOO", "IEF"),
        vix_tickers=(),
        core_tickers=("VOO", "IEF"),
        min_history_years=0.0,
    )

    with pytest.raises(RuntimeError, match="disagrees with existing VOO"):
        part0.download_market_data(cfg)


@pytest.mark.parametrize(
    "failure",
    [None, "invalid_manifest", "unverified", "backfill_disagrees",
     "current_session", "latest_missing", "price_scale_mismatch",
     "earlier_verified", "overwritten_backfill", "no_earlier_provenance", "legacy_protocol",
     "pre_inception"],
)
def test_part0_only_recovers_corroborated_historical_core_closes(
    tmp_path, monkeypatch, failure
):
    import artifact_integrity
    import part0_data_infrastructure as part0

    dates = ["2026-09-21", "2026-09-22", "2026-09-23"]
    if failure == "pre_inception":
        dates.insert(0, "2026-09-18")
    sessions = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close"]])
    prices = [[99.0 if failure == "price_scale_mismatch" else 100.0, 90.0],
              [None, None],
              [None if failure == "latest_missing" else 102.0, 92.0]]
    if failure == "pre_inception":
        prices.insert(0, [None, 89.0])
    raw = pd.DataFrame(prices, index=sessions, columns=columns)
    status_dir = tmp_path / "artifacts_part10_bot"
    meta_dir = tmp_path / "artifacts_part0"
    log_dir = tmp_path / "artifacts_part3"
    for directory in (status_dir, meta_dir, log_dir):
        directory.mkdir()
    previous_snapshot = failure in {"earlier_verified", "overwritten_backfill", "no_earlier_provenance"}
    protocol = (
        "causal-integrity-v3" if failure in {"earlier_verified", "overwritten_backfill", "legacy_protocol"}
        else artifact_integrity.PROTOCOL_VERSION
    )
    status = {
        "result": "verified" if failure != "unverified" else "failed",
        "protocol_version": protocol,
        "market_data_asof": "2026-09-23" if previous_snapshot or failure == "current_session" else "2026-09-22",
        "expected_completed_market_session": "2026-09-23" if previous_snapshot or failure == "current_session" else "2026-09-22",
        "github_run_id": "35796548754",
        "github_run_attempt": "1",
        "source_code_sha": "later-commit" if previous_snapshot else "source-commit",
    }
    (status_dir / "pipeline_status.json").write_text(json.dumps(status))
    (meta_dir / "part0_meta.json").write_text(json.dumps({
        "market_data_asof": status["market_data_asof"],
        "market_values_are_raw_observations": True,
        "last_raw_observation_by_ticker": {
            "VOO": status["market_data_asof"],
            "IEF": status["market_data_asof"],
        },
        "data_quality": {
            ticker: {
                "verified_archive_recovered_dates": ["2026-09-22"],
                "verified_archive_source_run_id": "35796548754",
            } for ticker in ("VOO", "IEF")
        } if failure in {"earlier_verified", "overwritten_backfill"} else {},
    }))
    with (log_dir / "prediction_log.csv").open("w", newline="") as handle:
        fields = [
            "decision_date", "target_date", "realized_target_date",
            "px_voo_t", "px_ief_t", "px_voo_realized", "px_ief_realized",
            "pipeline_run_id", "pipeline_run_attempt", "model_code_sha",
            "model_protocol_version",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "decision_date": "2026-09-21", "target_date": "2026-09-22",
            "realized_target_date": (
                "2026-09-23" if failure == "overwritten_backfill" else "2026-09-22"
            ),
            "px_voo_t": 100, "px_ief_t": 90,
            "px_voo_realized": 105 if failure == "backfill_disagrees" else 101,
            "px_ief_realized": 91,
            "model_protocol_version": protocol,
        })
        writer.writerow({
            "decision_date": "2026-09-22", "target_date": "2026-09-23",
            "px_voo_t": 101, "px_ief_t": 91,
            "pipeline_run_id": "35796548754.0", "pipeline_run_attempt": "1.0",
            "model_code_sha": "source-commit",
            "model_protocol_version": protocol,
        })

    monkeypatch.setattr(part0, "_resolve_project_root", lambda cfg: tmp_path)
    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        artifact_integrity, "verify_run_manifest",
        lambda root: ["manifest SHA-256 differs"] if failure == "invalid_manifest" else [],
    )
    cfg = part0.Part0Config(
        start="2026-09-21", end="2026-09-23", equity_tickers=("VOO", "IEF"),
        vix_tickers=(), core_tickers=("VOO", "IEF"), min_history_years=0,
    )
    if failure not in (None, "earlier_verified", "overwritten_backfill", "legacy_protocol", "pre_inception"):
        with pytest.raises(
            RuntimeError,
            match="2026-09-23" if failure == "latest_missing" else "2026-09-22",
        ):
            part0.download_market_data(cfg)
        return

    close, _, quality = part0.download_market_data(cfg)
    assert close.loc[pd.Timestamp("2026-09-22"), ["VOO", "IEF"]].tolist() == [101, 91]
    assert quality["VOO"]["verified_archive_recovered_dates"] == ["2026-09-22"]
    assert quality["IEF"]["verified_archive_source_run_id"] == "35796548754"


@pytest.mark.parametrize("failure", [
    None, "later_session", "replaced_backfill", "shifted_target", "bad_anchor",
    "missing_lineage", "unverified",
    "future_backfill", "stale_production", "invalid_manifest", "conflicting_close",
])
def test_part0_adjacent_close_requires_verified_exact_date_backfill(
    tmp_path, monkeypatch, failure
):
    import artifact_integrity
    import part0_data_infrastructure as part0

    dates = ["2026-09-23", "2026-09-24"]
    if failure in ("later_session", "replaced_backfill"):
        dates.append("2026-09-25")
    sessions = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close"]])
    prices = [[102.0, 92.0], [104.0 if failure == "conflicting_close" else None, None]]
    if failure in ("later_session", "replaced_backfill"):
        prices.append([105.0, 90.0])
    raw = pd.DataFrame(prices, index=sessions, columns=columns)
    for directory in ("artifacts_part10_bot", "artifacts_part0", "artifacts_part9", "artifacts_part3"):
        (tmp_path / directory).mkdir()
    status = {
        "result": "verified", "protocol_version": "causal-integrity-v3",
        "market_data_asof": "2026-09-22" if failure == "stale_production" else "2026-09-23",
        "expected_completed_market_session": "2026-09-23",
        "github_run_id": "35950967494", "github_run_attempt": "1",
        "source_code_sha": "production-sha",
    }
    meta = {
        "market_data_asof": "2026-09-23", "market_values_are_raw_observations": True,
        "last_raw_observation_by_ticker": {"VOO": "2026-09-23", "IEF": "2026-09-23"},
    }
    backfill = {
        "result": "failed" if failure == "unverified" else "verified",
        "protocol_version": artifact_integrity.PROTOCOL_VERSION,
        "backfill_run_date": "2026-09-25" if failure == "replaced_backfill" else "2026-09-24",
        "completed_at_utc": (
            "2027-09-24T21:30:00Z" if failure == "future_backfill"
            else "2026-09-24T21:30:00Z"
        ),
        "github_run_id": "36071978818", "github_run_attempt": "1",
        "source_code_sha": "" if failure == "missing_lineage" else "backfill-sha",
    }
    for path, payload in (
        ("artifacts_part10_bot/pipeline_status.json", status),
        ("artifacts_part0/part0_meta.json", meta),
        ("artifacts_part9/backfill_status.json", backfill),
    ):
        (tmp_path / path).write_text(json.dumps(payload))
    with (tmp_path / "artifacts_part3/prediction_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "decision_date", "target_date", "realized_target_date",
            "px_voo_t", "px_ief_t", "px_voo_realized", "px_ief_realized",
            "pipeline_run_id", "pipeline_run_attempt", "model_code_sha", "model_protocol_version",
        ])
        writer.writeheader()
        writer.writerow({
            "decision_date": "2026-09-23", "target_date": "2026-09-24",
            "realized_target_date": (
                "2026-09-25" if failure == "shifted_target" else "2026-09-24"
            ),
            "px_voo_t": 101 if failure == "bad_anchor" else 102,
            "px_ief_t": 92, "px_voo_realized": 103, "px_ief_realized": 91,
            "pipeline_run_id": "35950967494.0", "pipeline_run_attempt": "1.0",
            "model_code_sha": "production-sha", "model_protocol_version": "causal-integrity-v3",
        })

    monkeypatch.setattr(part0, "_resolve_project_root", lambda cfg: tmp_path)
    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(part0.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        artifact_integrity, "verify_run_manifest",
        lambda root: ["bad manifest"] if failure == "invalid_manifest" else [],
    )
    cfg = part0.Part0Config(
        start="2026-09-23", end=dates[-1], equity_tickers=("VOO", "IEF"),
        vix_tickers=(), core_tickers=("VOO", "IEF"), min_history_years=0,
    )
    if failure not in (None, "later_session"):
        with pytest.raises(RuntimeError, match="2026-09-24"):
            part0.download_market_data(cfg)
        return

    close, _, quality = part0.download_market_data(cfg)
    assert close.loc[pd.Timestamp("2026-09-24"), ["VOO", "IEF"]].tolist() == [103, 91]
    assert quality["VOO"]["verified_backfill_recovered_dates"] == ["2026-09-24"]
    assert quality["IEF"]["verified_backfill_source_run_id"] == "36071978818"
    if failure == "later_session":
        assert close.loc[pd.Timestamp("2026-09-25"), ["VOO", "IEF"]].tolist() == [105, 90]


def test_part0_ignores_expected_pre_inception_gaps(monkeypatch):
    import part0_data_infrastructure as part0

    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"]), name="Date"
    )
    columns = pd.MultiIndex.from_product([["VOO", "IEF"], ["Close"]])
    raw = pd.DataFrame(
        [[None, 90.0], [101.0, 91.0], [102.0, 92.0]],
        index=sessions, columns=columns,
    )
    monkeypatch.setattr(part0, "_business_day_calendar", lambda start, end: sessions)
    monkeypatch.setattr(part0.yf, "download", lambda *args, **kwargs: raw)
    monkeypatch.setattr(
        part0, "_recover_verified_historical_core_closes",
        lambda *args: pytest.fail("archive consulted for a pre-inception gap"),
    )
    cfg = part0.Part0Config(
        start="2026-09-21", end="2026-09-23", equity_tickers=("VOO", "IEF"),
        vix_tickers=(), core_tickers=("VOO", "IEF"), min_history_years=0,
    )
    close, _, _ = part0.download_market_data(cfg)
    assert close.index.min() == pd.Timestamp("2026-09-22")


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


def test_backfill_never_rolls_any_explicit_target_forward():
    from backfill_realized import _resolve_target_trading_date

    available = pd.DatetimeIndex(pd.to_datetime(["2026-09-04", "2026-09-09"]))
    target = pd.Timestamp("2026-09-08")

    assert _resolve_target_trading_date(
        pd.Timestamp("2026-09-04"),
        available,
        1,
        target,
    ) is None
    assert _resolve_target_trading_date(
        pd.Timestamp("2026-09-04"),
        available,
        1,
        target,
    ) is None
