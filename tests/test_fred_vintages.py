import pandas as pd
import pytest


def test_fred_history_uses_first_release_and_availability_date():
    from point_in_time_macro import first_release_series

    releases = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "realtime_start": ["2026-01-05", "2026-01-12", "2026-01-06"],
        "value": [1.0, 9.0, 2.0],
    })
    calendar = pd.bdate_range("2026-01-01", "2026-01-09")
    result = first_release_series(releases, calendar, "test_macro")
    assert pd.isna(result.loc[pd.Timestamp("2026-01-05")])
    assert result.loc[pd.Timestamp("2026-01-06")] == 1.0
    assert result.loc[pd.Timestamp("2026-01-07")] == 2.0


def test_weekend_macro_release_becomes_available_next_session():
    from point_in_time_macro import first_release_series

    releases = pd.DataFrame({
        "date": ["2026-01-02"],
        "realtime_start": ["2026-01-03"],  # Saturday
        "value": [3.0],
    })
    calendar = pd.DatetimeIndex([pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")])
    result = first_release_series(releases, calendar, "test_macro")
    assert pd.isna(result.loc[pd.Timestamp("2026-01-02")])
    assert result.loc[pd.Timestamp("2026-01-05")] == 3.0


def test_alfred_requests_are_bounded_and_deduplicated():
    from point_in_time_macro import get_series_releases_chunked

    class FakeFred:
        def __init__(self):
            self.calls = []

        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            self.calls.append((series_id, realtime_start, realtime_end))
            return pd.DataFrame({
                "date": ["2020-01-01"],
                "realtime_start": ["2020-01-02"],
                "value": [1.0],
            })

    fred = FakeFred()
    result = get_series_releases_chunked(fred, "TEST", "2005-01-01", "2026-09-03")
    assert len(fred.calls) == 6
    assert fred.calls[0][1:] == ("2005-01-01", "2008-12-31")
    assert fred.calls[-1][1:] == ("2025-01-01", "2026-09-03")
    assert len(result) == 1


def test_alfred_leading_prehistory_error_does_not_discard_later_chunks():
    from point_in_time_macro import get_series_releases_chunked

    class FakeFred:
        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            if realtime_start == "2005-01-01":
                raise ValueError("The series does not exist in ALFRED but may exist in FRED")
            return pd.DataFrame({
                "date": ["2009-01-01"],
                "realtime_start": [realtime_start],
                "value": [1.0],
            })

    result = get_series_releases_chunked(FakeFred(), "TEST", "2005-01-01", "2012-12-31")
    assert not result.empty
    assert result.attrs["continuous_retrieval_start"] == "2009-01-01"
    assert result.attrs["chunk_diagnostics"][0]["status"] == "prehistory_error"


def test_alfred_error_after_first_usable_chunk_is_fatal():
    from point_in_time_macro import get_series_releases_chunked

    class FakeFred:
        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            if realtime_start == "2009-01-01":
                raise RuntimeError("transient endpoint failure")
            return pd.DataFrame({
                "date": ["2005-01-01"],
                "realtime_start": [realtime_start],
                "value": [1.0],
            })

    try:
        get_series_releases_chunked(FakeFred(), "TEST", "2005-01-01", "2012-12-31")
    except RuntimeError as exc:
        assert "coverage failed for TEST" in str(exc)
    else:
        raise AssertionError("post-coverage ALFRED failure must not be ignored")


def test_initial_transport_error_is_never_classified_as_prehistory():
    from point_in_time_macro import AlfredCoverageError, get_series_releases_chunked

    class FakeFred:
        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            if realtime_start == "2005-01-01":
                raise TimeoutError("ALFRED temporarily unavailable")
            return pd.DataFrame({
                "date": ["2009-01-02"], "realtime_start": [realtime_start], "value": [2.0],
            })

    try:
        get_series_releases_chunked(FakeFred(), "TEST", "2005-01-01", "2012-12-31")
    except AlfredCoverageError as exc:
        assert exc.diagnostics[0]["status"] == "coverage_error"
    else:
        raise AssertionError("a network error before the first release must fail closed")


def test_prearchive_macro_is_missing_without_revised_history(tmp_path, monkeypatch):
    import json

    import point_in_time_macro as pit

    calendar = pd.bdate_range("2010-09-09", "2016-10-03", name="Date")
    close = pd.DataFrame({"VOO": 100.0, "IEF": 90.0}, index=calendar)
    output = tmp_path / "artifacts_part0"
    output.mkdir()
    (output / "close_prices.parquet").touch()
    (output / "macro_data.parquet").touch()  # Historical revised data must never be read.
    (output / "part0_meta.json").write_text("{}", encoding="utf-8")
    partial = {
        "T10Y2Y", "T10Y3M", "DTWEXBGS", "BAMLH0A0HYM2",
        "BAMLC0A0CM", "USREC", "TEDRATE",
    }

    class FakeFred:
        failing_series = None

        def __init__(self, api_key):
            assert api_key == "test-only"

        def get_series_all_releases(self, series_id, realtime_start, realtime_end):
            if series_id == self.failing_series:
                raise TimeoutError("ALFRED unavailable")
            if series_id in partial and realtime_start == "2008-09-09":
                raise ValueError("The series does not exist in ALFRED but may exist in FRED")
            return pd.DataFrame({
                "date": [realtime_start],
                "realtime_start": [realtime_start],
                "value": [3.0 if series_id in partial else 2.0],
            })

    results = {}

    def fake_read(path):
        assert path.name == "close_prices.parquet"
        return close.copy()

    monkeypatch.setenv("FRED_API_KEY", "test-only")
    monkeypatch.setattr(pit, "Fred", FakeFred)
    monkeypatch.setattr(pit.pd, "read_parquet", fake_read)
    monkeypatch.setattr(pit, "_atomic_parquet", lambda frame, path: results.__setitem__(path.name, frame.copy()))
    monkeypatch.setattr(pit, "sha256_file", lambda path: "synthetic-test-hash")
    meta = pit.rebuild_point_in_time_macro(tmp_path)
    macro = results["macro_data.parquet"]
    features = results["features_full.parquet"]
    assert meta["historical_point_in_time_complete"] is True
    assert meta["fred_revised_fallback_count"] == 0
    assert sum(v == "alfred_first_release_partial" for v in meta["fred_vintage_mode_by_series"].values()) == 7
    assert pd.isna(macro.loc["2011-02-01", "curve_2s10s"])
    assert pd.isna(features.loc["2011-02-01", "yield_curve_2s10s"])
    assert macro.loc["2013-01-02", "curve_2s10s"] == 3.0
    assert json.loads((output / "part0_meta.json").read_text())["historical_point_in_time_complete"]

    FakeFred.failing_series = "T10Y2Y"
    meta = pit.rebuild_point_in_time_macro(tmp_path)
    assert meta["historical_point_in_time_complete"] is False
    assert meta["fred_vintage_mode_by_series"]["curve_2s10s"] == "unavailable"
    assert results["macro_data.parquet"]["curve_2s10s"].isna().all()


def test_regime_loader_rejects_stale_or_missing_pit_feature_file(tmp_path):
    from artifact_integrity import write_json_strict
    from part6_regime_engine import Part6Config, _load_part0_features

    write_json_strict(tmp_path / "part0_meta.json", {
        "point_in_time_adapter": "point_in_time_macro.py",
        "features_file_sha256": "expected-vintage-hash",
    })
    cfg = Part6Config(part0_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="DuckDB fallback is unsafe"):
        _load_part0_features(cfg)
    (tmp_path / "features_full.parquet").write_bytes(b"revised-history-file")
    with pytest.raises(RuntimeError, match="differs from Part 0 metadata"):
        _load_part0_features(cfg)


def test_duckdb_tables_are_replaced_with_causal_values(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from point_in_time_macro import _refresh_duckdb

    db = tmp_path / "market_data.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE macro_data AS SELECT DATE '2010-09-09' AS Date, 999.0 AS curve_2s10s")
    con.execute("CREATE TABLE features_full AS SELECT DATE '2010-09-09' AS Date, 999.0 AS yield_curve_2s10s")
    con.close()
    dates = pd.DatetimeIndex(["2010-09-09", "2012-09-10"], name="Date")
    macro = pd.DataFrame({"curve_2s10s": [float("nan"), 2.0]}, index=dates)
    features = pd.DataFrame({"yield_curve_2s10s": [float("nan"), 2.0]}, index=dates)

    _refresh_duckdb(db, macro, features)

    con = duckdb.connect(str(db), read_only=True)
    try:
        for table, column in (("macro_data", "curve_2s10s"), ("features_full", "yield_curve_2s10s")):
            values = con.execute(f"SELECT {column} FROM {table} ORDER BY Date").fetchall()
            assert pd.isna(values[0][0])
            assert values[1][0] == 2.0
    finally:
        con.close()
