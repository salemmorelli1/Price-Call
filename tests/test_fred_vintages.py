import pandas as pd


def test_fred_history_uses_first_release_and_availability_date():
    from point_in_time_macro import first_release_series

    releases = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "realtime_start": ["2026-01-05", "2026-01-12", "2026-01-06"],
        "value": [1.0, 9.0, 2.0],
    })
    calendar = pd.bdate_range("2026-01-01", "2026-01-09")
    result = first_release_series(releases, calendar, "test_macro")
    assert pd.isna(result.loc[pd.Timestamp("2026-01-02")])
    assert result.loc[pd.Timestamp("2026-01-05")] == 1.0
    assert result.loc[pd.Timestamp("2026-01-06")] == 2.0


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
                raise ValueError("not archived yet")
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
        assert "coverage failed after the first usable" in str(exc)
    else:
        raise AssertionError("post-coverage ALFRED failure must not be ignored")
