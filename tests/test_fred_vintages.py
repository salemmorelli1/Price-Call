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
