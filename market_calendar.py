"""Exchange-session helpers shared by ingestion, freshness, and forecasts."""
from __future__ import annotations

import pandas as pd
import exchange_calendars as xcals


XNYS_CALENDAR_NAME = "XNYS"
XNYS_CALENDAR_START = "1990-01-01"
XNYS_CALENDAR_END = "2100-12-31"
DEFAULT_SETTLEMENT_DELAY_MINUTES = 20


def _calendar():
    """Return an XNYS schedule with stable, explicit historical bounds.

    ``exchange_calendars`` otherwise builds a rolling default schedule whose
    first session is only about 20 years before the runtime date.  Price-Call
    begins its research history in 2005, so relying on that default eventually
    makes the same configured history fall out of bounds as time advances.
    """
    return xcals.get_calendar(
        XNYS_CALENDAR_NAME,
        start=XNYS_CALENDAR_START,
        end=XNYS_CALENDAR_END,
    )


def _naive_date(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _utc_timestamp(value: object | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("America/New_York")
    return timestamp.tz_convert("UTC")


def latest_completed_xnys_session(
    now: object | None = None,
    *,
    settlement_delay_minutes: int = DEFAULT_SETTLEMENT_DELAY_MINUTES,
) -> pd.Timestamp:
    """Return the latest NYSE session whose close has had time to settle."""
    if settlement_delay_minutes < 0:
        raise ValueError("settlement_delay_minutes must be non-negative")
    current_utc = _utc_timestamp(now)
    current_et_date = current_utc.tz_convert("America/New_York").date().isoformat()
    calendar = _calendar()
    candidate = calendar.date_to_session(current_et_date, direction="previous")
    settled_at = calendar.session_close(candidate) + pd.Timedelta(
        minutes=settlement_delay_minutes
    )
    if current_utc < settled_at:
        candidate = calendar.previous_session(candidate)
    return _naive_date(candidate)


def completed_xnys_sessions(
    start: object,
    end: object | None = None,
    *,
    now: object | None = None,
) -> pd.DatetimeIndex:
    """Return XNYS sessions bounded by the latest completed session."""
    completed = latest_completed_xnys_session(now)
    requested_end = completed if end is None else min(_naive_date(end), completed)
    requested_start = _naive_date(start)
    if requested_start > requested_end:
        return pd.DatetimeIndex([], name="Date")
    sessions = _calendar().sessions_in_range(requested_start, requested_end)
    return pd.DatetimeIndex(sessions).tz_localize(None).normalize().rename("Date")


def xnys_session_age(last_observation: object, expected_session: object) -> int:
    """Count completed XNYS sessions after an observation through expected."""
    calendar = _calendar()
    last_date = _naive_date(last_observation)
    expected = _naive_date(expected_session)
    if last_date >= expected:
        return 0
    last_session = calendar.date_to_session(last_date, direction="previous")
    sessions = calendar.sessions_in_range(last_session, expected)
    return max(0, int(len(sessions) - 1))


def next_xnys_session(value: object) -> pd.Timestamp:
    """Return the first NYSE session strictly after ``value``.

    Unlike ``BDay(1)``, this observes exchange holidays and exceptional
    closures represented by the XNYS calendar.
    """
    decision = _naive_date(value)
    calendar = _calendar()
    session = calendar.date_to_session(decision, direction="previous")
    target = calendar.next_session(session)
    return _naive_date(target)
