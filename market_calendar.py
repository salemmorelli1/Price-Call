"""Exchange-session date helpers used by governed prediction records."""
from __future__ import annotations

import pandas as pd
import exchange_calendars as xcals


def next_xnys_session(value: object) -> pd.Timestamp:
    """Return the first NYSE session strictly after ``value``.

    Unlike ``BDay(1)``, this observes exchange holidays and exceptional
    closures represented by the XNYS calendar.
    """
    decision = pd.Timestamp(value)
    if decision.tzinfo is not None:
        decision = decision.tz_localize(None)
    decision = decision.normalize()
    calendar = xcals.get_calendar("XNYS")
    session = calendar.date_to_session(decision, direction="previous")
    target = calendar.next_session(session)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    return pd.Timestamp(target).normalize()
