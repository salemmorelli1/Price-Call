"""Provider-lag and immutable-outcome regressions for prospective paper rows."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from artifact_integrity import PROTOCOL_VERSION
from backfill_realized import (
    _part0_close_history,
    _verify_eligible_prices,
    pending_eligible_targets,
)


def _issued_row(**changes):
    row = {
        "decision_date": "2026-09-25",
        "target_date": "2026-09-28",
        "h_reb": 1,
        "px_voo_t": 710.79,
        "px_ief_t": 90.0,
        "px_voo_realized": np.nan,
        "px_ief_realized": np.nan,
        "realized_target_date": "",
        "model_protocol_version": PROTOCOL_VERSION,
        "evidence_eligible": 1,
    }
    return {**row, **changes}


def test_due_paper_outcome_remains_pending_even_when_backfill_marker_is_current():
    frame = pd.DataFrame([_issued_row()])
    end = pd.Timestamp("2026-09-28")
    assert pending_eligible_targets(frame, end) == [end]
    assert pending_eligible_targets(frame, pd.Timestamp("2026-09-25")) == []
    frame.loc[0, ["px_voo_realized", "px_ief_realized"]] = [703.61, 89.53]
    assert pending_eligible_targets(frame, end) == [end]  # Price pair lacks frozen anchors.
    frame.loc[0, ["px_voo_outcome_anchor", "px_ief_outcome_anchor"]] = [708.96698, 90.0]
    assert pending_eligible_targets(frame, end) == []
    frame.loc[0, "px_ief_realized"] = np.nan
    with pytest.raises(ValueError, match="partially realized"):
        pending_eligible_targets(frame, end)


def test_backfill_defers_without_writing_when_current_target_pair_is_absent(tmp_path, monkeypatch):
    import backfill_realized as backfill

    ledger = tmp_path / "artifacts_part3" / "prediction_log.csv"
    ledger.parent.mkdir()
    pd.DataFrame([_issued_row()]).to_csv(ledger, index=False)
    before = ledger.read_bytes()
    old_close = pd.DataFrame(
        {"VOO": [710.79], "IEF": [90.0]},
        index=pd.to_datetime(["2026-09-25"]),
    )
    monkeypatch.setattr(backfill, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(backfill, "PREDLOG_PATH", ledger)
    monkeypatch.setattr(backfill, "latest_completed_xnys_session", lambda: pd.Timestamp("2026-09-28"))
    monkeypatch.setattr(backfill, "_download_close_history", lambda start, end: old_close)

    assert backfill.main() == 75
    assert ledger.read_bytes() == before
    assert not (tmp_path / "artifacts_part9" / "backfill_run_date.txt").exists()
    # A successful status may never claim a date whose close pair is absent,
    # even when no current eligible paper target is due that day.
    pd.DataFrame([_issued_row(evidence_eligible=0)]).to_csv(ledger, index=False)
    before = ledger.read_bytes()
    assert backfill.main() == 75
    assert ledger.read_bytes() == before


def test_production_close_source_requires_observed_exact_pair(tmp_path, monkeypatch):
    folder = tmp_path / "artifacts_part0"
    folder.mkdir()
    days = pd.to_datetime(["2026-09-25", "2026-09-28"])
    close = pd.DataFrame({"VOO": [710.79, 703.61], "IEF": [90.0, 89.53]}, index=days)
    mask = pd.DataFrame({"VOO": [1, 1], "IEF": [1, 0]}, index=days)
    monkeypatch.setattr(pd, "read_parquet", lambda path: (
        close.copy() if path.name == "close_prices.parquet" else mask.copy()
    ))
    metadata = {
        "market_data_asof": "2026-09-28",
        "latest_completed_market_session": "2026-09-28",
        "market_values_are_raw_observations": True,
        "last_raw_observation_by_ticker": {"VOO": "2026-09-28", "IEF": "2026-09-28"},
    }
    (folder / "part0_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not observed"):
        _part0_close_history(tmp_path, days[-1])
    mask.loc[days[-1], "IEF"] = 1
    observed = _part0_close_history(tmp_path, days[-1])
    assert observed.loc[days[-1]].to_dict() == {"VOO": 703.61, "IEF": 89.53}


def test_eligible_realization_freezes_original_outcome_across_adjusted_history_revision():
    decision, target = pd.Timestamp("2026-09-25"), pd.Timestamp("2026-09-28")
    close = pd.DataFrame(
        {"VOO": [708.96698, 703.61], "IEF": [90.0, 89.53]},
        index=pd.DatetimeIndex([decision, target]),
    )
    row = pd.Series(_issued_row(
        px_voo_realized=703.60999, px_voo_outcome_anchor=708.96698,
        px_ief_realized=89.53,
        px_ief_outcome_anchor=90.0,
        realized_target_date="2026-09-28",
    ))
    assert _verify_eligible_prices(row, close, decision, target, (700.0, 89.0)) == (
        (703.60999, 89.53), (708.96698, 90.0),
    )
    new = pd.Series(_issued_row())
    assert _verify_eligible_prices(new, close, decision, target, (703.61, 89.53)) == (
        (703.61, 89.53), (708.96698, 90.0),
    )
    # The prior September 24 paper outcome was frozen before the history changed.
    earlier = pd.Series(_issued_row(
        decision_date="2026-09-24", target_date="2026-09-25", px_voo_t=706.98999,
        px_ief_t=89.690002,
        px_voo_realized=710.79, px_ief_realized=90.0, realized_target_date="2026-09-25",
    ))
    assert _verify_eligible_prices(
        earlier, close, pd.Timestamp("2026-09-24"), decision, (708.96698, 90.0),
        successor=pd.Series(_issued_row()),
    ) == ((710.79, 90.0), (706.98999, 89.690002))
    with pytest.raises(RuntimeError, match="corroborating next-session issuance"):
        _verify_eligible_prices(
            earlier, close, pd.Timestamp("2026-09-24"), decision, (708.96698, 90.0)
        )
    with pytest.raises(RuntimeError, match="partially recorded outcome anchor"):
        _verify_eligible_prices(
            pd.Series(_issued_row(px_voo_outcome_anchor=708.96698)), close, decision, target, (703.61, 89.53)
        )


def test_production_reconciles_due_outcome_before_publishing_cohort(tmp_path, monkeypatch):
    import backfill_realized as backfill

    ledger = tmp_path / "artifacts_part3" / "prediction_log.csv"
    ledger.parent.mkdir()
    pd.DataFrame([
        _issued_row(
            decision_date="2026-09-24", target_date="2026-09-25",
            px_voo_t=706.98999, px_ief_t=89.690002,
            px_voo_realized=710.79, px_ief_realized=90.0,
            realized_target_date="2026-09-25",
        ),
        _issued_row(px_voo_call_1d=704.0, px_ief_call_1d=89.5),
    ]).to_csv(ledger, index=False)
    summary = tmp_path / "artifacts_part3_v1" / "part3_summary.json"
    summary.parent.mkdir()
    summary.write_text('{"live_realized_dates": 0}', encoding="utf-8")
    # Part 9's full model inputs are outside this isolated integration fixture.
    (tmp_path / "part9_live_attribution.py").write_text(
        "class Part9Config:\n    pass\n"
        "def generate_live_report(config):\n"
        "    return {'n_live_realized': 2}\n",
        encoding="utf-8",
    )
    close = pd.DataFrame(
        {"VOO": [705.176758, 708.96698, 703.61], "IEF": [89.690002, 90.0, 89.53]},
        index=pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-28"]),
    )
    monkeypatch.setattr(backfill, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(backfill, "PREDLOG_PATH", ledger)
    monkeypatch.setattr(backfill, "latest_completed_xnys_session", lambda: pd.Timestamp("2026-09-28"))
    monkeypatch.setattr(backfill, "_part0_close_history", lambda root, end: close)

    original_part9 = sys.modules.get("part9_live_attribution")
    try:
        result = backfill.main(close_source="part0")
    finally:
        if original_part9 is None:
            sys.modules.pop("part9_live_attribution", None)
        else:
            sys.modules["part9_live_attribution"] = original_part9
    assert result == 0
    frozen, realized = list(pd.read_csv(ledger).itertuples(index=False))
    assert frozen.px_voo_realized == 710.79
    assert frozen.px_voo_outcome_anchor == 706.98999
    assert frozen.px_ief_outcome_anchor == 89.690002
    assert realized.realized_target_date == "2026-09-28"
    assert realized.px_voo_realized == 703.61
    assert realized.px_ief_realized == 89.53
    assert realized.px_voo_outcome_anchor == 708.96698
    assert realized.px_ief_outcome_anchor == 90.0
    assert json.loads(summary.read_text())["live_realized_dates"] == 2
