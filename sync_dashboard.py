#!/usr/bin/env python3
"""Build one verified dashboard snapshot and bind index.html to that snapshot."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from artifact_integrity import (
    LEGACY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    json_safe,
    read_json_strict,
    validate_execution_lineage,
    write_json_strict,
)
from market_calendar import (
    DEFAULT_SETTLEMENT_DELAY_MINUTES,
    _calendar,
    latest_completed_xnys_session,
    next_xnys_session,
    xnys_session_age,
)
from evaluate_prospective_cohort import MIN_OUTCOMES, evaluate
from part2_predictor import Part2Gen53Config


def _first(root: Path, candidates: list[str]) -> Path | None:
    return next((root / rel for rel in candidates if (root / rel).is_file()), None)


def _json(root: Path, candidates: list[str], *, required: bool = False) -> dict[str, Any]:
    path = _first(root, candidates)
    if path is None:
        if required:
            raise FileNotFoundError(f"required dashboard source missing: {candidates}")
        return {}
    return read_json_strict(path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _true(value: Any) -> bool:
    """JSON flags must be explicit booleans or numeric 0/1, never truthy text."""
    return value is True or (type(value) in (int, float) and value == 1)


def _parse_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse one ledger date column without dropping malformed provenance."""
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    raw = frame[column]
    parsed = pd.to_datetime(
        raw,
        errors="coerce",
        format="mixed",
        utc=True,
    ).dt.tz_localize(None)
    present = raw.notna() & raw.astype("string").str.strip().ne("")
    invalid = present & parsed.isna()
    if invalid.any():
        examples = raw.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"invalid {column} value(s) in dashboard ledger: {examples}")
    return parsed


def _latest_record(path: Path, date_columns: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    ordering_columns: list[str] = []
    for index, column in enumerate(date_columns):
        if column not in frame.columns:
            continue
        key = f"_dashboard_order_{index}"
        frame[key] = _parse_date_column(frame, column)
        ordering_columns.append(key)
    if ordering_columns:
        valid = frame[ordering_columns].notna().any(axis=1)
        ordered = frame.loc[valid].sort_values(
            ordering_columns, kind="stable", na_position="first"
        )
        row = ordered.iloc[-1] if not ordered.empty else frame.iloc[-1]
    else:
        row = frame.iloc[-1]
    row = row.drop(labels=ordering_columns, errors="ignore")
    return json_safe(row.to_dict())


def _csv_size(path: Path) -> int:
    return int(len(pd.read_csv(path))) if path.is_file() else 0


def _date_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _age_from_expected(value: Any, expected: pd.Timestamp) -> int | None:
    date_value = _date_text(value)
    if date_value is None:
        return None
    try:
        return int(xnys_session_age(date_value, expected))
    except (TypeError, ValueError):
        return None


def build_snapshot(root: Path, *, now: object | None = None) -> dict[str, Any]:
    part2 = _json(root, [
        "artifacts_part2_g532/predictions/part2_g532_summary.json",
        "artifacts_part2_g532/part2_g532_summary.json",
    ], required=True)
    part3 = _json(root, ["artifacts_part3_v1/part3_summary.json"], required=True)
    part9 = _json(root, ["artifacts_part9/live_attribution_report.json"], required=True)
    part7 = _json(root, ["artifacts_part7/current_target_weights.json"])
    part1 = _json(root, ["artifacts_part1/part1_meta.json"])
    part0 = _json(root, ["artifacts_part0/part0_meta.json"])
    bot = _json(root, [
        "artifacts_part10_bot/paper_state.json",
        "artifacts_part10_bot/portfolio_state.json",
    ])
    pipeline_status = _json(root, ["artifacts_part10_bot/pipeline_status.json"])
    backfill_status = _json(root, ["artifacts_part9/backfill_status.json"])
    execution = _json(root, ["artifacts_part8/execution_instructions.json"])
    execution_lineage_failures = validate_execution_lineage(root)
    execution_lineage_ok = not execution_lineage_failures

    part3_regime = str(part3.get("current_regime", "")).strip()
    part7_regime = str(part7.get("regime_label", "")).strip()
    if part3_regime and part7_regime and part3_regime != part7_regime:
        raise ValueError(
            f"dashboard source mismatch: Part 3 regime={part3_regime}, Part 7 regime={part7_regime}"
        )
    live = part9.get("classification_stats_live", {}) or {}
    backtest = part9.get("classification_stats_backtest", {}) or {}
    delong = part2.get("delong_overall_auc", {}) or {}
    distribution = part2.get("distributional_diagnostics", {}) or {}
    prediction_path = root / "artifacts_part3" / "prediction_log.csv"
    prospective = evaluate(pd.read_csv(prediction_path) if prediction_path.is_file() else pd.DataFrame())
    n_live = int(prospective.get("locked_realized", 0))
    minimum = MIN_OUTCOMES
    latest_prediction = _latest_record(
        root / "artifacts_part3" / "prediction_log.csv",
        ("target_date", "decision_date"),
    )
    latest_signal = _latest_record(
        root / "artifacts_part10_bot" / "signal_log.csv",
        ("source_decision_date", "date", "run_date"),
    )
    decision_date = (
        part7.get("Date") or part7.get("decision_date") or part2.get("decision_date")
    )
    auc = _number(backtest.get("auc", delong.get("auc")))
    auc_p = _number(delong.get("p_one_sided"))
    brier_skill = _number(
        part2.get("historical_brier_skill_causal", backtest.get("brier_skill_score"))
    )
    # Artifacts report the frozen policy; they cannot redefine its thresholds.
    auc_p_max = float(Part2Gen53Config.HISTORICAL_AUC_P_MAX)
    brier_skill_min = float(Part2Gen53Config.HISTORICAL_BRIER_SKILL_MIN)
    recorded_thresholds = {
        "auc_p_max": _number(part2.get("historical_auc_p_max")),
        "brier_skill_min": _number(part2.get("historical_brier_skill_min")),
    }
    required_thresholds = {"auc_p_max": auc_p_max, "brier_skill_min": brier_skill_min}
    thresholds_ok = recorded_thresholds == required_thresholds
    auc_ok = auc is not None and 0.50 < auc <= 1
    auc_p_ok = auc_p is not None and 0 <= auc_p <= auc_p_max
    brier_skill_ok = brier_skill is not None and brier_skill_min <= brier_skill <= 1
    macro_point_in_time_ok = _true(part2.get("macro_point_in_time_ok", False))
    governance_final_pass = _true(
        part3.get("final_pass", part2.get("final_pass", False))
    )
    core_publish_mode = part2.get("publish_mode", "UNKNOWN")
    governance_publish_mode = part3.get("publish_mode", "UNKNOWN")
    governance_deployment_mode = part3.get("deployment_mode", "UNKNOWN")
    core_ok = _true(part2.get("final_pass")) and core_publish_mode == "NORMAL"
    governance_ok = _true(part3.get("final_pass")) and governance_publish_mode == "NORMAL"
    source_freshness_ok = _true(part2.get("part1_data_freshness_ok", False))
    historical_ok = bool(
        _true(part2.get("historical_evidence_ok"))
        and thresholds_ok and auc_ok and auc_p_ok and brier_skill_ok
    )
    independent_ok = _true(part2.get("independent_validation_ok"))
    prospective_ok = n_live >= MIN_OUTCOMES and _true(prospective.get("prospective_thresholds_pass"))
    market_asof = (part0.get("date_range", {}) or {}).get("end")
    model_asof = part1.get("asof_date")
    pipeline_session = (
        pipeline_status.get("expected_completed_market_session")
        or pipeline_status.get("pipeline_run_date")
    )
    expected_session = (latest_completed_xnys_session() if now is None
                        else latest_completed_xnys_session(now))
    expected_session_text = expected_session.date().isoformat()
    publication_dates = {
        "market_data": _date_text(market_asof),
        "model_data": _date_text(model_asof),
        "decision": _date_text(decision_date),
        "pipeline": _date_text(pipeline_session),
        "execution": _date_text(
            execution.get("pipeline_run_date")
            or execution.get("source_decision_date")
            or execution.get("decision_date")
        ),
    }
    # A static page must close itself when the next exchange session settles,
    # even if both production and backfill stop publishing entirely.
    freshness_expires_at_utc = None
    if all(publication_dates.values()):
        first_asof = min(publication_dates.values())
        expires = _calendar().session_close(next_xnys_session(first_asof)) + pd.Timedelta(
            minutes=DEFAULT_SETTLEMENT_DELAY_MINUTES
        )
        freshness_expires_at_utc = expires.isoformat()
    publication_ages = {
        name: _age_from_expected(value, expected_session)
        for name, value in publication_dates.items()
    }
    component_dates_current = all(
        value == expected_session_text for value in publication_dates.values()
    )
    core_component_dates_current = all(
        publication_dates[name] == expected_session_text
        for name in ("market_data", "model_data", "decision", "pipeline")
    )
    publication_freshness_ok = component_dates_current and execution_lineage_ok
    known_ages = [age for age in publication_ages.values() if age is not None]
    publication_session_age = (
        max(known_ages) if known_ages else None
    )
    freshness_ok = source_freshness_ok and publication_freshness_ok
    def check(identifier, passed, observed, required):
        return {"id": identifier, "passed": bool(passed), "observed": observed, "required": required}

    gate_checks = [
        check("protocol_thresholds", thresholds_ok, recorded_thresholds, required_thresholds),
        check("core_publication", core_ok,
              {"final_pass": part2.get("final_pass"), "publish_mode": core_publish_mode},
              {"final_pass": True, "publish_mode": "NORMAL"}),
        check("governance_publication", governance_ok,
              {"final_pass": part3.get("final_pass"), "publish_mode": governance_publish_mode},
              {"final_pass": True, "publish_mode": "NORMAL"}),
        check("historical_auc", auc_ok, auc, {"greater_than": 0.50, "at_most": 1.0}),
        check("historical_auc_p", auc_p_ok, auc_p, {"at_least": 0.0, "at_most": auc_p_max}),
        check("historical_brier_skill", brier_skill_ok, brier_skill,
              {"at_least": brier_skill_min, "at_most": 1.0}),
        check("historical_approval", _true(part2.get("historical_evidence_ok")),
              part2.get("historical_evidence_ok"), True),
        check("source_freshness", source_freshness_ok, part2.get("part1_data_freshness_ok"), True),
        check("publication_freshness", publication_freshness_ok,
              {"component_dates": publication_dates, "execution_lineage_ok": execution_lineage_ok},
              {"every_component_session": expected_session_text, "execution_lineage_ok": True}),
        check("execution_lineage", execution_lineage_ok, execution_lineage_failures, []),
        check("point_in_time_macro", macro_point_in_time_ok, part2.get("macro_point_in_time_ok"), True),
        check("independent_review", independent_ok, part2.get("independent_validation_ok"), True),
        check("prospective_holdout", prospective_ok,
              {"locked_realized": n_live, "status": prospective["status"]},
              {"locked_realized": MIN_OUTCOMES, "prespecified_thresholds_pass": True}),
    ]
    final_pass = all(item["passed"] for item in gate_checks)
    publish_mode = (
        ("FAIL_CLOSED_STALE" if not publication_freshness_ok else "FAIL_CLOSED_NEUTRAL")
        if not final_pass and governance_publish_mode == "NORMAL"
        else governance_publish_mode
    )
    deployment_mode = (
        (governance_deployment_mode if final_pass else "DEFENSE_ONLY")
        if publication_freshness_ok else "NO_ACTION_STALE"
    )
    validation_reasons: list[str] = []
    if not thresholds_ok:
        validation_reasons.append("recorded historical thresholds differ from the frozen protocol")
    if not core_ok:
        validation_reasons.append(f"core publication has not cleared its gate ({core_publish_mode})")
    if not governance_ok:
        validation_reasons.append(f"governance publication has not cleared its gate ({governance_publish_mode})")
    if not auc_ok:
        validation_reasons.append("backtest AUC must be finite and in (0.50, 1.00]")
    if not auc_p_ok:
        validation_reasons.append(f"one-sided AUC p-value is invalid or exceeds {auc_p_max:.2f}")
    if not brier_skill_ok:
        validation_reasons.append(
            f"causal Brier skill is invalid or below {brier_skill_min:.3f}"
        )
    if not source_freshness_ok:
        validation_reasons.append("required market data are stale")
    if not publication_freshness_ok:
        age_text = (
            f"{publication_session_age} completed XNYS session(s)"
            if publication_session_age is not None
            else "an unknown number of sessions"
        )
        validation_reasons.append(
            f"published production is {age_text} behind "
            f"(expected {expected_session_text})"
        )
    if not execution_lineage_ok:
        validation_reasons.append(
            "execution artifacts do not match the governed production lineage"
        )
    if not macro_point_in_time_ok:
        validation_reasons.append("point-in-time macro coverage is incomplete")
    if not independent_ok:
        validation_reasons.append("prospective independent validation is pending")
    if not prospective_ok:
        validation_reasons.append(
            f"fixed prospective cohort: {prospective.get('locked_realized', 0)}/{minimum} "
            f"outcomes; {prospective['status']}"
        )
    if not historical_ok and auc is not None and auc_p is not None and brier_skill is not None:
        if auc_ok and auc_p_ok and brier_skill_ok and thresholds_ok:
            validation_reasons.append("historical evidence approval is absent")
    if not final_pass and not validation_reasons:
        validation_reasons.append("the complete governance gate did not pass")
    ticker_ages = part1.get("ticker_age_business_days_raw", {}) or {}
    ticker_limits = part1.get("ticker_freshness_limits_business_days", {}) or {}
    stale_tickers = {
        ticker: {"age_business_days": int(age), "limit": int(ticker_limits.get(ticker, 0))}
        for ticker, age in ticker_ages.items()
        if int(age) > int(ticker_limits.get(ticker, 0))
    }
    latest_prediction_target = latest_prediction.get("target_date")
    if not execution_lineage_ok and core_component_dates_current:
        history_message = (
            "Market, model, and decision artifacts are current, but the execution "
            "publication is quarantined because its date/run/SHA lineage does not "
            "match the governed production run."
        )
    elif not publication_freshness_ok:
        history_message = (
            f"The dashboard is serving the last successful production snapshot "
            f"(market {market_asof or 'unknown'}, model {model_asof or 'unknown'}); "
            f"the latest completed XNYS session is {expected_session_text}. Backfill "
            "updates do not make the forecast current."
        )
    elif not source_freshness_ok:
        history_message = (
            f"Market inputs extend through {market_asof}, but the governed prediction tape "
            f"stops at {latest_prediction_target} because validation failed closed at model "
            f"as-of {model_asof}. The observations were not deleted."
        )
    else:
        history_message = (
            "The prediction tape and market-data lineage are current under the published gate."
        )
    return json_safe({
        "protocol_version": PROTOCOL_VERSION,
        "decision_date": decision_date,
        "publish_mode": publish_mode,
        "deployment_mode": deployment_mode,
        "governance_publish_mode": governance_publish_mode,
        "governance_deployment_mode": governance_deployment_mode,
        "final_pass": final_pass,
        "gate_checks": gate_checks,
        "governance_final_pass": governance_final_pass,
        "data_freshness_ok": freshness_ok,
        "source_data_freshness_ok": source_freshness_ok,
        "publication_freshness_ok": publication_freshness_ok,
        "freshness_expires_at_utc": freshness_expires_at_utc,
        "execution_lineage_ok": execution_lineage_ok,
        "macro_point_in_time_ok": macro_point_in_time_ok,
        "alpha_state": part3.get("current_alpha_live_status", part3.get("latest_alpha_state", "UNKNOWN")),
        "alpha_reason": part3.get("current_alpha_reason", "not_recorded"),
        "regime": part3.get("current_regime", "unknown"),
        "weights": {
            "voo": _number(part7.get("w_target_voo", 0.60)),
            "ief": _number(part7.get("w_target_ief", 0.40)),
        },
        "live_health": (part9.get("health_status", part9.get("status", "IMMATURE"))
                        if n_live >= minimum else "IMMATURE"),
        "live_health_reasons": part9.get("health_reasons", []),
        "evidence": {
            "cohort": part9.get("evidence_cohort", PROTOCOL_VERSION),
            "eligible_realized": n_live,
            "total_current_realized": int(prospective["eligible_realized"]),
            "selection": "first_60_eligible_issuances",
            "minimum": minimum,
            "progress_pct": round(min(n_live / minimum, 1.0) * 100, 1),
            "legacy_realized_excluded": int(part9.get("legacy_realized_rows", 0) or 0),
            "inference_eligible": bool(live.get("inference_eligible", False)),
            "n_positive": int(live.get("n_positive", 0) or 0),
            "n_negative": int(live.get("n_negative", 0) or 0),
        },
        "metrics": {
            "backtest_auc": auc,
            "backtest_auc_p_value": auc_p,
            "backtest_auc_permutation_p_value": _number(backtest.get("p_value_auc_better")),
            "rolling_median_auc": _number(part2.get("raw_val_auc_median")),
            "backtest_brier": _number(backtest.get("brier")),
            "backtest_brier_skill": brier_skill,
            "live_auc": _number(live.get("auc")),
            "live_brier": _number(live.get("brier")),
            "live_brier_null": _number(live.get("brier_null")),
            "live_brier_skill": _number(live.get("brier_skill_score")),
            "balanced_accuracy": _number(live.get("balanced_accuracy")),
        },
        "prediction_interval": {
            "object": "next-day VOO-minus-IEF return spread",
            "nominal_coverage": 0.90,
            "empirical_coverage": _number(distribution.get("conf_coverage")),
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
        },
        "operator_validation": {
            "status": "VALIDATED" if final_pass else "NOT_VALIDATED",
            "reasons": validation_reasons,
            "historical_evidence_ok": historical_ok,
            "independent_validation_ok": independent_ok,
            "independent_validation_status": part2.get("independent_validation_status", "not_assessed"),
            "auc_p_value_max": auc_p_max,
            "brier_skill_min": brier_skill_min,
        },
        "prospective_validation": prospective,
        "lineage": {
            "market_data_asof": market_asof,
            "model_data_asof": model_asof,
            "latest_prediction_target_date": latest_prediction_target,
            "latest_signal_run_date": latest_signal.get("run_date") or latest_signal.get("date"),
            "latest_signal_source_decision_date": latest_signal.get("source_decision_date"),
            "prediction_tape_paused": not freshness_ok,
            "message": history_message,
            "stale_tickers": stale_tickers,
            "expected_completed_market_session": expected_session_text,
            "publication_session_age": publication_session_age,
            "publication_component_ages": publication_ages,
            "publication_component_dates": publication_dates,
            "pipeline_run_date": pipeline_status.get("pipeline_run_date"),
            "pipeline_run_id": pipeline_status.get("github_run_id"),
            "pipeline_source_code_sha": pipeline_status.get("source_code_sha"),
            "execution_decision_date": execution.get("decision_date"),
            "execution_pipeline_run_date": execution.get("pipeline_run_date"),
            "execution_pipeline_run_id": execution.get("pipeline_run_id"),
            "execution_source_code_sha": execution.get("source_code_sha"),
            "execution_lineage_ok": execution_lineage_ok,
            "execution_lineage_failures": execution_lineage_failures,
            "backfill_run_date": backfill_status.get("backfill_run_date"),
            "backfill_run_id": backfill_status.get("github_run_id"),
            "backfill_source_code_sha": backfill_status.get("source_code_sha"),
        },
        "latest_signal": {
            "signal_count": _csv_size(root / "artifacts_part10_bot" / "signal_log.csv"),
            "p_tail": _number(latest_signal.get("p_tail")),
            "base_rate": _number(latest_signal.get("base_rate")),
            "edge": _number(latest_signal.get("edge")),
            "target_voo": _number(
                latest_signal.get("target_w_voo", latest_signal.get("target_voo"))
            ),
            "raw_validation_auc": _number(
                latest_signal.get("raw_val_auc", latest_signal.get("auc"))
            ),
            "action_reason": latest_signal.get("action_reason"),
        },
        "bot_mode": bot.get("mode", bot.get("bot_mode", "DRY_RUN")),
    })


def _records(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    if "model_protocol_version" not in frame.columns:
        frame["model_protocol_version"] = LEGACY_PROTOCOL_VERSION
    keep = [col for col in columns if col in frame.columns]
    return json_safe(frame[keep].to_dict(orient="records"))


def _signal_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    run_date = _parse_date_column(frame, "run_date")
    source_date = _parse_date_column(frame, "date")
    recorded = run_date.fillna(source_date)
    frame["date"] = recorded.dt.date.astype("string")
    frame["target_voo"] = pd.to_numeric(
        frame.get("target_w_voo", frame.get("target_voo")), errors="coerce"
    )
    if "evidence" not in frame.columns:
        frame["evidence"] = float("nan")
    frame = frame.loc[recorded.notna()].assign(
        _recorded=recorded[recorded.notna()],
        _source_date=source_date[recorded.notna()],
    )
    frame = frame.sort_values(
        ["_recorded", "_source_date"], kind="stable", na_position="first"
    ).drop(columns=["_recorded", "_source_date"])
    keep = ["date", "p_tail", "base_rate", "target_voo", "raw_val_auc", "evidence"]
    return json_safe(
        frame[[column for column in keep if column in frame.columns]].to_dict(orient="records")
    )


_DASHBOARD_BINDING = r'''
<script id="pricecall-freshness-runtime" src="assets/snapshot_freshness.js"></script>
<script id="pricecall-verified-snapshot">
(() => {
  const fmt = (v, digits=3) => v == null ? '—' : Number(v).toFixed(digits);
  const metric = (label, value, note) => {
    const node = [...document.querySelectorAll('.metric')].find(x => x.querySelector('label')?.textContent.trim() === label);
    if (!node) return;
    node.querySelector('strong').textContent = value;
    if (note && node.querySelector('small')) node.querySelector('small').textContent = note;
  };
  const card = (label, value) => {
    const node = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === label);
    if (node?.querySelector('.display')) node.querySelector('.display').textContent = String(value).replaceAll('_', ' ');
  };
  const tableValue = (label, value) => {
    const row = [...document.querySelectorAll('.stat-table tr')].find(x => x.cells?.[0]?.textContent.trim() === label);
    if (row?.cells?.[1]) row.cells[1].textContent = value;
  };
  const pct = v => v == null ? '—' : `${(100 * Number(v)).toFixed(2)}%`;
  fetch('artifacts_dashboard/dashboard_snapshot.json', {cache: 'no-store'})
    .then(response => { if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`); return response.json(); })
    .then(original => {
      const render = () => {
      const s = PriceCallFreshness.derive(original, Date.now());
      metric('Backtest AUC', fmt(s.metrics.backtest_auc), 'current causal tape');
      metric('Rolling-median AUC', fmt(s.metrics.rolling_median_auc), 'current gate statistic');
      metric('AUC p-value', fmt(s.metrics.backtest_auc_p_value), `one-sided gate ≤ ${fmt(s.operator_validation.auc_p_value_max, 2)}`);
      metric('Live AUC', fmt(s.metrics.live_auc), `descriptive current cohort n = ${s.evidence.total_current_realized}`);
      metric('Brier skill', fmt(s.metrics.backtest_brier_skill), `causal gate ≥ ${fmt(s.operator_validation.brier_skill_min)}`);
      metric('90% interval coverage', pct(s.prediction_interval.empirical_coverage), 'next-day spread; nominal 90%');
      card('Sample maturity', s.live_health);
      card('Core publication', s.publish_mode);
      card('Operator validation', s.operator_validation.status);
      card('Auxiliary alpha', s.alpha_state);
      card('Trading status', s.bot_mode);
      tableValue('Core tape', s.publish_mode);
      tableValue('Alpha sleeve', s.alpha_state);
      tableValue('Regime', s.regime);
      tableValue('Weights', `VOO ${fmt(s.weights.voo, 2)} / IEF ${fmt(s.weights.ief, 2)}`);
      tableValue('Live health', s.live_health);
      const ticket = document.querySelector('.forecast-ticket');
      if (ticket) ticket.innerHTML = `<b>Latest verified snapshot</b><br>Decision ${s.decision_date || '—'} · ${s.publish_mode}<br>Freshness ${s.data_freshness_ok ? 'PASS' : 'FAIL-CLOSED'} · cohort ${s.protocol_version}`;
      const sample = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === 'Sample maturity');
      if (sample) {
        const p = sample.querySelector('p');
        if (p) p.textContent = `${s.evidence.eligible_realized} of ${s.evidence.minimum} outcomes in the fixed issuance window; ${s.evidence.legacy_realized_excluded} legacy rows are retained but excluded.`;
        const bar = sample.querySelector('.progress span');
        if (bar) bar.style.width = `${s.evidence.progress_pct}%`;
        const muted = sample.querySelector('.muted');
        if (muted) muted.textContent = `${s.evidence.progress_pct}% of the current-cohort minimum collected.`;
      }
      const status = document.getElementById('snapshot-status');
      if (status) status.textContent = `Verified market data through ${s.lineage.market_data_asof || '—'} · model as-of ${s.lineage.model_data_asof || '—'}`;
      const history = document.getElementById('history-note');
      if (history) history.textContent = s.lineage.message;
      const operator = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === 'Operator validation');
      if (operator?.querySelector('p')) operator.querySelector('p').textContent = s.operator_validation.reasons.join('; ') || 'All prespecified controls passed.';
      const latestCard = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === 'Latest verified signal');
      if (latestCard) latestCard.querySelector('.kicker').textContent = `Latest verified signal · ${s.lineage.latest_signal_run_date || '—'}`;
      tableValue('Tail probability', pct(s.latest_signal.p_tail));
      tableValue('Live base rate', pct(s.latest_signal.base_rate));
      tableValue('Probability edge', s.latest_signal.edge == null ? '—' : `${(100 * s.latest_signal.edge).toFixed(2)} pp`);
      tableValue('Paper target', s.latest_signal.target_voo == null ? '—' : `VOO ${(100*s.latest_signal.target_voo).toFixed(0)}% / IEF ${(100*(1-s.latest_signal.target_voo)).toFixed(0)}%`);
      tableValue('Raw validation AUC', fmt(s.latest_signal.raw_validation_auc, 4));
      tableValue('Execution result', s.final_pass ? 'Paper-only gate cleared' : `No action · ${s.latest_signal.action_reason || 'gate closed'}`);
      const heroValue = (id, value) => { const node = document.getElementById(id); if (node) node.textContent = value; };
      heroValue('hero-tail-probability', pct(s.latest_signal.p_tail));
      heroValue('hero-base-rate', pct(s.latest_signal.base_rate));
      heroValue('hero-gate-state', s.final_pass ? 'GATE OPEN' : 'FAIL-CLOSED');
      heroValue('hero-voo-weight', s.weights.voo == null ? '—' : `${(100 * Number(s.weights.voo)).toFixed(0)}%`);
      heroValue('hero-ief-weight', s.weights.ief == null ? '—' : `${(100 * Number(s.weights.ief)).toFixed(0)}%`);
      heroValue('hero-evidence', `${s.evidence.eligible_realized} / ${s.evidence.minimum}`);
      const heroGate = document.querySelector('.gate-node');
      if (heroGate) heroGate.classList.toggle('closed', !s.final_pass);
      metric('Live evidence', `${s.evidence.eligible_realized} / ${s.evidence.minimum}`, `${s.evidence.progress_pct}% of current gate`);
      metric('Live maturity', `${s.evidence.eligible_realized} / ${s.evidence.minimum}`, s.live_health);
      metric('Operator status', s.operator_validation.status, `AUC p=${fmt(s.metrics.backtest_auc_p_value)}`);
      metric('Signals logged', String(s.latest_signal.signal_count), `through ${s.lineage.latest_signal_run_date || '—'}`);
      const maturity = document.querySelector('.maturity-ring strong');
      if (maturity) maturity.textContent = `${s.evidence.eligible_realized} / ${s.evidence.minimum}`;
      const maturityRing = document.querySelector('.maturity-ring');
      if (maturityRing) maturityRing.style.setProperty('--maturity-angle', `${Math.min(100, Math.max(0, Number(s.evidence.progress_pct) || 0)) * 3.6}deg`);
      const promotion = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === 'Promotion gate');
      if (promotion) {
        const display = promotion.querySelector('.display');
        if (display) display.textContent = `${s.evidence.eligible_realized} of ${s.evidence.minimum} observations.`;
        const p = promotion.querySelector('p');
        if (p) p.textContent = `Only current-protocol, eligible realized predictions count. ${s.evidence.legacy_realized_excluded} legacy rows remain auditable but excluded.`;
        const bar = promotion.querySelector('.progress span');
        if (bar) bar.style.width = `${s.evidence.progress_pct}%`;
      }
      const botPlot = document.getElementById('bot-progress-chart');
      if (botPlot && window.Plotly) Plotly.relayout(botPlot, {'annotations[0].text': `${s.evidence.eligible_realized} / ${s.evidence.minimum} current cohort`});
      document.documentElement.dataset.snapshotProtocol = s.protocol_version;
      };
      render();
      window.setInterval(render, 60000);
      window.addEventListener('focus', render);
      document.addEventListener('visibilitychange', render);
    })
    .catch(error => {
      document.documentElement.dataset.snapshotError = error.message;
      card('Operator validation', 'NOT VALIDATED');
      card('Core publication', 'SNAPSHOT UNAVAILABLE');
      card('Auxiliary alpha', 'UNVERIFIED');
      card('Trading status', 'DRY RUN');
      const ticket = document.querySelector('.forecast-ticket');
      if (ticket) ticket.innerHTML = '<b>Verified snapshot unavailable</b><br>FAIL-CLOSED · published state could not be loaded.';
      const status = document.getElementById('snapshot-status');
      if (status) status.textContent = 'FAIL-CLOSED · verified dashboard snapshot unavailable';
      const mode = document.getElementById('snapshot-mode');
      if (mode) { mode.textContent = 'Snapshot unavailable'; mode.classList.add('fail'); }
      const gate = document.getElementById('hero-gate-state');
      if (gate) gate.textContent = 'FAIL-CLOSED';
      const gateNode = document.querySelector('.gate-node');
      if (gateNode) gateNode.classList.add('closed');
    });
})();
</script>
'''.strip()


def sync_html(root: Path, *, output_path: Path | None = None) -> None:
    path = root / "index.html"
    html = path.read_text(encoding="utf-8")
    pred_rows = _records(root / "artifacts_part3" / "prediction_log.csv", [
        "target_date", "px_voo_call_1d", "px_voo_realized", "px_ief_call_1d",
        "px_ief_realized", "p_final_cal", "publish_mode", "latest_alpha_state",
        "base_rate", "tail_threshold", "hit_direction", "model_protocol_version",
        "evidence_eligible",
    ])
    bot_rows = _signal_records(root / "artifacts_part10_bot" / "signal_log.csv")
    if not pred_rows:
        raise ValueError("prediction_log.csv is missing or empty; refusing to publish an empty dashboard")
    html, row_replacements = re.subn(
        r"const rows=.*?;\n",
        f"const rows={json.dumps(pred_rows, separators=(',', ':'))};\n",
        html,
        count=1,
    )
    if row_replacements != 1:
        raise ValueError("index.html is missing the expected 'const rows=' data binding")
    html, bot_replacements = re.subn(
        r"const botRows=.*?;\n",
        f"const botRows={json.dumps(bot_rows, separators=(',', ':'))};\n",
        html,
        count=1,
    )
    if bot_replacements != 1:
        raise ValueError("index.html is missing the expected 'const botRows=' data binding")
    html = re.sub(
        r"\s*<script id=\"pricecall-(?:verified-snapshot|freshness-runtime)\"[^>]*>.*?</script>\s*",
        "\n",
        html,
        flags=re.DOTALL,
    )
    if "</body>" not in html:
        raise ValueError("index.html is missing </body>; refusing to publish a partial page")
    html = html.replace("</body>", _DASHBOARD_BINDING + "\n</body>", 1)
    # Write encoded bytes so Windows cannot translate the repository's
    # canonical LF line endings to CRLF during deterministic regeneration.
    (output_path or path).write_bytes(html.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    write_json_strict(root / "artifacts_dashboard" / "dashboard_snapshot.json", build_snapshot(root))
    sync_html(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
