#!/usr/bin/env python3
"""Build one verified dashboard snapshot and bind index.html to that snapshot."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from artifact_integrity import LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, json_safe, write_json_strict


def _first(root: Path, candidates: list[str]) -> Path | None:
    return next((root / rel for rel in candidates if (root / rel).is_file()), None)


def _json(root: Path, candidates: list[str], *, required: bool = False) -> dict[str, Any]:
    path = _first(root, candidates)
    if path is None:
        if required:
            raise FileNotFoundError(f"required dashboard source missing: {candidates}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def _latest_record(path: Path, date_columns: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    ordering = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for column in date_columns:
        if column in frame.columns:
            ordering = ordering.fillna(pd.to_datetime(frame[column], errors="coerce"))
    valid = ordering.dropna()
    row = frame.loc[valid.idxmax()] if not valid.empty else frame.iloc[-1]
    return json_safe(row.to_dict())


def _csv_size(path: Path) -> int:
    return int(len(pd.read_csv(path))) if path.is_file() else 0


def build_snapshot(root: Path) -> dict[str, Any]:
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
    n_live = int(part9.get("n_live_realized", 0) or 0)
    minimum = 60
    latest_prediction = _latest_record(
        root / "artifacts_part3" / "prediction_log.csv",
        ("target_date", "decision_date"),
    )
    latest_signal = _latest_record(
        root / "artifacts_part10_bot" / "signal_log.csv",
        ("run_date", "date", "source_decision_date"),
    )
    decision_date = (
        part7.get("Date") or part7.get("decision_date") or part2.get("decision_date")
    )
    auc = _number(backtest.get("auc", delong.get("auc")))
    auc_p = _number(delong.get("p_one_sided"))
    brier_skill = _number(
        part2.get("historical_brier_skill_causal", backtest.get("brier_skill_score"))
    )
    auc_p_max = _number(part2.get("historical_auc_p_max")) or 0.10
    brier_skill_min = _number(part2.get("historical_brier_skill_min")) or 0.005
    macro_point_in_time_ok = bool(part2.get("macro_point_in_time_ok", False))
    final_pass = bool(part3.get("final_pass", part2.get("final_pass", False)))
    freshness_ok = bool(part2.get("part1_data_freshness_ok", False))
    validation_reasons: list[str] = []
    if auc is None or auc <= 0.50:
        validation_reasons.append("backtest AUC is not above 0.50")
    if auc_p is None or auc_p > auc_p_max:
        validation_reasons.append(f"one-sided AUC p-value exceeds {auc_p_max:.2f}")
    if brier_skill is None or brier_skill < brier_skill_min:
        validation_reasons.append(
            f"causal Brier skill is below {brier_skill_min:.3f}"
        )
    if not freshness_ok:
        validation_reasons.append("required market data are stale")
    if not macro_point_in_time_ok:
        validation_reasons.append("point-in-time macro coverage is incomplete")
    if not final_pass and not validation_reasons:
        validation_reasons.append("the complete governance gate did not pass")
    ticker_ages = part1.get("ticker_age_business_days_raw", {}) or {}
    ticker_limits = part1.get("ticker_freshness_limits_business_days", {}) or {}
    stale_tickers = {
        ticker: {"age_business_days": int(age), "limit": int(ticker_limits.get(ticker, 0))}
        for ticker, age in ticker_ages.items()
        if int(age) > int(ticker_limits.get(ticker, 0))
    }
    market_asof = (part0.get("date_range", {}) or {}).get("end")
    model_asof = part1.get("asof_date")
    latest_prediction_target = latest_prediction.get("target_date")
    history_message = (
        f"Market inputs extend through {market_asof}, but the governed prediction tape "
        f"stops at {latest_prediction_target} because validation failed closed at model "
        f"as-of {model_asof}. The observations were not deleted."
        if market_asof and latest_prediction_target and not freshness_ok
        else "The prediction tape and market-data lineage are current under the published gate."
    )
    return json_safe({
        "protocol_version": PROTOCOL_VERSION,
        "decision_date": decision_date,
        "publish_mode": part3.get("publish_mode", part2.get("publish_mode", "UNKNOWN")),
        "deployment_mode": part3.get("deployment_mode", "UNKNOWN"),
        "final_pass": final_pass,
        "data_freshness_ok": freshness_ok,
        "macro_point_in_time_ok": macro_point_in_time_ok,
        "alpha_state": part3.get("current_alpha_live_status", part3.get("latest_alpha_state", "UNKNOWN")),
        "regime": part3.get("current_regime", "unknown"),
        "weights": {
            "voo": _number(part7.get("w_target_voo", 0.60)),
            "ief": _number(part7.get("w_target_ief", 0.40)),
        },
        "live_health": part9.get("health_status", part9.get("status", "IMMATURE")),
        "live_health_reasons": part9.get("health_reasons", []),
        "evidence": {
            "cohort": part9.get("evidence_cohort", PROTOCOL_VERSION),
            "eligible_realized": n_live,
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
            "historical_evidence_ok": bool(part2.get("historical_evidence_ok", False)),
            "auc_p_value_max": auc_p_max,
            "brier_skill_min": brier_skill_min,
        },
        "lineage": {
            "market_data_asof": market_asof,
            "model_data_asof": model_asof,
            "latest_prediction_target_date": latest_prediction_target,
            "latest_signal_run_date": latest_signal.get("run_date") or latest_signal.get("date"),
            "latest_signal_source_decision_date": latest_signal.get("source_decision_date"),
            "prediction_tape_paused": not freshness_ok,
            "message": history_message,
            "stale_tickers": stale_tickers,
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
    run_date = (
        pd.to_datetime(frame["run_date"], errors="coerce")
        if "run_date" in frame.columns
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )
    source_date = pd.to_datetime(frame.get("date"), errors="coerce")
    recorded = run_date.fillna(source_date)
    frame["date"] = recorded.dt.date.astype("string")
    frame["target_voo"] = pd.to_numeric(
        frame.get("target_w_voo", frame.get("target_voo")), errors="coerce"
    )
    if "evidence" not in frame.columns:
        frame["evidence"] = float("nan")
    frame = frame.loc[recorded.notna()].assign(_recorded=recorded[recorded.notna()])
    frame = frame.sort_values("_recorded").drop(columns="_recorded")
    keep = ["date", "p_tail", "base_rate", "target_voo", "raw_val_auc", "evidence"]
    return json_safe(
        frame[[column for column in keep if column in frame.columns]].to_dict(orient="records")
    )


_DASHBOARD_BINDING = r'''
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
    if (node?.querySelector('.display')) node.querySelector('.display').textContent = value;
  };
  const tableValue = (label, value) => {
    const row = [...document.querySelectorAll('.stat-table tr')].find(x => x.cells?.[0]?.textContent.trim() === label);
    if (row?.cells?.[1]) row.cells[1].textContent = value;
  };
  const pct = v => v == null ? '—' : `${(100 * Number(v)).toFixed(2)}%`;
  fetch('artifacts_dashboard/dashboard_snapshot.json', {cache: 'no-store'})
    .then(response => { if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`); return response.json(); })
    .then(s => {
      metric('Backtest AUC', fmt(s.metrics.backtest_auc), 'current causal tape');
      metric('Rolling-median AUC', fmt(s.metrics.rolling_median_auc), 'current gate statistic');
      metric('AUC p-value', fmt(s.metrics.backtest_auc_p_value), `one-sided gate ≤ ${fmt(s.operator_validation.auc_p_value_max, 2)}`);
      metric('Live AUC', fmt(s.metrics.live_auc), `eligible cohort n = ${s.evidence.eligible_realized}`);
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
        if (p) p.textContent = `${s.evidence.eligible_realized} of ${s.evidence.minimum} eligible realized observations in the current cohort; ${s.evidence.legacy_realized_excluded} legacy rows are retained but excluded.`;
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
      metric('Live evidence', `${s.evidence.eligible_realized} / ${s.evidence.minimum}`, `${s.evidence.progress_pct}% of current gate`);
      metric('Live maturity', `${s.evidence.eligible_realized} / ${s.evidence.minimum}`, s.live_health);
      metric('Operator status', s.operator_validation.status, `AUC p=${fmt(s.metrics.backtest_auc_p_value)}`);
      metric('Signals logged', String(s.latest_signal.signal_count), `through ${s.lineage.latest_signal_run_date || '—'}`);
      const maturity = document.querySelector('.maturity-ring strong');
      if (maturity) maturity.textContent = `${s.evidence.eligible_realized} / ${s.evidence.minimum}`;
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
    })
    .catch(error => { document.documentElement.dataset.snapshotError = error.message; });
})();
</script>
'''.strip()


def sync_html(root: Path) -> None:
    path = root / "index.html"
    html = path.read_text(encoding="utf-8")
    pred_rows = _records(root / "artifacts_part3" / "prediction_log.csv", [
        "target_date", "px_voo_call_1d", "px_voo_realized", "px_ief_call_1d",
        "px_ief_realized", "p_final_cal", "publish_mode", "latest_alpha_state",
        "hit_direction", "model_protocol_version", "evidence_eligible",
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
    # Signal history is optional during a first paper-only run.  If no signal
    # ledger exists, retain the already-published history rather than replacing
    # it with an empty array that the existing chart code cannot annotate.
    if bot_rows:
        html, bot_replacements = re.subn(
            r"const botRows=.*?;\n",
            f"const botRows={json.dumps(bot_rows, separators=(',', ':'))};\n",
            html,
            count=1,
        )
        if bot_replacements != 1:
            raise ValueError("index.html is missing the expected 'const botRows=' data binding")
    html = re.sub(
        r"\s*<script id=\"pricecall-verified-snapshot\">.*?</script>\s*",
        "\n",
        html,
        flags=re.DOTALL,
    )
    if "</body>" not in html:
        raise ValueError("index.html is missing </body>; refusing to publish a partial page")
    html = html.replace("</body>", _DASHBOARD_BINDING + "\n</body>", 1)
    path.write_text(html, encoding="utf-8")


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
