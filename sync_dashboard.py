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


def build_snapshot(root: Path) -> dict[str, Any]:
    part2 = _json(root, [
        "artifacts_part2_g532/predictions/part2_g532_summary.json",
        "artifacts_part2_g532/part2_g532_summary.json",
    ], required=True)
    part3 = _json(root, ["artifacts_part3_v1/part3_summary.json"], required=True)
    part9 = _json(root, ["artifacts_part9/live_attribution_report.json"], required=True)
    part7 = _json(root, ["artifacts_part7/current_target_weights.json"])
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
    n_live = int(part9.get("n_live_realized", 0) or 0)
    minimum = 60
    decision_date = (
        part7.get("Date") or part7.get("decision_date") or part2.get("decision_date")
    )
    return json_safe({
        "protocol_version": PROTOCOL_VERSION,
        "decision_date": decision_date,
        "publish_mode": part3.get("publish_mode", part2.get("publish_mode", "UNKNOWN")),
        "deployment_mode": part3.get("deployment_mode", "UNKNOWN"),
        "final_pass": bool(part3.get("final_pass", part2.get("final_pass", False))),
        "data_freshness_ok": bool(part2.get("part1_data_freshness_ok", False)),
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
            "backtest_auc": _number(backtest.get("auc")),
            "backtest_brier": _number(backtest.get("brier")),
            "backtest_brier_skill": _number(backtest.get("brier_skill_score")),
            "live_auc": _number(live.get("auc")),
            "live_brier": _number(live.get("brier")),
            "live_brier_null": _number(live.get("brier_null")),
            "live_brier_skill": _number(live.get("brier_skill_score")),
            "balanced_accuracy": _number(live.get("balanced_accuracy")),
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
  fetch('artifacts_dashboard/dashboard_snapshot.json', {cache: 'no-store'})
    .then(response => { if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`); return response.json(); })
    .then(s => {
      metric('Backtest AUC', fmt(s.metrics.backtest_auc), 'current causal tape');
      metric('Live AUC', fmt(s.metrics.live_auc), `eligible cohort n = ${s.evidence.eligible_realized}`);
      metric('Backtest Brier', fmt(s.metrics.backtest_brier), 'per-row causal baseline');
      metric('Live Brier', fmt(s.metrics.live_brier), `null = ${fmt(s.metrics.live_brier_null)}`);
      metric('Live Brier skill', fmt(s.metrics.live_brier_skill), s.evidence.inference_eligible ? 'class-count gate cleared' : 'descriptive only');
      card('Sample maturity', s.live_health);
      card('Core publication', s.publish_mode);
      card('Auxiliary alpha', s.alpha_state);
      card('Trading status', s.bot_mode);
      const ticket = document.querySelector('.forecast-ticket');
      if (ticket) ticket.innerHTML = `<b>Latest verified snapshot</b><br>Decision ${s.decision_date || '—'} · ${s.publish_mode}<br>Freshness ${s.data_freshness_ok ? 'PASS' : 'FAIL-CLOSED'} · cohort ${s.protocol_version}`;
      const sample = [...document.querySelectorAll('.card')].find(x => x.querySelector('.kicker')?.textContent.trim() === 'Sample maturity');
      if (sample) {
        const p = sample.querySelector('p');
        if (p) p.textContent = `${s.evidence.eligible_realized} of ${s.evidence.minimum} eligible realized observations in the current cohort; ${s.evidence.legacy_realized_excluded} legacy rows are retained but excluded.`;
        const bar = sample.querySelector('.progress span');
        if (bar) bar.style.width = `${s.evidence.progress_pct}%`;
      }
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
    bot_rows = _records(root / "artifacts_part10_bot" / "signal_log.csv", [
        "date", "p_tail", "base_rate", "target_voo", "auc", "evidence",
    ])
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
