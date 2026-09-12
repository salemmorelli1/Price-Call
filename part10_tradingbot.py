#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @title Part 10 — Daily Trading Bot ($1,000 Paper Portfolio)
#
# AUDIT CHANGELOG (Quant-Guild Part 8 session, 2026-04)
# ──────────────────────────────────────────────────────
# Finding C (MEDIUM): compute_performance() annualized Sharpe using sqrt(252)
#   on between-trade returns, not between-day returns.  At ~1.6 trades/year the
#   interval between rows in the trade log is weeks to months.  Applying sqrt(252)
#   assumed each return represented one trading day, overstating Sharpe by
#   sqrt(252 / trades_per_year) — a factor of ~4.6× at 12 trades/year.
#   Fix: compute each trade period's annualized return using actual elapsed
#   calendar days (decision_date diff), then take mean/std across those values.
#   Sharpe is now NaN when fewer than 2 valid intervals exist, rather than
#   a numerically inflated finite value.
# ──────────────────────────────────────────────────────
from __future__ import annotations

import csv
import json
import os
import sys as _sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from artifact_integrity import PROTOCOL_VERSION, write_json_strict
from market_calendar import latest_completed_xnys_session

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("yfinance required: pip install yfinance") from exc

# ── Colab / environment detection ─────────────────────────────────────────────
_IN_COLAB = "google.colab" in _sys.modules
_DRIVE_ROOT = os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject")


def _colab_init(extra_packages=None):
    if _IN_COLAB:
        if not os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        os.makedirs(_DRIVE_ROOT, exist_ok=True)
        os.environ.setdefault("PRICECALL_ROOT", _DRIVE_ROOT)
    if extra_packages:
        import importlib
        import subprocess
        for pkg in extra_packages:
            mod = pkg.split("[")[0].replace("-", "_").split("==")[0]
            try:
                importlib.import_module(mod)
            except ImportError:
                subprocess.run([_sys.executable, "-m", "pip", "install", pkg, "-q"], capture_output=True)


@dataclass(frozen=True)
class BotConfig:
    version: str = "V2_DAILY_CANONICAL"

    root_dir: str = _DRIVE_ROOT
    part2_summary_path: str = _DRIVE_ROOT + "/artifacts_part2_g532/predictions/part2_g532_summary.json"
    part2_tape_path: str = _DRIVE_ROOT + "/artifacts_part2_g532/predictions/g532_final_consensus_tape.csv"
    part0_close_path: str = _DRIVE_ROOT + "/artifacts_part0/close_prices.parquet"
    part0_observation_mask_path: str = _DRIVE_ROOT + "/artifacts_part0/market_observation_mask.parquet"
    part0_meta_path: str = _DRIVE_ROOT + "/artifacts_part0/part0_meta.json"
    part7_current_target_path: str = _DRIVE_ROOT + "/artifacts_part7/current_target_weights.json"
    part7_weights_tape_path: str = _DRIVE_ROOT + "/artifacts_part7/portfolio_weights_tape.csv"
    part8_instructions_path: str = _DRIVE_ROOT + "/artifacts_part8/execution_instructions.json"
    part9_report_path: str = _DRIVE_ROOT + "/artifacts_part9/live_attribution_report.json"
    bot_dir: str = _DRIVE_ROOT + "/artifacts_part10_bot"

    starting_capital: float = 1000.0
    stop_loss_floor: float = 900.0
    max_position_voo: float = 0.90
    min_position_voo: float = 0.10

    dead_band: float = 0.02
    min_signal_edge: float = 0.03
    base_rate_default: float = 0.19

    require_significance: bool = True
    min_t_stat: float = 2.0
    min_live_n: int = 60
    fail_closed_on_nonfinite_t: bool = True
    enforce_tc_gate: bool = False

    use_prior_day_close: bool = True
    tc_bps_one_way: float = 0.60

    ticker_equity: str = "VOO"
    ticker_bond: str = "IEF"
    default_w_equity: float = 0.60
    default_w_bond: float = 0.40


CFG = BotConfig()


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _canonical_iso_date(value: Any, label: str) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise RuntimeError(f"{label} is missing or is not a valid date: {value!r}")
    return pd.Timestamp(parsed).date().isoformat()


def _identity_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _canonical_csv_flag(value: Any) -> str:
    """Serialize nullable CSV flags without pandas numeric coercion."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        raise RuntimeError(f"CSV flag must be scalar: {value!r}") from None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return "true"
    if normalized in {"false", "0", "0.0"}:
        return "false"
    raise RuntimeError(f"CSV flag has an invalid boolean value: {value!r}")


def _resolve_root() -> Path:
    candidates: List[Path] = []
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(_DRIVE_ROOT))
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd())
    seen = set()
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        key = str(rp)
        if key == "/content" or key in seen:
            continue
        seen.add(key)
        if rp.exists():
            return rp
    return Path.cwd().resolve()


def _abs_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_resolve_root() / path).resolve())


class PaperPortfolio:
    def __init__(self, cfg: BotConfig = CFG):
        self.cfg = cfg
        self.cash = 0.0
        self.shares_voo = 0.0
        self.shares_ief = 0.0
        self.peak_nav = cfg.starting_capital
        self.is_stopped = False
        self.trade_count = 0
        self._initialized = False

    def initialize(self, px_voo: float, px_ief: float) -> None:
        capital = self.cfg.starting_capital
        w_voo = self.cfg.default_w_equity
        self.shares_voo = (capital * w_voo) / px_voo
        self.shares_ief = (capital * (1.0 - w_voo)) / px_ief
        self.cash = 0.0
        self._initialized = True
        print(f"[Bot] Initialized LIVE: {self.shares_voo:.4f} VOO @ ${px_voo:.2f}, {self.shares_ief:.4f} IEF @ ${px_ief:.2f}")

    def initialize_cash_only(self) -> None:
        self.cash = self.cfg.starting_capital
        self.shares_voo = 0.0
        self.shares_ief = 0.0
        self.peak_nav = self.cfg.starting_capital
        self._initialized = True
        print("[Bot] Initialized in DRY_RUN cash mode ($1,000 uninvested)")

    def nav(self, px_voo: float, px_ief: float) -> float:
        return self.cash + self.shares_voo * px_voo + self.shares_ief * px_ief

    def weights(self, px_voo: float, px_ief: float) -> Tuple[float, float]:
        total = self.nav(px_voo, px_ief)
        if total <= 0:
            return 0.0, 0.0
        w_voo = (self.shares_voo * px_voo) / total
        w_ief = (self.shares_ief * px_ief) / total
        return float(w_voo), float(w_ief)

    def _trigger_stop_loss(self, px_voo: float, px_ief: float, decision_date: str) -> None:
        self.is_stopped = True
        nav_at_stop = self.nav(px_voo, px_ief)
        liquidation_notional = self.shares_voo * px_voo + self.shares_ief * px_ief
        liquidation_tc = liquidation_notional * (self.cfg.tc_bps_one_way / 10_000.0)
        self.cash = nav_at_stop - liquidation_tc
        self.shares_voo = 0.0
        self.shares_ief = 0.0
        loss_pct = (self.cash - self.cfg.starting_capital) / self.cfg.starting_capital
        print("\n" + "!" * 60)
        print(f"⛔ STOP-LOSS TRIGGERED on {decision_date}")
        print(f"   NAV at stop: ${self.cash:.2f}")
        print(f"   Total loss: {loss_pct:.2%} of starting capital")
        print("   All positions liquidated to cash. Trading suspended.")
        print("!" * 60 + "\n")

    def rebalance(
        self,
        target_w_voo: float,
        px_voo: float,
        px_ief: float,
        decision_date: str,
        signal: float,
        action_reason: str,
        dry_run: bool = False,
        target_source: str = "heuristic",
    ) -> Optional[Dict[str, Any]]:
        if self.is_stopped:
            print("[Bot] STOPPED — no trades allowed.")
            return None
        if not self._initialized:
            print("[Bot] Not initialized.")
            return None

        current_nav = self.nav(px_voo, px_ief)
        current_w_voo, current_w_ief = self.weights(px_voo, px_ief)
        delta_w = target_w_voo - current_w_voo

        if abs(delta_w) < self.cfg.dead_band:
            print(f"[Bot] Dead-band: |Δw|={abs(delta_w):.3f} < {self.cfg.dead_band:.2f} — no trade")
            return None

        target_nav_voo = current_nav * target_w_voo
        target_nav_ief = current_nav * (1.0 - target_w_voo)
        target_shares_voo = target_nav_voo / px_voo
        target_shares_ief = target_nav_ief / px_ief

        trade_voo = target_shares_voo - self.shares_voo
        trade_ief = target_shares_ief - self.shares_ief
        tc_voo = abs(trade_voo * px_voo) * (self.cfg.tc_bps_one_way / 10_000.0)
        tc_ief = abs(trade_ief * px_ief) * (self.cfg.tc_bps_one_way / 10_000.0)
        total_tc = tc_voo + tc_ief

        if dry_run:
            print(
                f"[Bot DRY_RUN] Would rebalance: VOO {current_w_voo:.1%}→{target_w_voo:.1%} "
                f"| TC ~${total_tc:.4f} | signal={signal:.4f} | src={target_source}"
            )
            return None

        self.shares_voo += trade_voo
        self.shares_ief += trade_ief
        self.cash -= total_tc
        self.trade_count += 1

        post_nav = self.nav(px_voo, px_ief)
        self.peak_nav = max(self.peak_nav, post_nav)
        drawdown = (post_nav - self.peak_nav) / self.peak_nav if self.peak_nav > 0 else 0.0

        if post_nav <= self.cfg.stop_loss_floor:
            self._trigger_stop_loss(px_voo, px_ief, decision_date)

        record = {
            "decision_date": decision_date,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "signal_p_tail": round(float(signal), 6),
            "action_reason": action_reason,
            "target_source": target_source,
            "w_voo_before": round(float(current_w_voo), 6),
            "w_voo_after": round(float(target_w_voo), 6),
            "delta_w_voo": round(float(delta_w), 6),
            "trade_voo_shares": round(float(trade_voo), 6),
            "trade_ief_shares": round(float(trade_ief), 6),
            "px_voo": round(float(px_voo), 4),
            "px_ief": round(float(px_ief), 4),
            "tc_dollars": round(float(total_tc), 6),
            "tc_bps": round(float(total_tc / max(current_nav, 1e-12) * 10_000.0), 4),
            "nav_before": round(float(current_nav), 4),
            "nav_after": round(float(post_nav), 4),
            "peak_nav": round(float(self.peak_nav), 4),
            "drawdown": round(float(drawdown), 6),
            "trade_count": self.trade_count,
            "is_stopped": self.is_stopped,
            "mode": "paper",
        }
        print(
            f"[Bot] TRADE #{self.trade_count} | {decision_date} | VOO {current_w_voo:.1%}→{target_w_voo:.1%} "
            f"| NAV ${post_nav:.2f} | TC ${total_tc:.4f} | DD {drawdown:.2%}"
        )
        return record

    def summary(self, px_voo: float, px_ief: float) -> Dict[str, Any]:
        nav = self.nav(px_voo, px_ief)
        w_voo, w_ief = self.weights(px_voo, px_ief)
        return {
            "nav": round(nav, 4),
            "starting_capital": self.cfg.starting_capital,
            "total_return_pct": round((nav - self.cfg.starting_capital) / self.cfg.starting_capital * 100.0, 4),
            "peak_nav": round(self.peak_nav, 4),
            "drawdown_from_peak": round((nav - self.peak_nav) / max(self.peak_nav, 1e-12) * 100.0, 4),
            "w_voo": round(w_voo, 4),
            "w_ief": round(w_ief, 4),
            "cash": round(self.cash, 4),
            "shares_voo": round(self.shares_voo, 6),
            "shares_ief": round(self.shares_ief, 6),
            "trade_count": self.trade_count,
            "is_stopped": self.is_stopped,
            "stop_loss_floor": self.cfg.stop_loss_floor,
            "initialized": self._initialized,
        }


class TradeLog:
    COLUMNS = [
        "decision_date", "executed_at", "signal_p_tail", "action_reason", "target_source",
        "w_voo_before", "w_voo_after", "delta_w_voo", "trade_voo_shares", "trade_ief_shares",
        "px_voo", "px_ief", "tc_dollars", "tc_bps", "nav_before", "nav_after", "peak_nav",
        "drawdown", "trade_count", "is_stopped", "mode", "protocol_version",
        "pipeline_run_date", "pipeline_run_id", "pipeline_run_attempt", "source_code_sha",
        "execution_lineage_verified", "price_source", "price_session",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
        else:
            existing = pd.read_csv(path)
            if list(existing.columns) != self.COLUMNS:
                for column in self.COLUMNS:
                    if column not in existing.columns:
                        existing[column] = None
                existing[self.COLUMNS].to_csv(path, index=False)

    def append(self, record: Dict[str, Any]) -> None:
        existing = self.load()
        if not existing.empty and "decision_date" in existing.columns:
            decision = _canonical_iso_date(
                record.get("decision_date"), "trade decision_date"
            )
            existing_dates = pd.to_datetime(
                existing["decision_date"], errors="coerce"
            ).dt.date.astype("string")
            if bool((existing_dates == decision).any()):
                raise RuntimeError(
                    f"trade log already contains a rebalance for {decision}; "
                    "refusing a same-session duplicate"
                )
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS, extrasaction="ignore").writerow(record)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_csv(self.path)
        if "decision_date" in df.columns:
            df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
        return df


class SignalLog:
    COLUMNS = [
        "date", "run_date", "source_decision_date", "model_protocol_version", "model_code_sha",
        "pipeline_run_date", "pipeline_run_id", "pipeline_run_attempt",
        "execution_source_code_sha", "execution_lineage_verified", "price_source", "price_session",
        "p_tail", "base_rate", "edge", "target_w_voo", "action_reason", "target_source",
        "dry_run", "accuracy_gate_passed", "accuracy_gate_reason", "publish_mode", "final_pass",
        "raw_val_auc", "px_voo", "px_ief", "nav",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
        else:
            existing = pd.read_csv(path)
            if list(existing.columns) != self.COLUMNS:
                for column in self.COLUMNS:
                    if column not in existing.columns:
                        existing[column] = None
                existing[self.COLUMNS].to_csv(path, index=False)

    def append(self, record: Dict[str, Any]) -> None:
        # A same-session rerun is a new provenance revision, not a second signal
        # and not a reason to retain the older SHA/run identity.  Replace all rows
        # for that completed session, then sort the tape chronologically.
        new_record = dict(record)
        decision = _canonical_iso_date(
            new_record.get("source_decision_date") or new_record.get("date"),
            "signal source_decision_date",
        )
        new_record["date"] = decision
        new_record["source_decision_date"] = decision
        existing = pd.read_csv(self.path)
        if not existing.empty:
            existing_dates = pd.to_datetime(
                existing["source_decision_date"].fillna(existing["date"]),
                errors="coerce",
            ).dt.date.astype("string")
            existing = existing.loc[existing_dates != decision].copy()
        output = pd.concat(
            [existing, pd.DataFrame([new_record], columns=self.COLUMNS)],
            ignore_index=True,
        )
        output["_decision_order"] = pd.to_datetime(
            output["source_decision_date"].fillna(output["date"]), errors="coerce"
        )
        output = output.sort_values(
            ["_decision_order", "run_date"], kind="stable", na_position="first"
        ).drop(columns="_decision_order")
        # Legacy tapes have no lineage column.  When pandas adds that nullable
        # column, concatenating a bool can coerce True to the float 1.0.  Keep a
        # stable textual representation so the publication contract is not
        # dependent on pandas' inferred dtype.
        output["execution_lineage_verified"] = output[
            "execution_lineage_verified"
        ].map(_canonical_csv_flag)
        output[self.COLUMNS].to_csv(self.path, index=False)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_row_from_csv(path: str) -> Optional[pd.Series]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    date_col = None
    for c in ["Date", "decision_date", "target_date", "asof_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if df.empty:
            return None
    return df.iloc[-1]


def read_latest_signal(cfg: BotConfig) -> Dict[str, Any]:
    signal: Dict[str, Any] = {
        "p_tail": cfg.base_rate_default,
        "base_rate": cfg.base_rate_default,
        "publish_mode": "UNKNOWN",
        "final_pass": False,
        "px_voo_t": np.nan,
        "px_ief_t": np.nan,
        "px_voo_call_1d": np.nan,
        "px_ief_call_1d": np.nan,
        "raw_val_auc": np.nan,
        "source_decision_date": None,
        "model_protocol_version": None,
        "model_code_sha": None,
        "source": "default",
    }

    # FIX (Finding 15, Audit 2026-04-21):
    # The prior code tried part2_g532_summary.json first. That file is a model-level
    # aggregate and contains neither p_final_cal nor base_rate at the live-row level.
    # Both fields fell through to cfg.base_rate_default (0.19), making edge=0.0 on
    # every run and corrupting the signal_log.csv diagnostic trail.
    #
    # Priority order:
    #   1. current_target_weights.json (Part 7 output) — carries p_tail_base,
    #      base_rate, edge, publish_mode, final_pass from the live run.
    #   2. Part 2 consensus tape last row — has p_final_cal and T (base_rate).
    #   3. Part 2 summary JSON — model-level aggregate; used only for AUC.
    ctw_path = cfg.part7_current_target_path
    if os.path.exists(ctw_path):
        try:
            ctw = _read_json(ctw_path)
            p_tail_raw = _safe_float(ctw.get("p_tail_base"), np.nan)
            base_rate_raw = _safe_float(ctw.get("base_rate"), np.nan)
            if np.isfinite(p_tail_raw) and np.isfinite(base_rate_raw):
                signal.update({
                    "p_tail":       float(p_tail_raw),
                    "base_rate":    float(base_rate_raw),
                    "publish_mode": str(ctw.get("publish_mode", "UNKNOWN")),
                    "final_pass":   bool(ctw.get("final_pass", False)),
                    "raw_val_auc":  _safe_float(ctw.get("raw_val_auc"), np.nan),
                    "source_decision_date": ctw.get("Date") or ctw.get("decision_date"),
                    "model_protocol_version": ctw.get("model_protocol_version"),
                    "model_code_sha": ctw.get("model_code_sha"),
                    "source": "part7_current_target_weights",
                })
                print(
                    f"[Bot] Signal from current_target_weights: p_tail={signal['p_tail']:.4f}, "
                    f"base_rate={signal['base_rate']:.4f}, edge={signal['base_rate']-signal['p_tail']:+.4f}, "
                    f"mode={signal['publish_mode']}"
                )
                return signal
        except Exception as e:
            print(f"[Bot] Warning: could not read current_target_weights ({e})")

    # Fallback 2: Part 2 tape last row (has p_final_cal and T)
    row = _latest_row_from_csv(cfg.part2_tape_path)
    if row is not None:
        p_tail_row = _safe_float(row.get("p_final_cal", cfg.base_rate_default), cfg.base_rate_default)
        base_rate_row = _safe_float(row.get("T", row.get("base_rate", cfg.base_rate_default)), cfg.base_rate_default)
        if np.isfinite(p_tail_row) and np.isfinite(base_rate_row):
            signal.update({
                "p_tail":        p_tail_row,
                "base_rate":     base_rate_row,
                "publish_mode":  str(row.get("publish_mode", "UNKNOWN")),
                "final_pass":    bool(row.get("final_pass", False)),
                "raw_val_auc":   _safe_float(row.get("raw_val_auc"), np.nan),
                "source_decision_date": row.get("Date", row.get("decision_date")),
                "model_protocol_version": row.get("model_protocol_version"),
                "model_code_sha": row.get("model_code_sha"),
                "px_voo_t":      _safe_float(row.get("px_voo_t"), np.nan),
                "px_ief_t":      _safe_float(row.get("px_ief_t"), np.nan),
                "px_voo_call_1d": _safe_float(row.get("px_voo_call_1d", row.get("px_voo_call_7d")), np.nan),
                "px_ief_call_1d": _safe_float(row.get("px_ief_call_1d", row.get("px_ief_call_7d")), np.nan),
                "source": "part2_tape_last_row",
            })
            print(f"[Bot] Signal from Part 2 tape: p_tail={signal['p_tail']:.4f}, "
                  f"base_rate={signal['base_rate']:.4f}, edge={signal['base_rate']-signal['p_tail']:+.4f}")
            return signal

    # Fallback 3: Part 2 summary JSON (model-level only — no row-level probabilities)
    if os.path.exists(cfg.part2_summary_path):
        try:
            s = _read_json(cfg.part2_summary_path)
            signal.update({
                "raw_val_auc":  _safe_float(s.get("raw_val_auc_median"), np.nan),
                "publish_mode": str(s.get("publish_mode", "UNKNOWN")),
                "final_pass":   bool(s.get("final_pass", False)),
                "source": "part2_summary_auc_only",
            })
            print("[Bot] Warning: Part 2 summary has no row-level probabilities. "
                  "p_tail and base_rate using defaults — signal_log edge will be ~0.")
        except Exception as e:
            print(f"[Bot] Warning: could not read Part 2 summary ({e})")

    print("[Bot] Warning: no reliable signal source found — using defaults.")
    return signal


def check_accuracy_gate(cfg: BotConfig) -> Tuple[bool, str]:
    if not cfg.require_significance:
        return True, "accuracy_gate_disabled"
    if not os.path.exists(cfg.part9_report_path):
        return False, f"Part 9 report not found at {cfg.part9_report_path}"

    try:
        report = _read_json(cfg.part9_report_path)
        n_live = int(report.get("n_live_realized", 0))
        if n_live < cfg.min_live_n:
            return False, f"Only {n_live} live realized obs (need {cfg.min_live_n})"

        health = str(report.get("health_status", "IMMATURE")).upper()
        if health == "SUSPEND":
            reasons = report.get("health_reasons", [])
            return False, f"Part 9 SUSPEND: {reasons}"

        cs = report.get("classification_stats_live", {}) or {}
        cal = report.get("calibration_live", {}) or {}

        t_auc = _safe_float(cs.get("t_stat_auc"), np.nan)
        acc = _safe_float(cs.get("balanced_accuracy"), np.nan)
        bss = _safe_float(cs.get("brier_skill_score"), np.nan)
        ece = _safe_float(cal.get("ece"), np.nan)

        if cfg.fail_closed_on_nonfinite_t and not np.isfinite(t_auc):
            return False, "Live AUC t-stat unavailable — fail closed"
        if np.isfinite(t_auc) and t_auc < cfg.min_t_stat:
            return False, f"Live AUC t-stat {t_auc:.2f} < {cfg.min_t_stat:.2f}"
        if np.isfinite(acc) and acc < 0.50:
            return False, f"Live balanced accuracy {acc:.2%} < 50%"
        if np.isfinite(bss) and bss <= 0.0:
            return False, f"Live Brier skill {bss:.4f} <= 0"
        if np.isfinite(ece) and ece > 0.15:
            return False, f"Live ECE {ece:.3f} > 0.15"

        if cfg.enforce_tc_gate:
            annual_tc_drag_bps = _safe_float(report.get("annual_tc_drag_bps"), np.nan)
            annual_edge_bps = _safe_float(cs.get("estimated_annual_edge_bps"), np.nan)
            if np.isfinite(annual_tc_drag_bps) and np.isfinite(annual_edge_bps) and annual_tc_drag_bps >= annual_edge_bps:
                return False, f"Annual TC drag {annual_tc_drag_bps:.1f} bps >= estimated annual edge {annual_edge_bps:.1f} bps"

        return True, f"Model passes live gate: t_auc={t_auc:.2f}, balanced_acc={acc:.2%}, bss={bss:.4f}, ece={ece:.3f}, n={n_live}"
    except Exception as e:
        return False, f"Error reading Part 9 report: {e}"


def read_strategy_target(cfg: BotConfig) -> Optional[Tuple[float, str]]:
    if os.path.exists(cfg.part7_current_target_path):
        try:
            obj = _read_json(cfg.part7_current_target_path)
            w = _safe_float(obj.get("w_target_voo"), np.nan)
            if np.isfinite(w):
                return float(np.clip(w, cfg.min_position_voo, cfg.max_position_voo)), "part7_current_target"
        except Exception as e:
            print(f"[Bot] Warning: could not read Part 7 current target ({e})")

    row = _latest_row_from_csv(cfg.part7_weights_tape_path)
    if row is not None:
        w = _safe_float(row.get("w_target_voo"), np.nan)
        if np.isfinite(w):
            return float(np.clip(w, cfg.min_position_voo, cfg.max_position_voo)), "part7_weights_tape"
    return None


def compute_target_weight(
    p_tail: float,
    base_rate: float,
    publish_mode: str,
    final_pass: bool,
    cfg: BotConfig,
) -> Tuple[float, str]:
    neutral = cfg.default_w_equity
    position_scale = 2.0
    fail_modes = {"FAIL_CLOSED_NEUTRAL", "FAIL_CLOSED", "SHADOW", "UNKNOWN"}
    if publish_mode in fail_modes or not final_pass:
        return neutral, f"neutral_model_not_cleared ({publish_mode})"

    edge = base_rate - p_tail
    if abs(edge) < cfg.min_signal_edge:
        return neutral, f"neutral_signal_too_weak (|edge|={abs(edge):.3f} < {cfg.min_signal_edge})"

    w_voo = neutral + edge * position_scale
    w_voo = float(np.clip(w_voo, cfg.min_position_voo, cfg.max_position_voo))
    direction = "long_VOO" if w_voo > neutral else "long_IEF"
    return w_voo, f"{direction} (edge={edge:+.3f} → w_voo={w_voo:.3f})"


def load_completed_session_prices(
    cfg: BotConfig, session: str
) -> Dict[str, float]:
    """Load exact, observed Part 0 closes for the governed session."""
    meta = _read_json(cfg.part0_meta_path)
    if meta.get("market_calendar") != "XNYS":
        raise RuntimeError("Part 0 metadata does not declare the XNYS calendar")
    if meta.get("market_values_are_raw_observations") is not True:
        raise RuntimeError("Part 0 metadata does not guarantee raw market observations")
    if _canonical_iso_date(meta.get("market_data_asof"), "Part 0 market_data_asof") != session:
        raise RuntimeError(
            "Part 0 market_data_asof does not match the governed bot session"
        )

    try:
        close = pd.read_parquet(cfg.part0_close_path)
        observed = pd.read_parquet(cfg.part0_observation_mask_path)
    except Exception as exc:
        raise RuntimeError(f"could not load verified Part 0 market inputs: {exc}") from exc

    for label, frame in (("close prices", close), ("observation mask", observed)):
        index = pd.to_datetime(frame.index, errors="coerce")
        if index.isna().any():
            raise RuntimeError(f"Part 0 {label} contain an invalid date index")
        if index.tz is not None:
            index = index.tz_convert(None)
        frame.index = index.normalize()

    expected = pd.Timestamp(session)
    close_rows = close.loc[close.index == expected]
    observed_rows = observed.loc[observed.index == expected]
    if len(close_rows) != 1 or len(observed_rows) != 1:
        raise RuntimeError(
            f"Part 0 does not contain exactly one observed row for {session}"
        )

    prices: Dict[str, float] = {}
    for ticker in (cfg.ticker_equity, cfg.ticker_bond):
        if ticker not in close_rows.columns or ticker not in observed_rows.columns:
            raise RuntimeError(f"Part 0 is missing required bot ticker {ticker}")
        observed_flag = _safe_float(observed_rows.iloc[0][ticker], np.nan)
        if observed_flag != 1.0:
            raise RuntimeError(f"Part 0 {ticker} close for {session} was not observed")
        price = _safe_float(close_rows.iloc[0][ticker], np.nan)
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(
                f"Part 0 {ticker} close for {session} is invalid: {price!r}"
            )
        prices[ticker] = float(price)
    return prices


def load_verified_bot_context(
    cfg: BotConfig,
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Bind the paper bot to the exact Part 8 and Part 0 production lineage."""
    session = _canonical_iso_date(
        os.environ.get("PRICECALL_RUN_DATE_ET")
        or latest_completed_xnys_session().date().isoformat(),
        "expected completed XNYS session",
    )
    instructions = _read_json(cfg.part8_instructions_path)
    if instructions.get("lineage_verified") is not True:
        raise RuntimeError("Part 8 instructions are not lineage-verified")
    if instructions.get("allocation_source") != "v1_fusion_allocations":
        raise RuntimeError("Part 8 instructions do not use the governed Part 3 allocation")
    if _identity_text(instructions.get("protocol_version")) != PROTOCOL_VERSION:
        raise RuntimeError("Part 8 protocol does not match the bot protocol")
    for label in ("decision_date", "source_decision_date", "pipeline_run_date"):
        actual = _canonical_iso_date(instructions.get(label), f"Part 8 {label}")
        if actual != session:
            raise RuntimeError(
                f"bot lineage date mismatch: Part 8 {label}={actual}, expected={session}"
            )

    identities = {
        "source_code_sha": (
            _identity_text(os.environ.get("PRICECALL_CODE_SHA") or os.environ.get("GITHUB_SHA")),
            _identity_text(instructions.get("source_code_sha")),
        ),
        "pipeline_run_id": (
            _identity_text(os.environ.get("GITHUB_RUN_ID")),
            _identity_text(instructions.get("pipeline_run_id")),
        ),
        "pipeline_run_attempt": (
            _identity_text(os.environ.get("GITHUB_RUN_ATTEMPT")),
            _identity_text(instructions.get("pipeline_run_attempt")),
        ),
    }
    for label, (expected, actual) in identities.items():
        if expected and actual != expected:
            raise RuntimeError(
                f"bot lineage {label} mismatch: actual={actual or 'missing'}, "
                f"expected={expected}"
            )

    prices = load_completed_session_prices(cfg, session)
    lineage = {
        "protocol_version": PROTOCOL_VERSION,
        "pipeline_run_date": session,
        "pipeline_run_id": identities["pipeline_run_id"][0]
        or identities["pipeline_run_id"][1]
        or None,
        "pipeline_run_attempt": identities["pipeline_run_attempt"][0]
        or identities["pipeline_run_attempt"][1]
        or None,
        "source_code_sha": identities["source_code_sha"][0]
        or identities["source_code_sha"][1]
        or None,
        "execution_lineage_verified": True,
        "price_source": "artifacts_part0/close_prices.parquet",
        "price_session": session,
    }
    return session, prices, lineage


def _latest_completed_close(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    # yfinance's end date is exclusive, so the last row is already the latest
    # completed session. Selecting -2 would skip an additional session.
    return float(values.iloc[-1])


def fetch_prices(tickers: List[str], use_prior_day: bool = True) -> Dict[str, float]:
    """Legacy diagnostic fetch; production uses ``load_completed_session_prices``."""
    prices: Dict[str, float] = {}
    try:
        completed = latest_completed_xnys_session().date()
        end = completed + timedelta(days=1)
        start = completed - timedelta(days=10)
        raw = yf.download(tickers, start=str(start), end=str(end), progress=False, auto_adjust=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for t in tickers:
            if t not in close.columns:
                continue
            price = _latest_completed_close(close[t])
            if np.isfinite(price):
                prices[t] = price
        print(f"[Bot] Prices fetched: {prices}")
    except Exception as e:
        print(f"[Bot] Warning: price fetch failed ({e})")
    return prices


def compute_performance(trade_log_df: pd.DataFrame, cfg: BotConfig) -> Dict[str, Any]:
    if trade_log_df.empty or "nav_after" not in trade_log_df.columns:
        return {"error": "No trades recorded yet", "as_of_date": date.today().isoformat()}

    df = trade_log_df.dropna(subset=["nav_after"]).copy().sort_values("decision_date")
    nav_series = pd.to_numeric(df["nav_after"], errors="coerce").dropna().values
    if len(nav_series) == 0:
        return {"error": "No valid NAV observations"}

    total_return = (nav_series[-1] - cfg.starting_capital) / cfg.starting_capital
    between_trade_returns = np.diff(nav_series) / nav_series[:-1] if len(nav_series) >= 2 else np.array([])
    if len(between_trade_returns) < 2:
        return {"total_return_pct": round(total_return * 100, 3), "n_trades": len(df)}

    # FIX (Finding C, Audit 2026-04): the original code applied sqrt(252) to
    # between-trade returns as if each represented one trading day.  At ~1.6
    # trades/year the actual interval is weeks to months; sqrt(252) overstates
    # the annualized Sharpe ratio by sqrt(252 / trades_per_year).
    #
    # Correct approach: convert each between-trade return to an annualized return
    # using the actual elapsed calendar days, then compute mean/std across those
    # annualized values.  This is equivalent to time-weighting each observation.
    dates = pd.to_datetime(df["decision_date"], errors="coerce").dropna()
    intervals_days = dates.diff().dt.days.dropna().values.astype(float)
    n_pairs = min(len(between_trade_returns), len(intervals_days))
    valid = (
        (intervals_days[:n_pairs] > 0)
        & np.isfinite(between_trade_returns[:n_pairs])
    )
    if valid.sum() >= 2:
        ann_returns = (
            between_trade_returns[:n_pairs][valid]
            * (252.0 / intervals_days[:n_pairs][valid])
        )
        mean_ann = float(np.mean(ann_returns))
        std_ann = float(np.std(ann_returns, ddof=1))
        sharpe = mean_ann / (std_ann + 1e-10) if std_ann > 0 else np.nan
    else:
        # Fewer than 2 valid intervals: cannot estimate annualized Sharpe reliably.
        sharpe = np.nan

    running_max = np.maximum.accumulate(nav_series)
    drawdowns = (nav_series - running_max) / running_max
    max_dd = float(drawdowns.min())
    win_rate = float((between_trade_returns > 0).mean())

    total_tc = float(
        pd.to_numeric(
            df.get("tc_dollars", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0).sum()
    )
    tc_drag_bps = float(total_tc / cfg.starting_capital * 10_000.0)

    return {
        "n_trades": len(df),
        "total_return_pct": round(total_return * 100.0, 4),
        "sharpe_ratio": round(sharpe, 4) if np.isfinite(sharpe) else None,
        "max_drawdown_pct": round(max_dd * 100.0, 4),
        "win_rate_pct": round(win_rate * 100.0, 2),
        "total_tc_dollars": round(total_tc, 4),
        "total_tc_drag_bps": round(tc_drag_bps, 2),
        "current_nav": round(float(nav_series[-1]), 4),
        "peak_nav": round(float(np.max(nav_series)), 4),
        "stop_loss_floor": cfg.stop_loss_floor,
        "is_stopped": bool(
            df.get("is_stopped", pd.Series(dtype=bool)).fillna(False).any()
        ),
    }


def run_daily(cfg: BotConfig = CFG) -> int:
    _colab_init([])
    root = _resolve_root()
    cfg = BotConfig(
        version=cfg.version,
        root_dir=str(root),
        part2_summary_path=_abs_path(cfg.part2_summary_path),
        part2_tape_path=_abs_path(cfg.part2_tape_path),
        part0_close_path=_abs_path(cfg.part0_close_path),
        part0_observation_mask_path=_abs_path(cfg.part0_observation_mask_path),
        part0_meta_path=_abs_path(cfg.part0_meta_path),
        part7_current_target_path=_abs_path(cfg.part7_current_target_path),
        part7_weights_tape_path=_abs_path(cfg.part7_weights_tape_path),
        part8_instructions_path=_abs_path(cfg.part8_instructions_path),
        part9_report_path=_abs_path(cfg.part9_report_path),
        bot_dir=_abs_path(cfg.bot_dir),
        starting_capital=cfg.starting_capital,
        stop_loss_floor=cfg.stop_loss_floor,
        max_position_voo=cfg.max_position_voo,
        min_position_voo=cfg.min_position_voo,
        dead_band=cfg.dead_band,
        min_signal_edge=cfg.min_signal_edge,
        base_rate_default=cfg.base_rate_default,
        require_significance=cfg.require_significance,
        min_t_stat=cfg.min_t_stat,
        min_live_n=cfg.min_live_n,
        fail_closed_on_nonfinite_t=cfg.fail_closed_on_nonfinite_t,
        enforce_tc_gate=cfg.enforce_tc_gate,
        use_prior_day_close=cfg.use_prior_day_close,
        tc_bps_one_way=cfg.tc_bps_one_way,
        ticker_equity=cfg.ticker_equity,
        ticker_bond=cfg.ticker_bond,
        default_w_equity=cfg.default_w_equity,
        default_w_bond=cfg.default_w_bond,
    )
    os.makedirs(cfg.bot_dir, exist_ok=True)

    run_date = date.today().isoformat()
    print("=" * 70)
    print(f"PART 10 — DAILY TRADING BOT | {run_date}")
    print("=" * 70)

    # Validate every upstream identity and the exact observed price row before
    # constructors migrate or otherwise touch persistent bot ledgers.
    try:
        decision_date, prices, execution_lineage = load_verified_bot_context(cfg)
    except Exception as exc:
        print(f"[Bot] ERROR: verified execution context unavailable ({exc})")
        return 1

    state_path = os.path.join(cfg.bot_dir, "portfolio_state.json")
    trade_log = TradeLog(os.path.join(cfg.bot_dir, "trade_log.csv"))
    signal_log = SignalLog(os.path.join(cfg.bot_dir, "signal_log.csv"))
    portfolio = PaperPortfolio(cfg)

    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        portfolio.cash = float(state.get("cash", 0.0))
        portfolio.shares_voo = float(state.get("shares_voo", 0.0))
        portfolio.shares_ief = float(state.get("shares_ief", 0.0))
        portfolio.peak_nav = float(state.get("peak_nav", cfg.starting_capital))
        portfolio.is_stopped = bool(state.get("is_stopped", False))
        portfolio.trade_count = int(state.get("trade_count", 0))
        portfolio._initialized = bool(state.get("initialized", True))
        print(f"[Bot] Portfolio state loaded (trade #{portfolio.trade_count})")
    else:
        print("[Bot] No saved state found — will initialize after gate decision")

    px_voo = prices.get(cfg.ticker_equity, np.nan)
    px_ief = prices.get(cfg.ticker_bond, np.nan)
    if not np.isfinite(px_voo) or not np.isfinite(px_ief):
        print(f"[Bot] ERROR: invalid prices VOO={px_voo}, IEF={px_ief}")
        return 1

    signal = read_latest_signal(cfg)
    p_tail = _safe_float(signal.get("p_tail"), cfg.base_rate_default)
    base_rate = _safe_float(signal.get("base_rate"), cfg.base_rate_default)
    publish_mode = str(signal.get("publish_mode", "UNKNOWN"))
    final_pass = bool(signal.get("final_pass", False))
    edge = base_rate - p_tail

    try:
        signal_date = _canonical_iso_date(
            signal.get("source_decision_date"), "bot signal source_decision_date"
        )
    except RuntimeError as exc:
        print(f"[Bot] ERROR: {exc}")
        return 1
    if signal_date != decision_date:
        print(
            f"[Bot] ERROR: signal session {signal_date} does not match "
            f"execution session {decision_date}"
        )
        return 1
    if _identity_text(signal.get("model_protocol_version")) != PROTOCOL_VERSION:
        print("[Bot] ERROR: signal protocol does not match the execution protocol")
        return 1
    signal_sha = _identity_text(signal.get("model_code_sha"))
    execution_sha = _identity_text(execution_lineage.get("source_code_sha"))
    if execution_sha and signal_sha != execution_sha:
        print(
            f"[Bot] ERROR: signal source SHA {signal_sha or 'missing'} does not "
            f"match execution source SHA {execution_sha}"
        )
        return 1

    gate_passed, gate_reason = check_accuracy_gate(cfg)
    dry_run = not gate_passed
    print(f"[Bot] {'DRY_RUN' if dry_run else 'LIVE'} mode — {gate_reason}")

    if not portfolio._initialized:
        if dry_run:
            portfolio.initialize_cash_only()
        else:
            portfolio.initialize(px_voo, px_ief)

    strategy_target = None
    fail_modes = {"FAIL_CLOSED_NEUTRAL", "FAIL_CLOSED", "SHADOW", "UNKNOWN"}
    if publish_mode not in fail_modes and final_pass:
        strategy_target = read_strategy_target(cfg)
    if strategy_target is not None:
        target_w_voo, target_src = strategy_target
        action_reason = f"{target_src} (w_voo={target_w_voo:.3f})"
    else:
        target_w_voo, action_reason = compute_target_weight(p_tail, base_rate, publish_mode, final_pass, cfg)
        target_src = "heuristic_fallback"

    print(f"[Bot] Signal: p_tail={p_tail:.4f}, base_rate={base_rate:.4f}, edge={edge:+.4f}")
    print(f"[Bot] Target w_VOO={target_w_voo:.3f} | Source: {target_src} | Reason: {action_reason}")

    current_nav = portfolio.nav(px_voo, px_ief)
    if (not portfolio.is_stopped and portfolio._initialized and current_nav <= cfg.stop_loss_floor):
        print(f"[Bot] Passive stop-loss check: NAV=${current_nav:.2f} ≤ floor=${cfg.stop_loss_floor:.2f}")
        portfolio._trigger_stop_loss(px_voo, px_ief, decision_date)

    trade_record = portfolio.rebalance(
        target_w_voo=target_w_voo,
        px_voo=px_voo,
        px_ief=px_ief,
        decision_date=decision_date,
        signal=p_tail,
        action_reason=action_reason,
        dry_run=(dry_run or portfolio.is_stopped),
        target_source=target_src,
    )
    if trade_record:
        trade_record.update(execution_lineage)
        trade_log.append(trade_record)

    post_nav = portfolio.nav(px_voo, px_ief)
    signal_log.append({
        "date": decision_date,
        "run_date": run_date,
        "source_decision_date": decision_date,
        "model_protocol_version": signal.get("model_protocol_version"),
        "model_code_sha": signal.get("model_code_sha"),
        "pipeline_run_date": execution_lineage["pipeline_run_date"],
        "pipeline_run_id": execution_lineage["pipeline_run_id"],
        "pipeline_run_attempt": execution_lineage["pipeline_run_attempt"],
        "execution_source_code_sha": execution_lineage["source_code_sha"],
        "execution_lineage_verified": True,
        "price_source": execution_lineage["price_source"],
        "price_session": execution_lineage["price_session"],
        "p_tail": round(p_tail, 6),
        "base_rate": round(base_rate, 6),
        "edge": round(edge, 6),
        "target_w_voo": round(target_w_voo, 4),
        "action_reason": action_reason,
        "target_source": target_src,
        "dry_run": dry_run,
        "accuracy_gate_passed": gate_passed,
        "accuracy_gate_reason": gate_reason,
        "publish_mode": publish_mode,
        "final_pass": final_pass,
        "raw_val_auc": round(_safe_float(signal.get("raw_val_auc"), np.nan), 4),
        "px_voo": round(px_voo, 4),
        "px_ief": round(px_ief, 4),
        "nav": round(post_nav, 4),
    })

    state = portfolio.summary(px_voo, px_ief)
    state.update(execution_lineage)
    state["decision_date"] = decision_date
    state["last_updated"] = run_date
    state["px_voo"] = round(float(px_voo), 4)
    state["px_ief"] = round(float(px_ief), 4)
    write_json_strict(state_path, state)

    perf = compute_performance(trade_log.load(), cfg)
    perf.update(execution_lineage)
    perf["decision_date"] = decision_date
    perf["as_of_date"] = run_date
    perf_path = os.path.join(cfg.bot_dir, "performance_report.json")
    write_json_strict(perf_path, perf)

    w_voo, w_ief = portfolio.weights(px_voo, px_ief)
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"  Decision session:  {decision_date}")
    print(f"  NAV:               ${post_nav:.2f} ({(post_nav - cfg.starting_capital) / cfg.starting_capital:+.2%})")
    print(f"  Stop-loss floor:   ${cfg.stop_loss_floor:.2f} (${post_nav - cfg.stop_loss_floor:.2f} buffer)")
    print(f"  Current weights:   VOO {w_voo:.1%} / IEF {w_ief:.1%}")
    print(f"  Trades executed:   {portfolio.trade_count}")
    print(f"  Sharpe (running):  {perf.get('sharpe_ratio', 'N/A')}")
    print(f"  Max drawdown:      {perf.get('max_drawdown_pct', 'N/A')}%")
    print(f"  Total TC drag:     {perf.get('total_tc_drag_bps', 'N/A')} bps")
    print(f"  Mode:              {'⛔ STOPPED' if portfolio.is_stopped else '🟡 DRY_RUN' if dry_run else '🟢 LIVE'}")
    print("\n✅ PART 10 COMPLETE")
    print(f"   Trade log:    {trade_log.path}")
    print(f"   Signal log:   {signal_log.path}")
    print(f"   State:        {state_path}")
    print(f"   Performance:  {perf_path}")
    return 0


def main() -> int:
    _colab_init([])
    return run_daily(CFG)


if __name__ == "__main__":
    raise SystemExit(main())
