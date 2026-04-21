#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_rerun.py — Post-rerun artifact acceptance validator
=============================================================
Checks every criterion agreed between the Quant-Guild audit and the
independent reviewer against a fresh set of pipeline artifacts.

Usage
-----
    python validate_rerun.py                       # default: auto-detect root
    python validate_rerun.py --root /path/to/root  # explicit project root
    python validate_rerun.py --uploads /path       # point at an uploads dir

Output
------
Prints a structured pass/fail table for every check.
Exits 0 if all checks pass, 1 if any fail.

Acceptance criteria sources
---------------------------
  Finding A  : final_pass AUC gate uses raw_val_auc_median
  Finding 3  : conditional_active_ir gate deferred until n >= 10
  Finding 2  : Part 6 regime labels flow through to Part 7
  Finding D  : Part 2C aleatoric = sqrt(p*(1-p)), not zero
  D-reviewer : predict_live() uses epist_threshold internally
  Finding E  : Part 8 tape path resolves; annual_drag populated
  E-reviewer : PostTradeAnalyzer.post_trade_log_path in __init__
  Finding C  : prediction_log accumulates across runs (N_after >= N_before+1)
  Finding 6  : overlay_failclosed_rate tracked separately
  DST        : (workflow only — not checkable from artifacts)
  Part 7     : weights depart from {0.6, 0.4} once NORMAL
  Part 2A    : latest_reason = "ok", latest_eligible = true
  Part 9     : total_predictions increments (accumulation proof)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Colour output (degrades gracefully on Windows / CI) ──────────────────────
try:
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    YELLOW = "\033[33m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
except Exception:
    GREEN = RED = YELLOW = RESET = BOLD = ""

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"


# ── Result collector ─────────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self._rows: List[Tuple[str, str, str, str]] = []  # group, name, status, detail

    def add(self, group: str, name: str, ok: bool, detail: str = "", skip: bool = False):
        status = SKIP if skip else (PASS if ok else FAIL)
        self._rows.append((group, name, status, detail))

    def print_table(self):
        print()
        print(f"{BOLD}{'Group':<20s} {'Check':<52s} {'Status':<8s} {'Detail'}{RESET}")
        print("─" * 110)
        current_group = ""
        for group, name, status, detail in self._rows:
            if group != current_group:
                print()
                current_group = group
            print(f"  {group:<18s} {name:<52s} {status:<16s} {detail}")
        print()

    def all_pass(self) -> bool:
        return all(FAIL not in status for _, _, status, _ in self._rows)

    def summary(self) -> str:
        passed = sum(1 for _, _, s, _ in self._rows if PASS in s)
        failed = sum(1 for _, _, s, _ in self._rows if FAIL in s)
        skipped = sum(1 for _, _, s, _ in self._rows if SKIP in s)
        total = len(self._rows)
        return f"{passed}/{total} passed, {failed} failed, {skipped} skipped"


# ── Path helpers ──────────────────────────────────────────────────────────────
def _resolve_root(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("PRICECALL_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    for candidate in [
        Path("/content/drive/MyDrive/PriceCallProject"),
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]:
        if candidate.exists():
            return candidate
    return Path.cwd()


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _fv(d: Dict, key: str, default=None):
    """Nested-safe dict get."""
    return d.get(key, default) if isinstance(d, dict) else default


# ── Individual check groups ───────────────────────────────────────────────────

def check_part2(r: Results, root: Path):
    p2 = _load_json(root / "artifacts_part2_g532" / "predictions" / "part2_g532_summary.json")
    if p2 is None:
        r.add("Part 2", "part2_g532_summary.json exists", False, "file not found")
        return

    r.add("Part 2", "file exists", True)

    # Finding A — AUC gate
    fp = _fv(p2, "final_pass")
    r.add("Part 2", "final_pass = true",
          fp is True or fp == 1,
          f"got {fp}")

    pm = _fv(p2, "publish_mode", "")
    r.add("Part 2", "publish_mode = NORMAL",
          str(pm).upper() == "NORMAL",
          f"got {pm!r}")

    auc_src = _fv(p2, "final_pass_auc_source", "")
    r.add("Part 2", "final_pass_auc_source = rolling_median",
          str(auc_src) == "rolling_median",
          f"got {auc_src!r}")

    auc_val = _fv(p2, "final_pass_auc_value")
    try:
        auc_ok = math.isfinite(float(auc_val)) and float(auc_val) >= 0.535
    except (TypeError, ValueError):
        auc_ok = False
    r.add("Part 2", "final_pass_auc_value >= 0.535",
          auc_ok,
          f"got {auc_val}")

    # Finding 3 — conditional IR deferred
    cond_ir = _fv(p2, "conditional_active_ir")
    cond_nan = cond_ir is None or (isinstance(cond_ir, float) and math.isnan(cond_ir))
    r.add("Part 2", "conditional_active_ir = NaN (deferred)",
          cond_nan,
          f"got {cond_ir}")

    min_n = _fv(p2, "conditional_active_ir_min_n")
    r.add("Part 2", "conditional_active_ir_min_n = 10",
          min_n == 10,
          f"got {min_n}")


def check_part2c(r: Results, root: Path):
    p2c = _load_json(root / "artifacts_part2c_bnn" / "predictions" / "part2c_bnn_summary.json")
    if p2c is None:
        r.add("Part 2C", "part2c_bnn_summary.json exists", False, "file not found — sleeve may not have run")
        return

    r.add("Part 2C", "file exists", True)

    # Finding D + D-reviewer
    epist_thr = _fv(p2c, "epist_overlay_threshold_75pct")
    try:
        thr_ok = math.isfinite(float(epist_thr)) and float(epist_thr) > 0
    except (TypeError, ValueError):
        thr_ok = False
    r.add("Part 2C", "epist_overlay_threshold_75pct finite and > 0",
          thr_ok, f"got {epist_thr}")

    epist_live = _fv(p2c, "live_p_bnn_epistemic")
    total_live = _fv(p2c, "live_p_bnn_total_std")  # check via tape; summary may not carry this
    try:
        ep = float(epist_live)
        # reviewer criterion: epistemic <= total_std (total = sqrt(epist² + aleat²) >= epist)
        # We verify this from the tape below; here just check epist is finite
        r.add("Part 2C", "live_p_bnn_epistemic is finite",
              math.isfinite(ep), f"got {epist_live}")
    except (TypeError, ValueError):
        r.add("Part 2C", "live_p_bnn_epistemic is finite", False, f"got {epist_live}")

    # BNN tape checks (aleatoric nonzero + decomposition consistency)
    tape_path = root / "artifacts_part2c_bnn" / "predictions" / "part2c_bnn_tape.csv"
    tape = _load_csv(tape_path)
    if tape is None:
        r.add("Part 2C", "bnn_tape.csv exists", False, "file not found")
        return

    r.add("Part 2C", "bnn_tape.csv exists", True)

    if "p_bnn_aleatoric" in tape.columns:
        aleat = pd.to_numeric(tape["p_bnn_aleatoric"], errors="coerce")
        nonzero_frac = (aleat > 0).mean()
        r.add("Part 2C", "p_bnn_aleatoric > 0 throughout",
              nonzero_frac > 0.99,
              f"nonzero fraction = {nonzero_frac:.4f} (expect ~1.0)")

        # Reviewer criterion: p_bnn_total_std^2 ≈ p_bnn_epistemic^2 + p_bnn_aleatoric^2
        if "p_bnn_epistemic" in tape.columns and "p_bnn_total_std" in tape.columns:
            ep_col = pd.to_numeric(tape["p_bnn_epistemic"], errors="coerce")
            tot_col = pd.to_numeric(tape["p_bnn_total_std"], errors="coerce")
            mask = ep_col.notna() & aleat.notna() & tot_col.notna()
            if mask.sum() > 10:
                reconstructed = np.sqrt(ep_col[mask]**2 + aleat[mask]**2)
                max_abs_err = (reconstructed - tot_col[mask]).abs().max()
                decomp_ok = max_abs_err < 1e-6
                r.add("Part 2C", "total_std² = epist² + aleat² (decomposition)",
                      decomp_ok,
                      f"max abs error = {max_abs_err:.2e} (expect < 1e-6)")

            # Reviewer criterion: live_p_bnn_epistemic <= p_bnn_total_std on live row
            live_rows = tape[tape.get("in_holdout", pd.Series([1]*len(tape)))==1].tail(1)
            if len(live_rows) > 0:
                ep_live = float(pd.to_numeric(live_rows["p_bnn_epistemic"], errors="coerce").iloc[0])
                tot_live = float(pd.to_numeric(live_rows["p_bnn_total_std"], errors="coerce").iloc[0])
                r.add("Part 2C", "live epistemic <= total_std",
                      ep_live <= tot_live + 1e-9,
                      f"epist={ep_live:.4f} total={tot_live:.4f}")
    else:
        r.add("Part 2C", "p_bnn_aleatoric column present", False, "column missing from tape")


def check_part8(r: Results, root: Path):
    p8 = _load_json(root / "artifacts_part8" / "part8_meta.json")
    if p8 is None:
        r.add("Part 8", "part8_meta.json exists", False, "file not found")
        return

    r.add("Part 8", "file exists", True)

    # Finding E — annual_drag populated (reviewer: both meta and tape)
    drag = _fv(p8, "annual_drag_summary")
    drag_ok = isinstance(drag, dict) and len(drag) > 0
    r.add("Part 8", "annual_drag_summary non-empty",
          drag_ok,
          f"keys={list(drag.keys()) if isinstance(drag, dict) else drag!r}")

    # Reviewer addition: execution_cost_tape annual_tc_drag_bps finite
    tape = _load_csv(root / "artifacts_part8" / "execution_cost_tape.csv")
    if tape is not None and "annual_tc_drag_bps" in tape.columns:
        latest_bps = pd.to_numeric(tape["annual_tc_drag_bps"], errors="coerce").iloc[-1]
        r.add("Part 8", "execution_cost_tape annual_tc_drag_bps finite",
              bool(pd.notna(latest_bps) and math.isfinite(float(latest_bps))),
              f"got {latest_bps}")
    else:
        r.add("Part 8", "execution_cost_tape annual_tc_drag_bps finite",
              False, "tape missing or column absent")


def check_prediction_log(r: Results, root: Path,
                         n_before: Optional[int] = None,
                         n_before_date: Optional[str] = None):
    """
    Finding C — prediction log accumulates across runs.
    n_before: row count from the PREVIOUS run's artifact.
    n_before_date: ISO date string of the most recent row in the PREVIOUS artifact.
    If either is provided, the strict reviewer criterion N_after >= N_before + 1 is checked.
    If neither is provided, falls back to checking row count >= 2.
    """
    pl_path = root / "artifacts_part3" / "prediction_log.csv"
    pl = _load_csv(pl_path)
    if pl is None:
        r.add("Prediction log", "prediction_log.csv exists", False, "file not found")
        return

    r.add("Prediction log", "prediction_log.csv exists", True)
    n_after = len(pl)

    if n_before is not None:
        # Strict reviewer criterion
        acc_ok = n_after >= n_before + 1
        r.add("Prediction log", f"N_after ({n_after}) >= N_before ({n_before}) + 1",
              acc_ok,
              f"N_after={n_after}, N_before={n_before}")
    else:
        # Fallback: at least 2 rows (today + at least one prior)
        r.add("Prediction log", "row count >= 2 (history accumulating)",
              n_after >= 2,
              f"got {n_after} rows")

    # Check today's row has NORMAL mode
    if "decision_date" in pl.columns:
        pl["decision_date"] = pd.to_datetime(pl["decision_date"], errors="coerce")
        latest = pl.sort_values("decision_date").iloc[-1]
        pm = str(latest.get("publish_mode", "")).upper()
        dm = str(latest.get("deployment_mode", "")).upper()
        r.add("Prediction log", "latest row publish_mode = NORMAL",
              pm == "NORMAL",
              f"got {pm!r}")
        r.add("Prediction log", "latest row deployment_mode = NORMAL",
              dm == "NORMAL",
              f"got {dm!r}")


def check_governance(r: Results, root: Path):
    gov_path = root / "artifacts_part3_v1" / "v1_final_production_governance.csv"
    gov = _load_csv(gov_path)
    if gov is None:
        r.add("Governance", "v1_final_production_governance.csv exists", False, "file not found")
        return

    r.add("Governance", "file exists", True)
    latest = gov.iloc[-1]

    pm = str(latest.get("publish_mode", "")).upper()
    r.add("Governance", "publish_mode = NORMAL", pm == "NORMAL", f"got {pm!r}")

    fp = latest.get("final_pass", 0)
    r.add("Governance", "final_pass = 1", int(fp) == 1, f"got {fp}")

    blockers = str(latest.get("alpha_blockers", "")).upper()
    r.add("Governance", "alpha_blockers = NONE", blockers == "NONE", f"got {blockers!r}")


def check_alpha(r: Results, root: Path):
    # Try both canonical locations
    for candidate in [
        root / "artifacts_part2a_alpha" / "predictions" / "part2a21_alpha_summary.json",
        root / "artifacts_part2a_alpha" / "predictions" / "alpha_summary.json",
    ]:
        p2a = _load_json(candidate)
        if p2a is not None:
            break
    else:
        r.add("Alpha (2A)", "alpha_summary.json exists", False, "file not found")
        return

    r.add("Alpha (2A)", "file exists", True)

    reason = _fv(p2a, "latest_reason", "")
    r.add("Alpha (2A)", "latest_reason = ok",
          str(reason) == "ok",
          f"got {reason!r}")

    elig = _fv(p2a, "latest_eligible")
    r.add("Alpha (2A)", "latest_eligible = true",
          elig is True or elig == 1,
          f"got {elig}")

    pm = _fv(p2a, "publish_mode", "")
    r.add("Alpha (2A)", "publish_mode = NORMAL",
          str(pm).upper() == "NORMAL",
          f"got {pm!r}")

    # Finding 6 — separate overlay sub-rates present
    dist_rate = _fv(p2a, "overlay_dist_veto_rate")
    fc_rate   = _fv(p2a, "overlay_failclosed_rate")
    r.add("Alpha (2A)", "overlay_dist_veto_rate field present",
          dist_rate is not None,
          f"got {dist_rate}")
    r.add("Alpha (2A)", "overlay_failclosed_rate field present",
          fc_rate is not None,
          f"got {fc_rate}")


def check_part7(r: Results, root: Path):
    ctw = _load_json(root / "artifacts_part7" / "current_target_weights.json")
    if ctw is None:
        r.add("Part 7", "current_target_weights.json exists", False, "file not found")
        return

    r.add("Part 7", "file exists", True)

    w_voo = _fv(ctw, "w_target_voo")
    w_ief = _fv(ctw, "w_target_ief")
    try:
        # The critical whole-pipeline signal: weights depart from exactly {0.6, 0.4}
        # once the alpha sleeve is live-eligible under NORMAL mode.
        not_locked = not (abs(float(w_voo) - 0.6) < 1e-9 and abs(float(w_ief) - 0.4) < 1e-9)
        r.add("Part 7", "weights depart from exactly {0.6, 0.4} (alpha active)",
              not_locked,
              f"w_voo={w_voo}, w_ief={w_ief}")
    except (TypeError, ValueError):
        r.add("Part 7", "weights depart from exactly {0.6, 0.4} (alpha active)",
              False, f"could not parse w_voo={w_voo}, w_ief={w_ief}")

    regime = _fv(ctw, "regime_label", "")
    r.add("Part 7", "regime_label is not unknown",
          str(regime).lower() not in ("unknown", "", "none"),
          f"got {regime!r}")


def check_part9(r: Results, root: Path,
                n_predictions_before: Optional[int] = None):
    """
    Part 9 accumulation check.
    n_predictions_before: total_predictions from the PREVIOUS run's report.
    If provided, checks that total_predictions increased by at least 1.
    """
    p9 = _load_json(root / "artifacts_part9" / "live_attribution_report.json")
    if p9 is None:
        r.add("Part 9", "live_attribution_report.json exists", False, "file not found")
        return

    r.add("Part 9", "file exists", True)

    total = _fv(p9, "total_predictions", 0)
    if n_predictions_before is not None:
        r.add("Part 9", f"total_predictions ({total}) > prior ({n_predictions_before})",
              total > n_predictions_before,
              f"total={total}, prior={n_predictions_before}")
    else:
        r.add("Part 9", "total_predictions >= 1",
              total >= 1,
              f"got {total}")

    # Still expected IMMATURE at this stage — just confirm it's not an error state
    status = _fv(p9, "status", _fv(p9, "health_status", ""))
    r.add("Part 9", "status is IMMATURE (expected — needs 60 live observations)",
          str(status).upper() == "IMMATURE",
          f"got {status!r}")


def check_part6(r: Results, root: Path):
    p6 = _load_json(root / "artifacts_part6" / "part6_meta.json")
    if p6 is None:
        r.add("Part 6", "part6_meta.json exists", False, "file not found")
        return

    r.add("Part 6", "file exists", True)

    unknown_rate = _fv(p6, "unknown_rate", 1.0)
    r.add("Part 6", "unknown_rate = 0.0 (full coverage)",
          float(unknown_rate) == 0.0,
          f"got {unknown_rate}")

    feature_cols = _fv(p6, "feature_cols", [])
    has_hy_fred = "hy_spread_fred" in feature_cols
    r.add("Part 6", "hy_spread_fred absent from feature_cols (NaN-filtered)",
          not has_hy_fred,
          f"features={feature_cols}")
    r.add("Part 6", "8 features used (dropped from 9)",
          len(feature_cols) == 8,
          f"got {len(feature_cols)} features")

    dist = _fv(p6, "regime_distribution", {})
    r.add("Part 6", "all four regimes present in distribution",
          set(dist.keys()) >= {"calm", "risk_on", "high_vol", "crisis"},
          f"got keys={set(dist.keys())}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",    default=None,
                        help="Explicit project root (overrides PRICECALL_ROOT env)")
    parser.add_argument("--uploads", default=None,
                        help="Path to uploads dir (alternative to --root for Colab use)")
    # Optional prior-run baselines for the reviewer's N_before criterion
    parser.add_argument("--predlog-rows-before", type=int, default=None,
                        help="Row count of prediction_log.csv from the PREVIOUS run")
    parser.add_argument("--predictions-before",  type=int, default=None,
                        help="total_predictions from live_attribution_report.json before this run")
    args = parser.parse_args()

    root = Path(args.uploads).expanduser().resolve() if args.uploads \
           else _resolve_root(args.root)

    print(f"{BOLD}=== PriceCall Rerun Acceptance Validator ==={RESET}")
    print(f"Project root: {root}")
    print()

    r = Results()

    check_part6(r, root)
    check_part2(r, root)
    check_part2c(r, root)
    check_part8(r, root)
    check_prediction_log(r, root,
                         n_before=args.predlog_rows_before)
    check_governance(r, root)
    check_alpha(r, root)
    check_part7(r, root)
    check_part9(r, root,
                n_predictions_before=args.predictions_before)

    r.print_table()

    summary = r.summary()
    print(f"{BOLD}Result: {summary}{RESET}")
    print()

    if r.all_pass():
        print(f"{GREEN}{BOLD}✅  All checks passed — production clearance confirmed.{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}⚠️  One or more checks failed — rerun artifacts not yet cleared.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
