#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical daily production runner for the current PriceCall stack.

Current authoritative behavior
------------------------------
- Daily H=1 model
- Prefers Part 5 validator as the authoritative orchestrator
- Falls back to direct execution only if Part 5 is missing
- Includes Part 10 trading bot in the canonical stack
- Keeps Part 4 optional and separate (HTML / GitHub dashboard can live independently)
- Part 2B (XGBoost ensemble) and Part 2C (BNN) are optional experimental sleeves

Authoritative daily execution order
-----------------------------------
Part 0 -> point-in-time macro -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10
(* Part 2B and Part 2C are optional: skipped if absent, non-blocking if they fail.
   Part 2C should only be activated after Part 2B's gate_validation_passed = true.)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ------------------------------------------------------------
# Colab / environment helpers
# ------------------------------------------------------------
def maybe_mount_drive() -> bool:
    try:
        from google.colab import drive  # type: ignore

        mount_root = Path("/content/drive")
        if not (mount_root / "MyDrive").exists():
            drive.mount(str(mount_root), force_remount=False)
        else:
            print("Drive already mounted.")
        return True
    except Exception:
        return False


IN_COLAB = maybe_mount_drive()


def resolve_project_dir() -> Path:
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    drive_root = Path("/content/drive/MyDrive/PriceCallProject")
    if IN_COLAB:
        return drive_root

    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_DIR = resolve_project_dir()
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PRICECALL_ROOT", str(PROJECT_DIR))
os.environ.setdefault("PRICECALL_STRICT_DRIVE_ONLY", "1")
os.environ.setdefault("PRICECALL_ALPHA_FAMILY", "part2a21")


# ------------------------------------------------------------
# Canonical files
# ------------------------------------------------------------
CANONICAL_FILES: Dict[str, str] = {
    "PART0":  "part0_data_infrastructure.py",
    "PIT_MACRO": "point_in_time_macro.py",
    "PART1":  "part1_builder.py",
    "PART2":  "part2_predictor.py",
    "PART2B": "part2b_xgb_ensemble.py",  # optional experimental sleeve
    "PART2C": "part2c_bnn_sleeve.py",    # optional experimental sleeve
    "PART2A": "part2a21_alpha.py",
    "PART3":  "part3_governance.py",
    "PART4":  "part4_gui.py",            # optional
    "PART5":  "part5_validator.py",
    "PART6":  "part6_regime_engine.py",
    "PART7":  "part7_portfolio_construction.py",
    "PART8":  "part8_execution_model.py",
    "PART9":  "part9_live_attribution.py",
}

PART10_CANDIDATES: Tuple[str, ...] = (
    "part10_trading_bot.py",
    "part10_tradingbot.py",
)

BACKFILL_CANDIDATES: Tuple[str, ...] = (
    "backfill_realized.py",
)

DIRECT_PIPELINE_ORDER: List[str] = [
    "PART0",
    "PIT_MACRO",
    "PART6",
    "PART1",
    "PART2",
    "PART2B",  # optional, non-blocking
    "PART2C",  # optional, non-blocking
    "PART2A",
    "PART7",
    "PART8",
    "PART3",
    "PART9",
]

# Core files required for a valid direct run.
# PART2B and PART2C are intentionally excluded because they are optional sleeves.
# PART5 is intentionally excluded so the advertised fallback is actually reachable.
REQUIRED_FOR_DIRECT_RUN: List[str] = [
    p for p in DIRECT_PIPELINE_ORDER if p not in {"PART2B", "PART2C"}
]


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
def first_existing(project_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        path = (project_dir / name).resolve()
        if path.exists():
            return path
    return None


# FIX (Quant-Guild Part 46 Audit): Part 2C must only run when Part 2B recommends it.
#
# ROOT CAUSE: the module docstring above has always stated "Part 2C should only be
# activated after Part 2B's gate_validation_passed = true," and part2c_bnn_sleeve.py
# itself prints "Gate check: ... bnn_sleeve_recommended = True" — but NEITHER file
# actually enforces this anywhere in code. The only real gate Part 2C checks is whether
# PyTorch is importable (HAVE_TORCH); once requirements.txt uncommented torch (audit
# 2026-05-07, justified at the time by bnn_sleeve_recommended=True), Part 2C began
# running unconditionally on every cycle, regardless of Part 2B's CURRENT recommendation.
#
# CONFIRMED IMPACT (S46 artifact): part2b_xgb_summary.json now reports
# bnn_sleeve_recommended=False (gate_validation_passed=False: Platt slope degenerate,
# walk-forward AUC not significant) — the exact opposite of the condition that
# justified uncommenting torch. With torch now permanently installed and no runtime
# check, Part 2C would run anyway, burning ~5 minutes of CI wall time per day on a
# sleeve its own upstream gate says is not currently viable, and writing a fresh
# part2c_bnn_tape.csv/summary every day that Part 3 then has to separately re-evaluate
# and discard via its own internal gate (part3_governance.py already does this
# correctly downstream — this fix prevents the wasted, gate-contradicting run upstream).
#
# Fix: read part2b_xgb_summary.json's bnn_sleeve_recommended flag (Part 2B always runs
# first in DIRECT_PIPELINE_ORDER, so a fresh summary is available by the time PART2C's
# turn comes up) and skip PART2C when it is not True. Fails safe: any error reading the
# flag (missing file, bad JSON, missing key) also skips PART2C, since "don't run an
# unrecommended, optional experimental sleeve" is the safe default in all uncertain cases.
def _part2b_recommends_bnn(project_dir: Path) -> Tuple[bool, str]:
    summary_path = (project_dir / "artifacts_part2b_xgb" / "predictions" / "part2b_xgb_summary.json").resolve()
    if not summary_path.exists():
        return False, f"part2b_xgb_summary.json not found at {summary_path}"
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        return False, f"could not parse {summary_path}: {exc}"
    recommended = bool(summary.get("bnn_sleeve_recommended", False))
    gate_ok = bool(summary.get("gate_validation_passed", False))
    reason = (
        f"bnn_sleeve_recommended={recommended}, gate_validation_passed={gate_ok} "
        f"(source={summary_path})"
    )
    return recommended, reason


def check_files(project_dir: Path) -> Tuple[List[str], List[Tuple[str, Path, bool]]]:
    audit: List[Tuple[str, Path, bool]] = []
    missing: List[str] = []

    for label, filename in CANONICAL_FILES.items():
        path = (project_dir / filename).resolve()
        exists = path.exists()
        audit.append((label, path, exists))

    part10_path = first_existing(project_dir, PART10_CANDIDATES)
    audit.append(("PART10", project_dir / PART10_CANDIDATES[0], part10_path is not None))

    backfill_path = first_existing(project_dir, BACKFILL_CANDIDATES)
    audit.append(("BACKFILL", project_dir / BACKFILL_CANDIDATES[0], backfill_path is not None))

    for label in REQUIRED_FOR_DIRECT_RUN:
        path = (project_dir / CANONICAL_FILES[label]).resolve()
        if not path.exists():
            missing.append(path.name)

    if part10_path is None:
        missing.append("part10_trading_bot.py or part10_tradingbot.py")

    return missing, audit


# ------------------------------------------------------------
# Subprocess helpers
# ------------------------------------------------------------
def run_subprocess(cmd: List[str], cwd: Path, extra_env: Optional[Dict[str, str]] = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print("\nLaunching:", " ".join(str(x) for x in cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print("\n--- STDERR ---")
        print(proc.stderr.rstrip())

    print(f"[exit={proc.returncode}]")
    return int(proc.returncode)


# ------------------------------------------------------------
# Preferred execution path: Part 5 validator
# ------------------------------------------------------------
def run_with_validator(project_dir: Path) -> int:
    validator = (project_dir / CANONICAL_FILES["PART5"]).resolve()
    return run_subprocess(
        [sys.executable, str(validator)],
        project_dir,
        extra_env={
            "PRICECALL_ROOT": str(project_dir),
            "PRICECALL_STRICT_DRIVE_ONLY": "1",
            "PRICECALL_ALPHA_FAMILY": "part2a21",
        },
    )


# ------------------------------------------------------------
# Fallback direct execution path
# ------------------------------------------------------------
def run_direct_pipeline(project_dir: Path) -> int:
    print("\n[INFO] part5_validator.py not found. Falling back to direct execution.")
    print("[INFO] Direct daily order:")
    print("       Part 0 -> point-in-time macro -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10")
    print("       * Part 2B and Part 2C are optional / experimental and non-blocking.")
    print("       Part 4 remains optional / separate.\n")

    common_env = {
        "PRICECALL_ROOT": str(project_dir),
        "PRICECALL_STRICT_DRIVE_ONLY": "1",
        "PRICECALL_ALPHA_FAMILY": "part2a21",
    }

    for label in DIRECT_PIPELINE_ORDER:
        script_name = CANONICAL_FILES.get(label)
        if script_name is None:
            continue
        script = (project_dir / script_name).resolve()

        if label in {"PART2B", "PART2C"}:
            if not script.exists():
                print(f"\n[INFO] {label} ({script_name}) not found — skipping.")
                continue
            # FIX (Quant-Guild Part 46 Audit): only launch Part 2C when Part 2B's
            # bnn_sleeve_recommended flag is True. Part 2B always runs first in
            # DIRECT_PIPELINE_ORDER, so its fresh summary is available here.
            if label == "PART2C":
                if os.environ.get("PRICECALL_ENABLE_BNN", "0") != "1":
                    print(f"\n[INFO] {label} skipped — set PRICECALL_ENABLE_BNN=1 and install requirements-bnn.txt to enable.")
                    continue
                recommended, reason = _part2b_recommends_bnn(project_dir)
                if not recommended:
                    print(f"\n[INFO] {label} ({script_name}) skipped — Part 2B does not "
                          f"currently recommend the BNN sleeve ({reason}).")
                    continue
                print(f"\n[INFO] {label} activation gate passed ({reason}).")
            rc_exp = run_subprocess([sys.executable, str(script)], project_dir, extra_env=common_env)
            if rc_exp != 0:
                print(f"\n[WARN] {label} exited with code {rc_exp} — continuing (experimental sleeve).")
            continue

        rc = run_subprocess([sys.executable, str(script)], project_dir, extra_env=common_env)
        if rc != 0:
            print(f"\n[ERROR] {label} failed with exit code {rc}.")
            return rc

    part10_path = first_existing(project_dir, PART10_CANDIDATES)
    if part10_path is None:
        print("\n[ERROR] Part 10 file not found.")
        return 1

    rc = run_subprocess([sys.executable, str(part10_path)], project_dir, extra_env=common_env)
    if rc != 0:
        print(f"\n[ERROR] PART10 failed with exit code {rc}.")
        return rc

    return 0


# ------------------------------------------------------------
# Optional GUI launch
# ------------------------------------------------------------
def launch_gui(project_dir: Path) -> int:
    gui = (project_dir / CANONICAL_FILES["PART4"]).resolve()
    if not gui.exists():
        print("[WARN] Part 4 GUI file not found. Skipping GUI launch.")
        return 0
    return run_subprocess([sys.executable, str(gui)], project_dir)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical daily PriceCall production pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility; the daily runner executes immediately.",
    )
    parser.add_argument(
        "--with-gui",
        action="store_true",
        help="Optionally launch the Python GUI after the stack finishes.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass Part 5 validator and run the pipeline directly.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring extra notebook/launcher args: {' '.join(unknown)}")

    print(f"ROOT: {PROJECT_DIR}")
    print(f"IN_COLAB: {IN_COLAB}")

    print("\n=== CANONICAL FILE AUDIT ===")
    missing, audit = check_files(PROJECT_DIR)
    for label, path, exists in audit:
        print(f"{label}: {path} | exists = {exists}")

    if missing:
        print("\n[ERROR] Required production files are missing:")
        for name in missing:
            print(f" - {name}")
        return 1

    print("\n=== AUTHORITATIVE DAILY EXECUTION ORDER ===")
    print("Part 0 -> point-in-time macro -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10")
    print("* Part 2B and Part 2C are optional / experimental and non-blocking.")
    if not args.with_gui:
        print("GUI note: HTML / GitHub dashboard is separate; Python GUI is not launched unless --with-gui is passed.")

    validator_exists = (PROJECT_DIR / CANONICAL_FILES["PART5"]).resolve().exists()

    if args.direct or not validator_exists:
        rc = run_direct_pipeline(PROJECT_DIR)
    else:
        rc = run_with_validator(PROJECT_DIR)

    if rc != 0:
        print(f"\n⚠️ Pipeline exited with code {rc}.")
        return rc

    if args.with_gui:
        gui_rc = launch_gui(PROJECT_DIR)
        if gui_rc != 0:
            print(f"\n[WARN] PART4 exited with code {gui_rc}.")
            return gui_rc

    print("\n✅ Daily pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


