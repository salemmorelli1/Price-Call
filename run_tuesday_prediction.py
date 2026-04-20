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
Part 0 -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10
(* Part 2B and Part 2C are optional: skipped if absent, non-blocking if they fail.
   Part 2C should only be activated after Part 2B's gate_validation_passed = true.)
"""

from __future__ import annotations

import argparse
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
    print("       Part 0 -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10")
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
    print("Part 0 -> Part 6 -> Part 1 -> Part 2 -> Part 2B* -> Part 2C* -> Part 2A -> Part 7 -> Part 8 -> Part 3 -> Part 9 -> Part 10")
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


