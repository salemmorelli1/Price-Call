#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Tuesday production runner for the current PriceCall stack.

Validated execution order
-------------------------
Part 0 -> Part 1 -> Part 2 -> Part 2A -> Part 3 -> Part 5 -> Part 6 -> Part 7 -> Part 8 -> Part 9
Part 4 is launched separately by default.

This script is Drive-first, Colab-safe, and prints full stdout/stderr so failures
are diagnosable in notebook environments and GitHub Actions logs.
"""

# @title run_tuesday_prediction.py

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


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

CANONICAL_FILES: Dict[str, str] = {
    "PART0": "part0_data_infrastructure.py",
    "PART1": "part1_builder.py",
    "PART2": "part2_predictor.py",
    "PART2A": "part2a21_alpha.py",
    "PART3": "part3_governance.py",
    "PART4": "part4_gui.py",
    "PART5": "part5_validator.py",
    "PART6": "part6_regime_engine.py",
    "PART7": "part7_portfolio_construction.py",
    "PART8": "part8_execution_model.py",
    "PART9": "part9_live_attribution.py",
}

PIPELINE_ORDER = [
    "PART0",
    "PART1",
    "PART2",
    "PART2A",
    "PART3",
    "PART5",
    "PART6",
    "PART7",
    "PART8",
    "PART9",
]

REQUIRED = PIPELINE_ORDER.copy()


def check_files(project_dir: Path) -> Tuple[List[str], List[Tuple[str, Path, bool]]]:
    audit: List[Tuple[str, Path, bool]] = []
    missing: List[str] = []
    for label, filename in CANONICAL_FILES.items():
        path = project_dir / filename
        exists = path.exists()
        audit.append((label, path, exists))
        if label in REQUIRED and not exists:
            missing.append(filename)
    return missing, audit


def run_subprocess(cmd: List[str], cwd: Path) -> int:
    print("\nLaunching:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print("\n--- STDERR ---")
        print(proc.stderr.rstrip())

    print(f"[exit={proc.returncode}]")
    return int(proc.returncode)


def launch_gui(project_dir: Path) -> int:
    gui = project_dir / CANONICAL_FILES["PART4"]
    return run_subprocess([sys.executable, str(gui)], project_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical Tuesday PriceCall production pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility; current runner always executes immediately.",
    )
    parser.add_argument(
        "--with-gui",
        action="store_true",
        help="Launch Part 4 GUI after the production stack finishes.",
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

    print("\n=== VALIDATED EXECUTION ORDER ===")
    print("Part 0 -> Part 1 -> Part 2 -> Part 2A -> Part 3 -> Part 5 -> Part 6 -> Part 7 -> Part 8 -> Part 9 -> Part 4")
    if not args.with_gui:
        print("GUI note: Part 4 is not launched unless --with-gui is passed.")

    for label in PIPELINE_ORDER:
        script = PROJECT_DIR / CANONICAL_FILES[label]
        rc = run_subprocess([sys.executable, str(script)], PROJECT_DIR)
        if rc != 0:
            print(f"\n[ERROR] {label} failed with exit code {rc}.")
            return rc

    if args.with_gui:
        rc = launch_gui(PROJECT_DIR)
        if rc != 0:
            print(f"\n[WARN] PART4 exited with code {rc}.")
            return rc

    print("\n✅ Tuesday pipeline completed successfully.")
    return 0



if __name__ == "__main__":
    main()

