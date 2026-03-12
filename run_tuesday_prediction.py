#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


def resolve_root() -> Path:
    try:
        import google.colab  # noqa: F401
        from google.colab import drive

        drive_root = Path("/content/drive")
        if not drive_root.exists():
            drive.mount("/content/drive")

        root = Path("/content/drive/MyDrive/PriceCallProject")
        root.mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        pass

    if "__file__" in globals():
        return Path(__file__).resolve().parent

    return Path.cwd()


ROOT = resolve_root()
ET = ZoneInfo("America/New_York")

PART1_SCRIPT = ROOT / "part1_builder.py"
PART2_SCRIPT = ROOT / "part2_predictor.py"
PART2A_SCRIPT = ROOT / "part2a21_alpha.py"
PART3_SCRIPT = ROOT / "part3_v1_fusion.py"

PRED_LOG_PATH = ROOT / "artifacts_part3" / "prediction_log.csv"
PART3_TAPE_PATH = ROOT / "artifacts_part3_v1" / "v1_final_production_tape.csv"
PART3_GOV_PATH = ROOT / "artifacts_part3_v1" / "v1_final_production_governance.csv"
PART3_ALLOC_PATH = ROOT / "artifacts_part3_v1" / "v1_fusion_allocations.csv"


def now_et() -> datetime:
    return datetime.now(ET)


def in_decision_window(ts: datetime) -> bool:
    return ts.weekday() == 1 and ts.hour == 9 and 35 <= ts.minute < 40


def load_prediction_log() -> pd.DataFrame:
    if not PRED_LOG_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(PRED_LOG_PATH)
    if "decision_date" in df.columns:
        df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce").dt.normalize()
    return df


def already_logged_today(decision_date: pd.Timestamp) -> bool:
    log = load_prediction_log()
    if log.empty or "decision_date" not in log.columns:
        return False
    return (log["decision_date"] == decision_date.normalize()).any()


def print_path_status() -> None:
    print(f"ROOT:        {ROOT}")
    print(f"PART1:       {PART1_SCRIPT} | exists={PART1_SCRIPT.exists()}")
    print(f"PART2:       {PART2_SCRIPT} | exists={PART2_SCRIPT.exists()}")
    print(f"PART2A21:    {PART2A_SCRIPT} | exists={PART2A_SCRIPT.exists()}")
    print(f"PART3 V1:    {PART3_SCRIPT} | exists={PART3_SCRIPT.exists()}")
    print(f"PRED_LOG:    {PRED_LOG_PATH} | exists={PRED_LOG_PATH.exists()}")
    print(f"TAPE V1:     {PART3_TAPE_PATH} | exists={PART3_TAPE_PATH.exists()}")
    print(f"GOV V1:      {PART3_GOV_PATH} | exists={PART3_GOV_PATH.exists()}")
    print(f"ALLOC V1:    {PART3_ALLOC_PATH} | exists={PART3_ALLOC_PATH.exists()}")


def run_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")

    print(f"\n>>> Running: {path.name}")
    subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=True)


def validate_outputs() -> None:
    if not PART3_TAPE_PATH.exists():
        raise FileNotFoundError(f"Missing Part 3 V1 tape: {PART3_TAPE_PATH}")
    if not PART3_GOV_PATH.exists():
        raise FileNotFoundError(f"Missing Part 3 V1 governance file: {PART3_GOV_PATH}")
    if not PART3_ALLOC_PATH.exists():
        raise FileNotFoundError(f"Missing Part 3 V1 fusion allocations file: {PART3_ALLOC_PATH}")
    if not PRED_LOG_PATH.exists():
        raise FileNotFoundError(f"Missing prediction log: {PRED_LOG_PATH}")

    log = load_prediction_log()
    if log.empty:
        raise RuntimeError("prediction_log.csv exists but is empty.")

    if "decision_date" not in log.columns:
        raise RuntimeError("prediction_log.csv is missing 'decision_date'.")

    tape = pd.read_csv(PART3_TAPE_PATH)
    if "Date" not in tape.columns:
        raise RuntimeError("v1_final_production_tape.csv is missing 'Date'.")

    tape["Date"] = pd.to_datetime(tape["Date"], errors="coerce").dt.normalize()
    tape = tape.dropna(subset=["Date"])
    if tape.empty:
        raise RuntimeError("v1_final_production_tape.csv has no valid Date rows.")

    expected_date = tape["Date"].max()
    last_logged = log["decision_date"].max()

    if pd.isna(last_logged):
        raise RuntimeError("prediction_log.csv has no valid decision_date values.")

    if last_logged != expected_date:
        raise RuntimeError(
            f"Pipeline finished, but prediction_log.csv last decision_date={last_logged.date()} "
            f"does not match production tape max Date={expected_date.date()}."
        )

    alloc = pd.read_csv(PART3_ALLOC_PATH)
    if "Date" not in alloc.columns or "weight" not in alloc.columns:
        raise RuntimeError("v1_fusion_allocations.csv is missing required columns: Date / weight")

    alloc["Date"] = pd.to_datetime(alloc["Date"], errors="coerce").dt.normalize()
    alloc = alloc.dropna(subset=["Date"])
    if alloc.empty:
        raise RuntimeError("v1_fusion_allocations.csv has no valid rows.")

    alloc_chk = alloc.groupby("Date", as_index=False)["weight"].sum()
    max_dev = float((alloc_chk["weight"] - 1.0).abs().max())
    if max_dev > 1e-6:
        raise RuntimeError(f"Fusion allocations fail sum-to-one check. max deviation={max_dev:.8f}")

    print("\n✅ Validation passed.")
    print(f"prediction_log:       {PRED_LOG_PATH}")
    print(f"production_tape_v1:   {PART3_TAPE_PATH}")
    print(f"governance_tape_v1:   {PART3_GOV_PATH}")
    print(f"fusion_allocations:   {PART3_ALLOC_PATH}")
    print(f"validated decision_date: {expected_date.date()}")


def required_scripts_exist() -> bool:
    return (
        PART1_SCRIPT.exists() and
        PART2_SCRIPT.exists() and
        PART2A_SCRIPT.exists() and
        PART3_SCRIPT.exists()
    )


def main(force: bool = False, dry_run: bool = False, show_paths: bool = True) -> int:
    ts = now_et()
    decision_date = pd.Timestamp(ts.date())

    print(f"Current ET time: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if show_paths:
        print_path_status()

    if not force:
        if not in_decision_window(ts):
            print("Skip: not inside Tuesday 09:35–09:39 ET decision window.")
            return 0

        if already_logged_today(decision_date):
            print(f"Skip: prediction for {decision_date.date()} already exists in prediction_log.csv.")
            return 0

    if dry_run:
        print("Dry run only. Pipeline not executed.")
        return 0

    if not required_scripts_exist():
        missing = []
        for p in [PART1_SCRIPT, PART2_SCRIPT, PART2A_SCRIPT, PART3_SCRIPT]:
            if not p.exists():
                missing.append(str(p))

        print("\nCannot run full Tuesday pipeline yet.")
        print("Missing required script file(s):")
        for m in missing:
            print(f" - {m}")
        return 1

    run_script(PART1_SCRIPT)
    run_script(PART2_SCRIPT)
    run_script(PART2A_SCRIPT)
    run_script(PART3_SCRIPT)

    validate_outputs()

    tape = pd.read_csv(PART3_TAPE_PATH)
    tape["Date"] = pd.to_datetime(tape["Date"], errors="coerce").dt.normalize()
    actual_decision_date = tape["Date"].max()
    print(f"\n✅ Tuesday prediction pipeline completed for decision_date={actual_decision_date.date()}.")
    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Tuesday prediction pipeline once near market open.")
    parser.add_argument("--force", action="store_true", help="Run regardless of day/time and duplicate checks.")
    parser.add_argument("--dry-run", action="store_true", help="Check schedule logic only.")
    parser.add_argument("--no-paths", action="store_true", help="Suppress path-status printing.")

    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        args, _ = parser.parse_known_args(args=[] if argv is None else argv)
    else:
        args = parser.parse_args(argv)

    return main(force=args.force, dry_run=args.dry_run, show_paths=not args.no_paths)


if __name__ == "__main__":
    rc = cli()
    if "ipykernel" not in sys.modules and "google.colab" not in sys.modules:
        raise SystemExit(rc)
