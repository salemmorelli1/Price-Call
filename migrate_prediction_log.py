#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""migrate_prediction_log.py — One-time migration for prediction_log.csv

Adds two schema improvements introduced in the Quant-Guild Part 3 audit:

  1. horizon_legacy (int, 0/1)
     Flags rows written by the legacy H=7 weekly pipeline, identified by their
     target_date being more than 3 business days after decision_date. These rows
     will never receive correct realized prices because their true H=1 target
     dates have already passed. Part 9 filters them out of the realized set to
     prevent stale H=7 entries from polluting live attribution metrics.

  2. deployment_mode (str)
     The user-facing operational label (DEFENSE_ONLY / NORMAL). Rows written
     before the schema change have publish_mode aliased as DEFENSE_ONLY when
     the governance value was FAIL_CLOSED_NEUTRAL. The new schema keeps both
     fields separate. For legacy rows, deployment_mode is back-filled from
     publish_mode using the same aliasing rule so downstream consumers reading
     deployment_mode always get a valid value.

Run once after deploying the updated part3_governance.py. Safe to re-run;
idempotent on already-migrated logs.

Usage:
    python migrate_prediction_log.py
    python migrate_prediction_log.py --dry-run   # print changes without saving
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Root resolution (matches all other parts) ────────────────────────────────

def _resolve_root() -> Path:
    candidates = []
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/content/drive/MyDrive/PriceCallProject"))
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    candidates.append(Path.cwd())

    seen: set = set()
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        s = str(rp)
        if s == "/content" or s in seen:
            continue
        seen.add(s)
        if rp.exists():
            return rp
    return Path.cwd().resolve()


def _predlog_path(root: Path) -> Path:
    return root / "artifacts_part3" / "prediction_log.csv"


# ── Migration helpers ─────────────────────────────────────────────────────────

_BDAY = pd.tseries.offsets.BDay


def _is_horizon_legacy(row: pd.Series) -> int:
    """Return 1 if this row was written by the legacy H=7 pipeline.

    Detection rule: target_date is more than 3 business days after
    decision_date. At H=1, target_date = decision_date + BDay(1), so any gap
    wider than 3 business days unambiguously identifies an H=7 entry.

    Also flags rows where h_reb is explicitly set to a value other than 1,
    if that column exists.
    """
    try:
        dd = pd.to_datetime(row.get("decision_date"), errors="coerce")
        td = pd.to_datetime(row.get("target_date"), errors="coerce")
        if pd.isna(dd) or pd.isna(td):
            return 0
        # Count business days between decision and target
        bday_gap = len(pd.bdate_range(dd + _BDAY(1), td))
        if bday_gap > 3:
            return 1
    except Exception:
        pass

    # Secondary check: explicit h_reb column
    h_reb = row.get("h_reb", None)
    if h_reb is not None:
        try:
            if int(float(h_reb)) > 1:
                return 1
        except Exception:
            pass

    return 0


def _deployment_mode_from_publish(publish_mode: object) -> str:
    """Back-fill deployment_mode from publish_mode for legacy rows.

    Applies the same aliasing rule as the updated Part 3 source: when the
    governance value is FAIL_CLOSED_NEUTRAL, the user-facing operational label
    is DEFENSE_ONLY. NORMAL passes through unchanged.
    """
    s = str(publish_mode).strip().upper() if publish_mode is not None else "UNKNOWN"
    if s == "FAIL_CLOSED_NEUTRAL":
        return "DEFENSE_ONLY"
    if s in {"NORMAL", "DEFENSE_ONLY"}:
        return s
    return "UNKNOWN"


# ── Main migration ────────────────────────────────────────────────────────────

def migrate(predlog_path: Path, dry_run: bool = False) -> int:
    if not predlog_path.exists():
        print(f"[migrate] Prediction log not found: {predlog_path}")
        print("[migrate] Nothing to migrate.")
        return 0

    df = pd.read_csv(predlog_path)
    original_columns = list(df.columns)
    n_rows = len(df)
    print(f"[migrate] Loaded {n_rows} rows from {predlog_path}")
    print(f"[migrate] Existing columns: {original_columns}")
    print()

    changes: list = []

    # ── 1. Add / refresh horizon_legacy ──────────────────────────────────────
    legacy_flags = df.apply(_is_horizon_legacy, axis=1).astype(int)
    if "horizon_legacy" not in df.columns:
        df["horizon_legacy"] = legacy_flags
        changes.append(f"  + Added   'horizon_legacy' column ({int(legacy_flags.sum())} rows flagged)")
    else:
        n_changed = int((df["horizon_legacy"].fillna(0).astype(int) != legacy_flags).sum())
        df["horizon_legacy"] = legacy_flags
        changes.append(f"  ~ Refreshed 'horizon_legacy' column ({n_changed} rows updated, {int(legacy_flags.sum())} total flagged)")

    # Report which rows are flagged
    legacy_rows = df[df["horizon_legacy"] == 1]
    if len(legacy_rows):
        for _, r in legacy_rows.iterrows():
            dd = r.get("decision_date", "?")
            td = r.get("target_date", "?")
            pm = r.get("publish_mode", "?")
            print(f"  ⚠️  horizon_legacy=1: decision={dd}  target={td}  publish_mode={pm}")
    else:
        print("  ✅ No legacy H=7 rows detected.")
    print()

    # ── 2. Add / refresh deployment_mode ─────────────────────────────────────
    if "publish_mode" in df.columns:
        new_deployment_mode = df["publish_mode"].apply(_deployment_mode_from_publish)
        if "deployment_mode" not in df.columns:
            df["deployment_mode"] = new_deployment_mode
            changes.append("  + Added   'deployment_mode' column (back-filled from publish_mode)")
        else:
            # Only overwrite rows where deployment_mode is null or UNKNOWN
            mask = df["deployment_mode"].isna() | (df["deployment_mode"].astype(str).str.upper() == "UNKNOWN")
            n_filled = int(mask.sum())
            df.loc[mask, "deployment_mode"] = new_deployment_mode[mask]
            changes.append(f"  ~ Refreshed 'deployment_mode' column ({n_filled} null/UNKNOWN rows filled)")
    else:
        changes.append("  ! Skipped 'deployment_mode': no 'publish_mode' column found to back-fill from")

    # ── 3. Column ordering: insert new columns after publish_mode ─────────────
    ordered_cols = list(original_columns)
    for new_col in ["deployment_mode", "horizon_legacy"]:
        if new_col not in ordered_cols:
            # Insert after publish_mode if present, else append
            if "publish_mode" in ordered_cols:
                idx = ordered_cols.index("publish_mode") + 1
                # Don't double-insert deployment_mode right after publish_mode
                # if it is already there from a prior migration
                if new_col == "deployment_mode":
                    ordered_cols.insert(idx, new_col)
                else:
                    ordered_cols.append(new_col)
            else:
                ordered_cols.append(new_col)

    # Ensure all df columns are accounted for (in case df gained cols not in ordered_cols)
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)

    df = df[[c for c in ordered_cols if c in df.columns]]

    # ── 4. Summary and write ──────────────────────────────────────────────────
    print("[migrate] Changes:")
    for c in changes:
        print(c)
    print()
    print(f"[migrate] Final columns: {list(df.columns)}")
    print(f"[migrate] Rows: {len(df)}")

    if dry_run:
        print()
        print("[migrate] DRY_RUN — no file written.")
        print("[migrate] Re-run without --dry-run to apply.")
        return 0

    # Write atomically: write to .tmp then rename
    tmp_path = predlog_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(predlog_path)
    print()
    print(f"[migrate] ✅ Written to {predlog_path}")

    # Verify round-trip
    verify = pd.read_csv(predlog_path)
    assert list(verify.columns) == list(df.columns), "Column order mismatch after write"
    assert len(verify) == len(df), "Row count mismatch after write"
    print("[migrate] ✅ Round-trip verification passed.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing")
    p.add_argument("--path", type=str, default=None, help="Explicit path to prediction_log.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.path:
        predlog_path = Path(args.path).expanduser().resolve()
    else:
        root = _resolve_root()
        predlog_path = _predlog_path(root)
        print(f"[migrate] Project root: {root}")

    print(f"[migrate] Target log:    {predlog_path}")
    print()
    return migrate(predlog_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
