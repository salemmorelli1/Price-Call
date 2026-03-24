


#!/usr/bin/env python3
"""
run_tuesday_prediction_current_model.py
===============================================================================
Colab/Jupyter-safe Tuesday runner for the CURRENT canonical model.

Current canonical pipeline
--------------------------
1) Part 2  : part2_predictor_gen4_v7h_active_ir_stability_patch.py
2) Part 2A : part2a23_alpha_selective_repair.py
3) Part 3  : part3_v3d_filename_cleanup.py

What this fixes
---------------
- Ignores notebook launcher args (-f kernel.json)
- Works when __file__ is unavailable
- Searches robustly across /content, /mnt/data, and Drive
- Supports explicit overrides for each stage
- Runs the full current pipeline, not just Part 2A + Part 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception as exc:
    raise RuntimeError("zoneinfo is required (Python 3.9+).") from exc


TRUE_SET = {"1", "true", "True", "YES", "yes", "y", "Y"}


def env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in TRUE_SET


def default_root_dir() -> Path:
    if "__file__" in globals():
        try:
            return Path(__file__).resolve().parent
        except Exception:
            pass

    drive_root = Path("/content/drive/MyDrive/PriceCallProject")
    if drive_root.exists():
        return drive_root.resolve()

    return Path.cwd().resolve()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tuesday prediction pipeline for the current model.")
    parser.add_argument("--force", action="store_true", help="Run even if it is not Tuesday.")
    parser.add_argument("--timezone", default=None, help="Override local scheduler timezone.")
    parser.add_argument("--artifact-dir", default=None, help="Override Part 3 artifact directory.")
    parser.add_argument("--strict-final-pass", dest="strict_final_pass", action="store_true",
                        help="Require Part 3 final_pass == True.")
    parser.add_argument("--no-strict-final-pass", dest="strict_final_pass", action="store_false",
                        help="Do not require Part 3 final_pass == True.")
    parser.set_defaults(strict_final_pass=None)
    parser.add_argument("--require-normal-mode", dest="require_normal_mode", action="store_true",
                        help="Require Part 3 publish_mode == NORMAL.")
    parser.add_argument("--no-require-normal-mode", dest="require_normal_mode", action="store_false",
                        help="Do not require Part 3 publish_mode == NORMAL.")
    parser.set_defaults(require_normal_mode=None)

    parser.add_argument("--part2-script", default=None, help="Explicit path to Part 2 script.")
    parser.add_argument("--part2a-script", default=None, help="Explicit path to Part 2A script.")
    parser.add_argument("--part3-script", default=None, help="Explicit path to Part 3 script.")

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[INFO] Ignoring unknown notebook/launcher args: {unknown}")
    return args


@dataclass(frozen=True)
class RunnerConfig:
    timezone_name: str = os.environ.get("RUN_TIMEZONE", "America/Phoenix")
    force_run: bool = env_bool("FORCE_RUN", "0")
    strict_final_pass: bool = env_bool("STRICT_FINAL_PASS", "1")
    require_normal_mode: bool = env_bool("REQUIRE_NORMAL_MODE", "1")
    root_dir: Path = field(default_factory=default_root_dir)
    artifact_dir_name: str = os.environ.get("ARTIFACT_DIR", "./artifacts_part3_v3b")
    part2_script_override: Optional[str] = os.environ.get("PART2_SCRIPT")
    part2a_script_override: Optional[str] = os.environ.get("PART2A_SCRIPT")
    part3_script_override: Optional[str] = os.environ.get("PART3_SCRIPT")


def build_config(args: argparse.Namespace) -> RunnerConfig:
    base = RunnerConfig()
    return RunnerConfig(
        timezone_name=args.timezone or base.timezone_name,
        force_run=bool(args.force) or base.force_run,
        strict_final_pass=base.strict_final_pass if args.strict_final_pass is None else bool(args.strict_final_pass),
        require_normal_mode=base.require_normal_mode if args.require_normal_mode is None else bool(args.require_normal_mode),
        root_dir=base.root_dir,
        artifact_dir_name=args.artifact_dir or base.artifact_dir_name,
        part2_script_override=args.part2_script or base.part2_script_override,
        part2a_script_override=args.part2a_script or base.part2a_script_override,
        part3_script_override=args.part3_script or base.part3_script_override,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def candidate_roots(cfg: RunnerConfig) -> List[Path]:
    roots = [
        cfg.root_dir,
        Path.cwd(),
        Path("/mnt/data"),
        Path("/content"),
        Path("/content/drive/MyDrive/PriceCallProject"),
        Path("./drive/MyDrive/PriceCallProject"),
    ]
    seen = set()
    out = []
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def build_script_candidates(cfg: RunnerConfig, stage: str) -> List[Path]:
    mapping = {
        "part2": [
            cfg.part2_script_override,
            "part2_predictor_gen4_v7h_active_ir_stability_patch.py",
            "part2_predictor.py",
            "part2_predictor_gen4_v7h_production_hardened.py",
        ],
        "part2a": [
            cfg.part2a_script_override,
            "part2a23_alpha_selective_repair.py",
            "part2a22_alpha_summary_backfill.py",
            "part2a21_alpha_production_ready.py",
        ],
        "part3": [
            cfg.part3_script_override,
            "part3_v3d_filename_cleanup.py",
            "part3_v3c_micro_cleanup.py",
            "part3_v3b_provenance_cleanup.py",
            "part3_v3a_directional_donor_tuned.py",
            "part3_v2f_family_locked_fix1.py",
        ],
    }
    names = [x for x in mapping[stage] if x]

    cands: List[Path] = []
    for root in candidate_roots(cfg):
        for nm in names:
            p = Path(nm)
            if p.is_absolute():
                cands.append(p)
            else:
                cands.append(root / nm)

    seen = set()
    out = []
    for p in cands:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def discover_first_existing(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "None of the candidate files were found:\n- " + "\n- ".join(str(x) for x in candidates)
    )


def run_python_script(script_path: Path, label: str) -> None:
    print(f"\n[{label}] Running: {script_path}")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}.")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_columns(df: pd.DataFrame, needed: List[str], label: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def latest_row(df: pd.DataFrame, date_col: str = "Date") -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Expected '{date_col}' in dataframe.")
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.loc[x[date_col].notna()].sort_values(date_col)
    if x.empty:
        raise ValueError("No valid dated rows found.")
    return x.iloc[-1]


def filter_latest_date(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.loc[x[date_col].notna()]
    if x.empty:
        return x.iloc[0:0].copy()
    last_dt = x[date_col].max()
    return x.loc[x[date_col] == last_dt].copy()


def export_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def check_tuesday_guard(cfg: RunnerConfig) -> Dict[str, Any]:
    tz = ZoneInfo(cfg.timezone_name)
    local_now = datetime.now(tz)
    is_tuesday = local_now.weekday() == 1

    info = {
        "runner_utc": utc_now_iso(),
        "runner_local": local_now.replace(microsecond=0).isoformat(),
        "timezone": cfg.timezone_name,
        "weekday_name": local_now.strftime("%A"),
        "is_tuesday": bool(is_tuesday),
        "force_run": bool(cfg.force_run),
    }

    if (not is_tuesday) and (not cfg.force_run):
        print("[SKIP] Not Tuesday in local scheduler timezone.")
        print(json.dumps(info, indent=2))
        return {**info, "skipped": True}

    print("[RUN] Tuesday guard passed.")
    print(json.dumps(info, indent=2))
    return {**info, "skipped": False}


def resolve_artifact_dir(cfg: RunnerConfig) -> Path:
    cands = [
        cfg.root_dir / cfg.artifact_dir_name,
        Path(cfg.artifact_dir_name),
        Path("/content") / cfg.artifact_dir_name.strip("./"),
        Path("/content/drive/MyDrive/PriceCallProject") / cfg.artifact_dir_name.strip("./"),
    ]
    for p in cands:
        if p.exists():
            return p.resolve()
    return (cfg.root_dir / cfg.artifact_dir_name).resolve()


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)

    guard = check_tuesday_guard(cfg)
    if guard["skipped"]:
        return

    part2_script = discover_first_existing(build_script_candidates(cfg, "part2"))
    part2a_script = discover_first_existing(build_script_candidates(cfg, "part2a"))
    part3_script = discover_first_existing(build_script_candidates(cfg, "part3"))

    print(f"[DISCOVER] Part 2 script:  {part2_script}")
    print(f"[DISCOVER] Part 2A script: {part2a_script}")
    print(f"[DISCOVER] Part 3 script:  {part3_script}")

    run_python_script(part2_script, "PART2")
    run_python_script(part2a_script, "PART2A23")
    run_python_script(part3_script, "PART3V3D")

    artifact_dir = resolve_artifact_dir(cfg)

    summary_path = artifact_dir / "part3_summary.json"
    tape_path = artifact_dir / "v3c_final_production_tape.csv"
    gov_path = artifact_dir / "v3c_final_production_governance.csv"
    alloc_path = artifact_dir / "v3c_fusion_allocations.csv"
    pred_log_path = artifact_dir / "prediction_log.csv"

    needed_paths = [summary_path, tape_path, gov_path, alloc_path]
    missing_paths = [str(p) for p in needed_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError("Expected production outputs missing:\n- " + "\n- ".join(missing_paths))

    summary = load_json(summary_path)
    if cfg.require_normal_mode and str(summary.get("publish_mode")) != "NORMAL":
        raise RuntimeError(f"Part 3 publish_mode is not NORMAL: {summary.get('publish_mode')}")
    if cfg.strict_final_pass and not bool(summary.get("final_pass", False)):
        raise RuntimeError("Part 3 final_pass is False.")

    tape = pd.read_csv(tape_path)
    gov = pd.read_csv(gov_path)
    alloc = pd.read_csv(alloc_path)

    ensure_columns(
        tape,
        [
            "Date", "px_voo_call_7d", "px_ief_call_7d", "w_fused_voo", "w_fused_ief",
            "alpha_state", "alpha_state_display", "fusion_live", "part3_version"
        ],
        "production tape",
    )
    ensure_columns(gov, ["Date", "alpha_state", "alpha_blocker_text", "budget_mult"], "governance")
    ensure_columns(alloc, ["Date", "Ticker", "weight", "alpha_state"], "fusion allocations")

    latest_tape = latest_row(tape)
    latest_gov = latest_row(gov)
    latest_alloc = filter_latest_date(alloc)

    alloc_map: Dict[str, float] = {}
    for _, row in latest_alloc.iterrows():
        tk = str(row.get("Ticker", "")).strip()
        wt = safe_float(row.get("weight"), 0.0)
        if tk:
            alloc_map[tk] = round(wt, 8)

    prediction_payload: Dict[str, Any] = {
        "part": "tuesday_prediction",
        "built_at_utc": utc_now_iso(),
        "runner": {
            "timezone": cfg.timezone_name,
            "force_run": bool(cfg.force_run),
            "part2_script": str(part2_script),
            "part2a_script": str(part2a_script),
            "part3_script": str(part3_script),
        },
        "part3": {
            "version": summary.get("version"),
            "publish_mode": summary.get("publish_mode"),
            "final_pass": bool(summary.get("final_pass", False)),
            "latest_alpha_state": summary.get("latest_alpha_state"),
            "latest_alpha_state_display": summary.get("latest_alpha_state_display"),
            "fusion_live_rate": safe_float(summary.get("fusion_live_rate")),
            "defense_ir_net": safe_float(summary.get("defense_ir_net")),
            "fused_ir_net": safe_float(summary.get("fused_ir_net")),
            "active_ir_vs_60_40": safe_float(summary.get("active_ir_vs_60_40")),
            "alpha_incremental_ir_net": safe_float(summary.get("alpha_incremental_ir_net")),
            "alpha_family_token_locked": summary.get("alpha_family_token_locked"),
            "summary_path": str(summary_path),
            "tape_path": str(tape_path),
            "governance_path": str(gov_path),
            "allocations_path": str(alloc_path),
            "prediction_log_path": str(pred_log_path),
        },
        "latest_decision": {
            "date": str(pd.to_datetime(latest_tape["Date"]).date()),
            "px_voo_call_7d": round(safe_float(latest_tape["px_voo_call_7d"]), 6),
            "px_ief_call_7d": round(safe_float(latest_tape["px_ief_call_7d"]), 6),
            "w_fused_voo": round(safe_float(latest_tape["w_fused_voo"]), 6),
            "w_fused_ief": round(safe_float(latest_tape["w_fused_ief"]), 6),
            "alpha_state": str(latest_tape["alpha_state"]),
            "alpha_state_display": str(latest_tape["alpha_state_display"]),
            "fusion_live": int(safe_float(latest_tape["fusion_live"], 0.0)),
            "part3_version": str(latest_tape["part3_version"]),
        },
        "latest_governance": {
            "date": str(pd.to_datetime(latest_gov["Date"]).date()),
            "alpha_state": str(latest_gov["alpha_state"]),
            "alpha_blocker_text": str(latest_gov.get("alpha_blocker_text", "")),
            "budget_mult": round(safe_float(latest_gov.get("budget_mult")), 6),
        },
        "latest_allocations": alloc_map,
    }

    csv_row: Dict[str, Any] = {
        "built_at_utc": prediction_payload["built_at_utc"],
        "decision_date": prediction_payload["latest_decision"]["date"],
        "px_voo_call_7d": prediction_payload["latest_decision"]["px_voo_call_7d"],
        "px_ief_call_7d": prediction_payload["latest_decision"]["px_ief_call_7d"],
        "w_fused_voo": prediction_payload["latest_decision"]["w_fused_voo"],
        "w_fused_ief": prediction_payload["latest_decision"]["w_fused_ief"],
        "alpha_state": prediction_payload["latest_decision"]["alpha_state"],
        "alpha_state_display": prediction_payload["latest_decision"]["alpha_state_display"],
        "fusion_live": prediction_payload["latest_decision"]["fusion_live"],
        "publish_mode": prediction_payload["part3"]["publish_mode"],
        "final_pass": prediction_payload["part3"]["final_pass"],
        "fusion_live_rate": prediction_payload["part3"]["fusion_live_rate"],
        "defense_ir_net": prediction_payload["part3"]["defense_ir_net"],
        "fused_ir_net": prediction_payload["part3"]["fused_ir_net"],
        "active_ir_vs_60_40": prediction_payload["part3"]["active_ir_vs_60_40"],
        "alpha_incremental_ir_net": prediction_payload["part3"]["alpha_incremental_ir_net"],
        "alpha_family_token_locked": prediction_payload["part3"]["alpha_family_token_locked"],
        "budget_mult": prediction_payload["latest_governance"]["budget_mult"],
        "alpha_blocker_text": prediction_payload["latest_governance"]["alpha_blocker_text"],
    }

    json_out = artifact_dir / "tuesday_prediction.json"
    csv_out = artifact_dir / "tuesday_prediction.csv"
    latest_json_alias = artifact_dir / "latest_prediction.json"

    export_json(json_out, prediction_payload)
    export_json(latest_json_alias, prediction_payload)
    export_csv(csv_out, csv_row)

    print("\n[OK] Tuesday prediction artifacts written:")
    print(f"- {json_out}")
    print(f"- {latest_json_alias}")
    print(f"- {csv_out}")
    print("\n[SUMMARY]")
    print(json.dumps(prediction_payload, indent=2))


if __name__ == "__main__":
    main()

