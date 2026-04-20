

# @title Part 5
from __future__ import annotations
import sys as _sys
import os as _os

# ── Colab / environment detection ─────────────────────────────────────────────
_IN_COLAB = "google.colab" in _sys.modules
_DRIVE_ROOT = _os.environ.get("PRICECALL_ROOT", "/content/drive/MyDrive/PriceCallProject")


def _colab_init(extra_packages=None):
    """Mount Google Drive (if in Colab) and pip-install any missing packages."""
    if _IN_COLAB:
        if not _os.path.exists("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        _os.makedirs(_DRIVE_ROOT, exist_ok=True)
        _os.environ.setdefault("PRICECALL_ROOT", _DRIVE_ROOT)
        _os.environ.setdefault("PRICECALL_STRICT_DRIVE_ONLY", "1")
        _os.environ.setdefault("PRICECALL_ALPHA_FAMILY", "part2a21")
    if extra_packages:
        import importlib, subprocess
        for pkg in extra_packages:
            mod = pkg.split("[")[0].replace("-", "_").split("==")[0]
            try:
                importlib.import_module(mod)
            except ImportError:
                print(f"[setup] pip install {pkg}")
                subprocess.run([_sys.executable, "-m", "pip", "install", pkg, "-q"],
                               capture_output=True)



import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class Part5Config:
    root_env_var: str = "PRICECALL_ROOT"
    strict_env_var: str = "PRICECALL_STRICT_DRIVE_ONLY"
    alpha_family_env_var: str = "PRICECALL_ALPHA_FAMILY"

    default_drive_root: str = "/content/drive/MyDrive/PriceCallProject"

    part0_candidates: Tuple[str, ...] = ("part0_data_infrastructure.py",)
    part1_candidates: Tuple[str, ...] = ("part1_builder.py",)
    part2_candidates: Tuple[str, ...] = ("part2_predictor.py",)
    # Part 2B and 2C are experimental sleeves — optional, validated at syntax level only.
    part2b_candidates: Tuple[str, ...] = ("part2b_xgb_ensemble.py",)
    part2c_candidates: Tuple[str, ...] = ("part2c_bnn_sleeve.py",)
    part3_candidates: Tuple[str, ...] = ("part3_governance.py",)
    part4_candidates: Tuple[str, ...] = ("part4_gui.py",)
    part6_candidates: Tuple[str, ...] = ("part6_regime_engine.py",)
    part7_candidates: Tuple[str, ...] = ("part7_portfolio_construction.py",)
    part8_candidates: Tuple[str, ...] = ("part8_execution_model.py",)
    part9_candidates: Tuple[str, ...] = ("part9_live_attribution.py",)
    part10_candidates: Tuple[str, ...] = ("part10_trading_bot.py", "part10_tradingbot.py")

    part2_summary_candidates: Tuple[str, ...] = (
        "artifacts_part2_g532/predictions/part2_g532_summary.json",
    )
    part3_summary_candidates: Tuple[str, ...] = (
        "artifacts_part3_v1/part3_summary.json",
    )
    predlog_relative: str = "artifacts_part3/prediction_log.csv"

    required_part3_summary_keys: Tuple[str, ...] = (
        "defense_source",
        "part2_summary_source",
        "alpha_positions_source",
        "alpha_summary_source",
        "alpha_eligibility_source",
        "alpha_summary_json_source",
        "alpha_contract",
        "alpha_family",
        "preferred_alpha_family",
        "strict_drive_only",
        "publish_mode",
        "final_pass",
        "latest_alpha_state",
        "prediction_log_path",
    )

    part2a_by_family: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "part2a21": (
                "part2a21_alpha.py",
                "part2a21_alpha_production_ready.py",
            ),
        }
    )


CFG = Part5Config()


def resolve_root(cfg: Part5Config = CFG) -> Path:
    raw: List[Path] = []

    env_root = os.environ.get(cfg.root_env_var, "").strip()
    if env_root:
        raw.append(Path(env_root))

    raw.append(Path(cfg.default_drive_root))

    try:
        raw.append(Path(_DRIVE_ROOT))
    except Exception:
        pass

    raw.append(Path.cwd())

    seen = set()
    resolved: List[Path] = []
    for p in raw:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        if str(rp) == "/content":
            continue
        key = str(rp)
        if key not in seen:
            seen.add(key)
            resolved.append(rp)

    for p in resolved:
        if p.exists():
            return p
    return Path.cwd().resolve()


def _find_first_existing(root: Path, candidates: Sequence[str], label: str) -> Path:
    tried: List[str] = []
    for name in candidates:
        p = (root / name).resolve()
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing required {label}. Tried: {', '.join(tried)}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _validate_python_syntax(path: Path) -> None:
    ast.parse(_read_text(path), filename=str(path))


def _count_main_blocks(path: Path) -> int:
    src = _read_text(path)
    return src.count('if __name__ == "__main__":') + src.count("if __name__ == '__main__':")


def _reject_notebook_export(path: Path, expected_part: Optional[str] = None) -> None:
    lowered = _read_text(path).lower()
    markers = {
        "part1": "# @title part 1",
        "part2": "# @title part 2",
        "part2a": "# @title part 2a",
        "part3": "# @title part 3",
        "part4": "# @title part 4",
        "part5": "# @title part 5",
    }

    if expected_part == "part3":
        forbidden = [markers["part1"], markers["part2"], markers["part2a"], markers["part4"], markers["part5"]]
        if any(m in lowered for m in forbidden):
            raise RuntimeError(f"{path.name} looks like a combined notebook export, not standalone Part 3.")

    if _count_main_blocks(path) > 1:
        raise RuntimeError(f"{path.name} contains multiple __main__ blocks and does not look standalone.")


def _inspect_part3_static_summary_contract(path: Path, cfg: Part5Config = CFG) -> None:
    src = _read_text(path)
    missing = []
    for key in cfg.required_part3_summary_keys:
        if f'"{key}"' not in src and f"'{key}'" not in src:
            missing.append(key)

    if missing:
        print(
            "[WARN] Static source scan did not find all Part 3 summary keys. "
            f"Missing in source scan: {missing}. "
            "Proceeding because runtime summary validation is authoritative."
        )


def _family_exists(root: Path, family: str, cfg: Part5Config = CFG) -> bool:
    return any((root / name).resolve().exists() for name in cfg.part2a_by_family[family])


def _installed_families(root: Path, cfg: Part5Config = CFG) -> List[str]:
    out: List[str] = []
    for fam in cfg.part2a_by_family:
        if _family_exists(root, fam, cfg):
            out.append(fam)
    return out


def _selected_alpha_family(root: Path, cfg: Part5Config = CFG) -> str:
    env_family = os.environ.get(cfg.alpha_family_env_var, "").strip().lower()
    if env_family:
        if env_family not in cfg.part2a_by_family:
            raise RuntimeError(
                f"Unsupported {cfg.alpha_family_env_var}={env_family}. "
                f"Supported: {sorted(cfg.part2a_by_family.keys())}"
            )
        if not _family_exists(root, env_family, cfg):
            tried = [str((root / name).resolve()) for name in cfg.part2a_by_family[env_family]]
            raise FileNotFoundError(
                f"{cfg.alpha_family_env_var}={env_family}, but no matching Part 2A file exists. "
                f"Tried: {', '.join(tried)}"
            )
        return env_family

    installed = _installed_families(root, cfg)
    if len(installed) == 1:
        print(f"[INFO] Auto-selected Part 2A family={installed[0]} because it is the only installed family.")
        return installed[0]

    if len(installed) == 0:
        raise FileNotFoundError(
            "No supported Part 2A family files were found under the project root. "
            f"Checked families: {sorted(cfg.part2a_by_family.keys())}"
        )

    raise RuntimeError(
        "Multiple Part 2A families are installed. "
        f"Set {cfg.alpha_family_env_var} explicitly to one of: {installed}"
    )


def _find_part2a_for_family(root: Path, family: str, cfg: Part5Config = CFG) -> Path:
    candidates = cfg.part2a_by_family[family]
    found = _find_first_existing(root, candidates, f"Part 2A ({family})")

    conflicting: List[str] = []
    for other_family, other_candidates in cfg.part2a_by_family.items():
        if other_family == family:
            continue
        for name in other_candidates:
            p = (root / name).resolve()
            if p.exists():
                conflicting.append(str(p))

    if conflicting:
        print(
            f"[INFO] Found non-selected Part 2A family files as well, but strict selection is enforced. "
            f"Using family={family}; ignoring: {conflicting}"
        )

    return found


def _expected_alpha_contract(alpha_family: str) -> str:
    if alpha_family == "part2a21":
        return "legacy_state_machine"
    raise RuntimeError(f"Unsupported alpha family: {alpha_family}")


def _cleanup_candidates_for_family(expected_alpha_family: str) -> List[Path]:
    if expected_alpha_family != "part2a21":
        raise RuntimeError(f"Unsupported alpha family: {expected_alpha_family}")

    common = [
        Path("/content/artifacts_part3"),
        Path("/content/artifacts_part3_v1"),
    ]
    return common + [
        Path("/content/artifacts_part2a_g5321"),
        Path("/content/artifacts_part2_g532"),
    ]


def _cleanup_conflicting_session_artifacts(expected_alpha_family: str) -> List[str]:
    removed: List[str] = []
    for p in _cleanup_candidates_for_family(expected_alpha_family):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        rp_str = str(rp)
        if not rp_str.startswith("/content/artifacts_"):
            continue
        if rp.exists():
            shutil.rmtree(rp, ignore_errors=True)
            if not rp.exists():
                removed.append(rp_str)
    return removed


def _run_subprocess(
    script_path: Path,
    root: Path,
    *,
    extra_args: Optional[Sequence[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env[CFG.root_env_var] = str(root)
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        text=True,
        capture_output=True,
    )


def _print_proc(label: str, proc: subprocess.CompletedProcess) -> None:
    print(f"\n=== {label} ===")
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print("--- STDERR ---")
        print(proc.stderr.rstrip())
    print(f"[exit={proc.returncode}]")


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_part2_summary(root: Path) -> Dict[str, object]:
    return _load_json(_find_first_existing(root, CFG.part2_summary_candidates, "Part 2 summary JSON"))


def _load_part3_summary(root: Path) -> Dict[str, object]:
    return _load_json(_find_first_existing(root, CFG.part3_summary_candidates, "Part 3 summary JSON"))


def _prediction_log_path(root: Path) -> Path:
    return (root / CFG.predlog_relative).resolve()


def _validate_persistent_predlog(root: Path, part3_summary: Dict[str, object]) -> Path:
    predlog = _prediction_log_path(root)
    if not predlog.exists():
        raise FileNotFoundError(f"Prediction log not found under project root: {predlog}")

    reported = str(part3_summary.get("prediction_log_path", "")).strip()
    if reported:
        rep = Path(reported).expanduser().resolve()
        if rep != predlog:
            raise RuntimeError(
                f"Part 3 reported prediction_log_path={rep}, but expected {predlog}."
            )

    if not str(predlog).startswith(str(root.resolve())):
        raise RuntimeError(f"Prediction log is not under project root: {predlog}")
    return predlog


def _validate_source_paths(part3_summary: Dict[str, object], root: Path) -> str:
    required = {
        "defense_source": "defense_source",
        "alpha_positions_source": "alpha_positions_source",
        "alpha_summary_source": "alpha_summary_source",
        "alpha_eligibility_source": "alpha_eligibility_source",
        "alpha_summary_json_source": "alpha_summary_json_source",
    }

    root_resolved = root.resolve()
    messages: List[str] = []
    for label, key in required.items():
        raw = str(part3_summary.get(key, "")).strip()
        if not raw:
            raise RuntimeError(
                f"Missing {label} in Part 3 summary. "
                f"The Part 3 file being executed is likely not the polished summary-contract version."
            )
        p = Path(raw).expanduser().resolve()
        p_str = str(p).replace("\\", "/")
        if "/content/artifacts_" in p_str and "/content/drive/" not in p_str:
            raise RuntimeError(f"{label} is contaminated by session artifacts: {p}")
        if not str(p).startswith(str(root_resolved)):
            raise RuntimeError(f"{label} is outside project root: {p}")
        messages.append(f"{label}={p}")
    return "; ".join(messages)


def _normalize_publish_mode(x: object) -> str:
    s = str(x).strip().upper() if x is not None else "UNKNOWN"
    return s if s in {"NORMAL", "DEFENSE_ONLY", "FAIL_CLOSED_NEUTRAL"} else "UNKNOWN"


def _normalize_alpha_family_tag(x: object) -> str:
    s = str(x).strip().lower() if x is not None else ""
    if not s:
        return ""
    if "part2a21" in s:
        return "part2a21"
    return s


def _validate_part2_contract(part2_summary: Dict[str, object]) -> Tuple[str, bool]:
    publish_mode = _normalize_publish_mode(part2_summary.get("publish_mode", "UNKNOWN"))
    final_pass = bool(part2_summary.get("final_pass", False))
    if publish_mode == "UNKNOWN":
        raise RuntimeError(f"Unexpected publish_mode from Part 2: {part2_summary.get('publish_mode')}")
    return publish_mode, final_pass


def _validate_part3_contract(part3_summary: Dict[str, object], expected_alpha_family: str) -> Tuple[str, bool, str, str]:
    alpha_family_raw = str(part3_summary.get("alpha_family", "UNKNOWN"))
    alpha_family = _normalize_alpha_family_tag(alpha_family_raw)
    alpha_contract = str(part3_summary.get("alpha_contract", "UNKNOWN"))

    preferred_alpha_family_raw = part3_summary.get("preferred_alpha_family", "")
    preferred_alpha_family = "" if preferred_alpha_family_raw is None else str(preferred_alpha_family_raw)
    preferred_alpha_family_norm = _normalize_alpha_family_tag(preferred_alpha_family)

    strict_drive_only = int(part3_summary.get("strict_drive_only", 0))
    publish_mode = _normalize_publish_mode(part3_summary.get("publish_mode", "UNKNOWN"))
    final_pass = bool(part3_summary.get("final_pass", False))
    latest_alpha_state = str(part3_summary.get("latest_alpha_state", "UNKNOWN"))

    if alpha_family != expected_alpha_family:
        raise RuntimeError(
            f"Alpha family mismatch. Expected {expected_alpha_family}, got {alpha_family_raw}."
        )
    if preferred_alpha_family and preferred_alpha_family_norm not in {expected_alpha_family, "unknown", "none", "null"}:
        raise RuntimeError(
            f"preferred_alpha_family mismatch. Expected {expected_alpha_family}, got {preferred_alpha_family}."
        )
    if strict_drive_only != 1:
        raise RuntimeError("Part 3 did not run with strict_drive_only=1.")
    if alpha_contract != _expected_alpha_contract(expected_alpha_family):
        raise RuntimeError(
            f"Unexpected alpha contract for {expected_alpha_family}: {alpha_contract}"
        )
    if publish_mode == "UNKNOWN":
        raise RuntimeError(f"Unexpected publish_mode from Part 3: {part3_summary.get('publish_mode')}")
    if latest_alpha_state not in {"SHADOW", "ELIGIBLE", "LIVE_TRIAL", "LIVE_FUSED"}:
        raise RuntimeError(f"Unexpected latest_alpha_state from Part 3: {latest_alpha_state}")

    return publish_mode, final_pass, latest_alpha_state, alpha_contract


def _predlog_stats(predlog_path: Path) -> Tuple[int, int, dict]:
    """Return (total_rows, realized_rows, schema_health).

    schema_health keys:
      has_deployment_mode  bool  — True if the new deployment_mode column is present
      has_horizon_legacy   bool  — True if the horizon_legacy column is present
      n_legacy_rows        int   — number of rows flagged as horizon_legacy
      rows_realized_fused  int   — from part3_summary (cross-checked against realized count)
    """
    schema: dict = {
        "has_deployment_mode": False,
        "has_horizon_legacy": False,
        "n_legacy_rows": 0,
    }
    if not predlog_path.exists():
        return 0, 0, schema
    try:
        df = pd.read_csv(predlog_path)
    except Exception:
        return 0, 0, schema

    total_rows = int(len(df))
    realized_rows = 0

    voo_real_cols = [c for c in ["px_voo_realized", "voo_realized"] if c in df.columns]
    ief_real_cols = [c for c in ["px_ief_realized", "ief_realized"] if c in df.columns]
    if voo_real_cols and ief_real_cols:
        mask = df[voo_real_cols[0]].notna() & df[ief_real_cols[0]].notna()
        if "horizon_legacy" in df.columns:
            mask = mask & (df["horizon_legacy"].fillna(0).astype(int) == 0)
        realized_rows = int(mask.sum())

    schema["has_deployment_mode"] = "deployment_mode" in df.columns
    schema["has_horizon_legacy"] = "horizon_legacy" in df.columns
    if schema["has_horizon_legacy"]:
        schema["n_legacy_rows"] = int(df["horizon_legacy"].fillna(0).astype(int).sum())

    return total_rows, realized_rows, schema



def run_pipeline(root: Path, validate_only: bool = False) -> None:
    part0 = _find_first_existing(root, CFG.part0_candidates, "Part 0")
    part1 = _find_first_existing(root, CFG.part1_candidates, "Part 1")
    part2 = _find_first_existing(root, CFG.part2_candidates, "Part 2")
    # Optional experimental sleeves — syntax-checked if present, never required.
    part2b_path = root / CFG.part2b_candidates[0]
    part2c_path = root / CFG.part2c_candidates[0]
    expected_alpha_family = _selected_alpha_family(root, CFG)
    part2a = _find_part2a_for_family(root, expected_alpha_family, CFG)
    part3 = _find_first_existing(root, CFG.part3_candidates, "Part 3")
    part6 = _find_first_existing(root, CFG.part6_candidates, "Part 6")
    part7 = _find_first_existing(root, CFG.part7_candidates, "Part 7")
    part8 = _find_first_existing(root, CFG.part8_candidates, "Part 8")
    part9 = _find_first_existing(root, CFG.part9_candidates, "Part 9")
    part10 = _find_first_existing(root, CFG.part10_candidates, "Part 10")

    print(f"ROOT: {root}")
    # Core pipeline — all required
    ordered_scripts = [
        ("PART0",  part0),
        ("PART6",  part6),
        ("PART1",  part1),
        ("PART2",  part2),
        ("PART2A", part2a),
        ("PART7",  part7),
        ("PART8",  part8),
        ("PART3",  part3),
        ("PART9",  part9),
        ("PART10", part10),
    ]
    for label, script in ordered_scripts:
        print(f"{label}: {script} | exists = {script.exists()}")
        _validate_python_syntax(script)

    # Optional experimental sleeves — syntax-checked only if file exists
    for label, path in [("PART2B", part2b_path), ("PART2C", part2c_path)]:
        if path.exists():
            print(f"{label}: {path} | exists = True (optional sleeve)")
            _validate_python_syntax(path)
        else:
            print(f"{label}: {path} | exists = False (optional, not required)")
    _reject_notebook_export(part3, expected_part="part3")
    _inspect_part3_static_summary_contract(part3, CFG)
    print("[OK] All standalone Python files passed syntax + standalone checks")

    if validate_only:
        return

    common_env = {
        CFG.root_env_var: str(root),
        CFG.strict_env_var: "1",
        CFG.alpha_family_env_var: expected_alpha_family,
    }

    ordered = [
        ("PART 0",  part0),
        ("PART 6",  part6),
        ("PART 1",  part1),
        ("PART 2",  part2),
    ]

    # Insert optional experimental sleeves between PART2 and PART2A.
    # Both are non-blocking: a non-zero exit code logs a warning and continues.
    optional_sleeves = [
        ("PART 2B", part2b_path),   # XGBoost ensemble uncertainty (step 5)
        ("PART 2C", part2c_path),   # BNN sleeve (step 6, only after 2B passes)
    ]

    ordered_tail = [
        ("PART 2A", part2a),
        ("PART 7",  part7),
        ("PART 8",  part8),
        ("PART 3",  part3),
        ("PART 9",  part9),
        ("PART 10", part10),
    ]

    for label, script in ordered:
        if label in {"PART 2A", "PART 3"}:
            removed = _cleanup_conflicting_session_artifacts(expected_alpha_family)
            if removed:
                print(f"[Cleanup before {label}] Removed conflicting session artifact trees: {removed}")
        proc = _run_subprocess(script, root, extra_env=common_env)
        _print_proc(label, proc)
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

    # Run optional sleeves — never raise, always continue
    for label, path in optional_sleeves:
        if not path.exists():
            print(f"[{label}] not found — skipping (optional experimental sleeve)")
            continue
        proc = _run_subprocess(path, root, extra_env=common_env)
        _print_proc(label, proc)
        if proc.returncode != 0:
            print(f"[WARN] {label} exited {proc.returncode} — continuing (experimental, non-blocking)")

    for label, script in ordered_tail:
        if label in {"PART 2A", "PART 3"}:
            removed = _cleanup_conflicting_session_artifacts(expected_alpha_family)
            if removed:
                print(f"[Cleanup before {label}] Removed conflicting session artifact trees: {removed}")
        proc = _run_subprocess(script, root, extra_env=common_env)
        _print_proc(label, proc)
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

    part2_summary = _load_part2_summary(root)
    part3_summary = _load_part3_summary(root)

    source_health = _validate_source_paths(part3_summary, root)
    part2_publish_mode, part2_final_pass = _validate_part2_contract(part2_summary)
    part3_publish_mode, part3_final_pass, latest_alpha_state, alpha_contract = _validate_part3_contract(
        part3_summary, expected_alpha_family
    )

    predlog_path = _validate_persistent_predlog(root, part3_summary)
    predlog_rows, predlog_realized_rows, predlog_schema = _predlog_stats(predlog_path)
    rows_realized_fused_summary = int(part3_summary.get("rows_realized_fused", -1))

    print("\n=== PART 5 VALIDATION ===")
    print(f"Source health: {source_health}")
    print(f"Prediction log path: {predlog_path}")
    print(f"Prediction log rows: {predlog_rows}")
    print(f"Prediction log realized rows: {predlog_realized_rows}")
    # Schema health checks — warn only, do not raise; migration script adds these columns.
    if not predlog_schema["has_deployment_mode"]:
        print("[WARN] Prediction log missing 'deployment_mode' column — run migrate_prediction_log.py")
    else:
        print("[OK] Prediction log has 'deployment_mode' column")
    if not predlog_schema["has_horizon_legacy"]:
        print("[WARN] Prediction log missing 'horizon_legacy' column — run migrate_prediction_log.py")
    else:
        n_leg = predlog_schema["n_legacy_rows"]
        if n_leg > 0:
            print(f"[OK] Prediction log has 'horizon_legacy' column ({n_leg} legacy rows excluded from realized count)")
        else:
            print("[OK] Prediction log has 'horizon_legacy' column (0 legacy rows)")
    # Cross-check: part3_summary rows_realized_fused vs predlog realized count
    if rows_realized_fused_summary >= 0 and rows_realized_fused_summary != predlog_realized_rows:
        # This is expected when rows_realized_fused was computed by the old proxy
        # formula (len(tape)-2). After the Part 3 fix it should match.
        print(
            f"[WARN] rows_realized_fused mismatch: part3_summary={rows_realized_fused_summary}, "
            f"predlog_realized={predlog_realized_rows}. "
            f"Expected to align after Part 3 fix is deployed."
        )
    print(f"Part 2 status: publish_mode={part2_publish_mode} | final_pass={part2_final_pass}")
    print(f"Part 3 status: publish_mode={part3_publish_mode} | final_pass={part3_final_pass} | latest_alpha_state={latest_alpha_state} | alpha_contract={alpha_contract}")
    print("[OK] Daily canonical pipeline completed successfully")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and run standalone daily Price Call Parts 0/6/1/2/2A/7/8/3/9/10 from the project root."
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the files; do not execute the pipeline.",
    )

    args, extras = p.parse_known_args(argv)
    if extras:
        print(f"[INFO] Ignoring extra notebook/launcher args: {' '.join(extras)}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root = resolve_root(CFG)
    os.environ[CFG.root_env_var] = str(root)
    run_pipeline(root, validate_only=args.validate_only)


if __name__ == "__main__":
    main()


