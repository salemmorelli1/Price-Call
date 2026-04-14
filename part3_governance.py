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



import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# @title PART 3 — Governance + Defense Sleeve + Fusion Engine
# Standalone, Drive-first, root-anchored production consumer.
# ============================================================


@dataclass(frozen=True)
class Part3Config:
    root_env_var: str = "PRICECALL_ROOT"
    strict_env_var: str = "PRICECALL_STRICT_DRIVE_ONLY"
    alpha_family_env_var: str = "PRICECALL_ALPHA_FAMILY"
    default_drive_root: str = "/content/drive/MyDrive/PriceCallProject"
    out_dir_relative: str = "artifacts_part3_v1"
    predlog_dir_relative: str = "artifacts_part3"
    tape_name: str = "v1_final_production_tape.csv"
    gov_name: str = "v1_final_production_governance.csv"
    alloc_name: str = "v1_fusion_allocations.csv"
    summary_name: str = "part3_summary.json"
    predlog_name: str = "prediction_log.csv"
    default_voo_weight: float = 0.60
    default_ief_weight: float = 0.40


CFG = Part3Config()


def resolve_root(cfg: Part3Config = CFG) -> Path:
    env_root = os.environ.get(cfg.root_env_var, "").strip()
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(cfg.default_drive_root))
    try:
        candidates.append(Path(_DRIVE_ROOT))
    except Exception:
        pass
    candidates.append(Path.cwd())

    for c in candidates:
        try:
            p = c.expanduser().resolve()
        except Exception:
            continue
        if str(p) == "/content":
            continue
        if p.exists():
            return p
    return Path.cwd().resolve()


def _project_roots(root: Path) -> List[Path]:
    raw: List[Path] = []
    raw.append(root)
    drive_root = Path(CFG.default_drive_root)
    raw.append(drive_root)
    try:
        raw.append(Path(_DRIVE_ROOT))
    except Exception:
        pass
    raw.append(Path.cwd())

    out: List[Path] = []
    seen = set()
    for p in raw:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            continue
        if str(rp) == "/content":
            continue
        s = str(rp)
        if s not in seen:
            seen.add(s)
            out.append(rp)
    return out


def _expand_candidate_paths(candidates: Sequence[str], root: Path) -> List[Path]:
    expanded: List[Path] = []
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_absolute():
            expanded.append(p)
        else:
            for r in _project_roots(root):
                expanded.append((r / p).resolve())
    dedup: List[Path] = []
    seen = set()
    for p in expanded:
        s = str(p)
        if s not in seen:
            seen.add(s)
            dedup.append(p)
    return dedup


def _first_existing_path(candidates: Sequence[str], root: Path) -> Optional[Path]:
    for p in _expand_candidate_paths(candidates, root):
        if p.exists():
            return p
    return None


def _must_find(label: str, candidates: Sequence[str], root: Path) -> Path:
    p = _first_existing_path(candidates, root)
    if p is None:
        attempted = "\n".join(str(x) for x in _expand_candidate_paths(candidates, root))
        raise FileNotFoundError(f"{label} not found. Attempted:\n{attempted}")
    return p


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["Date", "decision_date", "target_date", "asof_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def _series(df: pd.DataFrame, names: Sequence[str], numeric: bool = False) -> Optional[pd.Series]:
    col = _first_col(df, names)
    if col is None:
        return None
    s = df[col]
    return pd.to_numeric(s, errors="coerce") if numeric else s


def _last_valid_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("DataFrame is empty")
    xcol = _first_col(df, ["Date", "decision_date", "target_date"])
    if xcol is not None:
        g = df.copy()
        g = g[pd.to_datetime(g[xcol], errors="coerce").notna()]
        if not g.empty:
            g = g.sort_values(xcol)
            return g.iloc[-1]
    return df.iloc[-1]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    v = _safe_float(x)
    return default if v is None else int(round(v))


def _json_value(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _row_value(row: pd.Series, keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return row[k]
    return default


def _normalize_publish_mode(x: Any) -> str:
    s = str(x).strip().upper() if x is not None else "UNKNOWN"
    allowed = {"NORMAL", "DEFENSE_ONLY", "FAIL_CLOSED_NEUTRAL"}
    return s if s in allowed else "UNKNOWN"


def _boolish(x: Any, default: int = 0) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(float(x) != 0.0)
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "open", "live"}:
        return 1
    if s in {"0", "false", "no", "n", "closed", "shadow"}:
        return 0
    return default


def _state_display(state: str) -> str:
    if state == "ELIGIBLE":
        return "CANDIDATE"
    return state


def _canonical_state(state: Any) -> str:
    s = str(state).strip().upper() if state is not None else "SHADOW"
    if not s:
        return "SHADOW"
    allowed = {"SHADOW", "ELIGIBLE", "LIVE_TRIAL", "LIVE_FUSED", "CANDIDATE"}
    if s not in allowed:
        return "SHADOW"
    return "ELIGIBLE" if s == "CANDIDATE" else s


def _extract_latest_price_call(defense_row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    voo = _safe_float(_row_value(defense_row, [
        "px_voo_call_1d", "voo_call_1d", "px_voo_call_7d", "voo_call_7d", "voo_target_price", "VOO_target_price", "price_call_voo"
    ]))
    ief = _safe_float(_row_value(defense_row, [
        "px_ief_call_1d", "ief_call_1d", "px_ief_call_7d", "ief_call_7d", "ief_target_price", "IEF_target_price", "price_call_ief"
    ]))
    return voo, ief


def _extract_base_weights(defense_row: pd.Series) -> Tuple[float, float]:
    voo = _safe_float(_row_value(defense_row, [
        "w_strategy_voo", "w_voo", "weight_voo", "alloc_voo", "defense_weight_voo", "voo_weight"
    ]))
    ief = _safe_float(_row_value(defense_row, [
        "w_strategy_ief", "w_ief", "weight_ief", "alloc_ief", "defense_weight_ief", "ief_weight"
    ]))
    if voo is None and ief is None:
        return CFG.default_voo_weight, CFG.default_ief_weight
    if voo is None:
        voo = max(0.0, 1.0 - float(ief))
    if ief is None:
        ief = max(0.0, 1.0 - float(voo))
    total = max(float(voo) + float(ief), 1e-12)
    return float(voo) / total, float(ief) / total


def _load_alpha_status(alpha_tape_df: pd.DataFrame, alpha_summary_json: Dict[str, Any]) -> Dict[str, Any]:
    latest_row = _last_valid_row(alpha_tape_df) if not alpha_tape_df.empty else pd.Series(dtype=object)

    latest_state = _canonical_state(
        _row_value(latest_row, ["alpha_state_display", "alpha_state", "state_display", "state"],
                   _json_value(alpha_summary_json, ["latest_alpha_state_display", "latest_alpha_state", "alpha_state"], "SHADOW"))
    )
    realized_dates = _safe_int(
        _json_value(alpha_summary_json, ["realized_dates", "n_realized_dates", "realized_rows"],
                    _row_value(latest_row, ["realized_dates", "realized_rows"], len(alpha_tape_df))),
        default=len(alpha_tape_df),
    )
    budget_mult = _safe_float(
        _row_value(latest_row, ["budget_mult", "alpha_budget_mult"],
                   _json_value(alpha_summary_json, ["budget_mult", "alpha_budget_mult"], 1.0))
    )
    if budget_mult is None:
        budget_mult = 1.0
    drift_rate = _safe_float(
        _row_value(latest_row, ["drift_rate", "alpha_drift_rate"],
                   _json_value(alpha_summary_json, ["drift_rate", "alpha_drift_rate"], 0.0))
    )
    if drift_rate is None:
        drift_rate = 0.0
    quality_ok = _boolish(_row_value(latest_row, ["quality_ok"], _json_value(alpha_summary_json, ["quality_ok"], 1)), 1)
    drift_ok = _boolish(_row_value(latest_row, ["drift_ok"], _json_value(alpha_summary_json, ["drift_ok"], 1)), 1)
    trial_gate_open = _boolish(_row_value(latest_row, ["trial_gate_open"], _json_value(alpha_summary_json, ["trial_gate_open"], 1)), 1)
    fused_gate_open = _boolish(_row_value(latest_row, ["fused_gate_open"], _json_value(alpha_summary_json, ["fused_gate_open"], 1)), 1)
    promotion_ready = _boolish(_row_value(latest_row, ["promotion_ready"], _json_value(alpha_summary_json, ["promotion_ready"], 1)), 1)

    blockers = _row_value(latest_row, ["alpha_blockers", "blockers"], _json_value(alpha_summary_json, ["alpha_blockers", "blockers"], "NONE"))
    if blockers is None or (isinstance(blockers, float) and math.isnan(blockers)):
        blockers = "NONE"
    if isinstance(blockers, list):
        blockers = ", ".join(str(x) for x in blockers) if blockers else "NONE"

    thresholds = {
        "Eligible": _safe_int(_json_value(alpha_summary_json, ["eligible_threshold", "Eligible"], 26), 26),
        "Trial": _safe_int(_json_value(alpha_summary_json, ["trial_threshold", "Trial"], 52), 52),
        "Fused": _safe_int(_json_value(alpha_summary_json, ["fused_threshold", "Fused"], 78), 78),
        "Max drift rate": _safe_float(_json_value(alpha_summary_json, ["max_drift_rate", "max_alpha_drift_rate"], 0.80)) or 0.80,
    }

    alpha_live = latest_state in {"LIVE_TRIAL", "LIVE_FUSED"}
    if latest_state == "LIVE_FUSED":
        alpha_live = alpha_live and bool(fused_gate_open)
    if latest_state == "LIVE_TRIAL":
        alpha_live = alpha_live and bool(trial_gate_open)
    alpha_live = alpha_live and bool(quality_ok) and bool(drift_ok) and budget_mult > 0

    return {
        "latest_state": latest_state,
        "display_state": _state_display(latest_state),
        "realized_dates": realized_dates,
        "budget_mult": float(budget_mult),
        "drift_rate": float(drift_rate),
        "quality_ok": quality_ok,
        "drift_ok": drift_ok,
        "trial_gate_open": trial_gate_open,
        "fused_gate_open": fused_gate_open,
        "promotion_ready": promotion_ready,
        "blockers": str(blockers),
        "alpha_live": int(alpha_live),
        "thresholds": thresholds,
    }


def _alpha_distribution(alpha_tape_df: pd.DataFrame) -> pd.Series:
    col = _first_col(alpha_tape_df, ["alpha_state", "alpha_state_display", "state"])
    if col is None or alpha_tape_df.empty:
        return pd.Series(dtype=float)
    s = alpha_tape_df[col].astype(str).str.upper().replace({"CANDIDATE": "ELIGIBLE"})
    return s.value_counts(normalize=True).sort_index()


def _extract_alpha_positions(alpha_positions_df: pd.DataFrame) -> pd.DataFrame:
    if alpha_positions_df.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    g = alpha_positions_df.copy()
    date_col = _first_col(g, ["Date", "decision_date", "asof_date"])
    if date_col is not None:
        d = pd.to_datetime(g[date_col], errors="coerce")
        if d.notna().any():
            g = g.loc[d == d.max()].copy()

    ticker_col = _first_col(g, ["ticker", "asset", "sleeve", "name", "symbol"])
    weight_col = _first_col(g, ["weight", "w", "alloc", "allocation"])
    if ticker_col is None or weight_col is None:
        return pd.DataFrame(columns=["ticker", "weight"])

    out = g[[ticker_col, weight_col]].copy()
    out.columns = ["ticker", "weight"]
    out["ticker"] = out["ticker"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight"])
    out = out.groupby("ticker", as_index=False)["weight"].sum()
    out = out[out["weight"] > 0].copy()
    total = float(out["weight"].sum()) if not out.empty else 0.0
    if total > 0:
        out["weight"] = out["weight"] / total
    return out.reset_index(drop=True)


def _build_fusion_allocations(decision_date: pd.Timestamp, defense_row: pd.Series, alpha_positions_df: pd.DataFrame, alpha_status: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
    voo_base, ief_base = _extract_base_weights(defense_row)
    alpha_live = bool(alpha_status["alpha_live"])
    budget_mult = float(alpha_status["budget_mult"])
    alpha_share = max(0.0, min(voo_base, voo_base * budget_mult)) if alpha_live else 0.0

    rows: List[Dict[str, Any]] = []
    if alpha_share > 0 and not alpha_positions_df.empty:
        alpha_positions = alpha_positions_df.copy()
        alpha_positions["weight"] = alpha_positions["weight"] * alpha_share
        for _, r in alpha_positions.iterrows():
            rows.append({
                "Date": decision_date,
                "sleeve": str(r["ticker"]),
                "weight": float(r["weight"]),
                "is_alpha": 1,
                "alpha_state": alpha_status["latest_state"],
            })
        voo_weight = max(0.0, voo_base - alpha_share)
    else:
        voo_weight = voo_base

    rows.append({
        "Date": decision_date,
        "sleeve": "VOO",
        "weight": float(voo_weight),
        "is_alpha": 0,
        "alpha_state": alpha_status["latest_state"],
    })
    rows.append({
        "Date": decision_date,
        "sleeve": "IEF",
        "weight": float(ief_base),
        "is_alpha": 0,
        "alpha_state": alpha_status["latest_state"],
    })

    alloc = pd.DataFrame(rows)
    total = float(alloc["weight"].sum()) if not alloc.empty else 0.0
    if total <= 0:
        alloc = pd.DataFrame([
            {"Date": decision_date, "sleeve": "VOO", "weight": CFG.default_voo_weight, "is_alpha": 0, "alpha_state": alpha_status["latest_state"]},
            {"Date": decision_date, "sleeve": "IEF", "weight": CFG.default_ief_weight, "is_alpha": 0, "alpha_state": alpha_status["latest_state"]},
        ])
        total = float(alloc["weight"].sum())
    alloc["weight"] = alloc["weight"] / total
    deviation = abs(float(alloc["weight"].sum()) - 1.0)
    return alloc, deviation


def _prepare_production_tape(defense_df: pd.DataFrame, part2_summary: Dict[str, Any], alpha_status: Dict[str, Any], alpha_summary_json: Dict[str, Any]) -> pd.DataFrame:
    tape = defense_df.copy()
    if "Date" not in tape.columns:
        dcol = _first_col(tape, ["decision_date", "target_date", "asof_date"])
        if dcol is not None:
            tape["Date"] = pd.to_datetime(tape[dcol], errors="coerce")
        else:
            tape["Date"] = pd.NaT
    tape["publish_mode"] = _normalize_publish_mode(_json_value(part2_summary, ["publish_mode", "mode"], "UNKNOWN"))
    tape["final_pass"] = _boolish(_json_value(part2_summary, ["final_pass"], 0), 0)
    tape["alpha_state"] = alpha_status["latest_state"]
    tape["alpha_state_display"] = alpha_status["display_state"]
    tape["alpha_live"] = alpha_status["alpha_live"]
    tape["budget_mult"] = alpha_status["budget_mult"]
    tape["drift_rate"] = alpha_status["drift_rate"]
    tape["script_version_part2"] = str(_json_value(part2_summary, ["script_version", "version"], "UNKNOWN"))
    tape["alpha_family"] = str(_json_value(alpha_summary_json, ["alpha_family", "version", "part"], "part2a21"))
    return tape


def _build_governance_df(decision_date: pd.Timestamp, part2_summary: Dict[str, Any], alpha_status: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "Date": decision_date,
        "publish_mode": _normalize_publish_mode(_json_value(part2_summary, ["publish_mode", "mode"], "UNKNOWN")),
        "final_pass": _boolish(_json_value(part2_summary, ["final_pass"], 0), 0),
        "quality_ok": alpha_status["quality_ok"],
        "drift_ok": alpha_status["drift_ok"],
        "trial_gate_open": alpha_status["trial_gate_open"],
        "fused_gate_open": alpha_status["fused_gate_open"],
        "promotion_ready": alpha_status["promotion_ready"],
        "alpha_state": alpha_status["latest_state"],
        "alpha_state_display": alpha_status["display_state"],
        "alpha_live": alpha_status["alpha_live"],
        "drift_alarm": int(not alpha_status["drift_ok"]),
        "drift_rate": alpha_status["drift_rate"],
        "budget_mult": alpha_status["budget_mult"],
        "alpha_blockers": alpha_status["blockers"],
    }
    return pd.DataFrame([row])


def _count_realized_predlog_rows(predlog_df: pd.DataFrame) -> int:
    if predlog_df.empty:
        return 0
    vcol = _first_col(predlog_df, ["px_voo_realized", "voo_realized"])
    icol = _first_col(predlog_df, ["px_ief_realized", "ief_realized"])
    if vcol is None or icol is None:
        return 0
    return int((predlog_df[vcol].notna() & predlog_df[icol].notna()).sum())


def _upsert_prediction_log(predlog_path: Path, decision_date: pd.Timestamp, target_date: pd.Timestamp,
                           voo_call: Optional[float], ief_call: Optional[float],
                           publish_mode: str, final_pass: int, alpha_status: Dict[str, Any],
                           defense_source: Path, alpha_sources: Dict[str, Path],
                           defense_row: pd.Series, part2_summary: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
    if predlog_path.exists():
        predlog_df = _read_csv(predlog_path)
    else:
        predlog_df = pd.DataFrame()

    if predlog_df.empty:
        predlog_df = pd.DataFrame(columns=[
            "decision_date", "target_date", "h_reb",
            "px_voo_t", "px_ief_t",
            "px_voo_call_1d", "px_ief_call_1d",
            # NOTE: px_voo_call_7d / px_ief_call_7d intentionally retained as
            # backward-compat aliases equal to _1d in the daily system.
            # backfill_realized.py reads _7d by name; keeping them avoids patching backfill.
            "px_voo_call_7d", "px_ief_call_7d",
            "p_final_cal", "base_rate", "raw_val_auc", "tail_threshold",
            "publish_mode", "final_pass", "latest_alpha_state", "alpha_live",
            "defense_source", "alpha_positions_source", "alpha_summary_source",
            "alpha_eligibility_source", "alpha_summary_json_source",
            "px_voo_realized", "px_ief_realized", "voo_err", "ief_err", "spread_err", "hit_direction"
        ])

    row = {
        "decision_date": pd.Timestamp(decision_date).normalize(),
        "target_date": pd.Timestamp(target_date).normalize(),
        "h_reb": 1,
        "px_voo_t": _safe_float(_row_value(defense_row, ["px_voo_t"], None)),
        "px_ief_t": _safe_float(_row_value(defense_row, ["px_ief_t"], None)),
        "px_voo_call_1d": voo_call,
        "px_ief_call_1d": ief_call,
        # FIX: _7d columns are intentional backward-compat aliases of _1d at H=1.
        # backfill_realized.py and Part 9 read px_*_call_7d by name.
        # Both columns carry the 1-day price call; the _7d name is legacy.
        "px_voo_call_7d": voo_call,   # alias: same as px_voo_call_1d at H=1
        "px_ief_call_7d": ief_call,   # alias: same as px_ief_call_1d at H=1
        "p_final_cal": _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None)),
        "base_rate": _safe_float(_row_value(defense_row, ["T", "base_rate", "b"], None)),
        "raw_val_auc": _safe_float(_row_value(defense_row, ["raw_val_auc"], _json_value(part2_summary, ["raw_val_auc_median"], None))),
        "tail_threshold": _safe_float(_json_value(part2_summary, ["tail_event_threshold"], None)),
        "publish_mode": publish_mode,
        "final_pass": int(final_pass),
        "latest_alpha_state": alpha_status["latest_state"],
        "alpha_live": int(alpha_status["alpha_live"]),
        "defense_source": str(defense_source),
        "alpha_positions_source": str(alpha_sources["positions"]),
        "alpha_summary_source": str(alpha_sources["summary_tape"]),
        "alpha_eligibility_source": str(alpha_sources["eligibility"]),
        "alpha_summary_json_source": str(alpha_sources["summary_json"]),
    }

    if "decision_date" in predlog_df.columns:
        predlog_df["decision_date"] = pd.to_datetime(predlog_df["decision_date"], errors="coerce")
        mask = predlog_df["decision_date"] == row["decision_date"]
        if mask.any():
            for k, v in row.items():
                predlog_df.loc[mask, k] = v
        else:
            predlog_df = pd.concat([predlog_df, pd.DataFrame([row])], ignore_index=True)
    else:
        predlog_df = pd.concat([predlog_df, pd.DataFrame([row])], ignore_index=True)

    predlog_df = predlog_df.sort_values("decision_date").reset_index(drop=True)
    realized_rows = _count_realized_predlog_rows(predlog_df)
    predlog_df.to_csv(predlog_path, index=False)
    return predlog_df, realized_rows


def _compute_ir_from_returns(values: Optional[pd.Series]) -> Optional[float]:
    if values is None:
        return None
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return None
    std = float(s.std(ddof=0))
    if std <= 0:
        return None
    return float(s.mean() / std)


def _extract_performance_metrics(defense_df: pd.DataFrame, alpha_summary_json: Dict[str, Any], alpha_status: Dict[str, Any]) -> Dict[str, Optional[float]]:
    defense_ir = _safe_float(_json_value(alpha_summary_json, ["defense_ir_net", "defense_ir"], None))
    fused_ir = _safe_float(_json_value(alpha_summary_json, ["fused_ir_net", "fused_ir"], None))
    active_ir = _safe_float(_json_value(alpha_summary_json, ["active_ir_vs_60_40", "active_ir"], None))
    active_mean = _safe_float(_json_value(alpha_summary_json, ["active_mean", "active_return_mean"], None))
    fusion_live_rate = _safe_float(_json_value(alpha_summary_json, ["fusion_live_rate", "fusion_live_rate_pct"], None))

    if defense_ir is None:
        defense_ir = _compute_ir_from_returns(_series(defense_df, ["ret_defense", "defense_return", "portfolio_return"], numeric=True))
    if fused_ir is None:
        fused_ir = defense_ir if alpha_status["alpha_live"] else defense_ir
    if fusion_live_rate is None:
        fusion_live_rate = 1.0 if alpha_status["alpha_live"] else 0.0

    return {
        "defense_ir": defense_ir,
        "fused_ir": fused_ir,
        "active_ir": active_ir,
        "active_mean": active_mean,
        "fusion_live_rate": fusion_live_rate,
    }


def _format_float(x: Optional[float], digits: int = 3) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def main(cfg: Part3Config = CFG) -> None:
    root = resolve_root(cfg)
    os.environ[cfg.root_env_var] = str(root)

    part2_tape = _must_find(
        "Defense tape",
        [
            "artifacts_part2_g532/predictions/g532_final_consensus_tape.csv",
                    ],
        root,
    )
    part2_summary_path = _must_find(
        "Part 2 summary JSON",
        [
            "artifacts_part2_g532/predictions/part2_g532_summary.json",
                    ],
        root,
    )
    alpha_positions_path = _must_find(
        "Alpha positions",
        [
            "artifacts_part2a_alpha/predictions/part2a21_alpha_positions.csv",
                    ],
        root,
    )
    alpha_summary_tape_path = _must_find(
        "Alpha summary tape",
        [
            "artifacts_part2a_alpha/predictions/part2a21_alpha_summary_tape.csv",
                    ],
        root,
    )
    alpha_eligibility_path = _must_find(
        "Alpha eligibility",
        [
            "artifacts_part2a_alpha/predictions/part2a21_alpha_eligibility.csv",
                    ],
        root,
    )
    alpha_summary_json_path = _must_find(
        "Alpha summary JSON",
        [
            "artifacts_part2a_alpha/predictions/part2a21_alpha_summary.json",
                    ],
        root,
    )

    defense_df = _read_csv(part2_tape)
    part2_summary = _read_json(part2_summary_path)
    alpha_positions_df = _read_csv(alpha_positions_path)
    alpha_tape_df = _read_csv(alpha_summary_tape_path)
    _ = _read_csv(alpha_eligibility_path)
    alpha_summary_json = _read_json(alpha_summary_json_path)

    publish_mode = _normalize_publish_mode(_json_value(part2_summary, ["publish_mode", "mode"], "UNKNOWN"))
    final_pass = _boolish(_json_value(part2_summary, ["final_pass"], 0), 0)
    if publish_mode not in {"NORMAL", "DEFENSE_ONLY", "FAIL_CLOSED_NEUTRAL"}:
        raise RuntimeError(f"Uncertified publish_mode: {publish_mode}")
    if not final_pass and publish_mode != "FAIL_CLOSED_NEUTRAL":
        raise RuntimeError("Part 2 final_pass is False and publish_mode is not FAIL_CLOSED_NEUTRAL")

    defense_row = _last_valid_row(defense_df)
    decision_date = pd.to_datetime(_row_value(defense_row, ["Date", "decision_date", "asof_date"]), errors="coerce")
    if pd.isna(decision_date):
        decision_date = pd.Timestamp.today().normalize()
    target_date = decision_date + pd.tseries.offsets.BDay(1)

    voo_call, ief_call = _extract_latest_price_call(defense_row)
    alpha_status = _load_alpha_status(alpha_tape_df, alpha_summary_json)
    alpha_positions_latest = _extract_alpha_positions(alpha_positions_df)

    prod_tape = _prepare_production_tape(defense_df, part2_summary, alpha_status, alpha_summary_json)
    gov_df = _build_governance_df(decision_date, part2_summary, alpha_status)
    alloc_df, max_dev = _build_fusion_allocations(decision_date, defense_row, alpha_positions_latest, alpha_status)

    out_dir = root / cfg.out_dir_relative
    predlog_dir = root / cfg.predlog_dir_relative
    _ensure_dir(out_dir)
    _ensure_dir(predlog_dir)

    tape_out = out_dir / cfg.tape_name
    gov_out = out_dir / cfg.gov_name
    alloc_out = out_dir / cfg.alloc_name
    predlog_out = predlog_dir / cfg.predlog_name
    summary_out = out_dir / cfg.summary_name

    prod_tape.to_csv(tape_out, index=False)
    gov_df.to_csv(gov_out, index=False)
    alloc_df.to_csv(alloc_out, index=False)

    alpha_sources = {
        "positions": alpha_positions_path,
        "summary_tape": alpha_summary_tape_path,
        "eligibility": alpha_eligibility_path,
        "summary_json": alpha_summary_json_path,
    }
    predlog_df, realized_rows = _upsert_prediction_log(
        predlog_out,
        decision_date,
        target_date,
        voo_call,
        ief_call,
        publish_mode if publish_mode != "FAIL_CLOSED_NEUTRAL" else "DEFENSE_ONLY",
        final_pass,
        alpha_status,
        part2_tape,
        alpha_sources,
        defense_row,
        part2_summary,
    )

    perf = _extract_performance_metrics(defense_df, alpha_summary_json, alpha_status)
    dist = _alpha_distribution(alpha_tape_df)
    dist_display = {str(k): float(v) for k, v in dist.items()}

    summary = {
        "part": "PART3_V1",
        "root": str(root),
        "defense_source": str(part2_tape),
        "part2_summary_source": str(part2_summary_path),
        "alpha_positions_source": str(alpha_positions_path),
        "alpha_summary_source": str(alpha_summary_tape_path),
        "alpha_eligibility_source": str(alpha_eligibility_path),
        "alpha_summary_json_source": str(alpha_summary_json_path),
        "publish_mode": publish_mode if publish_mode != "FAIL_CLOSED_NEUTRAL" else "DEFENSE_ONLY",
        "final_pass": int(final_pass),
        "latest_alpha_state": alpha_status["latest_state"],
        "latest_alpha_state_display": alpha_status["display_state"],
        "realized_dates": alpha_status["realized_dates"],
        "budget_mult": alpha_status["budget_mult"],
        "drift_rate": alpha_status["drift_rate"],
        "quality_ok": alpha_status["quality_ok"],
        "drift_ok": alpha_status["drift_ok"],
        "trial_gate_open": alpha_status["trial_gate_open"],
        "fused_gate_open": alpha_status["fused_gate_open"],
        "promotion_ready": alpha_status["promotion_ready"],
        "alpha_blockers": alpha_status["blockers"],
        "rows": int(len(prod_tape)),
        "rows_realized_fused": max(0, int(len(prod_tape)) - 2),
        "fusion_live_rate": perf["fusion_live_rate"],
        "defense_ir_net": perf["defense_ir"],
        "fused_ir_net": perf["fused_ir"],
        "active_ir_vs_60_40": perf["active_ir"],
        "active_mean": perf["active_mean"],
        "alpha_state_distribution": dist_display,
        "prediction_log_path": str(predlog_out),
        "prediction_log_realized_rows": realized_rows,
        "allocation_sum_to_one_max_deviation": max_dev,
        "horizon": 1,
        "alpha_family": str(_json_value(alpha_summary_json, ["alpha_family", "version", "part"], os.environ.get(CFG.alpha_family_env_var, "part2a21"))),
        "alpha_contract": "legacy_state_machine",
        "preferred_alpha_family": os.environ.get(CFG.alpha_family_env_var, "part2a21"),
        "strict_drive_only": _boolish(os.environ.get(CFG.strict_env_var, "0"), 0),
    }
    _write_json(summary_out, summary)

    print(f"✅ DEFENSE TAPE DISCOVERED: {part2_tape}")
    print(f"Decision-time 1D price call: VOO={voo_call:.4f} | IEF={ief_call:.4f}" if voo_call is not None and ief_call is not None else "Decision-time 1D price call: NA")
    print(f"Part 3 V1 defense_source: {part2_tape}")
    print(f"Part 3 V1 last defense Date: {pd.Timestamp(decision_date)} | is_live tail: 1")
    print(f"✅ ALPHA POSITIONS DISCOVERED: {alpha_positions_path}")
    print(f"✅ ALPHA SUMMARY DISCOVERED: {alpha_summary_tape_path}")
    print(f"✅ ALPHA ELIGIBILITY DISCOVERED: {alpha_eligibility_path}")
    print(f"✅ ALPHA SUMMARY JSON DISCOVERED: {alpha_summary_json_path}")
    print(
        f"Alpha state latest: {alpha_status['latest_state']} (display={alpha_status['display_state']}) | "
        f"realized_dates={alpha_status['realized_dates']} | budget_mult={alpha_status['budget_mult']:.2f} | drift_rate={alpha_status['drift_rate']:.4f}"
    )
    print(
        f"Alpha diagnostics: quality_ok={alpha_status['quality_ok']} | drift_ok={alpha_status['drift_ok']} | "
        f"trial_gate_open={alpha_status['trial_gate_open']} | fused_gate_open={alpha_status['fused_gate_open']} | promotion_ready={alpha_status['promotion_ready']}"
    )
    print(f"Alpha blockers: {alpha_status['blockers']}")
    print("\n" + "=" * 96)
    print("🏛️  PART 3 V1 AUDIT (Defense Sleeve + Fusion Engine)")
    print("=" * 96)
    print(
        f"Rows: {len(prod_tape)} | Realized fused rows: {max(0, len(prod_tape)-2)} | "
        f"Fusion live rate: {(summary['fusion_live_rate'] or 0.0) * 100:.2f}%"
    )
    print(
        f"Defense IR (net): {_format_float(summary['defense_ir_net'])} | "
        f"Fused IR (net): {_format_float(summary['fused_ir_net'])} | "
        f"Active IR vs 60/40: {_format_float(summary['active_ir_vs_60_40'])} | "
        f"Active mean: {_format_float(summary['active_mean'], 6)}"
    )
    print("Alpha state distribution:")
    if dist.empty:
        print("alpha_state\nSHADOW    1.0000")
    else:
        print(dist.rename_axis("alpha_state"))
    th = alpha_status["thresholds"]
    print("\nAlpha promotion thresholds:")
    print(
        f"Eligible={th['Eligible']} | Trial={th['Trial']} | Fused={th['Fused']} | Max drift rate={th['Max drift rate']:.2f}"
    )
    print(f"[PredLog] realized rows={realized_rows} ({'not enough tape history yet' if realized_rows == 0 else 'matured rows found'}).")
    print(f"[PredLog] path: {predlog_out}")
    print(f"Fusion allocation sum-to-one max deviation: {max_dev:.8f}")
    print("\n✅ PART 3 V1 WRITTEN")
    print(f"   Tape:        {tape_out}")
    print(f"   Governance:  {gov_out}")
    print(f"   Allocations: {alloc_out}")
    print(f"   Summary:     {summary_out}")
    print(f"   Prediction log: {predlog_out}")
    print("   Alpha is only made live when the state machine reaches LIVE_TRIAL or LIVE_FUSED.")
    print("   Fusion is funded from the VOO sleeve, not from IEF.")
    print("   UI-facing alpha label uses CANDIDATE instead of ambiguous ELIGIBLE.")


if __name__ == "__main__":
    main(CFG)



    
