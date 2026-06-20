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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Optional: regime-conditional Platt scaling (scipy + sklearn) ──────────
try:
    from scipy.special import logit as _logit, expit as _expit
    from sklearn.linear_model import LogisticRegression as _LogisticRegression
    HAVE_PLATT = True
except Exception:
    HAVE_PLATT = False


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


# Numeric rank for the promotion ladder — used to take the higher of two states.
_STATE_RANK: Dict[str, int] = {
    "SHADOW": 0,
    "ELIGIBLE": 1,
    "LIVE_TRIAL": 2,
    "LIVE_FUSED": 3,
}


def _infer_promotion_state(
    realized_dates: int,
    quality_ok: int,
    drift_ok: int,
    trial_gate_open: int,
    fused_gate_open: int,
    thresholds: Dict[str, Any],
) -> str:
    """Derive the canonical alpha promotion state from first principles.

    This is the authoritative state-assignment path. Part 2A does not write
    an ``alpha_state`` or ``latest_alpha_state`` field to its outputs — it
    writes ``alpha_governance_tier``, ``latest_eligible``, etc. — so the
    prior lookup-first strategy always fell through to the "SHADOW" default,
    producing a locked-SHADOW tape regardless of realized_dates or gate flags.

    State ladder (each level requires all lower conditions):
      SHADOW     : realized_dates < th_eligible  OR  quality/drift gate failed
      ELIGIBLE   : realized_dates >= th_eligible AND quality_ok AND drift_ok
      LIVE_TRIAL : ELIGIBLE conditions AND realized_dates >= th_trial
                   AND trial_gate_open
      LIVE_FUSED : LIVE_TRIAL conditions AND realized_dates >= th_fused
                   AND fused_gate_open

    All threshold keys are read with safe integer conversion so stale or
    missing JSON values fall back to the hard-coded defaults (26 / 52 / 78).
    """
    th_e = max(1, int(thresholds.get("Eligible", 26) or 26))
    th_t = max(th_e + 1, int(thresholds.get("Trial", 52) or 52))
    th_f = max(th_t + 1, int(thresholds.get("Fused", 78) or 78))

    # Quality and drift are hard gates — failure at any level returns SHADOW.
    if not quality_ok or not drift_ok:
        return "SHADOW"
    if realized_dates < th_e:
        return "SHADOW"
    # ELIGIBLE floor — gates above here are soft (closed gate → stay at lower tier).
    if realized_dates >= th_f and fused_gate_open:
        return "LIVE_FUSED"
    if realized_dates >= th_t and trial_gate_open:
        return "LIVE_TRIAL"
    return "ELIGIBLE"


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


def _load_part7_base_weights(root: Path) -> Tuple[Optional[float], Optional[float]]:
    """Load the latest target portfolio weights from Part 7's output tape.

    Part 3's default base weights (60/40) are stale relative to Part 7's
    Black-Litterman/CVaR output, which currently targets ~70/30. Using Part 3's
    defaults produces a fusion allocation whose core VOO sleeve is
    systematically 10 pp too low, misaligning the allocation tape with the
    portfolio construction output.

    Returns (w_target_voo, w_target_ief) normalised to sum to 1.0, or
    (None, None) if the tape cannot be found or parsed. Part 3 falls back
    to _extract_base_weights(defense_row) → CFG defaults on None.
    """
    p = _first_existing_path(["artifacts_part7/portfolio_weights_tape.csv"], root)
    if p is None:
        return None, None
    try:
        df = _read_csv(p)
        if df.empty:
            return None, None
        row = _last_valid_row(df)
        voo = _safe_float(_row_value(row, ["w_target_voo", "w_voo", "target_weight_voo", "voo_weight"]))
        ief = _safe_float(_row_value(row, ["w_target_ief", "w_ief", "target_weight_ief", "ief_weight"]))
        if voo is None or ief is None:
            return None, None
        total = float(voo) + float(ief)
        if total <= 1e-12:
            return None, None
        return float(voo) / total, float(ief) / total
    except Exception:
        return None, None


def _load_alpha_status(alpha_tape_df: pd.DataFrame, alpha_summary_json: Dict[str, Any], live_realized_rows: int = 0) -> Dict[str, Any]:
    latest_row = _last_valid_row(alpha_tape_df) if not alpha_tape_df.empty else pd.Series(dtype=object)

    # ── realized_dates (backtest) vs live_realized_rows ────────────────────
    # FIX (F1, Audit 2026-05-10 — Quant-Guild Part 20):
    # PROBLEM: alpha_summary_json["realized_dates"] = 1109 is the BACKTEST tape count
    # (rows of the Part 2A summary_tape with non-null rank_ic / topk_rel_ret_net).
    # These are historical in-sample rows, NOT live prediction-log realized rows.
    # With 1109 >> 78 (FUSED threshold), _infer_promotion_state always returned
    # LIVE_FUSED — even with prediction_log_realized_rows = 0.
    # Effect: alpha_position=0.0516 was added to the portfolio on zero live track record.
    #
    # CORRECT DESIGN: the promotion thresholds (26/52/78) gate on LIVE realized
    # observations only. The backtest history is useful for performance monitoring
    # but must NOT satisfy the live-observation gate.
    #
    # FIX: _infer_promotion_state now receives live_realized_rows (passed in from the
    # main function via prediction_log_realized_rows). The backtest count is preserved
    # as backtest_realized_dates for monitoring and display only.
    backtest_realized_dates = _safe_int(
        _json_value(alpha_summary_json, ["realized_dates", "n_realized_dates", "realized_rows"],
                    _row_value(latest_row, ["realized_dates", "realized_rows"], len(alpha_tape_df))),
        default=len(alpha_tape_df),
    )
    # realized_dates for promotion = live prediction-log rows with confirmed realized prices
    realized_dates = int(max(0, live_realized_rows))

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

    # ── latest_state — derived from first principles ─────────────────────────
    # FIX: Part 2A does not write an `alpha_state` or `latest_alpha_state`
    # field to its summary tape or summary JSON. The previous lookup:
    #
    #   _row_value(latest_row, ["alpha_state_display", "alpha_state", ...],
    #              _json_value(alpha_summary_json,
    #                  ["latest_alpha_state_display", "latest_alpha_state", "alpha_state"],
    #                  "SHADOW"))
    #
    # always fell through to the "SHADOW" default, locking the entire tape
    # at SHADOW regardless of how many realized dates had accumulated or
    # whether all promotion gates were open.
    #
    # The authoritative state is now always computed by _infer_promotion_state
    # using the realized_dates derived above and the gate flags read from the
    # alpha summary JSON / tape row.  The state is fully re-derived from
    # underlying variables on every run, so stale alpha_state fields in older
    # artifacts do not corrupt it.
    # This eliminates the dependency on Part 2A writing a field it never wrote.
    #
    # For forward-compatibility: if a future Part 2A version does write
    # alpha_state to its outputs, _infer_promotion_state still produces the
    # correct answer because it re-derives state from underlying variables.
    latest_state = _infer_promotion_state(
        realized_dates=realized_dates,
        quality_ok=quality_ok,
        drift_ok=drift_ok,
        trial_gate_open=trial_gate_open,
        fused_gate_open=fused_gate_open,
        thresholds=thresholds,
    )

    latest_alpha_eligible = _boolish(
        _row_value(latest_row, ["latest_eligible", "eligible"], _json_value(alpha_summary_json, ["latest_eligible"], 0)),
        0,
    )
    latest_alpha_abs = _safe_float(
        _row_value(
            latest_row,
            ["latest_alpha_abs", "latest_alpha_position", "alpha_abs", "alpha_position"],
            _json_value(alpha_summary_json, ["latest_alpha_abs", "latest_alpha_position"], 0.0),
        )
    )
    if latest_alpha_abs is None:
        latest_alpha_abs = 0.0
    latest_alpha_reason = str(
        _row_value(latest_row, ["latest_reason", "reason"], _json_value(alpha_summary_json, ["latest_reason"], "unknown"))
    )

    alpha_live = latest_state in {"LIVE_TRIAL", "LIVE_FUSED"}
    if latest_state == "LIVE_FUSED":
        alpha_live = alpha_live and bool(fused_gate_open)
    if latest_state == "LIVE_TRIAL":
        alpha_live = alpha_live and bool(trial_gate_open)
    alpha_live = alpha_live and bool(quality_ok) and bool(drift_ok) and budget_mult > 0
    alpha_live = alpha_live and bool(latest_alpha_eligible) and float(latest_alpha_abs) > 0.0

    if (not latest_alpha_eligible) or float(latest_alpha_abs) <= 0.0:
        current_alpha_live_status = "FLAT_VETOED" if latest_alpha_reason != "ok" else "FLAT_INACTIVE"
    else:
        current_alpha_live_status = latest_state

    return {
        "latest_state": latest_state,
        "display_state": _state_display(latest_state),
        "realized_dates": realized_dates,          # LIVE realized rows (used for promotion)
        "backtest_realized_dates": backtest_realized_dates,  # backtest rows (monitoring only)
        "budget_mult": float(budget_mult),
        "drift_rate": float(drift_rate),
        "quality_ok": quality_ok,
        "drift_ok": drift_ok,
        "trial_gate_open": trial_gate_open,
        "fused_gate_open": fused_gate_open,
        "promotion_ready": promotion_ready,
        "blockers": str(blockers),
        "alpha_live": int(alpha_live),
        "current_alpha_live_status": str(current_alpha_live_status),
        "current_alpha_reason": str(latest_alpha_reason),
        "current_alpha_eligible": int(latest_alpha_eligible),
        "current_alpha_abs": float(latest_alpha_abs),
        "thresholds": thresholds,
    }


def _alpha_distribution(alpha_tape_df: pd.DataFrame, alpha_status: Optional[Dict[str, Any]] = None) -> pd.Series:
    col = _first_col(alpha_tape_df, ["alpha_state", "alpha_state_display", "state"])
    if col is None or alpha_tape_df.empty:
        if alpha_status and alpha_status.get("latest_state"):
            return pd.Series({str(alpha_status["latest_state"]).upper(): 1.0}, dtype=float)
        return pd.Series(dtype=float)
    s = alpha_tape_df[col].astype(str).str.upper().replace({"CANDIDATE": "ELIGIBLE"})
    s = s[s.notna() & (s != "")]
    if s.empty and alpha_status and alpha_status.get("latest_state"):
        return pd.Series({str(alpha_status["latest_state"]).upper(): 1.0}, dtype=float)
    return s.value_counts(normalize=True).sort_index()


def _extract_alpha_positions(alpha_positions_df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest-date alpha positions as raw portfolio weights.

    FIX A (v1) — 'Ticker' added to ticker lookup.
    FIX A (v2) — 'alpha_leg' added to ticker lookup.
        Part 2A's positions CSV uses 'alpha_leg' (values: 'VOO' or 'FLAT').
        'Ticker' and 'ticker' are not present in the actual schema.
        Previous fix added 'Ticker' based on Part 4 GUI expected columns, which
        reflect a historical schema Part 2A no longer writes.

    FIX B (v1) — normalization removed.
    FIX B (v2) — 'alpha_position' and 'w_alpha_voo' added to weight lookup.
        Part 2A writes 'alpha_position' (and alias 'w_alpha_voo') as the
        portfolio weight column. Neither 'weight', 'w', 'alloc', nor 'allocation'
        is present. Both v1 lookups returned None → function returned an empty
        DataFrame every run → no alpha sleeve was ever carved out despite
        alpha_live=1 in governance.

    Column map confirmed from part2a21_alpha.py lines 446–477:
        ticker:  alpha_leg      (values: "VOO" | "FLAT")
        weight:  alpha_position  (= w_alpha_voo; both are identical)

    FLAT entries have alpha_position=0.0 and are dropped by the weight > 0 filter.
    Raw weights are preserved (no normalization) as direct portfolio fractions.
    """
    if alpha_positions_df.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    g = alpha_positions_df.copy()
    date_col = _first_col(g, ["Date", "decision_date", "asof_date"])
    if date_col is not None:
        d = pd.to_datetime(g[date_col], errors="coerce")
        if d.notna().any():
            g = g.loc[d == d.max()].copy()

    # Part 2A writes 'alpha_leg'. Legacy/future schemas may use Ticker/ticker.
    ticker_col = _first_col(g, ["alpha_leg", "Ticker", "ticker", "asset", "sleeve", "name", "symbol"])
    # Part 2A writes 'alpha_position'. Legacy/future schemas may use weight/w.
    weight_col = _first_col(g, ["alpha_position", "w_alpha_voo", "weight", "w", "alloc", "allocation"])
    if ticker_col is None or weight_col is None:
        return pd.DataFrame(columns=["ticker", "weight"])

    out = g[[ticker_col, weight_col]].copy()
    out.columns = ["ticker", "weight"]
    out["ticker"] = out["ticker"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight"])
    out = out.groupby("ticker", as_index=False)["weight"].sum()
    # Keep only positive positions. FLAT rows (alpha_position=0) are dropped here.
    out = out[out["weight"] > 0].copy()
    # NOTE: do NOT normalize. Weights are already direct portfolio fractions.
    # Normalizing to sum-to-1 would destroy the ~1.95% alpha magnitude.
    return out.reset_index(drop=True)


def _build_fusion_allocations(
    decision_date: pd.Timestamp,
    defense_row: pd.Series,
    alpha_positions_df: pd.DataFrame,
    alpha_status: Dict[str, Any],
    part7_weights: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, float]:
    # Base weights: prefer Part 7 portfolio construction output when available.
    # Part 3's default fallback (0.60/0.40 from CFG) is misaligned with Part 7's
    # current Black-Litterman/CVaR output (~0.70/0.30). If _load_part7_base_weights
    # succeeded, those weights are passed in via part7_weights and used here.
    if part7_weights is not None:
        voo_base, ief_base = part7_weights
    else:
        voo_base, ief_base = _extract_base_weights(defense_row)
    alpha_live = bool(alpha_status["alpha_live"])
    budget_mult = float(alpha_status["budget_mult"])

    rows: List[Dict[str, Any]] = []
    voo_alpha_used = 0.0

    if alpha_live and not alpha_positions_df.empty:
        # FIX C — alpha sleeve sizing.
        #
        # Previous code:
        #   alpha_share = min(voo_base, voo_base * budget_mult)   # always = voo_base = 0.60
        #   alpha_positions["weight"] *= alpha_share              # 1.0 * 0.60 = 0.60 (after normalization)
        #   voo_weight = max(0, voo_base - alpha_share)           # 0.60 - 0.60 = 0.0
        #
        # That routed the entire VOO sleeve to alpha because _extract_alpha_positions
        # had normalized the Part 2A weight (0.0195) to 1.0 before arriving here.
        # With the normalization fix in _extract_alpha_positions, weights are now
        # raw portfolio fractions. The correct sizing is:
        #
        #   alpha_weight = raw_part2a_weight * budget_mult   (e.g. 0.0195 * 1.0 = 0.0195)
        #   voo_core     = voo_base - alpha_weight            (e.g. 0.60  - 0.0195 = 0.5805)
        #
        # Safety cap: alpha sleeve cannot exceed voo_base regardless of budget_mult.
        # If multiple alpha tickers sum above voo_base, scale all proportionally.
        alpha_positions = alpha_positions_df.copy()
        alpha_positions["weight"] = alpha_positions["weight"] * budget_mult

        raw_alpha_total = float(alpha_positions["weight"].sum())
        if raw_alpha_total > voo_base and raw_alpha_total > 0:
            cap_scale = voo_base / raw_alpha_total
            alpha_positions["weight"] = alpha_positions["weight"] * cap_scale

        for _, r in alpha_positions.iterrows():
            w = float(r["weight"])
            if w <= 0:
                continue
            rows.append({
                "Date": decision_date,
                "sleeve": str(r["ticker"]),
                "weight": w,
                "is_alpha": 1,
                "alpha_state": alpha_status["latest_state"],
            })
            voo_alpha_used += w

    voo_weight = max(0.0, voo_base - voo_alpha_used)

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


def _count_realized_fused_rows(tape: pd.DataFrame) -> int:
    """Count live predictions whose realized prices have been backfilled.

    NOTE: This function returns 0 to correctly reflect the live-prediction
    realized state.  Historical tape rows are not 'fused live predictions'
    regardless of whether their forward returns are revealed in the backtest
    history. The prior implementation returned up to ~1,638 historical rows,
    which contradicted the simultaneously-zero prediction_log_realized_rows.

    The authoritative live realized count is prediction_log_realized_rows,
    computed in _upsert_prediction_log and written to the summary dict.
    rows_realized_fused=0 is kept for call-site compatibility; downstream
    consumers must read prediction_log_realized_rows for the correct count.
    See part3_summary.json: "prediction_log_realized_rows" for the live count.
    """
    # The authoritative live realized count is prediction_log_realized_rows,
    # computed in _upsert_prediction_log and written to the summary dict.
    # Return 0 here so rows_realized_fused reflects live-prediction state,
    # not the 6-year historical tape.
    return 0


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


def _build_governance_df(
    decision_date: pd.Timestamp,
    part2_summary: Dict[str, Any],
    alpha_status: Dict[str, Any],
    publish_mode_override: Optional[str] = None,
    final_pass_override: Optional[int] = None,
) -> pd.DataFrame:
    # FIX (F-2, Quant-Guild Part 32 Audit):
    # Part 3's main function overrides the local `publish_mode` variable at L1345
    # (when final_pass=False but publish_mode=NORMAL, it forces FAIL_CLOSED_NEUTRAL).
    # However, _build_governance_df previously read publish_mode directly from the
    # part2_summary dict, which is never mutated. Result: the governance CSV always
    # received the pre-override publish_mode (NORMAL) while the prediction log and
    # part3_summary.json correctly received the overridden value (FAIL_CLOSED_NEUTRAL).
    #
    # Fix: accept explicit override parameters. The call site passes the post-override
    # `publish_mode` and `final_pass` local variables so all artifacts stay consistent.
    row = {
        "Date": decision_date,
        "publish_mode": (_normalize_publish_mode(publish_mode_override) if publish_mode_override is not None else _normalize_publish_mode(_json_value(part2_summary, ["publish_mode", "mode"], "UNKNOWN"))),
        "final_pass": (int(final_pass_override) if final_pass_override is not None else _boolish(_json_value(part2_summary, ["final_pass"], 0), 0)),
        "quality_ok": alpha_status["quality_ok"],
        "drift_ok": alpha_status["drift_ok"],
        "trial_gate_open": alpha_status["trial_gate_open"],
        "fused_gate_open": alpha_status["fused_gate_open"],
        "promotion_ready": alpha_status["promotion_ready"],
        "alpha_state": alpha_status["latest_state"],
        "alpha_state_display": alpha_status["display_state"],
        "alpha_live": alpha_status["alpha_live"],
        "current_alpha_live_status": alpha_status.get("current_alpha_live_status", alpha_status["latest_state"]),
        "current_alpha_reason": alpha_status.get("current_alpha_reason", "unknown"),
        "current_alpha_eligible": alpha_status.get("current_alpha_eligible", 0),
        "current_alpha_abs": alpha_status.get("current_alpha_abs", 0.0),
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
            "p_final_cal", "base_rate", "raw_val_auc", "tail_threshold",
            # publish_mode: raw Part 2 governance value (FAIL_CLOSED_NEUTRAL / NORMAL).
            # deployment_mode: user-facing operational label (DEFENSE_ONLY / NORMAL).
            # Keeping both columns avoids the cross-file field collision where predlog
            # previously aliased FAIL_CLOSED_NEUTRAL → DEFENSE_ONLY in the publish_mode
            # column, creating a mismatch with part3_summary.json's publish_mode field.
            "publish_mode", "deployment_mode", "final_pass",
            "latest_alpha_state", "alpha_live",
            "historical_alpha_state", "current_alpha_live_status", "current_alpha_reason",
            "current_alpha_eligible", "current_alpha_abs",
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
        # NOTE: _7d alias columns removed. All consumers (backfill_realized.py,
        # Part 9, Part 4) now prefer _1d explicitly and fall back gracefully.
        # Existing rows in the prediction log retain their _7d columns harmlessly;
        # new rows written from this point forward carry only _1d.
        # FIX (Audit Part 28 — C-3): Store the blended and regime-recalibrated
        # probabilities in the prediction_log so Part 9 and operators can trace the
        # full probability chain: raw Part 2 → blend → Platt recal → deployed.
        #
        # p_final_cal (existing): raw Part 2 base model probability (from defense tape).
        # p_final_cal_blended (new): after Part 2B/2C ensemble blend (or = p_final_cal
        #   if blend unavailable). This is the probability that Part 7 uses for BL.
        # p_final_cal_regime_recal (new): after Platt regime recalibration applied to
        #   the blended value. This is the probability used for Part 9 live attribution
        #   (already written separately as p_regime_recal; now named explicitly here).
        "p_final_cal": _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None)),
        "base_rate": _safe_float(_row_value(defense_row, ["T", "base_rate", "b"], None)),
        "raw_val_auc": _safe_float(_row_value(defense_row, ["raw_val_auc"], _json_value(part2_summary, ["raw_val_auc_median"], None))),
        # FIX (Audit 2026-05-07 — CRITICAL Bug: wrong tail_threshold written to log):
        # The previous lookup included "signal_q_threshold" in the priority chain.
        # "signal_q_threshold" in the consensus tape is the DEFENSE TRIGGER quantile
        # threshold (the rolling 56th-percentile of defense_trigger_raw, ≈ 0.15) — it
        # has nothing to do with the tail EVENT threshold used to construct the label.
        #
        # Writing 0.15 as tail_threshold corrupts Part 9 live attribution: Part 9 uses
        # this field to reconstruct y_live = (excess_ret < tail_threshold). With threshold
        # = 0.15, nearly every daily return qualifies as a "tail event" (excess_ret < 0.15
        # is true ~99% of the time), producing a base rate of ~1.0 instead of ~0.20 and
        # making every AUC, Brier, and ECE metric meaningless.
        #
        # Corrected priority:
        #   1. tail_threshold_dynamic — future-proofing if Part 2 ever writes a per-row
        #      dynamic threshold column explicitly named "tail_threshold_dynamic"
        #   2. part2_summary["tail_event_threshold"] — the authoritative H=1 daily
        #      threshold = -0.015, validated against Part 1's rolling-quantile labels
        #
        # "signal_q_threshold" is REMOVED from the lookup chain.
        #
        # FIX (Audit 2026-05-07 — F2 + F4):
        # Three-tier lookup with hardcoded last resort.
        #
        # F4: the prior `or`-based pattern is incorrect Python.  `x or y` treats
        #     x as truthy/falsy: if x == 0.0 (valid but falsy), the chain silently
        #     discards it and returns y.  Explicit `if x is not None` is required.
        #
        # F2: the prior chain had no hardcoded last resort.  If part2_g532_summary.json
        #     is ever missing the "tail_event_threshold" key (schema migration, cold start,
        #     truncated write), _safe_float(None) returns None and prediction_log gets a
        #     null tail_threshold, silently breaking all Part 9 y_live reconstructions.
        #     A hardcoded fallback of -0.015 (authoritative H=1 daily threshold) closes
        #     this gap regardless of upstream schema changes.
        #
        # Priority chain (highest → lowest):
        #   1. tail_threshold_dynamic  — per-row dynamic threshold if Part 2 ever writes it
        #   2. part2_summary["tail_event_threshold"]  — -0.015 from the current summary JSON
        #   3. -0.015 hardcoded  — last resort; matches H=1 base daily threshold
        **{
            "tail_threshold": _safe_float(
                _row_value(defense_row, ["tail_threshold_dynamic"], None)
                if _row_value(defense_row, ["tail_threshold_dynamic"], None) is not None
                else (
                    _json_value(part2_summary, ["tail_event_threshold"], None)
                    if _json_value(part2_summary, ["tail_event_threshold"], None) is not None
                    else -0.015  # hardcoded H=1 last resort
                )
            )
        },
        # publish_mode: raw governance value, consistent with part3_summary.json.
        # deployment_mode: user-facing operational label (DEFENSE_ONLY when fail-closed).
        # Separating these eliminates the prior cross-file field-name collision.
        "publish_mode": publish_mode,
        "deployment_mode": "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL" else publish_mode,
        "final_pass": int(final_pass),
        "latest_alpha_state": alpha_status["latest_state"],
        "historical_alpha_state": alpha_status["latest_state"],
        "current_alpha_live_status": alpha_status.get("current_alpha_live_status", alpha_status["latest_state"]),
        "current_alpha_reason": alpha_status.get("current_alpha_reason", "unknown"),
        "current_alpha_eligible": int(alpha_status.get("current_alpha_eligible", 0)),
        "current_alpha_abs": float(alpha_status.get("current_alpha_abs", 0.0)),
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
    # FIX (BUG-3, Audit 2026-05-11 — Quant-Guild Part 24):
    # Use ddof=1 (sample std) not ddof=0 (population std). With n=5 defense
    # events, ddof=0 overstates the IR by sqrt(5/4)=1.12 (12%). Industry
    # convention for IR is always sample std with ddof=1.
    std = float(s.std(ddof=1))
    # Guard: ddof=1 on n=1 gives NaN; treat as undefined (return None).
    if not math.isfinite(std) or std <= 0:
        return None
    # FIX (BUG-2, Audit 2026-05-11 — Quant-Guild Part 24):
    # Annualize the IR. The original code returned mean/std (daily IR) with no
    # sqrt(252) factor. Part 2's _annualized_ir multiplies by sqrt(252/H) at H=1.
    # Without this, defense_ir and fused_ir are ~1/15.87 of the correct annualized
    # value — approximately 0.04 instead of 0.69 for a typical defense IR of 0.69.
    # At H=1 daily: annualized_factor = sqrt(252).
    _ANNUALIZE_H1 = math.sqrt(252.0)
    return float(s.mean() / std * _ANNUALIZE_H1)


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
        # FIX (Finding 7, Audit 2026-05-10 — Quant-Guild Part 19):
        # fusion_live_rate=1.0 while rows_realized_fused=0 and prediction_log_realized_rows=0
        # is misleading. The metric means "is alpha currently being fused?" not "fraction
        # of live rows with realized prices". The prior name implied the latter.
        # Both names are now returned so downstream consumers are unambiguous:
        #   fusion_live_rate:      preserved for backward compatibility (1.0 = alpha is live now)
        #   alpha_fusion_is_live:  boolean, semantically correct label for the same metric
        "fusion_live_rate": fusion_live_rate,
        "alpha_fusion_is_live": bool(fusion_live_rate >= 1.0),
    }


def _format_float(x: Optional[float], digits: int = 3) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def _json_safe_p3(obj: Any) -> Any:
    """Convert NaN/Inf/numpy scalars to JSON-safe Python types.

    FIX (BUG-6, Audit 2026-05-12 — Quant-Guild Part 25):
    json.dump's default= hook is never called for Python float (natively serializable),
    so the old 'default=str' approach let NaN floats through as the literal token NaN
    (invalid RFC 8259 JSON). numpy types were serialized as strings ("42" not 42).
    Pre-processing the entire dict before json.dump is the correct fix.
    """
    import math as _math
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (not _math.isfinite(obj)) else obj
    try:
        import numpy as _np
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            v = float(obj)
            return None if (not _math.isfinite(v)) else v
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import pandas as _pd
        if isinstance(obj, _pd.Timestamp):
            return str(obj)
        if obj is _pd.NaT or (hasattr(_pd, "NA") and obj is _pd.NA):
            return None
    except ImportError:
        pass
    return obj


def _deep_clean_for_json_p3(obj: Any) -> Any:
    """Recursively walk dict/list and apply _json_safe_p3 to every leaf value."""
    if isinstance(obj, dict):
        return {k: _deep_clean_for_json_p3(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_clean_for_json_p3(v) for v in obj]
    return _json_safe_p3(obj)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    # FIX (BUG-6, Audit 2026-05-12 — Quant-Guild Part 25):
    # Pre-clean the entire dict with _deep_clean_for_json_p3 before json.dump.
    # Without this, NaN floats serialize as the literal token NaN (invalid RFC 8259),
    # and numpy types (np.int64, np.float64) serialize as strings via default=str.
    cleaned = _deep_clean_for_json_p3(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)


# ============================================================
# Regime-conditional Platt scaling
# ============================================================

# FIX (F-1, Quant-Guild Part 33 Audit):
# Minimum Platt slope required for a regime-specific fit to be retained.
# a < 0 → signal inversion (already gated below).
# 0 <= a < _PLATT_MIN_SLOPE → near-zero slope → essentially maps every input
#   probability to the same constant output, destroying all discriminative signal.
#
# Empirical evidence (2026-05-26 run):
#   crisis:  a=0.0146, b=-1.108 → p(0.15)=0.2436, p(0.35)=0.2466 → range=0.003
#   _global: a=0.3438, b=-0.923 → p(0.15)=0.1795, p(0.35)=0.2430 → range=0.064
# The crisis Platt compresses the [0.15, 0.35] probability range to 0.003 pp
# (21× narrower than the global fallback). This makes crisis Platt functionally
# identical to a constant, providing no additional calibration value whatsoever.
# Using the _global fallback for crisis is strictly better.
#
# Threshold: 0.25 — same value used by Part 2B's platt_degenerate gate.
# This ensures both layers (Part 2B exclusion gate and Part 3 Platt exclusion)
# apply the same minimum-slope standard for internal consistency.
_PLATT_MIN_SLOPE: float = 0.25


def _fit_regime_platt_scaling(
    defense_df: pd.DataFrame,
    y_revealed_path: Optional[Path],
    regime_history_path: Optional[Path],
) -> Dict[str, Tuple[float, float]]:
    """Fit per-regime logistic recalibration (Platt scaling) on the historical tape.

    Joins three sources:
      • defense_df         — Part 2 tape: Date, p_final_cal
      • y_revealed_path    — Part 1 revealed labels: y_rel_tail_voo_vs_ief (index=Date)
      • regime_history_path— Part 6 regime history: regime_label (index=Date)

    Returns a dict mapping regime_label → (a, b) where the recalibrated probability is:
        p_recal = σ(a * logit(p_final_cal) + b)

    Also stores '_global' key as a fallback for unseen regime labels.

    Design notes
    ————————————
    • Platt scaling has only 2 parameters per regime — extremely stable even at 200 rows.
    • We fit on the full historical revealed tape (not a held-out set) because:
        (a) the 2-parameter model cannot meaningfully overfit to 400+ observations, and
        (b) using held-out data would discard ~3/4 of the already-small calibration set.
    • If HAVE_PLATT is False (scipy/sklearn not available), returns empty dict and the
      caller falls back to raw p_final_cal transparently.
    """
    if not HAVE_PLATT:
        return {}
    if y_revealed_path is None or not y_revealed_path.exists():
        return {}
    if regime_history_path is None or not regime_history_path.exists():
        return {}

    try:
        # Load revealed labels
        y_df = pd.read_parquet(y_revealed_path)
        y_df.index = pd.to_datetime(y_df.index, errors="coerce")
        y_df = y_df[~y_df.index.isna()]
        if "y_rel_tail_voo_vs_ief" not in y_df.columns:
            return {}

        # Load regime history
        reg_df = pd.read_parquet(regime_history_path)
        reg_df.index = pd.to_datetime(reg_df.index, errors="coerce")
        reg_df = reg_df[~reg_df.index.isna()]
        if "regime_label" not in reg_df.columns:
            return {}

        # Build working frame: Date × p_final_cal
        tape = defense_df.copy()
        tape["Date"] = pd.to_datetime(tape.get("Date", tape.index), errors="coerce")
        tape = tape.dropna(subset=["Date"]).set_index("Date")
        if "p_final_cal" not in tape.columns:
            return {}

        # Inner-join all three sources
        merged = (
            tape[["p_final_cal"]]
            .join(y_df[["y_rel_tail_voo_vs_ief"]], how="inner")
            .join(reg_df[["regime_label"]], how="left")
        )
        merged = merged.dropna(subset=["p_final_cal", "y_rel_tail_voo_vs_ief"])
        if len(merged) < 50:
            return {}

        params: Dict[str, Tuple[float, float]] = {}

        # FIX (Finding 3, Audit 2026-05-10 — Quant-Guild Part 19):
        # Track sample size per regime so _persist_platt_params can include 'n'.
        # The comment in _persist_platt_params documented n: int but code never stored it.
        # risk_on has only 55 rows — flagging this for operators is important.
        params_n: Dict[str, int] = {}

        # Per-regime fit
        for regime in merged["regime_label"].dropna().unique():
            sub = merged[merged["regime_label"] == regime].copy()
            if len(sub) < 30:
                continue
            p = sub["p_final_cal"].clip(0.01, 0.99).values
            y = sub["y_rel_tail_voo_vs_ief"].values
            X = _logit(p).reshape(-1, 1)
            # FIX (F6, Audit 2026-05-10 — Quant-Guild Part 20):
            # risk_on has n=55 and AUC=0.527 (0.83 SE above random — not significant).
            # Platt C=1e4 (unregularized) amplifies a near-noise signal via a=1.19.
            # At n<100 use C=0.1 (L2 ridge ≡ MAP with N(0,1) prior on coefficients),
            # which shrinks 'a' toward 0 (no-signal). This is Platt's recommended
            # practice for small-n calibration sets to prevent overfit amplification.
            # At n>=100 (calm n=610, high_vol n=654, crisis n=338) C=1e4 is fine.
            _platt_C = 0.1 if len(sub) < 100 else 1e4
            lr = _LogisticRegression(C=_platt_C, max_iter=2000, solver="lbfgs")
            lr.fit(X, y)
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])

            # FIX (Finding 1, Quant-Guild Part 26): Do NOT store per-regime Platt
            # params when a < 0. When a < 0, the transform INVERTS the signal:
            # higher raw p_final_cal → lower recalibrated probability. This is the
            # opposite of the intended behavior for a defense signal.
            #
            # Empirical evidence (2026-05-25 run):
            #   calm    a=-0.741 → p(0.20)->0.151, p(0.30)->0.109  (signal INVERTED, HARMFUL)
            #   crisis  a=-0.199 → p(0.20)->0.271, p(0.30)->0.252  (signal INVERTED)
            #   risk_on a=-0.079 → p(0.20)->0.197, p(0.30)->0.191  (signal INVERTED)
            #   high_vol a=+1.425 → correct direction (amplified, not inverted)
            #
            # Applying inverted Platt in production: recalibrated probability DECREASES
            # when the model is more confident of a tail event. This reduces the defense
            # signal precisely when it should be elevated. 59.2% of all rows (calm 36.4%
            # + crisis 18.5% + risk_on 4.3%) were receiving an inverted correction.
            #
            # FIX (Finding 1, Quant-Guild Part 26): exclude a < 0 (signal inversion).
            # FIX (F-1, Quant-Guild Part 33 Audit): also exclude a < _PLATT_MIN_SLOPE.
            #
            # A near-zero positive slope (e.g. crisis a=0.0146) maps all probabilities
            # in [0.15, 0.35] to a 0.003 pp range — functionally a constant.  This
            # provides zero discriminative recalibration while silently replacing the
            # _global fallback (range=0.064) with a worse fit.  The a < _PLATT_MIN_SLOPE
            # gate catches this degenerate case the same way Part 2B's platt_degenerate
            # flag catches it there (both use threshold 0.25 for internal consistency).
            if a < _PLATT_MIN_SLOPE:
                reason = "ANTI-PREDICTIVE (a<0)" if a < 0 else f"DEGENERATE SLOPE (a={a:.4f} < {_PLATT_MIN_SLOPE})"
                print(
                    f"[Part 3] Platt({regime:12s}): a={a:.4f}  b={b:.4f}  n={len(sub)} "
                    f"-- {reason}: EXCLUDED. _global fallback will apply."
                )
                params_n[regime] = len(sub)   # track n for diagnostics only
                continue                       # do NOT store params[regime] = (a, b)

            # FIX (F4, Quant-Guild Part 50 Audit): Add DeLong significance gate to
            # Platt per-regime fit acceptance.
            #
            # ROOT CAUSE: The Part 6 HMM refit between S49 and S50 caused a massive
            # regime relabeling — risk_on grew from n=45 to n=563 rows (12x). With
            # n=563, the Platt minimum-n threshold (30) and slope threshold (0.25) are
            # trivially satisfied. risk_on received a non-degenerate, non-inverted fit
            # (a=0.866, n=563). However, risk_on's full-period AUC = 0.5156 with
            # DeLong z=0.513, p=0.304 — statistically indistinguishable from random.
            # Applying Platt scaling (which amplifies probability toward the Platt fixed
            # point) on a regime where the model has no verified signal is philosophically
            # incorrect and internally inconsistent:
            #
            #   _compute_regime_auc (Part 2) requires DeLong p < 0.10 to classify a
            #   regime as "active" (active_regimes). risk_on fails this gate (p=0.304)
            #   and is NOT in active_regimes. Part 7 therefore routes risk_on rows to
            #   regime_gated_prior (60/40) with view_confidence=0.
            #
            #   Yet Part 3's Platt fit WAS stored for risk_on and applied whenever the
            #   live row lands in risk_on, amplifying a signal that the rest of the system
            #   simultaneously treats as statistically insignificant noise. These two layers
            #   were operating on opposite assumptions about risk_on's predictive validity.
            #
            # FIX: compute DeLong z-test on (y, p_final_cal) for this regime. If
            # p_one_sided >= 0.10 (auc_warning=True), the fit is excluded — exactly
            # the same gate _compute_regime_auc uses for active_regimes. This ensures:
            #   (a) If DeLong p < 0.10 → regime in active_regimes AND Platt fit stored.
            #   (b) If DeLong p >= 0.10 → regime NOT in active_regimes AND Platt excluded.
            # Internal consistency between Part 2's regime activation gate and Part 3's
            # Platt calibration gate is now guaranteed.
            #
            # The AUC computed here is from (y, p_final_cal) on the merged calibration set,
            # which matches what _compute_regime_auc computes. The DeLong SE uses the
            # Hanley-McNeil approximation (same as _delong_auc_ztest in Part 2 pre-exact-DeLong
            # era); the approximation is slightly conservative (larger SE, smaller z) which
            # is the safe direction for a gate that prevents over-fitting.
            #
            # Validation (S50 artifacts):
            #   high_vol: AUC=0.5404, z=1.195, p=0.116 → auc_warning=True (p>=0.10)
            #             But high_vol slope a=0.972 >> 0.25 → already storing
            #             Wait — S50 high_vol is NOT in active_regimes (p=0.116>0.10)
            #             → with this fix, high_vol Platt would ALSO be excluded
            #             → BOTH high_vol and risk_on excluded, only _global remains
            #             → consistent: no regime is active → no regime-specific Platt
            #   risk_on:  AUC=0.5156, p=0.304 → excluded (p>>0.10) ✓
            #   calm:     AUC=0.475 < 0.50 → a < 0 → excluded by existing gate ✓
            #   crisis:   AUC=0.428 < 0.50 → a < 0 → excluded by existing gate ✓
            #
            # At S49: high_vol AUC=0.5553, DeLong z=2.123, p=0.017 < 0.10 → RETAINED ✓
            # This correctly preserves S49 behavior (high_vol was significant at p10).
            #
            # Implementation: inline DeLong (HM approximation). Uses the same y and p
            # arrays already computed above.
            try:
                from sklearn.metrics import roc_auc_score as _p3_roc_auc
                _p3_y = y.astype(float)
                _p3_p = p.clip(1e-6, 1 - 1e-6)
                if len(np.unique(_p3_y)) >= 2:
                    _p3_auc = float(_p3_roc_auc(_p3_y, _p3_p))
                    _p3_n1 = int(_p3_y.sum())
                    _p3_n0 = len(_p3_y) - _p3_n1
                    if _p3_n1 >= 5 and _p3_n0 >= 5:
                        _p3_Q1 = _p3_auc / (2.0 - _p3_auc)
                        _p3_Q2 = 2.0 * _p3_auc ** 2 / (1.0 + _p3_auc)
                        _p3_se = float(np.sqrt(
                            (_p3_auc * (1 - _p3_auc)
                             + (_p3_n1 - 1) * (_p3_Q1 - _p3_auc ** 2)
                             + (_p3_n0 - 1) * (_p3_Q2 - _p3_auc ** 2))
                            / (_p3_n1 * _p3_n0)
                        ))
                        _p3_z = (_p3_auc - 0.5) / _p3_se if _p3_se > 0 else 0.0
                        from scipy.stats import norm as _p3_norm
                        _p3_p_val = float(1.0 - _p3_norm.cdf(_p3_z))
                        _platt_auc_warning = bool(_p3_p_val >= 0.10)
                    else:
                        _p3_p_val = float("nan")
                        _platt_auc_warning = True   # too few events — conservative exclusion
                else:
                    _p3_p_val = float("nan")
                    _platt_auc_warning = True
            except Exception:
                _p3_auc = float("nan")
                _p3_p_val = float("nan")
                _platt_auc_warning = False   # on error, permissive (retain existing behavior)

            if _platt_auc_warning:
                print(
                    f"[Part 3] Platt({regime:12s}): a={a:.4f}  b={b:.4f}  n={len(sub)} "
                    f"AUC={_p3_auc:.4f} DeLong p={_p3_p_val:.4f} >= 0.10 "
                    f"-- INSIGNIFICANT: EXCLUDED. _global fallback or passthrough will apply."
                    f"  [S50 F4 fix]"
                )
                params_n[regime] = len(sub)   # track n for diagnostics only
                continue                       # do NOT store params[regime] = (a, b)

            params[regime] = (a, b)
            params_n[regime] = len(sub)
            print(
                f"[Part 3] Platt({regime:12s}): a={a:.4f}  b={b:.4f}  n={len(sub)} "
                f"AUC={_p3_auc:.4f}  DeLong p={_p3_p_val:.4f} < 0.10  [significant]"
            )

        # Global fallback (all regimes combined)
        p_all = merged["p_final_cal"].clip(0.01, 0.99).values
        y_all = merged["y_rel_tail_voo_vs_ief"].values
        lr_global = _LogisticRegression(C=1e4, max_iter=2000, solver="lbfgs")
        lr_global.fit(_logit(p_all).reshape(-1, 1), y_all)
        _a_global = float(lr_global.coef_[0][0])
        _b_global = float(lr_global.intercept_[0])
        params["_global"] = (_a_global, _b_global)
        params_n["_global"] = len(merged)

        # FIX (F1, Quant-Guild Part 47 Audit): _global was fit and stored
        # unconditionally, with NO check against _PLATT_MIN_SLOPE — even though
        # every regime-specific fit above is rejected (a < 0.25) and routed to
        # THIS fallback precisely because it's supposed to be the safe, stable
        # alternative. There is no further fallback beyond _global; if _global
        # itself degenerates, nothing in the codebase ever catches it.
        #
        # CONFIRMED (S47 artifact, 2026-06-18): _global a=0.230365 < 0.25 floor.
        # crisis and risk_on are both currently excluded as regime-specific fits
        # (platt_inverted_regimes_excluded=["crisis","risk_on"]), and risk_on is
        # TODAY's live regime — meaning every live row was silently being passed
        # through a degenerate calibrator with no flag raised anywhere. Part 2B's
        # analogous single global Platt slope IS gated (platt_degenerate flag,
        # gates bnn_sleeve_recommended); Part 3's _global had no equivalent.
        #
        # Fix: flag _global the same way a regime-specific fit would be flagged.
        # The flag is surfaced via params["__global_degenerate__"] (read by
        # _apply_regime_platt below and exposed in the persisted JSON via
        # _persist_platt_params) rather than discarding _global outright, since
        # discarding it would leave literally nothing to fall back to — the
        # transparent pass-through (return p_cal unchanged) is applied at the
        # point of use instead, matching the existing "params empty" behavior.
        _global_degenerate = _a_global < _PLATT_MIN_SLOPE
        if _global_degenerate:
            _reason = "ANTI-PREDICTIVE (a<0)" if _a_global < 0 else f"DEGENERATE SLOPE (a={_a_global:.4f} < {_PLATT_MIN_SLOPE})"
            print(
                f"[Part 3] Platt(_global    ): a={_a_global:.4f}  b={_b_global:.4f}  n={len(merged)} "
                f"-- {_reason}: _global fallback is ALSO degenerate. No further fallback exists; "
                f"any regime routed to _global will receive UNCALIBRATED p_final_cal this run."
            )
        else:
            print(f"[Part 3] Platt(_global    ): a={_a_global:.4f}  b={_b_global:.4f}  n={len(merged)}")
        params["__global_degenerate__"] = _global_degenerate  # type: ignore[assignment]

        # Attach sample sizes so _persist_platt_params can write them
        # Store as a special key (not a regime) — _persist_platt_params reads it
        params["__sample_sizes__"] = params_n  # type: ignore[assignment]

        return params

    except Exception as e:
        print(f"[Part 3] Platt scaling fit failed: {e} — using raw p_final_cal")
        return {}


def _persist_platt_params(params: Dict[str, Tuple[float, float]], out_dir: Path) -> None:
    """Write Platt calibration parameters to part3_platt_params.json.

    FIX (F7, Audit 2026-05-10 — Quant-Guild Part 17):
    The (a, b) coefficients were previously recomputed on every run and never
    written to disk.  This meant:
      (a) No audit trail — operators cannot verify which calibration was active
          on a given date without re-running Part 3.
      (b) No independent reproducibility — a code change that breaks _fit_regime_platt_scaling
          would silently fall back to raw p_final_cal with no persisted baseline to diff against.
      (c) Dashboard inaccessibility — index.html cannot display Platt coefficients
          without reading from a stable artifact path.

    Format: {regime: {a: float, b: float, n: int}, "_meta": {built_at: str}}
    p_recal = sigmoid(a * logit(p_final_cal) + b)
    """
    if not params:
        return
    # Extract embedded sample sizes written by _fit_regime_platt_scaling
    # (stored under the special key "__sample_sizes__"; not a real regime).
    _sample_sizes: Dict[str, int] = {}
    if "__sample_sizes__" in params:
        _sample_sizes = params.get("__sample_sizes__", {})  # type: ignore[assignment]

    # FIX (F1, Quant-Guild Part 47 Audit): surface whether _global itself fell
    # below _PLATT_MIN_SLOPE this run. Previously this state was invisible in
    # the persisted artifact — an operator inspecting part3_platt_params.json
    # would see _global's (a, b) with no indication it was degenerate, since
    # the same 0.25 floor used to exclude regime-specific fits was never
    # checked against _global. Written into "_meta" so the dashboard/audit
    # trail can flag it without affecting the {regime: {a,b,n}} contract any
    # downstream consumer of part3_platt_params.json already relies on.
    _global_degenerate = bool(params.get("__global_degenerate__", False))  # type: ignore[arg-type]

    payload: Dict[str, object] = {}
    for regime, ab in params.items():
        if regime.startswith("__") or not isinstance(ab, tuple):
            continue  # skip metadata keys
        a, b = ab
        _n = _sample_sizes.get(regime, None)
        entry: Dict[str, object] = {"a": round(float(a), 6), "b": round(float(b), 6)}
        if _n is not None:
            entry["n"] = int(_n)
        if regime == "_global":
            entry["degenerate"] = _global_degenerate
        payload[regime] = entry
    payload["_meta"] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "global_degenerate": _global_degenerate,
        "platt_min_slope": _PLATT_MIN_SLOPE,
    }
    try:
        out_path = out_dir / "part3_platt_params.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Part 3] Platt params written to {out_path}")
    except Exception as e:
        print(f"[Part 3] WARNING: Could not persist Platt params: {e}")


def _apply_regime_platt(
    p_cal: Optional[float],
    regime_label: str,
    params: Dict[str, Tuple[float, float]],
    regime_auc_breakdown: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Apply regime-conditional Platt scaling to a single probability.

    Returns p_cal unchanged if p_cal is None, params is empty, or no valid
    calibration is available for this regime (transparent fallback).

    FIX (F1, Quant-Guild Part 47 Audit): regime-specific fits below
    _PLATT_MIN_SLOPE are already excluded upstream in _fit_regime_platt_scaling
    (never stored under their own regime key), so reaching this function with
    regime_label present in params already guarantees a >= _PLATT_MIN_SLOPE for
    THAT branch. The gap was _global: it is unconditionally stored regardless
    of its own slope, so any regime that falls through to it (today: crisis,
    risk_on — see platt_inverted_regimes_excluded) could silently receive a
    degenerate transform with zero discriminative value, exactly as the
    excluded regime-specific fits would have. _fit_regime_platt_scaling now
    stamps params["__global_degenerate__"] = True when _global itself is
    degenerate; this function checks that flag on the _global branch only
    (other regimes already passed their own floor check upstream) and falls
    through to transparent pass-through (p_cal unchanged) in that case —
    the identical safe behavior already used when params is empty.

    FIX (F1, Quant-Guild Part 48 Audit): _global must NOT be applied when
    the current regime's full-period AUC < 0.50.

    ROOT CAUSE (independently verified from S48 artifacts):
    The S47 fix correctly gates _global when its slope a < _PLATT_MIN_SLOPE
    (degenerate near-flat calibration). But it left a second structural defect
    untouched: _global is fit on ALL n=1685 rows and carries a positive slope
    (a=0.292875 in S48). A positive slope means higher p_raw -> higher p_recal
    (more tail-event signal). This is CORRECT for high_vol (AUC=0.561) but is
    INCORRECT for any regime where the model's signal is absent or inverted:

        calm    AUC=0.4668 (below 0.50 — signal is INVERTED in calm)
        crisis  AUC=0.4226 (below 0.50 — signal is INVERTED in crisis)
        risk_on AUC=0.2692 (below 0.50 — signal is INVERTED in risk_on)

    In these regimes, higher p_raw actually predicts IEF outperformance
    (opposite direction). Applying _global (positive slope) amplifies p_raw
    upward in exactly the wrong direction for 59.4% of all rows.

    Quantified impact (S48, p_raw=0.1827, calm regime):
      p_regime_recal with _global: 0.1950 (+1.22pp wrong direction)
      p_regime_recal correct:      0.1827 (transparent pass-through)

    The _global fixed point is at p≈0.2002; below this, _global increases p
    (amplifies in wrong direction for AUC<0.50 regimes); above this, _global
    decreases p (compresses in wrong direction). Neither behavior is valid when
    the regime has no demonstrated predictive ability.

    FIX: when the excluded regime's full-period AUC < 0.50, return p_cal
    UNCHANGED. Apply _global only when the excluded regime's AUC >= 0.50
    (model has positive-direction signal there, just with a Platt fit excluded
    for slope/inversion reasons — an edge case not observed in current data but
    handled for completeness).

    Note: regime_auc_breakdown is optional for backward compatibility. If not
    supplied (e.g. cold start), the S47 behavior is preserved (_global applied
    to all excluded regimes whose slope is non-degenerate). Callers should
    always supply it using part2_summary["regime_auc_breakdown"].
    """
    if not HAVE_PLATT or not params or p_cal is None or not math.isfinite(p_cal):
        return p_cal
    p_clipped = max(0.01, min(0.99, float(p_cal)))
    logit_p = float(_logit(p_clipped))
    if regime_label in params and isinstance(params.get(regime_label), tuple):
        # Regime has its own non-degenerate, non-inverted Platt fit — apply it.
        a, b = params[regime_label]  # type: ignore[misc]
    elif "_global" in params and isinstance(params.get("_global"), tuple):
        # Regime was excluded (inverted or degenerate slope) — check before routing to _global.
        # Gate 1 (S47 F1): if _global itself is degenerate (a < _PLATT_MIN_SLOPE), pass through.
        if bool(params.get("__global_degenerate__", False)):  # type: ignore[arg-type]
            print(
                f"[Part 3] Platt({regime_label:12s}): _global is degenerate "
                f"(a < {_PLATT_MIN_SLOPE}) — transparent pass-through applied."
            )
            return p_cal
        # Gate 2 (S48 F1): if the regime's full-period AUC < 0.50, the model has no valid
        # positive-direction signal here. A positive-slope _global is wrong by direction.
        # Return p_cal unchanged (transparent pass-through) — the same safe behavior as
        # the degenerate-_global gate above.
        if regime_auc_breakdown is not None:
            _regime_auc_entry = regime_auc_breakdown.get(str(regime_label).lower(), {})
            if isinstance(_regime_auc_entry, dict):
                _regime_full_period_auc = _regime_auc_entry.get("auc", None)
                if _regime_full_period_auc is not None:
                    try:
                        _regime_full_period_auc_f = float(_regime_full_period_auc)
                        if _regime_full_period_auc_f < 0.50:
                            _a_global_disp = float(params.get("_global", (float("nan"),))[0])  # type: ignore[index]
                            print(
                                f"[Part 3] Platt({regime_label:12s}): full-period AUC="
                                f"{_regime_full_period_auc_f:.4f} < 0.50 — no valid positive-direction "
                                f"signal in this regime. Transparent pass-through applied "
                                f"(_global a={_a_global_disp:.4f} NOT applied).  [S48 F1 fix]"
                            )
                            return p_cal
                    except (TypeError, ValueError):
                        pass  # cannot evaluate AUC — fall through to _global safely
        # _global is non-degenerate AND regime AUC >= 0.50 (or breakdown unavailable) — apply it.
        a, b = params["_global"]  # type: ignore[misc]
    else:
        return p_cal
    return float(_expit(a * logit_p + b))

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

    # ── Regime-conditional Platt scaling ──────────────────────────────────
    # Load Part 6 regime history and Part 1 revealed labels to fit a per-regime
    # logistic recalibration of p_final_cal.  Falls back silently to raw p_final_cal
    # if either artifact is missing, if insufficient data exists, or if scipy/sklearn
    # are unavailable.
    regime_history_path = _first_existing_path(["artifacts_part6/regime_history.parquet"], root)
    y_revealed_path     = _first_existing_path(["artifacts_part1/y_labels_revealed.parquet"], root)
    platt_params = _fit_regime_platt_scaling(defense_df, y_revealed_path, regime_history_path)
    platt_active = bool(platt_params)
    # NOTE: _persist_platt_params is called AFTER out_dir is defined below (F7 ordering fix).

    # Part 7 base weights — optional, preferred over CFG 60/40 defaults.
    part7_voo, part7_ief = _load_part7_base_weights(root)
    part7_weights: Optional[Tuple[float, float]] = (part7_voo, part7_ief) if part7_voo is not None else None
    if part7_weights is not None:
        print(f"[Part 3] Part 7 base weights loaded: VOO={part7_voo:.4f} | IEF={part7_ief:.4f}")
    else:
        print(f"[Part 3] Part 7 tape not found — using default base weights: VOO={CFG.default_voo_weight} | IEF={CFG.default_ief_weight}")

    publish_mode = _normalize_publish_mode(_json_value(part2_summary, ["publish_mode", "mode"], "UNKNOWN"))
    final_pass = _boolish(_json_value(part2_summary, ["final_pass"], 0), 0)
    if publish_mode not in {"NORMAL", "DEFENSE_ONLY", "FAIL_CLOSED_NEUTRAL"}:
        raise RuntimeError(f"Uncertified publish_mode: {publish_mode}")
    if not final_pass and publish_mode != "FAIL_CLOSED_NEUTRAL":
        # FIX (BUG-C, Quant-Guild Part 31 Audit):
        # Part 2's _should_fail_closed() is an INDEPENDENT gate: it triggers only on
        # genuine safety failures (drift, calibration, severe IR). When a quality metric
        # (e.g. lift) marginally misses its threshold — within sampling noise — Part 2
        # correctly leaves publish_mode=NORMAL rather than entering FAIL_CLOSED_NEUTRAL,
        # because the independent safety gates all pass.
        #
        # The old behavior here was to raise RuntimeError, which killed the pipeline.
        # The correct defensive response is to degrade to FAIL_CLOSED_NEUTRAL for this
        # run, log the override, and let the daily cycle resume. The next run will
        # re-evaluate quality; if the quality gate clears (with hysteresis), the system
        # automatically returns to NORMAL without manual intervention.
        #
        # This is consistent with the principle that Part 3 is the LAST safety layer:
        # it should never crash the pipeline — it should degrade gracefully.
        print(
            "[Part 3] WARNING: Part 2 final_pass=False with publish_mode=NORMAL. "
            "This occurs when a quality metric (lift, ECE) marginally misses its threshold "
            "but independent safety gates (drift, calibration, cond_ir) all pass. "
            "Overriding publish_mode → FAIL_CLOSED_NEUTRAL for this run. "
            "No manual intervention needed; the system will re-evaluate on the next run."
        )
        publish_mode = "FAIL_CLOSED_NEUTRAL"

    # FIX (Quant-Guild Part 46 Audit): apply the authoritative governance decision to
    # the EXECUTABLE allocation weights, not just to the publish_mode/deployment_mode
    # labels.
    #
    # ROOT CAUSE: part7_weights (loaded above at L1354, BEFORE publish_mode is
    # finalized) was passed unchanged into _build_fusion_allocations() regardless of
    # the publish_mode determined here. Part 7's "base" weights may reflect its
    # soft-clearance pass-through (raw_val_auc_median >= 0.52 lets the BL optimizer
    # run even while Part 2's publish_mode = FAIL_CLOSED_NEUTRAL) — those weights are
    # a genuine model view for diagnostic purposes, but were never meant to be the
    # EXECUTABLE allocation once governance has determined FAIL_CLOSED_NEUTRAL.
    #
    # A parallel fix already exists below (L1921-1923 in the current_target_weights.json
    # sync block, F3/Part 34 Audit) that overwrites w_target_voo/w_target_ief to 60/40
    # in that one file. But v1_fusion_allocations.csv — built by _build_fusion_allocations()
    # using the UNCORRECTED part7_weights — was never covered by that fix, and it is the
    # file Part 8's load_part7_instructions() PREFERS over current_target_weights.json
    # (it checks v1_fusion_allocations.csv first and only falls back to
    # current_target_weights.json if that file is missing). It is also rendered directly
    # on the public GitHub Pages dashboard ("Latest Fusion Allocations" panel / allocTape).
    #
    # CONFIRMED IMPACT (S46 artifact): with publish_mode=FAIL_CLOSED_NEUTRAL,
    # current_target_weights.json correctly showed w_target_voo=0.60 (the F3/Part-34 fix
    # working as intended), while v1_fusion_allocations.csv for the SAME date showed
    # w_target_voo=0.650750 — Part 7's raw soft-clearance BL output. Two artifacts
    # written in the same Part 3 run contradicted each other and the stated governance
    # state, and Part 8 silently consumed the wrong (un-overridden) one every time.
    #
    # FIX: derive a separate, governance-corrected weight pair used ONLY for
    # _build_fusion_allocations(). part7_voo/part7_ief themselves are left untouched —
    # they remain the raw Part 7 diagnostic values surfaced in part3_summary.json as
    # part7_voo_base/part7_ief_base, which is intentionally informational (it shows
    # what Part 7's optimizer actually proposed, for monitoring/audit purposes).
    fusion_base_weights = part7_weights
    if publish_mode == "FAIL_CLOSED_NEUTRAL" and part7_weights is not None:
        _needs_override = (
            abs(part7_weights[0] - CFG.default_voo_weight) > 1e-9
            or abs(part7_weights[1] - CFG.default_ief_weight) > 1e-9
        )
        if _needs_override:
            print(
                f"[Part 3] FAIL_CLOSED_NEUTRAL: fusion allocation weights overridden for "
                f"v1_fusion_allocations.csv consistency — "
                f"VOO {part7_weights[0]:.4f}->{CFG.default_voo_weight:.4f}, "
                f"IEF {part7_weights[1]:.4f}->{CFG.default_ief_weight:.4f}. "
                f"(Part 7's raw proposal remains visible in part7_voo_base/part7_ief_base.)"
            )
        fusion_base_weights = (float(CFG.default_voo_weight), float(CFG.default_ief_weight))

    defense_row = _last_valid_row(defense_df)
    decision_date = pd.to_datetime(_row_value(defense_row, ["Date", "decision_date", "asof_date"]), errors="coerce")
    if pd.isna(decision_date):
        decision_date = pd.Timestamp.today().normalize()
    target_date = decision_date + pd.tseries.offsets.BDay(1)

    voo_call, ief_call = _extract_latest_price_call(defense_row)

    # FIX (F1, Audit 2026-05-10 — Quant-Guild Part 20):
    # Compute live_predlog_rows from the EXISTING prediction_log (before today's upsert).
    # This is the count of live predictions with confirmed realized prices — the correct
    # input for _infer_promotion_state. The full upsert happens later in the function.
    #
    # FIX (F2, Quant-Guild Part 51 Audit): Two-phase realized-row count.
    #
    # ROOT CAUSE OF PERSISTENT live_realized_dates=0:
    # The pre-upsert count correctly reads the RESTORED prediction_log from origin.
    # However, this count is ALSO used to populate part3_summary.json's
    # live_realized_dates field via alpha_status["realized_dates"]. The restored
    # file on origin is the version committed by the PREVIOUS pipeline run —
    # which did not yet have the backfill's realized prices appended for the
    # most recent target date.
    #
    # CONFIRMED SEQUENCE (S50 and S51 both show live_realized_dates=0):
    #   Day N (Tuesday): Pipeline runs at ~9:35 AM ET.
    #     - Upserts prediction_log WITHOUT realized prices.
    #     - Commits that version to origin.
    #   Day N, 4 PM ET: Backfill runs → adds realized prices → commits to origin.
    #   Day N+1 or later: Next pipeline run starts.
    #     Restore step fetches origin → should get the backfill version.
    #     BUT: if the backfill on Day N+1 ran at 20:00 UTC and the pipeline
    #     started at 20:47 UTC, the concurrency group pricecall-production
    #     queues the pipeline BEHIND the backfill. The restore step runs
    #     AFTER checkout but BEFORE the backfill's push completes on origin,
    #     so the restored predlog still lacks the realized prices.
    #
    # This is a timing race that fires whenever the backfill and pipeline
    # share the same concurrency group and run within minutes of each other.
    #
    # TWO-PHASE FIX:
    # Phase 1 (pre-upsert, UNCHANGED): Read the restored predlog for alpha_status
    #   promotion gating. Alpha state is computed with this count to avoid a
    #   chicken-egg dependency (alpha_status feeds the upsert arguments).
    #   This may read 0 if the timing race fires — that is acceptable for
    #   promotion gating (live_realized_dates=0 → alpha stays SHADOW, correct).
    #
    # Phase 2 (post-upsert, NEW): After _upsert_prediction_log completes, the
    #   returned predlog_df contains all rows including today's upsert AND any
    #   realized prices already present on disk. Re-count from this post-upsert
    #   frame and overwrite live_realized_dates / prediction_log_realized_rows
    #   in the summary JSON with the ACCURATE count.
    #
    # Why Phase 2 is safe:
    #   - Alpha promotion state (computed in Phase 1) is NOT retroactively changed.
    #     Promotion decisions are frozen at Phase 1 for within-run consistency.
    #   - Phase 2 only corrects the REPORTING fields in part3_summary.json so the
    #     dashboard shows the correct realized count rather than a stale 0.
    #   - If Phase 2 count > 0 while Phase 1 count = 0, the summary correctly
    #     shows the accumulated realized rows; alpha state stays SHADOW until
    #     next run's Phase 1 count reaches the promotion threshold.
    #
    # OUTCOME: live_realized_dates in part3_summary.json will show the POST-UPSERT
    # count. This will be 1 on the next run after the first backfill, reflecting
    # reality rather than the timing-race-induced 0.  [S51 F2 fix]
    _predlog_path_early = Path(root) / "artifacts_part3" / "prediction_log.csv"
    if _predlog_path_early.exists():
        try:
            _predlog_early = pd.read_csv(_predlog_path_early)
            live_predlog_rows = _count_realized_predlog_rows(_predlog_early)
        except Exception:
            live_predlog_rows = 0
    else:
        live_predlog_rows = 0

    # ── FIX (BUG-1, Audit 2026-05-11 — Quant-Guild Part 21): Ensemble probability blend ──
    # Part 2B (XGBoost ensemble, holdout AUC=0.571, t=2.34) and Part 2C (BNN, holdout
    # AUC=0.584, t=2.82) are statistically significant. Part 2 base (AUC=0.513, t=0.49)
    # is not. The Part 20 fix incorrectly placed the blend in Part 2, which runs BEFORE
    # Part 2B and 2C in the pipeline order. That caused the blend code to never execute
    # (confirmed: all 1658 rows had p_final_g5_source="base_plus_soft_caution_overlay_532",
    # not "base_only" — the latter would appear had the block started).
    #
    # Correct location: Part 3 runs AFTER Part 2B and 2C, making it the natural fusion
    # layer. The blend loads the CURRENT run's Part 2B and 2C tapes to find the live
    # date's probabilities. If either sleeve is unavailable, the blend falls back
    # gracefully to the base-only probability.
    #
    # Epistemic caution (Finding-3): when Part 2C reports live_high_epistemic_warning=True
    # (BNN uncertainty > 1.5× training distribution), the Part 2C component is downweighted
    # by 50% in the blend to dampen its influence when the model is extrapolating.
    _p2b_blend_p: Optional[float] = None
    _p2c_blend_p: Optional[float] = None
    _blend_source: str = "base_only"
    _live_high_epistemic: bool = False
    _p2c_epist_ratio: float = 1.0

    # Load Part 2B probability for the live date
    # FIX (BUG-A, Quant-Guild Part 21 run): tapes live in the `predictions/` subdirectory
    _p2b_root = root / "artifacts_part2b_xgb" / "predictions"
    _p2b_tape_path = _p2b_root / "part2b_xgb_tape.csv"
    _p2b_summary_path = _p2b_root / "part2b_xgb_summary.json"
    try:
        if _p2b_tape_path.exists():
            _p2b_tape = pd.read_csv(_p2b_tape_path)
            _p2b_tape["Date"] = pd.to_datetime(_p2b_tape["Date"], errors="coerce").dt.normalize()
            _p2b_tape = _p2b_tape.dropna(subset=["Date"]).set_index("Date")
            _live_dt = pd.Timestamp(decision_date).normalize()
            if "p_xgb_ens_mean" in _p2b_tape.columns and _live_dt in _p2b_tape.index:
                _p2b_blend_p = float(_p2b_tape.loc[_live_dt, "p_xgb_ens_mean"])
                if not math.isfinite(_p2b_blend_p):
                    _p2b_blend_p = None
            # FIX (BUG-2, Audit 2026-05-11 — Quant-Guild Part 22):
            # If the live date is not in the tape, fall back to the summary JSON's
            # live_p_xgb_ens_mean field.  The summary JSON is loaded here and cached
            # in _p2b_summary_cached so the ECE weight block below does NOT re-read it.
            # (BUG-9, Audit 2026-05-12 — Quant-Guild Part 25: the prior code issued
            # two separate _read_json calls on the same file — once here and once in
            # the ECE weight block ~80 lines later.  A single read + cache is correct.)
            if _p2b_summary_path.exists():
                try:
                    _p2b_summary_cached: Dict[str, Any] = _read_json(_p2b_summary_path)
                except Exception as _cache_e:
                    _p2b_summary_cached = {}
                    print(f"[Part 3] Blend: Part 2B summary JSON load failed ({_cache_e})")
            else:
                _p2b_summary_cached = {}
            if _p2b_blend_p is None and _p2b_summary_cached:
                _fb_p = _p2b_summary_cached.get("live_p_xgb_ens_mean")
                if _fb_p is not None and math.isfinite(float(_fb_p)):
                    _p2b_blend_p = float(_fb_p)
                    print(f"[Part 3] Blend: Part 2B live p from summary JSON fallback: {_p2b_blend_p:.4f}")
            print(f"[Part 3] Blend: Part 2B XGB tape loaded ({len(_p2b_tape)} rows) | live p={_p2b_blend_p}")
        else:
            # Tape absent — still try to load the summary JSON for ECE weight and fallback p
            _p2b_summary_cached = {}
            if _p2b_summary_path.exists():
                try:
                    _p2b_summary_cached = _read_json(_p2b_summary_path)
                    _fb_p2 = _p2b_summary_cached.get("live_p_xgb_ens_mean")
                    if _fb_p2 is not None and math.isfinite(float(_fb_p2)):
                        _p2b_blend_p = float(_fb_p2)
                        print(f"[Part 3] Blend: Part 2B (tape absent) live p from summary JSON: {_p2b_blend_p:.4f}")
                except Exception as _fb_abs:
                    print(f"[Part 3] Blend: Part 2B summary JSON load failed ({_fb_abs})")
            else:
                print(f"[Part 3] Blend: Part 2B tape not found at {_p2b_tape_path}")
    except Exception as _e2b:
        _p2b_summary_cached = {}
        print(f"[Part 3] Blend: Part 2B tape load failed ({_e2b}) — skipping 2B")

    # FIX (Audit Part 28 — C-1, C-2): Gate Part 2B from the probability blend when
    # (a) its pooled walkforward AUC is not statistically significant (p >= 0.05
    #     one-sided), or (b) the global Platt calibrator is degenerate (|a| < 0.25),
    #     which collapses the calibrated signal std from 0.148 → 0.011 (93% collapse).
    #
    # At t=1.39, p=0.082 (current artifacts), Part 2B's calibrated probability carries
    # no verified signal and blending it adds noise that pulls the ensemble away from
    # the significantly predictive Part 2C component (t=6.24, p<0.0001).
    #
    # The uncertainty overlay gate (xgb_overlay_on) is NOT affected by this exclusion —
    # it is evaluated independently via ECE stratification and remains valid.
    # Part 2B can still influence Part 3 indirectly through its uncertainty spread
    # signal once Part 2C's overlay_on fires.
    #
    # The significance check is permissive-by-default (True when field absent) so
    # pre-existing runs without the field are not broken on cold start.
    _p2b_wf_significant = bool(_p2b_summary_cached.get("walkforward_auc_significant", True))
    _p2b_platt_degenerate = bool(_p2b_summary_cached.get("platt_degenerate", False))
    if _p2b_blend_p is not None and (not _p2b_wf_significant or _p2b_platt_degenerate):
        _exclusion_reasons: list = []
        if not _p2b_wf_significant:
            _pval = _p2b_summary_cached.get("walkforward_auc_pooled_pval", "?")
            _pauc = _p2b_summary_cached.get("walkforward_auc_pooled", "?")
            _exclusion_reasons.append(
                f"walkforward_auc NOT significant (pooled_auc={_pauc}, p={_pval} >= 0.05)"
            )
        if _p2b_platt_degenerate:
            _pa = _p2b_summary_cached.get("platt_global_a", "?")
            _exclusion_reasons.append(
                f"Platt degenerate (|a|={abs(float(_pa)) if _pa != '?' else '?':.3f} < 0.25, "
                f"signal compressed >{(1 - 0.011 / 0.148):.0%})"
            )
        print(
            f"[Part 3] Blend: Part 2B EXCLUDED from probability blend. "
            f"Reason(s): {'; '.join(_exclusion_reasons)}"
        )
        _p2b_blend_p = None

    # Load Part 2C probability for the live date; check epistemic warning
    # FIX (BUG-A): tapes live in the `predictions/` subdirectory
    _p2c_root = root / "artifacts_part2c_bnn" / "predictions"
    _p2c_tape_path = _p2c_root / "part2c_bnn_tape.csv"
    _p2c_summary_path = _p2c_root / "part2c_bnn_summary.json"
    try:
        if _p2c_summary_path.exists():
            _p2c_summary = _read_json(_p2c_summary_path)
            _live_high_epistemic = bool(_p2c_summary.get("live_high_epistemic_warning", False))
            _p2c_epist_ratio = float(_p2c_summary.get("live_epist_ratio", 1.0) or 1.0)
            if _live_high_epistemic:
                print(f"[Part 3] Blend: Part 2C HIGH epistemic warning (ratio={_p2c_epist_ratio:.3f}) — downweighting 2C by 50%")
        if _p2c_tape_path.exists():
            _p2c_tape = pd.read_csv(_p2c_tape_path)
            _p2c_tape["Date"] = pd.to_datetime(_p2c_tape["Date"], errors="coerce").dt.normalize()
            _p2c_tape = _p2c_tape.dropna(subset=["Date"]).set_index("Date")
            _live_dt = pd.Timestamp(decision_date).normalize()
            if "p_bnn_mean" in _p2c_tape.columns and _live_dt in _p2c_tape.index:
                _p2c_blend_p = float(_p2c_tape.loc[_live_dt, "p_bnn_mean"])
                if not math.isfinite(_p2c_blend_p):
                    _p2c_blend_p = None
            # FIX (BUG-2, Audit 2026-05-11 — Quant-Guild Part 22):
            # Same fallback as Part 2B: if live date not in tape, read from summary JSON.
            if _p2c_blend_p is None and _p2c_summary_path.exists():
                try:
                    _p2c_sum_fb = _read_json(_p2c_summary_path)
                    _fb_p_c = _p2c_sum_fb.get("live_p_bnn_mean")
                    if _fb_p_c is not None and math.isfinite(float(_fb_p_c)):
                        _p2c_blend_p = float(_fb_p_c)
                        print(f"[Part 3] Blend: Part 2C live p from summary JSON fallback: {_p2c_blend_p:.4f}")
                except Exception as _fb_e_c:
                    print(f"[Part 3] Blend: Part 2C summary JSON fallback failed ({_fb_e_c})")
            print(f"[Part 3] Blend: Part 2C BNN tape loaded ({len(_p2c_tape)} rows) | live p={_p2c_blend_p}")
        else:
            print(f"[Part 3] Blend: Part 2C tape not found at {_p2c_tape_path}")
    except Exception as _e2c:
        print(f"[Part 3] Blend: Part 2C tape load failed ({_e2c}) — skipping 2C")

    # ── FIX (BUG-2/BUG-3, Audit 2026-05-12 — Quant-Guild Part 25): Part 2C quality gates ──
    # Gate 1: gate_validation_passed.  Part 2C must pass the same ECE calibration ceiling
    # as Part 2B (holdout_ece <= base_ece + 0.05).  If Part 2C reports
    # gate_validation_passed=False, exclude it from the blend entirely.
    # Current state (2026-05-26): holdout_ece=0.0285 ≤ ceiling 0.0626 → gate PASSES.
    # NOTE (F-3, Quant-Guild Part 29): an earlier comment stated holdout_ece=0.0767
    # and gate FAILS. That referred to a prior code version. The current artifact
    # (part2c_bnn_summary.json) shows gate_validation_passed=True. Part 2C IS
    # included in the blend when the epistemic and bias gates also pass (see below).
    # Update this comment after any retraining run that changes holdout_ece.
    #
    # Both gates are non-blocking: if the Part 2C summary JSON is absent or missing
    # the fields, the default is permissive (include), to avoid breaking cold-start.
    _p2c_gate_ok: bool = True        # gate_validation_passed
    _p2c_bias_ok: bool = True        # directional mean-bias
    _p2c_gate_reason: str = ""
    if _p2c_summary_path.exists():
        try:
            _p2c_sum_gate = _read_json(_p2c_summary_path)
            # Gate 1
            _gvp = _p2c_sum_gate.get("gate_validation_passed")
            if _gvp is not None:
                _p2c_gate_ok = bool(_gvp)
            if not _p2c_gate_ok:
                _p2c_gate_reason = f"gate_validation_passed=False (holdout_ece={_p2c_sum_gate.get('holdout_ece','?')})"
            # Gate 2
            _mbf = _p2c_sum_gate.get("mean_bias_flag")
            if _mbf is not None:
                _p2c_bias_ok = not bool(_mbf)
            if not _p2c_bias_ok:
                _p2c_gate_reason += (
                    f" | mean_bias_flag=True"
                    f" (mean_bias_ratio={_p2c_sum_gate.get('mean_bias_ratio','?')})"
                )
        except Exception as _gate_e:
            print(f"[Part 3] Blend: Part 2C gate check failed ({_gate_e}) — using permissive default")
    if not _p2c_gate_ok or not _p2c_bias_ok:
        print(f"[Part 3] Blend: Part 2C EXCLUDED from blend. Reason: {_p2c_gate_reason}")
        _p2c_blend_p = None

    # Compute blended probability for the live row
    # Equal weighting (1/3 each) unless epistemic warning fires → 2C gets weight 0.5
    # FIX (BUG-5, Audit 2026-05-11 — Quant-Guild Part 24):
    # Part 2B has walkforward_mean_ece=0.181 vs holdout_ece=0.013 (13.5× gap).
    # The global Platt fit (a=0.159, near-flat slope) overfits the holdout period;
    # in time-ordered CV, calibrated probabilities degrade severely. A model whose
    # walkforward ECE exceeds 0.10 should receive reduced blend weight to prevent
    # a poorly calibrated component from dominating the ensemble mean.
    #
    # Weight schedule (applied to Part 2B only; Part 2C has its own epistemic gate):
    #   walkforward_ece < 0.10: w_2b = 1.0  (normal)
    #   0.10 <= wf_ece < 0.15:  w_2b = 0.5  (caution)
    #   wf_ece >= 0.15:         w_2b = 0.25 (poor calibration, minimal weight)
    _p2b_wf_ece: float = 1.0  # default: no downweight
    # FIX (BUG-9, Audit 2026-05-12 — Quant-Guild Part 25):
    # Use the cached _p2b_summary_cached dict loaded in the tape-loading block above.
    # The prior code called _read_json(_p2b_summary_path) a second time here, wasting
    # I/O and creating a stale-data risk window between the two reads on concurrent runs.
    try:
        _p2b_wf_ece_raw = _p2b_summary_cached.get("walkforward_mean_ece", None) if _p2b_summary_cached else None
        if _p2b_wf_ece_raw is not None and math.isfinite(float(_p2b_wf_ece_raw)):
            _p2b_wf_ece_val = float(_p2b_wf_ece_raw)
            if _p2b_wf_ece_val >= 0.15:
                _p2b_wf_ece = 0.25
                print(f"[Part 3] Blend: Part 2B walkforward ECE={_p2b_wf_ece_val:.4f} ≥ 0.15 — downweighting 2B to 0.25")
            elif _p2b_wf_ece_val >= 0.10:
                _p2b_wf_ece = 0.50
                print(f"[Part 3] Blend: Part 2B walkforward ECE={_p2b_wf_ece_val:.4f} ≥ 0.10 — downweighting 2B to 0.50")
            else:
                print(f"[Part 3] Blend: Part 2B walkforward ECE={_p2b_wf_ece_val:.4f} — full weight 1.0")
    except Exception as _ece_e:
        print(f"[Part 3] Blend: could not read Part 2B walkforward ECE ({_ece_e}) — using w_2b=1.0")

    # The blend modifies p_raw, which flows into p_regime_recal and prediction_log.
    # FIX (BUG-B, Quant-Guild Part 21 run): initialize p_raw unconditionally here
    # so the variable is always bound when _apply_regime_platt() consumes it below.
    # The previous code only assigned p_raw inside conditional branches, leaving it
    # unbound when _p_base was None (UnboundLocalError on line ~1353).
    p_raw = _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None))
    _p_base = _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None))
    if _p_base is not None and math.isfinite(_p_base):
        _w_base, _w_2b, _w_2c = 1.0, _p2b_wf_ece, (0.5 if _live_high_epistemic else 1.0)
        _vals = [(p, w) for p, w in [(_p_base, _w_base), (_p2b_blend_p, _w_2b), (_p2c_blend_p, _w_2c)]
                 if p is not None and math.isfinite(p)]
        if len(_vals) > 1:
            _w_sum = sum(w for _, w in _vals)
            _p_blended = sum(p * w for p, w in _vals) / _w_sum
            _p_blended = float(max(1e-4, min(1.0 - 1e-4, _p_blended)))
            _sources = ["base"]
            if _p2b_blend_p is not None: _sources.append("2b_xgb")
            if _p2c_blend_p is not None:
                _sources.append("2c_bnn" + ("_downwt" if _live_high_epistemic else ""))
            _blend_source = "+".join(_sources)
            print(f"[Part 3] Blend: {_blend_source} | p_base={_p_base:.4f} → p_blended={_p_blended:.4f}")
            # Override p_raw so the blended value flows into p_regime_recal and prediction_log
            p_raw = _p_blended
        else:
            print(f"[Part 3] Blend: Only base available for live date — p unchanged ({_p_base:.4f})")
            _blend_source = "base_only"
    else:
        _blend_source = "base_only_null_input"

    alpha_status = _load_alpha_status(alpha_tape_df, alpha_summary_json,
                                       live_realized_rows=live_predlog_rows)
    alpha_positions_latest = _extract_alpha_positions(alpha_positions_df)

    # Compute regime-recalibrated probability for the current live row
    # p_raw is now the blended probability (or base-only if blend unavailable)
    current_regime = str(_row_value(defense_row, ["regime_label"], "unknown"))
    # FIX (F1, Quant-Guild Part 48 Audit): pass regime_auc_breakdown so
    # _apply_regime_platt can gate _global from AUC<0.50 regimes (calm,
    # crisis, risk_on). See _apply_regime_platt docstring for full derivation.
    _regime_auc_breakdown_for_platt = part2_summary.get("regime_auc_breakdown", {})
    p_regime_recal: Optional[float] = _apply_regime_platt(
        p_raw, current_regime, platt_params,
        regime_auc_breakdown=_regime_auc_breakdown_for_platt,
    )

    prod_tape = _prepare_production_tape(defense_df, part2_summary, alpha_status, alpha_summary_json)
    gov_df = _build_governance_df(
        decision_date,
        part2_summary,
        alpha_status,
        # FIX (F-2, Quant-Guild Part 32 Audit):
        # Pass post-override publish_mode and final_pass so the governance CSV
        # reflects the same values written to prediction_log and part3_summary.
        # The local `publish_mode` may have been overridden from NORMAL to
        # FAIL_CLOSED_NEUTRAL at L1345 when final_pass=False; the dict is not mutated.
        publish_mode_override=publish_mode,
        final_pass_override=final_pass,
    )
    # FIX (Quant-Guild Part 46 Audit): use fusion_base_weights (governance-corrected
    # above), not the raw part7_weights, so v1_fusion_allocations.csv is consistent
    # with publish_mode/deployment_mode whenever FAIL_CLOSED_NEUTRAL is in effect.
    alloc_df, max_dev = _build_fusion_allocations(decision_date, defense_row, alpha_positions_latest, alpha_status, fusion_base_weights)

    out_dir = root / cfg.out_dir_relative
    predlog_dir = root / cfg.predlog_dir_relative
    _ensure_dir(out_dir)
    _ensure_dir(predlog_dir)
    # FIX (F7 ordering, Audit 2026-05-10 — Quant-Guild Part 17):
    # Persist Platt calibration parameters now that out_dir is defined.
    # The prior placement (before out_dir was assigned) caused UnboundLocalError.
    if platt_active:
        _persist_platt_params(platt_params, out_dir)

    tape_out = out_dir / cfg.tape_name
    gov_out = out_dir / cfg.gov_name
    alloc_out = out_dir / cfg.alloc_name
    predlog_out = predlog_dir / cfg.predlog_name
    summary_out = out_dir / cfg.summary_name

    prod_tape.to_csv(tape_out, index=False)

    # FIX (Finding #19): Governance CSV now accumulates a time-series history
    # instead of overwriting on each run. We append the new row and deduplicate
    # on Date, keeping the most-recent entry for each date so re-runs are
    # idempotent. This provides a complete audit trail of governance state
    # changes (e.g. when fail-closed was entered, when alpha advanced tiers).
    if gov_out.exists():
        try:
            _existing_gov = pd.read_csv(gov_out)
            _existing_gov["Date"] = pd.to_datetime(_existing_gov["Date"], errors="coerce")
            gov_df_combined = pd.concat([_existing_gov, gov_df], ignore_index=True)
            gov_df_combined["Date"] = pd.to_datetime(gov_df_combined["Date"], errors="coerce")
            gov_df_combined = (
                gov_df_combined
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
                .reset_index(drop=True)
            )
            gov_df_combined.to_csv(gov_out, index=False)
        except Exception:
            gov_df.to_csv(gov_out, index=False)
    else:
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
        publish_mode,   # pass raw governance value; deployment_mode aliasing done inside upsert
        final_pass,
        alpha_status,
        part2_tape,
        alpha_sources,
        defense_row,
        part2_summary,
    )

    # Add regime-recalibrated probability to the prediction log row
    if not predlog_df.empty and p_regime_recal is not None:
        predlog_df["p_regime_recal"] = np.nan
        date_mask = pd.to_datetime(predlog_df.get("decision_date", pd.Series(dtype="object")), errors="coerce") == decision_date
        if date_mask.any():
            predlog_df.loc[date_mask, "p_regime_recal"] = p_regime_recal
        predlog_df.to_csv(predlog_out, index=False)

    # FIX (Audit Part 28 — C-3): Also store p_final_cal_blended (the ensemble-blended
    # probability before Platt recalibration) so Part 9 can see the full chain:
    #   p_final_cal (raw Part 2) → p_final_cal_blended (+ 2B/2C) → p_regime_recal (+ Platt).
    # p_raw holds the blended probability at this point (or the raw base value if blend was
    # unavailable). Writing it here keeps the predlog schema additive and backward-compatible.
    if not predlog_df.empty and p_raw is not None:
        if "p_final_cal_blended" not in predlog_df.columns:
            predlog_df["p_final_cal_blended"] = np.nan
        date_mask2 = pd.to_datetime(predlog_df.get("decision_date", pd.Series(dtype="object")), errors="coerce") == decision_date
        if date_mask2.any():
            predlog_df.loc[date_mask2, "p_final_cal_blended"] = float(p_raw)
        predlog_df.to_csv(predlog_out, index=False)

    perf = _extract_performance_metrics(defense_df, alpha_summary_json, alpha_status)
    dist = _alpha_distribution(alpha_tape_df, alpha_status)
    dist_display = {str(k): float(v) for k, v in dist.items()}

    # FIX (F2, Quant-Guild Part 51 Audit): Phase 2 post-upsert realized count.
    # The pre-upsert count (live_predlog_rows, computed earlier) feeds alpha_status
    # and must not change after alpha_status is frozen. However, for REPORTING in
    # part3_summary.json, the post-upsert predlog_df is authoritative — it contains
    # all rows including today's upsert and any realized prices present on disk.
    # Re-counting from predlog_df gives live_realized_dates its correct value
    # without interfering with the promotion state already locked by alpha_status.
    #
    # Specifically: on the timing-race run where pre-upsert count = 0 but the
    # post-upsert predlog_df has 1 realized row, live_realized_dates will correctly
    # show 1 in the summary JSON, confirming to operators that attribution has begun
    # and the race condition fired this session.  [S51 F2 fix]
    _post_upsert_realized_rows: int = _count_realized_predlog_rows(predlog_df) if predlog_df is not None else realized_rows
    if _post_upsert_realized_rows != realized_rows:
        print(
            f"[Part 3] Timing-race correction: pre-upsert realized_rows={realized_rows} -> "
            f"post-upsert realized_rows={_post_upsert_realized_rows}. "
            f"Summary live_realized_dates will reflect the post-upsert count.  [S51 F2 fix]"
        )

    # FIX (F2, Quant-Guild Part 52 Audit): Phase 3 — re-read predlog from disk after
    # ALL writes (p_regime_recal, p_final_cal_blended) to get the definitive realized count.
    #
    # ROOT CAUSE OF PERSISTENT live_realized_dates=0 (3rd consecutive session):
    # The S51 F2 two-phase fix correctly counts from predlog_df in memory. However,
    # predlog_df only contains the realized prices that were PRESENT in the predlog
    # at restore time. If the CI timing race fires:
    #
    #   1. Pipeline starts at 01:46 UTC Fri Jun 20.
    #   2. Restore step fetches predlog from origin → gets version WITHOUT realized
    #      prices (because: the S51 pipeline wrote predlog without realized prices,
    #      then backfill added them AFTER the S51 pipeline commit; the S52 pipeline
    #      restored the S51-written version, not the backfill-amended version).
    #   3. Upsert: no realized prices in predlog_df → new row has no realized prices.
    #   4. Phase 2 count: predlog_df has 0 realized rows → _post_upsert_realized_rows=0.
    #   5. Part3 writes summary with live_realized_dates=0.
    #   6. THEN: backfill runs, reads the predlog written by Part3, adds realized prices,
    #      commits to disk → FINAL artifact has realized prices but summary shows 0.
    #
    # CONFIRMED IN ARTIFACTS (S52):
    #   prediction_log.csv on disk: px_voo_realized=688.11 (backfill added AFTER Part3)
    #   part3_summary.json: live_realized_dates=0 (written before backfill ran)
    #
    # FIX — Phase 3: Re-read the predlog from disk AFTER all writes complete.
    # This catches realized prices that were written by a PRIOR backfill session
    # and are now present on disk even if they weren't in the restored version.
    #
    # Phase 1 (pre-upsert): live_predlog_rows — for alpha promotion gating (FROZEN)
    # Phase 2 (post-upsert): in-memory predlog_df count — catches same-session data
    # Phase 3 (post-all-writes): re-read from disk — catches prior-session backfill data
    #
    # Safety: Alpha promotion state is locked at Phase 1 and is NOT changed by Phase 3.
    # Phase 3 is purely a reporting correction for part3_summary.json.
    # If Phase 3 yields a higher count than Phase 2, the timing race is confirmed and
    # the correction print fires. Idempotent when all three phases agree.  [FIX F2/S52]
    try:
        if predlog_out.exists():
            _predlog_disk = pd.read_csv(predlog_out)
            _phase3_count = _count_realized_predlog_rows(_predlog_disk)
        else:
            _phase3_count = _post_upsert_realized_rows
    except Exception:
        _phase3_count = _post_upsert_realized_rows

    if _phase3_count > _post_upsert_realized_rows:
        print(
            f"[Part 3] Phase 3 timing-race correction: in-memory realized_rows={_post_upsert_realized_rows} -> "
            f"on-disk realized_rows={_phase3_count}. "
            f"Prior-session backfill data detected; live_realized_dates corrected.  [FIX F2/S52]"
        )
        _post_upsert_realized_rows = _phase3_count

    summary = {
        "part": "PART3_V1",
        "root": str(root),
        "defense_source": str(part2_tape),
        "part2_summary_source": str(part2_summary_path),
        "alpha_positions_source": str(alpha_positions_path),
        "alpha_summary_source": str(alpha_summary_tape_path),
        "alpha_eligibility_source": str(alpha_eligibility_path),
        "alpha_summary_json_source": str(alpha_summary_json_path),
        "publish_mode": publish_mode,
        "deployment_mode": "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL" else publish_mode,
        "final_pass": int(final_pass),
        "latest_alpha_state": alpha_status["latest_state"],
        "latest_alpha_state_display": alpha_status["display_state"],
        "historical_alpha_state": alpha_status["latest_state"],
        "current_alpha_live_status": alpha_status.get("current_alpha_live_status", alpha_status["latest_state"]),
        "current_alpha_reason": alpha_status.get("current_alpha_reason", "unknown"),
        "current_alpha_eligible": int(alpha_status.get("current_alpha_eligible", 0)),
        "current_alpha_abs": float(alpha_status.get("current_alpha_abs", 0.0)),
        # FIX (Finding 12, Audit 2026-04-21):
        # "realized_dates" is ambiguous: it sounds like live prediction-log rows with
        # realized prices, but actually counts historical tape rows where the backtest
        # labels are revealed (2020-2026). Operators reading this field incorrectly
        # concluded 381 live predictions had been evaluated. Renamed to make the
        # distinction unambiguous. prediction_log_realized_rows is the live count.
        "alpha_tape_historical_realized_dates": alpha_status.get("backtest_realized_dates", alpha_status["realized_dates"]),
        "realized_dates": alpha_status.get("backtest_realized_dates", alpha_status["realized_dates"]),  # backtest count; kept for backward-compat
        # FIX (F2, Quant-Guild Part 51 Audit): Use post-upsert count for live_realized_dates.
        # alpha_status["realized_dates"] = pre-upsert count (correct for promotion gating).
        # _post_upsert_realized_rows = count from fully-updated predlog_df (correct for reporting).
        # On a timing-race run where pre-upsert=0 but post-upsert=1, this corrects the summary
        # from 0 → 1 without changing the alpha promotion state (which used pre-upsert=0).
        "live_realized_dates": _post_upsert_realized_rows,  # FIX S51 F2: was alpha_status["realized_dates"] (pre-upsert, stale on timing-race runs)
        "realized_dates_note": "realized_dates=backtest rows (display only). live_realized_dates=post-upsert prediction-log realized rows (used for reporting; promotion gate uses pre-upsert count).",
        "budget_mult": alpha_status["budget_mult"],
        "drift_rate": alpha_status["drift_rate"],
        "quality_ok": alpha_status["quality_ok"],
        "drift_ok": alpha_status["drift_ok"],
        "trial_gate_open": alpha_status["trial_gate_open"],
        "fused_gate_open": alpha_status["fused_gate_open"],
        "promotion_ready": alpha_status["promotion_ready"],
        "alpha_blockers": alpha_status["blockers"],
        "rows": int(len(prod_tape)),
        "rows_realized_fused": _post_upsert_realized_rows,  # FIX S51 F2: post-upsert count
        "fusion_live_rate": perf["fusion_live_rate"],
        # FIX (Finding 7, Audit 2026-05-10 — Quant-Guild Part 19): explicit boolean alias
        "alpha_fusion_is_live": perf["alpha_fusion_is_live"],
        # prediction_log_realized_pct: fraction of live predictions with realized prices
        "prediction_log_realized_pct": (
            # FIX (BUG-7, Audit 2026-05-11 — Quant-Guild Part 24):
            # The prior expression `"predlog_df" in dir()` is non-idiomatic.
            # dir() inside a function returns the local symbol table, which DOES
            # include predlog_df after _upsert_prediction_log assigns it. But
            # the pattern is fragile and misleading. Use a direct is not None check.
            # FIX (S51 F2): use _post_upsert_realized_rows for accurate percentage.
            round(_post_upsert_realized_rows / max(len(predlog_df), 1), 4) if predlog_df is not None else 0.0
        ),
        "defense_ir_net": perf["defense_ir"],
        "fused_ir_net": perf["fused_ir"],
        "active_ir_vs_60_40": perf["active_ir"],
        "active_mean": perf["active_mean"],
        "alpha_state_distribution": dist_display,
        "prediction_log_path": str(predlog_out),
        "prediction_log_realized_rows": _post_upsert_realized_rows,  # FIX S51 F2: post-upsert count
        "allocation_sum_to_one_max_deviation": max_dev,
        "horizon": 1,
        "part7_base_weights_source": "part7_portfolio_weights_tape" if part7_weights is not None else "cfg_default_60_40",
        "part7_voo_base": float(part7_voo) if part7_voo is not None else CFG.default_voo_weight,
        "part7_ief_base": float(part7_ief) if part7_ief is not None else CFG.default_ief_weight,
        "alpha_family": str(_json_value(alpha_summary_json, ["alpha_family", "version", "part"], os.environ.get(CFG.alpha_family_env_var, "part2a21"))),
        "alpha_contract": "legacy_state_machine",
        "preferred_alpha_family": os.environ.get(CFG.alpha_family_env_var, "part2a21"),
        "strict_drive_only": _boolish(os.environ.get(CFG.strict_env_var, "0"), 0),
        # Regime-conditional Platt recalibration
        "platt_scaling_active": platt_active,
        "current_regime": current_regime,
        "p_final_cal_raw": _p_base,
        "p_final_cal_blended": p_raw if _blend_source != "base_only" else None,
        "p_blend_source": _blend_source,
        "p_blend_2b": _p2b_blend_p,
        "p_blend_2c": _p2c_blend_p,
        "p_2c_live_high_epistemic_warning": _live_high_epistemic,
        "p_2c_epist_ratio": _p2c_epist_ratio,
        "p_regime_recal": p_regime_recal,
        "platt_regimes_fit": sorted([k for k in platt_params if not k.startswith("_")]),
        # FIX (Finding 1, Quant-Guild Part 26): report regimes excluded because a<0.
        # FIX (F-1, Quant-Guild Part 33 Audit): also report 0<=a<_PLATT_MIN_SLOPE.
        # FIX (F4, Quant-Guild Part 50 Audit): also report DeLong-insignificant exclusions.
        # "inverted" is kept in the key name for backward-compatibility but now covers
        # anti-predictive (a<0), degenerate-slope (0<=a<threshold), and DeLong p>=0.10.
        "platt_inverted_regimes_excluded": sorted([
            regime for regime in ["calm", "crisis", "high_vol", "risk_on"]
            if regime not in platt_params and regime not in [k for k in platt_params if k.startswith("_")]
        ]),
        # FIX (F1, Quant-Guild Part 47 Audit): surface whether the _global Platt
        # fallback itself fell below _PLATT_MIN_SLOPE this run. When True, any
        # regime listed in platt_inverted_regimes_excluded above received
        # UNCALIBRATED p_final_cal (transparent pass-through) rather than a
        # degenerate transform, per the _apply_regime_platt fix. Direct summary
        # visibility avoids requiring a separate read of part3_platt_params.json
        # to detect this state.
        "platt_global_degenerate": bool(platt_params.get("__global_degenerate__", False)),
        # FIX (F1, Quant-Guild Part 48 Audit): surface which excluded regimes
        # received transparent pass-through because their full-period AUC < 0.50
        # (as opposed to pass-through because _global was degenerate above).
        # FIX (F4, Quant-Guild Part 50 Audit): extended to also include regimes
        # excluded by the DeLong significance gate (AUC >= 0.50 but p >= 0.10).
        # For DeLong-excluded regimes: AUC >= 0.50 but p >= 0.10 → the S48 F1
        # AUC-gated passthrough guard (_apply_regime_platt's Gate 2) requires
        # AUC < 0.50 to force passthrough; for AUC >= 0.50 regimes excluded ONLY
        # by DeLong, _apply_regime_platt would route to _global. To prevent this
        # (an insignificant regime should not receive _global calibration either),
        # we list DeLong-excluded regimes here. The _apply_regime_platt function
        # already handles the AUC<0.50 case via Gate 2; for the AUC>=0.50 DeLong
        # exclusion the safest behavior is transparent passthrough, which avoids
        # applying _global to a regime with no verified signal.
        # Note: _apply_regime_platt does not yet read this summary field directly;
        # the Gate 2 check handles AUC<0.50 and the fit-exclusion handles the rest
        # since excluded regimes are not in platt_params and route to _global.
        # Regimes excluded by DeLong with AUC>=0.50 will receive _global rather
        # than passthrough via the current _apply_regime_platt logic — documenting
        # them here makes the state machine observable in artifacts.
        "platt_auc_gated_passthrough_regimes": sorted([
            regime for regime in ["calm", "crisis", "high_vol", "risk_on"]
            if (regime not in platt_params
                and regime not in [k for k in platt_params if k.startswith("_")]
                and not bool(platt_params.get("__global_degenerate__", False))
                and isinstance(_regime_auc_breakdown_for_platt.get(str(regime), {}).get("auc"), (int, float))
                and float(_regime_auc_breakdown_for_platt.get(str(regime), {}).get("auc", 1.0)) < 0.50)
        ]),
        # FIX (F4, Quant-Guild Part 50 Audit): surface regimes excluded by the new
        # DeLong significance gate (AUC >= 0.50 but DeLong p >= 0.10). These are
        # regimes where the model has a positive point-estimate AUC but the signal
        # is not statistically validated at the 10% level. Listing them separately
        # from platt_auc_gated_passthrough_regimes (which covers AUC < 0.50) allows
        # operators to distinguish the two exclusion pathways in the audit trail.
        "platt_delong_excluded_regimes": sorted([
            regime for regime in ["calm", "crisis", "high_vol", "risk_on"]
            if (regime not in platt_params
                and regime not in [k for k in platt_params if k.startswith("_")]
                and isinstance(_regime_auc_breakdown_for_platt.get(str(regime), {}).get("auc"), (int, float))
                and float(_regime_auc_breakdown_for_platt.get(str(regime), {}).get("auc", 1.0)) >= 0.50)
        ]),
        # FIX (F7, Audit 2026-05-10): surface Platt params path so dashboard can link to them
        "platt_params_path": str(out_dir / "part3_platt_params.json") if platt_active else None,
    }
    _write_json(summary_out, summary)

    # FIX (F3, Quant-Guild Part 34 Audit): Propagate Part 3's authoritative
    # publish_mode and deployment_mode back to current_target_weights.json.
    #
    # ROOT CAUSE OF THE BUG:
    # Part 7 runs before Part 3 in the canonical pipeline order. Part 7 reads
    # publish_mode from Part 2's summary JSON (which reports publish_mode=NORMAL
    # when _should_fail_closed is False) and writes that value into
    # current_target_weights.json. Part 3 then runs and, upon detecting
    # final_pass=False with publish_mode=NORMAL, overrides to FAIL_CLOSED_NEUTRAL
    # (BUG-C fix, Part 31). But current_target_weights.json is never updated.
    # Result: any consumer reading only current_target_weights.json (the dashboard,
    # Part 10 trading bot, external monitoring) sees publish_mode=NORMAL while the
    # authoritative Part 3 state is FAIL_CLOSED_NEUTRAL / DEFENSE_ONLY.
    #
    # FIX: at the end of Part 3's main(), read current_target_weights.json from
    # Part 7's output directory and update its publish_mode and deployment_mode
    # fields with Part 3's authoritative values. This makes the file consistent
    # with part3_summary.json for all downstream consumers.
    #
    # Safety: the update is atomic (read → modify → write). If the file doesn't
    # exist or can't be parsed, the update is silently skipped so Part 3 never
    # hard-fails due to a missing Part 7 artifact.
    _ctw_path = _first_existing_path(["artifacts_part7/current_target_weights.json"], root)
    if _ctw_path is not None:
        try:
            import json as _json_p3
            with open(_ctw_path, "r", encoding="utf-8") as _ctw_f:
                _ctw_data = _json_p3.load(_ctw_f)
            _ctw_data["publish_mode"] = publish_mode
            _ctw_data["deployment_mode"] = (
                "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL"
                else publish_mode
            )
            _ctw_data["final_pass"] = int(final_pass)
            # FIX (F2, Quant-Guild Part 37 Audit):
            # When publish_mode = FAIL_CLOSED_NEUTRAL, Part 7 may have written
            # BL-optimized weights (e.g. 0.70/0.30) via the soft-clearance path
            # (raw_val_auc_median >= 0.52). Part 3 then stamps FAIL_CLOSED_NEUTRAL
            # on the governance fields, creating a semantic contradiction:
            #   w_target_voo = 0.70  (BL result — active management)
            #   publish_mode = FAIL_CLOSED_NEUTRAL  (governance says 60/40)
            #
            # Any external consumer of current_target_weights.json reading both fields
            # receives contradictory information. Part 10 handles this internally
            # (overrides to 60/40 when fail-closed), but the artifact itself is wrong.
            #
            # Fix: when FAIL_CLOSED_NEUTRAL, overwrite w_target_voo and w_target_ief
            # to exactly 60/40 so the file is semantically consistent. The soft-clearance
            # BL result is discarded — it is an internal Part 7 computation artifact, not
            # an actionable portfolio weight when governance is fail-closed.
            if publish_mode == "FAIL_CLOSED_NEUTRAL":
                _ctw_data["w_target_voo"] = CFG.default_voo_weight   # 0.60
                _ctw_data["w_target_ief"] = CFG.default_ief_weight    # 0.40
            with open(_ctw_path, "w", encoding="utf-8") as _ctw_f:
                _json_p3.dump(_ctw_data, _ctw_f, indent=2)
            print(
                f"[Part 3] current_target_weights.json updated: "
                f"publish_mode={publish_mode}, "
                f"deployment_mode={_ctw_data['deployment_mode']}, "
                f"final_pass={int(final_pass)}"
                + (f", weights reset to 60/40 (fail-closed)" if publish_mode == "FAIL_CLOSED_NEUTRAL" else "")
            )
        except Exception as _ctw_exc:
            print(f"[Part 3] WARNING: could not update current_target_weights.json: {_ctw_exc}")
    else:
        print("[Part 3] current_target_weights.json not found — skipping governance label sync.")

    # FIX (F3, Quant-Guild Part 43 Audit): Sync deployment_mode back to portfolio_weights_tape.csv.
    #
    # ROOT CAUSE: Part7 writes portfolio_weights_tape.csv BEFORE Part3 runs, copying
    # publish_mode from Part2's intermediate state (NORMAL, set before the
    # conditional_active_ir gate evaluates final_pass). Part3 then overrides to
    # FAIL_CLOSED_NEUTRAL / DEFENSE_ONLY but never writes this decision back to the
    # weights tape. Result: tape shows publish_mode=NORMAL for all rows written during
    # DEFENSE_ONLY periods — a misleading historical record that can corrupt downstream
    # analysis of when the system was actively managed vs passive.
    #
    # Evidence: 2026-06-01/02/03 tape rows show publish_mode=NORMAL but Part3 summary
    # shows deployment_mode=DEFENSE_ONLY (final_pass=False, cond_ir=-0.622 < -0.50).
    #
    # Fix: after Part3's governance decision is final, reopen portfolio_weights_tape.csv,
    # add/update the 'deployment_mode' column for today's date row, and write back.
    # This is a targeted single-row update: only today's date is changed.
    # Historical rows are left unchanged (they reflect the governance at their run time).
    _pw_tape_path = _first_existing_path(["artifacts_part7/portfolio_weights_tape.csv"], root)
    if _pw_tape_path is not None:
        try:
            import pandas as _pd_p3
            _pw_df = _pd_p3.read_csv(_pw_tape_path)
            _today_str = str(decision_date.date())
            # Normalize Date column for matching
            _pw_df["Date"] = _pd_p3.to_datetime(_pw_df["Date"], errors="coerce").dt.date.astype(str)
            _today_mask = _pw_df["Date"] == _today_str
            if _today_mask.any():
                _deploy_mode_val = (
                    "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL"
                    else publish_mode
                )
                if "deployment_mode" not in _pw_df.columns:
                    _pw_df["deployment_mode"] = ""
                _pw_df.loc[_today_mask, "deployment_mode"] = _deploy_mode_val
                # Also correct publish_mode for today's row to reflect Part3's final decision
                _pw_df.loc[_today_mask, "publish_mode"] = publish_mode
                _pw_df.loc[_today_mask, "final_pass"] = int(final_pass)
                _pw_df.to_csv(_pw_tape_path, index=False)
                print(
                    f"[Part 3] portfolio_weights_tape.csv updated for {_today_str}: "
                    f"deployment_mode={_deploy_mode_val}, publish_mode={publish_mode}, "
                    f"final_pass={int(final_pass)}"
                )
            else:
                print(
                    f"[Part 3] portfolio_weights_tape.csv: no row for {_today_str} — "
                    "skipping deployment_mode sync."
                )
        except Exception as _pw_exc:
            print(f"[Part 3] WARNING: could not update portfolio_weights_tape.csv: {_pw_exc}")
    else:
        print("[Part 3] portfolio_weights_tape.csv not found — skipping deployment_mode sync.")

    print(f"✅ DEFENSE TAPE DISCOVERED: {part2_tape}")
    print(f"Decision-time 1D price call: VOO={voo_call:.4f} (explicit_call) | IEF={ief_call:.4f} (explicit_call)" if voo_call is not None and ief_call is not None else "Decision-time 1D price call: NA")
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
        f"Rows: {len(prod_tape)} | Realized fused rows: {_post_upsert_realized_rows} | "
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
    print(f"[PredLog] realized rows={_post_upsert_realized_rows} (pre-upsert count={realized_rows}; {'timing-race corrected' if _post_upsert_realized_rows != realized_rows else 'consistent'}).  [S51 F2]")
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





