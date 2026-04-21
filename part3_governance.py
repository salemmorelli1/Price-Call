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


def _load_alpha_status(alpha_tape_df: pd.DataFrame, alpha_summary_json: Dict[str, Any]) -> Dict[str, Any]:
    latest_row = _last_valid_row(alpha_tape_df) if not alpha_tape_df.empty else pd.Series(dtype=object)

    # ── realized_dates ──────────────────────────────────────────────────────
    # Primary source: alpha_summary_json["realized_dates"] — set by Part 2A
    # to the count of historical dates where realized returns are available.
    # Fallback chain: tape row columns → len(tape).
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
            "p_final_cal", "base_rate", "raw_val_auc", "tail_threshold",
            # publish_mode: raw Part 2 governance value (FAIL_CLOSED_NEUTRAL / NORMAL).
            # deployment_mode: user-facing operational label (DEFENSE_ONLY / NORMAL).
            # Keeping both columns avoids the cross-file field collision where predlog
            # previously aliased FAIL_CLOSED_NEUTRAL → DEFENSE_ONLY in the publish_mode
            # column, creating a mismatch with part3_summary.json's publish_mode field.
            "publish_mode", "deployment_mode", "final_pass",
            "latest_alpha_state", "alpha_live",
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
        "p_final_cal": _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None)),
        "base_rate": _safe_float(_row_value(defense_row, ["T", "base_rate", "b"], None)),
        "raw_val_auc": _safe_float(_row_value(defense_row, ["raw_val_auc"], _json_value(part2_summary, ["raw_val_auc_median"], None))),
        # FIX (Finding 14 / Finding 26, Audit 2026-04-21):
        # Write the live row's actual dynamic threshold (from Part 1's rolling-quantile
        # label) rather than the fixed summary JSON value (-0.00567, H=7 formula).
        # Part 9 uses this value to reconstruct the live binary label for realized rows.
        # The classification model was trained on the rolling-quantile label; Part 9
        # must evaluate it against the same definition.
        # Priority: tail_threshold_dynamic from the defense_row (if present) →
        # signal_q_threshold (rolling quantile threshold in the consensus tape) →
        # fallback to summary JSON tail_event_threshold.
        "tail_threshold": _safe_float(
            _row_value(defense_row, ["tail_threshold_dynamic", "signal_q_threshold"], None)
            or _json_value(part2_summary, ["tail_event_threshold"], None)
        ),
        # publish_mode: raw governance value, consistent with part3_summary.json.
        # deployment_mode: user-facing operational label (DEFENSE_ONLY when fail-closed).
        # Separating these eliminates the prior cross-file field-name collision.
        "publish_mode": publish_mode,
        "deployment_mode": "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL" else publish_mode,
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


# ============================================================
# Regime-conditional Platt scaling
# ============================================================

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

        # Per-regime fit
        for regime in merged["regime_label"].dropna().unique():
            sub = merged[merged["regime_label"] == regime].copy()
            if len(sub) < 30:
                continue
            p = sub["p_final_cal"].clip(0.01, 0.99).values
            y = sub["y_rel_tail_voo_vs_ief"].values
            X = _logit(p).reshape(-1, 1)
            lr = _LogisticRegression(C=1e4, max_iter=2000, solver="lbfgs")
            lr.fit(X, y)
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])
            params[regime] = (a, b)
            print(f"[Part 3] Platt({regime:12s}): a={a:.4f}  b={b:.4f}  n={len(sub)}")

        # Global fallback (all regimes combined)
        p_all = merged["p_final_cal"].clip(0.01, 0.99).values
        y_all = merged["y_rel_tail_voo_vs_ief"].values
        lr_global = _LogisticRegression(C=1e4, max_iter=2000, solver="lbfgs")
        lr_global.fit(_logit(p_all).reshape(-1, 1), y_all)
        params["_global"] = (float(lr_global.coef_[0][0]), float(lr_global.intercept_[0]))
        print(f"[Part 3] Platt(_global    ): a={params['_global'][0]:.4f}  b={params['_global'][1]:.4f}  n={len(merged)}")

        return params

    except Exception as e:
        print(f"[Part 3] Platt scaling fit failed: {e} — using raw p_final_cal")
        return {}


def _apply_regime_platt(
    p_cal: Optional[float],
    regime_label: str,
    params: Dict[str, Tuple[float, float]],
) -> Optional[float]:
    """Apply regime-conditional Platt scaling to a single probability.

    Returns None if p_cal is None or params is empty (transparent fallback).
    """
    if not HAVE_PLATT or not params or p_cal is None or not math.isfinite(p_cal):
        return p_cal
    p_clipped = max(0.01, min(0.99, float(p_cal)))
    logit_p = float(_logit(p_clipped))
    if regime_label in params:
        a, b = params[regime_label]
    elif "_global" in params:
        a, b = params["_global"]
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
        raise RuntimeError("Part 2 final_pass is False and publish_mode is not FAIL_CLOSED_NEUTRAL")

    defense_row = _last_valid_row(defense_df)
    decision_date = pd.to_datetime(_row_value(defense_row, ["Date", "decision_date", "asof_date"]), errors="coerce")
    if pd.isna(decision_date):
        decision_date = pd.Timestamp.today().normalize()
    target_date = decision_date + pd.tseries.offsets.BDay(1)

    voo_call, ief_call = _extract_latest_price_call(defense_row)
    alpha_status = _load_alpha_status(alpha_tape_df, alpha_summary_json)
    alpha_positions_latest = _extract_alpha_positions(alpha_positions_df)

    # Compute regime-recalibrated probability for the current live row
    p_raw = _safe_float(_row_value(defense_row, ["p_final_cal", "p_final_g5"], None))
    current_regime = str(_row_value(defense_row, ["regime_label"], "unknown"))
    p_regime_recal: Optional[float] = _apply_regime_platt(p_raw, current_regime, platt_params)

    prod_tape = _prepare_production_tape(defense_df, part2_summary, alpha_status, alpha_summary_json)
    gov_df = _build_governance_df(decision_date, part2_summary, alpha_status)
    alloc_df, max_dev = _build_fusion_allocations(decision_date, defense_row, alpha_positions_latest, alpha_status, part7_weights)

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

    perf = _extract_performance_metrics(defense_df, alpha_summary_json, alpha_status)
    dist = _alpha_distribution(alpha_tape_df, alpha_status)
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
        "publish_mode": publish_mode,
        "deployment_mode": "DEFENSE_ONLY" if publish_mode == "FAIL_CLOSED_NEUTRAL" else publish_mode,
        "final_pass": int(final_pass),
        "latest_alpha_state": alpha_status["latest_state"],
        "latest_alpha_state_display": alpha_status["display_state"],
        # FIX (Finding 12, Audit 2026-04-21):
        # "realized_dates" is ambiguous: it sounds like live prediction-log rows with
        # realized prices, but actually counts historical tape rows where the backtest
        # labels are revealed (2020-2026). Operators reading this field incorrectly
        # concluded 381 live predictions had been evaluated. Renamed to make the
        # distinction unambiguous. prediction_log_realized_rows is the live count.
        "alpha_tape_historical_realized_dates": alpha_status["realized_dates"],
        "realized_dates": alpha_status["realized_dates"],  # kept for backward-compat; deprecated
        "realized_dates_note": "Backtest tape rows only — NOT live realized predictions. See prediction_log_realized_rows for live count.",
        "budget_mult": alpha_status["budget_mult"],
        "drift_rate": alpha_status["drift_rate"],
        "quality_ok": alpha_status["quality_ok"],
        "drift_ok": alpha_status["drift_ok"],
        "trial_gate_open": alpha_status["trial_gate_open"],
        "fused_gate_open": alpha_status["fused_gate_open"],
        "promotion_ready": alpha_status["promotion_ready"],
        "alpha_blockers": alpha_status["blockers"],
        "rows": int(len(prod_tape)),
        "rows_realized_fused": realized_rows,
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
        "p_final_cal_raw": p_raw,
        "p_regime_recal": p_regime_recal,
        "platt_regimes_fit": sorted([k for k in platt_params if not k.startswith("_")]),
    }
    _write_json(summary_out, summary)

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
        f"Rows: {len(prod_tape)} | Realized fused rows: {realized_rows} | "
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




    



    
