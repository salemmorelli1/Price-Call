#!/usr/bin/env python3
# @title PART 9 — Live Attribution & Statistical Significance
# =============================================================================
# Industry-grade live performance analytics for PriceCallProject v2
#
# This is the part that makes the system TRUSTWORTHY.
# No backtest metric matters. Only this file tells you if the model is real.
#
# Responsibilities:
#   1. Consume the prediction log (live realized trades only)
#   2. Compute statistically valid live performance metrics
#   3. Diebold-Mariano test: model vs naive benchmark
#   4. Factor attribution: how much of return is explained by beta, duration, etc.
#   5. Model health diagnostics: feature drift, concept drift, calibration
#   6. Stopping rules: when to suspend live trading
# =============================================================================
from __future__ import annotations

import os
import dataclasses
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Part9Config:
    version: str = "V1"
    predlog_path: str = "./artifacts_part3/prediction_log.csv"
    part2_tape_path: str = "./artifacts_part2_g532/predictions/g532_final_consensus_tape.csv"
    part6_dir: str = "./artifacts_part6"
    out_dir: str = "./artifacts_part9"
    part8_cost_path: str = "./artifacts_part8/execution_cost_tape.csv"
    horizon: int = 7

    # Statistical thresholds
    t_stat_min: float = 2.0             # Minimum t-stat for edge significance
    t_stat_suspend: float = -1.5        # Suspend live trading below this
    max_drawdown_hard_stop: float = 0.15  # 15% max drawdown → hard stop
    min_live_n: int = 26                # Minimum observations to make statements

    # Calibration thresholds
    ece_warn: float = 0.08
    ece_suspend: float = 0.15
    brier_warn: float = 0.20
    brier_suspend: float = 0.25

    # Feature drift thresholds (Kolmogorov-Smirnov test)
    drift_ks_pvalue_warn: float = 0.05    # Flag if p < 0.05
    drift_ks_pvalue_suspend: float = 0.01 # Suspend if p < 0.01


CFG = Part9Config()

def _resolve_root() -> str:
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
    seen = set()
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
            return s
    return str(Path.cwd().resolve())


def _abs_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((Path(_resolve_root()) / path).resolve())



# ============================================================
# Statistical testing
# ============================================================

def t_stat_sign_accuracy(y_true: np.ndarray, p_pred: np.ndarray, base_rate: float) -> Dict:
    """
    Test if model's sign accuracy is statistically better than base rate.
    Null hypothesis: P(correct direction) = 0.5

    For a binary tail event:
    - Correct = model predicted high risk AND tail event occurred
              OR model predicted low risk AND no tail event
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    n = len(y)

    if n < 2:
        return {"n": n, "t_stat": np.nan, "p_value": np.nan, "significant": False}

    # Brier skill score vs base rate
    brier_model = float(np.mean((y - p) ** 2))
    brier_null = float(np.mean((y - base_rate) ** 2))
    bss = 1.0 - brier_model / (brier_null + 1e-10)  # Positive = better than base rate

    # Direction accuracy
    pred_up = p < base_rate  # Model predicts VOO outperforms
    actual_up = y < 0.5      # No tail event = VOO outperformed
    correct = (pred_up == actual_up).astype(float)
    accuracy = float(correct.mean())

    # t-test: is accuracy > 0.5?
    t, pval = stats.ttest_1samp(correct, 0.5)

    # AUC
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y.astype(int), p)) if len(np.unique(y.astype(int))) >= 2 else np.nan

    # AUC t-stat
    se_auc = float(np.sqrt(auc * (1 - auc) / max(n, 1)))
    t_auc = (auc - 0.5) / se_auc if se_auc > 0 else np.nan

    return {
        "n": n,
        "accuracy": accuracy,
        "t_stat_accuracy": float(t),
        "p_value_accuracy": float(pval),
        "auc": auc,
        "t_stat_auc": float(t_auc),
        "brier": brier_model,
        "brier_null": brier_null,
        "brier_skill_score": float(bss),
        "significant_5pct": bool(abs(t) > 1.96 or (np.isfinite(t_auc) and abs(t_auc) > 1.96)),
        "significant_1pct": bool(abs(t) > 2.58 or (np.isfinite(t_auc) and abs(t_auc) > 2.58)),
    }


def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_benchmark: np.ndarray,
    h: int = 1,
) -> Dict:
    """
    Diebold-Mariano test: Is model's forecast error significantly smaller than benchmark?

    Benchmark options:
    1. Naive: always predict base rate (p = 0.20)
    2. Historical average: predict rolling 63-day average

    H0: Model and benchmark have equal forecast accuracy
    H1: Model is significantly better (one-tailed)

    For weekly predictions (h=1 in our rebalanced dataset), standard DM applies.
    """
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_benchmark, dtype=float)
    finite = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[finite], e2[finite]
    n = len(e1)

    if n < 8:
        return {"n": n, "dm_stat": np.nan, "p_value": np.nan}

    # Loss differential
    d = e1**2 - e2**2  # Squared error loss
    d_bar = d.mean()
    d_var = d.var(ddof=1)

    if d_var <= 0:
        return {"n": n, "dm_stat": 0.0, "p_value": 0.5, "model_better": False}

    # DM statistic (Harvey, Leybourne, Newbold small-sample correction)
    dm = float(d_bar / np.sqrt(d_var / n))
    # One-tailed: positive DM = model worse, negative = model better
    p_value = float(stats.t.sf(-dm, df=n - 1))  # P(model better)

    return {
        "n": n,
        "dm_stat": dm,
        "p_value": p_value,
        "model_better": bool(dm < 0),
        "significant_5pct": bool(p_value < 0.05),
        "significant_1pct": bool(p_value < 0.01),
        "mean_loss_model": float(np.mean(e1**2)),
        "mean_loss_benchmark": float(np.mean(e2**2)),
        "mse_reduction_pct": float(100 * (1 - np.mean(e1**2) / np.mean(e2**2))),
    }


def ece_score(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        idx = (p >= edges[i]) & (p < (1.0 if i == bins - 1 else edges[i + 1]))
        if idx.sum() == 0:
            continue
        ece += (idx.sum() / len(y)) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


# ============================================================
# Feature drift detection
# ============================================================

def detect_feature_drift(
    historical_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    recent_window: int = 63,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test for each feature:
    H0: recent distribution = historical distribution
    Low p-value → feature has drifted → model may be stale
    """
    results = []
    hist = historical_features.dropna()
    rec = recent_features.dropna().tail(recent_window)

    for col in hist.columns:
        if col not in rec.columns:
            continue
        h_vals = hist[col].dropna().values
        r_vals = rec[col].dropna().values
        if len(h_vals) < 20 or len(r_vals) < 5:
            continue
        ks_stat, ks_pval = stats.ks_2samp(h_vals, r_vals)
        results.append({
            "feature": col,
            "ks_stat": float(ks_stat),
            "ks_pval": float(ks_pval),
            "hist_mean": float(h_vals.mean()),
            "recent_mean": float(r_vals.mean()),
            "mean_shift": float(r_vals.mean() - h_vals.mean()),
            "drifted": bool(ks_pval < 0.05),
            "severely_drifted": bool(ks_pval < 0.01),
        })

    df = pd.DataFrame(results).sort_values("ks_pval")
    n_drifted = int((df["drifted"]).sum()) if len(df) else 0
    print(f"[Part 9] Feature drift: {n_drifted}/{len(df)} features drifted (KS p<0.05)")
    return df


# ============================================================
# Performance attribution
# ============================================================

def factor_attribution(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Decompose strategy returns into:
    1. Beta to benchmark (market exposure)
    2. Alpha (residual after market)
    3. Factor loadings (if factor_returns provided: value, momentum, quality)
    4. Information Ratio of pure alpha
    """
    r = strategy_returns.dropna()
    b = benchmark_returns.reindex(r.index).dropna()
    idx = r.index.intersection(b.index)
    r, b = r[idx], b[idx]

    if len(r) < 8:
        return {"n": len(r), "alpha": np.nan, "beta": np.nan, "ir": np.nan}

    # OLS: r = alpha + beta * b + epsilon
    slope, intercept, r_val, p_val, se = stats.linregress(b.values, r.values)
    alpha_daily = float(intercept)
    beta = float(slope)
    r_squared = float(r_val ** 2)

    # Annualize alpha
    periods_per_year = 252.0 / 7  # weekly rebalancing
    alpha_ann = alpha_daily * periods_per_year

    # IR of pure alpha
    residuals = r.values - (alpha_daily + beta * b.values)
    ir_alpha = float(residuals.mean() / (residuals.std(ddof=1) + 1e-10) * np.sqrt(periods_per_year))

    result = {
        "n": len(r),
        "alpha_daily": alpha_daily,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "r_squared": r_squared,
        "ir_alpha": ir_alpha,
        "t_stat_alpha": float(intercept / (se + 1e-10)),
        "tracking_error_ann": float(residuals.std(ddof=1) * np.sqrt(periods_per_year)),
        "info_ratio": float(r.mean() / (r.std(ddof=1) + 1e-10) * np.sqrt(periods_per_year)),
    }

    # Maximum drawdown
    cumret = (1 + r).cumprod()
    rolling_max = cumret.expanding().max()
    drawdown = (cumret - rolling_max) / rolling_max
    result["max_drawdown"] = float(drawdown.min())
    result["current_drawdown"] = float(drawdown.iloc[-1]) if len(drawdown) > 0 else np.nan

    return result


# ============================================================
# Stopping rules
# ============================================================

def evaluate_stopping_rules(
    live_stats: Dict,
    calibration_stats: Dict,
    drift_df: pd.DataFrame,
    cfg: Part9Config,
) -> Dict:
    """
    Automated stopping rules. Returns:
    - status: ACTIVE | WARN | SUSPEND
    - reasons: list of triggered conditions
    """
    reasons = []
    status = "ACTIVE"

    n = live_stats.get("n", 0)
    if n < cfg.min_live_n:
        return {
            "status": "IMMATURE",
            "reasons": [f"Only {n} live observations ({cfg.min_live_n} required)"],
        }

    # 1. Statistical edge check
    t_auc = live_stats.get("t_stat_auc", np.nan)
    if np.isfinite(t_auc) and t_auc < cfg.t_stat_suspend:
        reasons.append(f"AUC t-stat={t_auc:.2f} < {cfg.t_stat_suspend} SUSPEND THRESHOLD")
        status = "SUSPEND"
    elif np.isfinite(t_auc) and t_auc < cfg.t_stat_min:
        reasons.append(f"AUC t-stat={t_auc:.2f} < {cfg.t_stat_min} WARN THRESHOLD")
        if status == "ACTIVE":
            status = "WARN"

    # 2. Drawdown check
    max_dd = live_stats.get("max_drawdown", np.nan)
    if np.isfinite(max_dd) and abs(max_dd) > cfg.max_drawdown_hard_stop:
        reasons.append(f"Max drawdown={max_dd:.1%} > {cfg.max_drawdown_hard_stop:.1%} HARD STOP")
        status = "SUSPEND"

    # 3. Calibration check
    ece = calibration_stats.get("ece", np.nan)
    if np.isfinite(ece) and ece > cfg.ece_suspend:
        reasons.append(f"ECE={ece:.4f} > {cfg.ece_suspend} SUSPEND")
        status = "SUSPEND"
    elif np.isfinite(ece) and ece > cfg.ece_warn:
        reasons.append(f"ECE={ece:.4f} > {cfg.ece_warn} WARN")
        if status == "ACTIVE":
            status = "WARN"

    # 4. Feature drift check
    if len(drift_df) > 0:
        severe_drifts = drift_df[drift_df.get("severely_drifted", False)]["feature"].tolist() if "severely_drifted" in drift_df.columns else []
        if len(severe_drifts) > 2:
            reasons.append(f"Feature drift: {len(severe_drifts)} features severely drifted: {severe_drifts[:3]}")
            if status == "ACTIVE":
                status = "WARN"

    if not reasons:
        reasons.append("All checks passed")

    return {"status": status, "reasons": reasons}


# ============================================================
# Main report
# ============================================================

def generate_live_report(cfg: Part9Config) -> Dict:
    """Read prediction log and generate the full live performance report."""
    if not os.path.exists(cfg.predlog_path):
        return {"error": f"Prediction log not found at {cfg.predlog_path}"}

    predlog = pd.read_csv(cfg.predlog_path)
    predlog["decision_date"] = pd.to_datetime(predlog["decision_date"], errors="coerce")

    # Load Part 2 tape for backtest comparison
    tape = None
    if os.path.exists(cfg.part2_tape_path):
        tape = pd.read_csv(cfg.part2_tape_path)
        tape["Date"] = pd.to_datetime(tape["Date"], errors="coerce")

    # Isolate realized rows
    realized_cols = ["px_voo_realized", "voo_realized"]
    real_col = next((c for c in realized_cols if c in predlog.columns), None)
    if real_col is None:
        return {"error": "No realized price column found in prediction log", "n_live": 0}

    realized = predlog[predlog[real_col].notna()].copy()
    n_live = len(realized)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_predictions": len(predlog),
        "n_live_realized": n_live,
        "status": "IMMATURE" if n_live < cfg.min_live_n else None,
    }

    if n_live < 2:
        report["message"] = (
            f"Only {n_live} realized predictions available. "
            f"Accumulate {cfg.min_live_n} before making statistical inferences. "
            f"At weekly rebalancing, this takes {cfg.min_live_n / 52:.1f} years."
        )
        return report

    # Prediction accuracy from predlog
    voo_pred = predlog.get("px_voo_call_7d", pd.Series(dtype=float))
    voo_real = predlog.get(real_col, pd.Series(dtype=float))
    if not voo_pred.empty and not voo_real.empty:
        errors = (voo_pred - voo_real).dropna()
        mae_live = float(errors.abs().mean())
        rmse_live = float(np.sqrt((errors ** 2).mean()))
        report["voo_mae_live"] = mae_live
        report["voo_rmse_live"] = rmse_live
        report["voo_hit_rate"] = float((np.sign(voo_pred) == np.sign(voo_real)).mean()) if len(voo_pred) > 0 else np.nan

    # Load P2 tape for classification metrics
    if tape is not None and "p_final_cal" in tape.columns and "y_rel_tail_voo_vs_ief" in tape.columns:
        tape_real = tape[tape["y_avail"] == 1].dropna(subset=["p_final_cal", "y_rel_tail_voo_vs_ief"])
        base_rate = float(tape_real["y_rel_tail_voo_vs_ief"].mean())

        live_stat = t_stat_sign_accuracy(
            tape_real["y_rel_tail_voo_vs_ief"].values,
            tape_real["p_final_cal"].values,
            base_rate,
        )
        report["classification_stats_backtest"] = live_stat
        report["base_rate"] = base_rate

        # Naive benchmark: always predict base rate
        naive_errors = tape_real["p_final_cal"].values - base_rate
        model_errors = tape_real["p_final_cal"].values - tape_real["y_rel_tail_voo_vs_ief"].values
        dm_result = diebold_mariano_test(model_errors, naive_errors)
        report["diebold_mariano_vs_naive"] = dm_result

        # Calibration (backtest)
        ece_bt = ece_score(
            tape_real["y_rel_tail_voo_vs_ief"].values,
            tape_real["p_final_cal"].values
        )
        report["calibration_backtest"] = {
            "ece": ece_bt,
            "brier": float(np.mean((tape_real["y_rel_tail_voo_vs_ief"].values
                                    - tape_real["p_final_cal"].values)**2)),
        }

        # Strategy performance attribution
        if "strategy_ret_net" in tape_real.columns and "benchmark_ret" in tape_real.columns:
            attr = factor_attribution(
                tape_real["strategy_ret_net"],
                tape_real["benchmark_ret"],
            )
            report["factor_attribution"] = attr

    # Health status
    stopping = evaluate_stopping_rules(
        live_stat if "classification_stats_backtest" in report else {},
        report.get("calibration_backtest", {}),
        pd.DataFrame(),
        cfg,
    )
    report["health_status"] = stopping["status"]
    report["health_reasons"] = stopping["reasons"]

    # Print summary
    print("=" * 70)
    print("PART 9 — Live Attribution Report")
    print("=" * 70)
    print(f"Live realized: {n_live} / {len(predlog)} predictions")
    print(f"Health status: {report['health_status']}")
    for reason in stopping["reasons"]:
        print(f"  • {reason}")

    if "classification_stats_backtest" in report:
        cs = report["classification_stats_backtest"]
        print(f"\nBacktest classification (N={cs['n']}):")
        print(f"  AUC:         {cs.get('auc', np.nan):.4f}  (t={cs.get('t_stat_auc', np.nan):.2f})")
        print(f"  Direction:   {cs.get('accuracy', np.nan):.2%}  (t={cs.get('t_stat_accuracy', np.nan):.2f})")
        print(f"  BSS:         {cs.get('brier_skill_score', np.nan):.4f}")
        print(f"  Significant: {'YES ✓' if cs.get('significant_5pct') else 'NO ✗'} (5%) | "
              f"{'YES ✓' if cs.get('significant_1pct') else 'NO ✗'} (1%)")

    if "diebold_mariano_vs_naive" in report:
        dm = report["diebold_mariano_vs_naive"]
        print(f"\nDiebold-Mariano vs naive baseline:")
        print(f"  DM stat:     {dm.get('dm_stat', np.nan):.3f}")
        print(f"  p-value:     {dm.get('p_value', np.nan):.4f}")
        print(f"  MSE reduction: {dm.get('mse_reduction_pct', np.nan):.1f}%")
        print(f"  Significant: {'YES ✓' if dm.get('significant_5pct') else 'NO ✗'}")

    return report


def main() -> int:
    cfg = Part9Config()
    cfg = dataclasses.replace(cfg, predlog_path=_abs_path(cfg.predlog_path))
    cfg = dataclasses.replace(cfg, part2_tape_path=_abs_path(cfg.part2_tape_path))
    cfg = dataclasses.replace(cfg, part6_dir=_abs_path(cfg.part6_dir))
    cfg = dataclasses.replace(cfg, out_dir=_abs_path(cfg.out_dir))
    cfg = dataclasses.replace(cfg, part8_cost_path=_abs_path(cfg.part8_cost_path))
    os.makedirs(cfg.out_dir, exist_ok=True)

    report = generate_live_report(cfg)

    out_path = os.path.join(cfg.out_dir, "live_attribution_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ PART 9 COMPLETE → {out_path}")
    return 0


if __name__ == "__main__":
    main()

