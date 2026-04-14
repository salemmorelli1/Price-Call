#!/usr/bin/env python3
from __future__ import annotations

import sys as _sys

import json
import os
import dataclasses
from pathlib import Path
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm
    HAVE_HMM = True
except Exception:
    try:
        import subprocess as _subprocess
        _subprocess.run(
            [_sys.executable, "-m", "pip", "install", "hmmlearn", "-q"],
            capture_output=True,
            check=False,
        )
        from hmmlearn import hmm
        HAVE_HMM = True
    except Exception:
        hmm = None
        HAVE_HMM = False
        print("[Part 6] hmmlearn unavailable after install attempt — falling back to GMM regime.")

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False


def _resolve_root() -> str:
    candidates = []
    env_root = os.environ.get("PRICECALL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/content/drive/MyDrive/PriceCallProject"))
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


@dataclass(frozen=True)
class Part6Config:
    version: str = "V2_DAILY_CANONICAL"
    part0_dir: str = "artifacts_part0"
    out_dir: str = "artifacts_part6"
    horizon: int = 1

    n_regimes: int = 3
    hmm_covariance_type: str = "full"
    hmm_n_iter: int = 500
    hmm_min_train_rows: int = 252
    seed: int = 42

    regime_features: Tuple[str, ...] = (
        "vix_z21",
        "vix_term_ratio",
        "vix_rv_gap",
        "hy_ig_z21",
        "spread_vol10",
        "duration_spread_z21",
        "breadth_rsp_voo_z21",
        "yield_curve_2s10s",
        "hy_spread_fred",
    )
    regime_features_nofed: Tuple[str, ...] = (
        "vix_z21",
        "vix_term_ratio",
        "vix_rv_gap",
        "hy_ig_z21",
        "spread_vol10",
        "duration_spread_z21",
        "breadth_rsp_voo_z21",
        "skew_z21",
    )


CFG = Part6Config()

KNOWN_FOMC_DATES_2020_2026 = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


def build_fomc_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.to_datetime(index)
    fomc = pd.to_datetime(KNOWN_FOMC_DATES_2020_2026)
    out = pd.DataFrame(index=idx)
    out.index.name = "Date"

    days_to_fomc = []
    fomc_in_7d = []
    fomc_this_week = []
    for dt in idx:
        future = fomc[fomc >= dt]
        d = int((future.min() - dt).days) if len(future) else 999
        days_to_fomc.append(d)
        fomc_in_7d.append(int(d <= 7))
        fomc_this_week.append(int(-2 <= d <= 7))
    out["days_to_fomc"] = days_to_fomc
    out["fomc_in_7d"] = fomc_in_7d
    out["fomc_this_week"] = fomc_this_week

    is_nfp = (idx.weekday == 4) & (idx.day <= 7)
    nfp_dates = idx[is_nfp]
    days_to_nfp = []
    for dt in idx:
        future = nfp_dates[nfp_dates >= dt]
        days_to_nfp.append(int((future.min() - dt).days) if len(future) else 999)
    out["days_to_nfp"] = days_to_nfp
    out["nfp_in_7d"] = (out["days_to_nfp"] <= 7).astype(int)
    return out


def _load_part0_features(cfg: Part6Config) -> pd.DataFrame:
    parquet_path = os.path.join(cfg.part0_dir, "features_full.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        df = df.sort_index()
        df.index.name = "Date"
        return df

    duckdb_path = os.path.join(cfg.part0_dir, "market_data.duckdb")
    if HAVE_DUCKDB and os.path.exists(duckdb_path):
        con = duckdb.connect(duckdb_path, read_only=True)
        try:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            target = "features_full" if "features_full" in tables else ("features" if "features" in tables else None)
            if target is None:
                raise FileNotFoundError("DuckDB found but no features_full/features table present.")
            df = con.execute(f"SELECT * FROM {target} ORDER BY Date").df()
        finally:
            con.close()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        df.index.name = "Date"
        return df

    raise FileNotFoundError(
        f"Part 0 features not found in either {parquet_path} or {duckdb_path}. Run Part 0 first."
    )


def _label_regimes_by_vol(cluster_labels: np.ndarray, feature_df: pd.DataFrame, vol_col: str = "vix_z21") -> Dict[int, str]:
    if vol_col not in feature_df.columns:
        return {0: "risk_on", 1: "high_vol", 2: "crisis"}
    means = {}
    for k in np.unique(cluster_labels):
        mask = cluster_labels == k
        means[int(k)] = float(np.nanmean(feature_df.loc[mask, vol_col].values))
    ordered = sorted(means.keys(), key=lambda k: means[k])
    names = {}
    if len(ordered) >= 1:
        names[ordered[0]] = "risk_on"
    if len(ordered) >= 2:
        names[ordered[1]] = "high_vol"
    if len(ordered) >= 3:
        names[ordered[2]] = "crisis"
    return names


class RegimeEngine:
    def __init__(self, cfg: Part6Config = CFG):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.model_type: str = "none"
        self.feature_cols: List[str] = []
        self.regime_map: Dict[int, str] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        preferred = [c for c in self.cfg.regime_features if c in df.columns]
        if len(preferred) >= 3:
            return preferred
        fallback = [c for c in self.cfg.regime_features_nofed if c in df.columns]
        if len(fallback) >= 3:
            return fallback
        raise RuntimeError(
            f"Insufficient regime features available. Preferred found={preferred}, fallback found={fallback}."
        )

    def fit(self, df: pd.DataFrame) -> None:
        self.feature_cols = self._select_features(df)
        X = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) < self.cfg.hmm_min_train_rows:
            raise RuntimeError(
                f"Insufficient data to fit regime model: {len(X)} rows < {self.cfg.hmm_min_train_rows} required"
            )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)

        labels = None
        if HAVE_HMM:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.cfg.n_regimes,
                    covariance_type=self.cfg.hmm_covariance_type,
                    n_iter=self.cfg.hmm_n_iter,
                    random_state=self.cfg.seed,
                )
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                self.transition_matrix = np.asarray(model.transmat_, dtype=float)
                self.model = model
                self.model_type = "hmm"
                print(f"[Part 6] HMM converged in {int(model.monitor_.iter)} iterations")
            except Exception as e:
                print(f"[Part 6] HMM failed ({e}), falling back to GMM")

        if self.model is None:
            model = GaussianMixture(
                n_components=self.cfg.n_regimes,
                covariance_type="full",
                random_state=self.cfg.seed,
                n_init=5,
            )
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            self.model = model
            self.model_type = "gmm"
            self.transition_matrix = None

        feat_notna = X.reset_index(drop=True)
        self.regime_map = _label_regimes_by_vol(labels, feat_notna, "vix_z21" if "vix_z21" in self.feature_cols else self.feature_cols[0])
        self.is_fitted = True

        for cid, name in self.regime_map.items():
            frac = float((labels == cid).mean())
            print(f"  Regime {cid} ({name:12s}): {frac:.1%} of history")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("RegimeEngine not fitted. Call fit() first.")

        out = pd.DataFrame(index=df.index)
        out.index.name = "Date"

        X = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        mask = X.notna().all(axis=1)
        Xg = X.loc[mask]
        if Xg.empty:
            out["regime_label"] = "unknown"
            out["regime_id"] = -1
            for name in ["risk_on", "high_vol", "crisis"]:
                out[f"regime_prob_{name}"] = np.nan
            out["regime_persistence"] = np.nan
            out["transition_prob_crisis"] = np.nan
            return out

        X_scaled = self.scaler.transform(Xg.values)
        if self.model_type == "hmm":
            labels = self.model.predict(X_scaled)
            probs = self.model.predict_proba(X_scaled)
        else:
            labels = self.model.predict(X_scaled)
            probs = self.model.predict_proba(X_scaled)

        result_labels = np.full(len(df), "unknown", dtype=object)
        result_ids = np.full(len(df), -1, dtype=int)
        result_probs = np.full((len(df), self.cfg.n_regimes), np.nan)
        positions = np.where(mask.values)[0]
        for i, pos in enumerate(positions):
            result_labels[pos] = self.regime_map.get(int(labels[i]), f"regime_{int(labels[i])}")
            result_ids[pos] = int(labels[i])
            result_probs[pos, :] = probs[i]

        out["regime_label"] = result_labels
        out["regime_id"] = result_ids
        for cid, name in self.regime_map.items():
            out[f"regime_prob_{name}"] = result_probs[:, cid]
        for name in ["risk_on", "high_vol", "crisis"]:
            col = f"regime_prob_{name}"
            if col not in out.columns:
                out[col] = np.nan

        regime_series = pd.Series(result_labels, index=df.index)
        persist = []
        for i in range(len(regime_series)):
            hist = regime_series.iloc[max(0, i - 9): i + 1]
            cur = regime_series.iloc[i]
            if len(hist) < 3 or cur == "unknown":
                persist.append(np.nan)
            else:
                persist.append(float((hist == cur).mean()))
        out["regime_persistence"] = persist

        if self.transition_matrix is not None:
            crisis_ids = [k for k, v in self.regime_map.items() if v == "crisis"]
            crisis_id = crisis_ids[0] if crisis_ids else None
            trans = []
            for rid in result_ids:
                if crisis_id is None or rid < 0 or rid >= self.transition_matrix.shape[0]:
                    trans.append(np.nan)
                else:
                    trans.append(float(self.transition_matrix[rid, crisis_id]))
            out["transition_prob_crisis"] = trans
        else:
            out["transition_prob_crisis"] = np.nan
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Part 6] Regime engine saved to {path}")


def main() -> int:
    cfg = CFG
    cfg = dataclasses.replace(cfg, part0_dir=_abs_path(cfg.part0_dir))
    cfg = dataclasses.replace(cfg, out_dir=_abs_path(cfg.out_dir))
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 70)
    print("PART 6 — Regime & Macro Engine v1")
    print("=" * 70)

    features = _load_part0_features(cfg)
    print(f"[Part 6] Loaded {len(features)} rows × {len(features.columns)} features from Part 0")

    engine = RegimeEngine(cfg)
    engine.fit(features)
    regime_df = engine.predict(features)

    calendar = build_fomc_calendar(features.index)
    regime_df = regime_df.join(calendar, how="left")

    regime_path = os.path.join(cfg.out_dir, "regime_history.parquet")
    regime_df.to_parquet(regime_path)
    feat_used_path = os.path.join(cfg.out_dir, "regime_features_used.parquet")
    features[engine.feature_cols].to_parquet(feat_used_path)

    engine_path = os.path.join(cfg.out_dir, "regime_engine.pkl")
    engine.save(engine_path)

    meta = {
        "version": cfg.version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_type": engine.model_type,
        "n_regimes": cfg.n_regimes,
        "feature_cols": engine.feature_cols,
        "regime_map": engine.regime_map,
        "transition_matrix": engine.transition_matrix.tolist() if engine.transition_matrix is not None else None,
        "regime_distribution": regime_df["regime_label"].value_counts(normalize=True).to_dict(),
        "fomc_dates_included": len(KNOWN_FOMC_DATES_2020_2026),
        "source_part0_dir": cfg.part0_dir,
    }
    with open(os.path.join(cfg.out_dir, "part6_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\nRegime distribution:")
    print(regime_df["regime_label"].value_counts(normalize=True))
    print("\n✅ PART 6 COMPLETE")
    print(f"   Model type:      {engine.model_type.upper()}")
    print(f"   Regime features: {len(engine.feature_cols)}")
    print("   Outputs:         regime_history.parquet, regime_features_used.parquet, part6_meta.json, regime_engine.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
