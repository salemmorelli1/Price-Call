"""Small dependency-light statistical tests shared by optional model sleeves."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


def delong_auc_statistics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Read-only DeLong AUC uncertainty for the independently reviewed cohort.

    Var(AUC) = sample_var(V10)/n_positive + sample_var(V01)/n_negative.
    Tied scores contribute one half. This is the structural-component estimator,
    not the older Q1/Q2 approximation in the frozen forecasting implementation.
    Its normal approximation assumes independent observations; this report cannot
    authorize allocation or replace the separate time-series integrity review.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or y.shape != probability.shape:
        raise ValueError("AUC outcomes and probabilities must be aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(probability).all():
        raise ValueError("AUC input must be finite")
    if not np.isin(y, (0, 1)).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("AUC requires binary labels and bounded probabilities")
    positive, negative = probability[y == 1], probability[y == 0]
    n1, n0 = len(positive), len(negative)
    result = {
        "auc": float("nan"), "n": len(y), "n1": n1, "n0": n0,
        "se": float("nan"), "z": float("nan"),
        "p_one_sided": float("nan"), "inference_eligible": False,
        "method": "delong_structural_components_v1",
        "serial_dependence_adjusted": False,
    }
    if not n1 or not n0:
        return result
    # Search sorted scores instead of allocating an n_positive by n_negative matrix.
    pos_sorted, neg_sorted = np.sort(positive), np.sort(negative)
    left = np.searchsorted(neg_sorted, positive, side="left")
    right = np.searchsorted(neg_sorted, positive, side="right")
    v10 = (left + right) / (2.0 * n0)
    left = np.searchsorted(pos_sorted, negative, side="left")
    right = np.searchsorted(pos_sorted, negative, side="right")
    v01 = 1.0 - (left + right) / (2.0 * n1)
    auc = float(v10.mean())
    result["auc"] = auc
    if n1 < 5 or n0 < 5 or len(y) < 20:
        return result
    variance = float(np.var(v10, ddof=1) / n1 + np.var(v01, ddof=1) / n0)
    se = float(np.sqrt(variance))
    result["se"] = se
    # Zero empirical variance cannot support a normal-approximation significance claim.
    if not np.isfinite(se) or se <= 0:
        return result
    z = (auc - 0.5) / se
    return {**result, "z": z, "p_one_sided": float(norm.sf(z)),
            "inference_eligible": True}


def uncertainty_gate_statistics(wf_df: pd.DataFrame) -> dict[str, Any]:
    """Test paired fold-level ECE gaps with a one-sided sign permutation."""
    if not {"ece_high_spread", "ece_low_spread"}.issubset(wf_df.columns):
        differences = np.asarray([], dtype=float)
    else:
        high = pd.to_numeric(wf_df["ece_high_spread"], errors="coerce")
        low = pd.to_numeric(wf_df["ece_low_spread"], errors="coerce")
        differences = (high - low).to_numpy(dtype=float)
        differences = differences[np.isfinite(differences)]
    n_folds = int(differences.size)
    observed = float(np.mean(differences)) if n_folds else float("nan")
    minimum_folds = 8
    effect_minimum = 0.002
    alpha = 0.10
    p_value = float("nan")
    method = "insufficient_folds"
    if n_folds:
        if n_folds <= 20:
            extreme = 0
            total = 1 << n_folds
            for signs in itertools.product((-1.0, 1.0), repeat=n_folds):
                permuted = float(np.mean(differences * np.asarray(signs)))
                extreme += int(permuted >= observed - 1e-15)
            p_value = extreme / total
            method = "exact_paired_sign_permutation"
        else:
            rng = np.random.default_rng(532)
            signs = rng.choice((-1.0, 1.0), size=(50_000, n_folds))
            permuted = np.mean(signs * differences, axis=1)
            p_value = float((np.count_nonzero(permuted >= observed) + 1) / 50_001)
            method = "monte_carlo_paired_sign_permutation_50000"
    inference_eligible = n_folds >= minimum_folds
    validated = bool(
        inference_eligible
        and np.isfinite(observed)
        and observed > effect_minimum
        and np.isfinite(p_value)
        and p_value <= alpha
    )
    return {
        "n_folds": n_folds,
        "minimum_folds": minimum_folds,
        "ece_gap": observed,
        "effect_minimum": effect_minimum,
        "p_one_sided": p_value,
        "alpha": alpha,
        "method": method,
        "inference_eligible": inference_eligible,
        "validated": validated,
    }
