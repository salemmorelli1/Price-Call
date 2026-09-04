"""Small dependency-light statistical tests shared by optional model sleeves."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


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
