import pandas as pd


def test_uncertainty_claim_requires_paired_statistical_evidence():
    from statistical_tests import uncertainty_gate_statistics

    weak = pd.DataFrame({
        "ece_high_spread": [0.051, 0.049, 0.052, 0.048, 0.051, 0.049, 0.052, 0.048],
        "ece_low_spread": [0.050] * 8,
    })
    result = uncertainty_gate_statistics(weak)
    assert result["inference_eligible"] is True
    assert result["validated"] is False


def test_uncertainty_claim_passes_only_prespecified_effect_and_pvalue():
    from statistical_tests import uncertainty_gate_statistics

    strong = pd.DataFrame({
        "ece_high_spread": [0.08] * 8,
        "ece_low_spread": [0.05] * 8,
    })
    result = uncertainty_gate_statistics(strong)
    assert result["ece_gap"] > result["effect_minimum"]
    assert result["p_one_sided"] <= result["alpha"]
    assert result["validated"] is True
