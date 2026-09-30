import numpy as np

from src.extended_data.short_horizon_reversal import P, weights


def test_cross_sectional_weights_are_neutral_and_budgeted():
    prev = np.linspace(-0.1, 0.1, P)
    w = weights(prev, "cross_sectional_5x5", np.ones(P, dtype=bool))
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(np.abs(w).sum(), 0.90)
    assert np.count_nonzero(w > 0) == 5
    assert np.count_nonzero(w < 0) == 5


def test_zero_volume_assets_are_not_traded():
    prev = np.linspace(-0.1, 0.1, P)
    valid = np.ones(P, dtype=bool); valid[:4] = False
    w = weights(prev, "directional_sign", valid)
    assert np.all(w[:4] == 0.0)
    assert np.isclose(np.abs(w).sum(), 0.90)
