import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from audit_ridge_candidate import (
    CONFIG, SYMBOLS, fit_ridge, funding_cashflow, make_features, repair_trigger,
)


def test_repair_direction_has_counterexample_and_fix():
    # Long entered at -4%; -6% is farther away, -2% is a 50% repair.
    assert not repair_trigger(1, -0.06, -0.04)
    assert repair_trigger(1, -0.02, -0.04)
    # Same invariant for a short entered at +4%.
    assert not repair_trigger(-1, 0.06, 0.04)
    assert repair_trigger(-1, 0.02, 0.04)


def test_ridge_intercept_and_scale_are_explicit():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 3))
    y = 0.7 + x @ np.array([0.3, -0.2, 0.1]) + rng.normal(0, 0.01, len(x))
    beta, alpha = fit_ridge(x, y, scaled=True)
    assert np.isfinite(beta).all() and abs(alpha) > 0.2
    raw, _ = fit_ridge(x * np.array([1.0, 20.0, 0.2]), y, scaled=False)
    assert not np.allclose(beta, raw)


def test_center_and_scale_are_causal():
    idx = pd.date_range("2026-04-01", periods=900, freq="5min", tz="UTC")
    rng = np.random.default_rng(2)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.001, (len(idx), len(SYMBOLS))), axis=0)), index=idx, columns=SYMBOLS)
    beta = np.zeros(len(SYMBOLS) - 1)
    fits = {idx[0].normalize(): [(beta, 0.0) for _ in SYMBOLS]}
    a = make_features(close, fits)["BTCUSDT"]
    changed = close.copy(); changed.iloc[-1, 0] *= 2
    b = make_features(changed, fits)["BTCUSDT"]
    assert np.allclose(a.iloc[:-1].to_numpy(), b.iloc[:-1].to_numpy(), equal_nan=True)


def test_funding_sign_and_actual_mark_price():
    assert funding_cashflow(1, 30000, 100, 110, 0.001) == -33.0
    assert funding_cashflow(-1, 30000, 100, 110, 0.001) == 33.0


def test_written_spec_constants_are_not_silent():
    assert CONFIG["fit_hours"] == 672
    assert CONFIG["center_minutes"] == 180
    assert CONFIG["max_positions"] == 3
    assert CONFIG["allocation"] == 0.30
    assert CONFIG["max_new_budget"] == 0.90
