"""Small causal and ledger tests for the real pair backtester."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
import run_statarb_backtest as bt  # noqa: E402
from statarb_models import _tls, build_models  # noqa: E402


def _toy():
    n = 1_000
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    z = np.zeros(n); z[1:10] = 3.0
    close = np.ones((n, len(bt.SYMBOLS))) * 100.0
    close[:, 0] = np.exp(0.01 * z) * 100.0
    open_ = close.copy()
    selected = pd.DataFrame([{
        "model": "ols_log", "model_end_time": "2024-01-01T00:00:00Z",
        "a": 0, "b": 1, "alpha": 0.0, "beta": 1.0,
        "cal_mean": 0.0, "cal_scale": 0.01, "half_life_hours": 12.0,
    }])
    rates = np.zeros_like(close); marks = close.copy()
    return index, open_, close, rates, marks, selected


def _run(close, *, fee=5.0, mutant=False):
    index, op, _, rates, marks, selected = _toy()
    op = close.copy()
    return bt.run_model(index, op, close, rates, marks, selected, "ols_log", 0, len(index),
                        fee_bp=fee, use_funding=False, direction_mutant=mutant)


def test_future_close_perturbation_does_not_change_prior_orders():
    index, op, close, rates, marks, selected = _toy()
    base = bt.run_model(index, op, close, rates, marks, selected, "ols_log", 0, len(index),
                        fee_bp=5.0, use_funding=False)
    changed = close.copy(); changed[100:, 0] *= 1.7
    perturbed = bt.run_model(index, op, changed, rates, changed, selected, "ols_log", 0, len(index),
                             fee_bp=5.0, use_funding=False)
    cutoff = str(index[100])
    a = base[2].query("time < @cutoff").reset_index(drop=True)
    b = perturbed[2].query("time < @cutoff").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_direction_mutant_reverses_first_entry():
    normal = _run(_toy()[2], mutant=False)[2]
    reverse = _run(_toy()[2], mutant=True)[2]
    a = normal[normal.kind.eq("entry")].sort_values("symbol_index").quantity_change.to_numpy()
    b = reverse[reverse.kind.eq("entry")].sort_values("symbol_index").quantity_change.to_numpy()
    assert len(a) == 2 and np.allclose(a, -b)


def test_fee_is_monotone_and_independent_replay_matches():
    zero = _run(_toy()[2], fee=0.0)
    five = _run(_toy()[2], fee=5.0)
    assert five[3]["return"] <= zero[3]["return"]
    assert abs(zero[3]["return"] - five[3]["return"] - five[3]["fees"]) < 1e-12
    assert abs(bt.replay_orders(five[0], five[1], five[2], use_funding=False) - five[0].equity.iloc[-1]) < 1e-12
    tampered = five[2].copy()
    tampered.loc[tampered.index[0], "fee"] = 0.0
    assert abs(bt.replay_orders(five[0], five[1], tampered, use_funding=False) - five[0].equity.iloc[-1]) > 1e-9


def test_tls_normal_vector_and_formation_suffix_causality():
    x = np.linspace(-2.0, 2.0, 100); alpha, beta = _tls(3.0 + 2.0 * x, x)
    assert abs(alpha - 3.0) < 1e-10 and abs(beta - 2.0) < 1e-10
    n = 24 * 12 * 160
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7); common = np.cumsum(rng.normal(0, .001, n))
    values = np.column_stack([np.exp(common + rng.normal(0, .02, n)) for _ in range(3)]) * 100.0
    args = dict(index=index, fit_days=30, cal_days=14, symbols=["a", "b", "c"],
                formation_start="2024-03-01", formation_end="2024-06-01")
    _, before = build_models(values, **args)
    changed = values.copy(); changed[index >= pd.Timestamp("2024-05-15", tz="UTC"), 0] *= 1.8
    _, after = build_models(changed, **args)
    cutoff = pd.Timestamp("2024-05-01", tz="UTC")
    b = before[pd.to_datetime(before.model_end_time, utc=True) < cutoff].reset_index(drop=True)
    a = after[pd.to_datetime(after.model_end_time, utc=True) < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)
