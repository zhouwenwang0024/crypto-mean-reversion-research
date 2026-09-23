import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from research_v2 import Feature, SYMBOLS, backtest, net_turnover, rolling_z, signed_weights


def _synthetic(z_values, target_fill=(90.0, 80.0), cost=0.0, funding_events=None):
    starts = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    mins = pd.date_range("2026-01-01", periods=16, freq="1min", tz="UTC")
    close = pd.DataFrame(100.0, index=starts, columns=SYMBOLS)
    close.iloc[1:, 0] = 80.0
    open1 = pd.DataFrame(100.0, index=mins, columns=SYMBOLS)
    close1 = open1.copy()
    open1.loc[pd.Timestamp("2026-01-01 00:06", tz="UTC"), SYMBOLS[0]] = target_fill[0]
    open1.loc[pd.Timestamp("2026-01-01 00:11", tz="UTC"), SYMBOLS[0]] = target_fill[1]
    close1.loc[:, SYMBOLS[0]] = open1.loc[:, SYMBOLS[0]]
    z = pd.DataFrame(0.0, index=starts, columns=SYMBOLS)
    z.iloc[:, :] = 0.0; z.iloc[0, 0] = z_values
    hedge = np.full((20, 20), -0.5 / 19.0); np.fill_diagonal(hedge, 0.5)
    ft = Feature("PEER", 5, close, close.copy(), {starts[0].normalize(): hedge, starts[1].normalize(): hedge, starts[2].normalize(): hedge})
    return backtest(ft, z, open1, close1, 2.0, 0.5, 4, cost, flat_boundaries=(), funding_events=funding_events)


def test_positive_residual_shorts_target_and_is_dollar_neutral():
    w = signed_weights(2.0, np.array([0.5, -0.5 / 19, -0.5 / 19] + [0.0] * 17))
    assert w[0] < 0 and w[1] > 0
    assert abs(w.sum()) < 1e-12


def test_net_turnover_uses_aggregate_order():
    # Two same-symbol orders (+2 and -1) are aggregated to a net +1 before fees.
    assert net_turnover(np.array([1.0]), np.array([100.0])) == 100.0


def test_zero_signal_does_not_trade():
    eq, trades, _ = _synthetic(0.0)
    assert trades.empty
    assert np.isclose(eq.iloc[-1], 100000.0)


def test_delayed_fill_and_directional_pnl():
    eq, trades, _ = _synthetic(3.0)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t.entry_time.endswith("00:06:00+00:00")
    assert t.price_pnl > 0
    assert t.reason == "mean"
    assert np.isclose(eq.iloc[-1], 100000.0 + t.net_pnl)


def test_cost_changes_cash_but_not_signal_pnl():
    free_eq, free_trades, _ = _synthetic(3.0, cost=0.0)
    paid_eq, paid_trades, _ = _synthetic(3.0, cost=3.0)
    assert np.isclose(free_eq.iloc[0], paid_eq.iloc[0])
    assert np.isclose(free_eq.iloc[-1] - paid_eq.iloc[-1], paid_trades.fee.iloc[0])


def test_funding_before_delayed_entry_is_not_charged():
    at = pd.Timestamp("2026-01-01 00:05", tz="UTC")
    rate = np.zeros(20); rate[0] = 0.10
    mark = np.full(20, 100.0)
    _, _, meta = _synthetic(3.0, funding_events=[(at, rate, mark)])
    assert np.isclose(meta["funding_cash"], 0.0)


def test_rolling_z_is_causal():
    ix = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    a = pd.DataFrame({"x": np.sin(np.arange(100) / 4)}, index=ix)
    z1 = rolling_z(a.iloc[:80], 5, 4, 3)
    z2 = rolling_z(a, 5, 4, 3)
    assert np.allclose(z1.iloc[-10:].to_numpy(), z2.iloc[70:80].to_numpy(), equal_nan=True)
