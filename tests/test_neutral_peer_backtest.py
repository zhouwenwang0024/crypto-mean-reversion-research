import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__file__.replace("\\tests\\test_neutral_peer_backtest.py", "\\src\\extended_data")))
from run_neutral_peer_backtest import replay_portfolio, run_portfolio  # noqa: E402


def test_portfolio_caps_positions_and_replays_orders():
    n, m = 80, 20
    index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    op = np.full((n, m), 100.0); close = op.copy(); rates = np.zeros((n, m)); marks = close.copy()
    z = np.full((n, m), np.nan); z[5, :4] = 4.0
    hedge = {(5, j): {k: -1.0 / 19 for k in range(20) if k != j} for j in range(4)}
    bars, trades, orders, summary = run_portfolio(index, op, close, rates, marks, z, hedge, 0, n,
                                                   fee_bp=5.0, force_month_boundary=False,
                                                   observe_bars=1, hold_bars=20)
    assert summary["max_open_positions"] <= 3
    assert len(trades) == 3
    replay = replay_portfolio(index, op, close, rates, marks, orders, 0, n, 5.0, True)
    assert abs(replay - bars.equity.iloc[-1]) < 1e-10


def test_same_bar_opposite_peer_fills_are_net_fee_charged():
    n, m = 40, 20
    index = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    op = np.full((n, m), 100.0); close = op.copy(); rates = np.zeros((n, m)); marks = close.copy()
    z = np.full((n, m), np.nan); z[5, 0] = 4.0; z[5, 1] = -4.0
    hedge = {(5, 0): {2: -1.0}, (5, 1): {2: -1.0}}
    _, _, orders, summary = run_portfolio(index, op, close, rates, marks, z, hedge, 0, n,
                                           fee_bp=5.0, force_month_boundary=False,
                                           observe_bars=1, hold_bars=20)
    # Sequential budgets differ slightly, but the shared coin-2 leg is one net order.
    coin2 = orders[(orders.symbol_index == 2) & (orders.kind == "entry")]
    assert len(coin2) == 1
    assert abs(float(coin2.fee.iloc[0]) - abs(float(coin2.quantity_change.iloc[0]) * 100.0) * 5e-4) < 1e-12
    assert summary["fees"] < 0.0015
