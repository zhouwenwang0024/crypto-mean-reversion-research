from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from run_target_predictor_backtest import run_target_basket, replay_basket  # noqa: E402


def _arrays(index):
    close = np.full((len(index), 20), 100.0)
    return close.copy(), close, np.zeros_like(close), close.copy()


def test_positive_residual_short_target_long_peer_and_replay():
    idx = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    op, close, rates, marks = _arrays(idx)
    z = np.full((len(idx), 20), np.nan); z[:, 0] = [3.0, 3.0, 0.0, 0.0]
    bars, trades, orders, summary = run_target_basket(
        idx, op, close, rates, marks, z, {(0, 0): {1: -0.5}}, 0, 0, len(idx),
        fee_bp=0.0, use_funding=False, force_month_boundary=False,
    )
    entry = orders[orders.kind == "entry"].set_index("symbol").quantity_change
    assert entry["BTC"] < 0 and entry["ETH"] > 0
    assert len(trades) == 1 and abs(summary["reconciliation_error"]) < 1e-12
    replay = replay_basket(idx, op, close, rates, marks, orders, 0, 0, len(idx), 0.0, False)
    assert abs(replay - bars.equity.iloc[-1]) < 1e-12


def test_month_boundary_closes_when_next_model_is_missing():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2025-01-31 23:50", tz="UTC"),
        pd.Timestamp("2025-01-31 23:55", tz="UTC"),
        pd.Timestamp("2025-02-01 00:00", tz="UTC"),
        pd.Timestamp("2025-02-01 00:05", tz="UTC"),
    ])
    op, close, rates, marks = _arrays(idx)
    z = np.full((len(idx), 20), np.nan); z[:2, 0] = 3.0
    _, trades, _, _ = run_target_basket(
        idx, op, close, rates, marks, z, {(0, 0): {1: -0.5}, (1, 0): {1: -0.5}},
        0, 0, len(idx), fee_bp=0.0, use_funding=False, force_month_boundary=True,
    )
    assert len(trades) == 1 and trades.iloc[0].reason == "month_boundary"
