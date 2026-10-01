import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from funding_relative_value import (event_slots, market_betas, neutral_weights,
                                    rank_signal, replay_orders, run)  # noqa: E402


def test_neutral_weights_are_dollar_and_beta_neutral_with_cap():
    beta = np.array([0.8, 1.0, 1.2, 0.9, 1.1, 1.3])
    weights, ok = neutral_weights(beta, np.array([0, 2, 4]), np.array([1, 3, 5]))
    assert ok
    assert weights is not None
    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(weights @ beta, 0.0)
    assert np.isclose(np.abs(weights).sum(), 0.90)
    assert np.max(np.abs(weights)) <= 0.25 + 1e-10


def test_rank_uses_lagged_completed_event():
    rates = np.full((10, 4), np.nan)
    slots = np.array([1, 3, 5, 7, 9])
    rates[1] = [4, 3, 2, 1]
    rates[3] = [1, 2, 3, 4]
    rates[5] = rates[3]
    rates[7] = rates[3]
    rates[9] = rates[3]
    assert event_slots(rates).tolist() == slots.tolist()
    rates[5] = [5, 0, 1, 2]
    long_ids, short_ids = rank_signal(rates, slots, 3, 2, 1, 1)
    assert long_ids.tolist() == [1]
    assert short_ids.tolist() == [3]
    latest_long, latest_short = rank_signal(rates, slots, 3, 1, 1, 1)
    assert latest_long.tolist() == [1]
    assert latest_short.tolist() == [0]


def _synthetic(future_shift=0.0):
    n, m = 96 * 150, 6
    index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(7)
    logp = np.cumsum(rng.normal(0.0, 0.0005, size=(n, m)), axis=0)
    close = np.exp(logp)
    op = close.copy(); marks = np.full((n, m), np.nan)
    rates = np.full((n, m), np.nan)
    slots = np.arange(96, n, 96)
    for p, t in enumerate(slots):
        rates[t] = np.sin(np.arange(m) + p * 0.7) * 0.0004
        marks[t] = close[t]
    if future_shift:
        close[slots[-20]:, 1] *= 1.0 + future_shift
        op[slots[-20]:, 1] = close[slots[-20]:, 1]
    return index, op, close, rates, marks, slots


def test_run_replays_and_future_changes_do_not_rewrite_past_orders():
    base = _synthetic()
    future = _synthetic(0.15)
    cfg = {"name": "test", "lookback_events": 21, "hold_events": 21,
           "k": 3, "lag_events": 1}
    kwargs = dict(cfg=cfg, start=0, end=len(base[0]), fee_bp=5.0)
    r1, _, o1, t1 = run(base[0], base[1], base[2], base[3], base[4], **kwargs)
    r2, _, o2, t2 = run(future[0], future[1], future[2], future[3], future[4], **kwargs)
    assert abs(r1["replay_error"]) < 1e-10
    assert np.isclose(r1["gross_pnl"], r1["return"] - r1["funding"] + r1["fees"])
    assert np.isclose(r1["net_pnl"], r1["return"])
    assert np.isclose(replay_orders(o1, t1, r1["funding"]), 1.0 + r1["return"])
    cutoff = base[5][-20]
    cols = ["time", "kind", "symbol_index", "quantity_change", "price"]
    left = o1[pd.to_datetime(o1.time, utc=True) < base[0][cutoff]][cols].reset_index(drop=True)
    right = o2[pd.to_datetime(o2.time, utc=True) < base[0][cutoff]][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_market_beta_is_causal_shape():
    idx = np.arange(100)
    close = np.exp(np.cumsum(np.ones((100, 3)) * 0.001, axis=0))
    out = market_betas(close, idx, 70, 30)
    assert out.shape == (3,)
    assert np.isfinite(out).all()
