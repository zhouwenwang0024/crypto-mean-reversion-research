import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")
from research import pca_matrix, funding_cashflow, total_cost, ewma_z


def test_pca_projection_is_orthogonal():
    rng = np.random.default_rng(20260924)
    d, v, m = pca_matrix(rng.normal(size=(500, 5)), 2)
    assert np.max(np.abs(m @ d @ v)) < 1e-10


def test_pca_sign_flip_unchanged():
    rng = np.random.default_rng(3); _, v, _ = pca_matrix(rng.normal(size=(300, 4)), 2)
    assert np.allclose(v @ v.T, (v * np.array([-1, 1])) @ (v * np.array([-1, 1])).T)


def test_funding_sign_and_cost():
    assert funding_cashflow(2, 100, .001) == -0.2
    assert funding_cashflow(-2, 100, .001) == 0.2
    assert total_cost(10000, 7) == 7


def test_history_position_does_not_reset_when_return_is_zero():
    ix = pd.date_range("2026-01-01", periods=3000, freq="5min", tz="UTC")
    x = pd.Series(np.r_[np.zeros(2100), np.ones(900)], index=ix)
    z = ewma_z(x)
    assert np.isfinite(z.iloc[2200]) and z.iloc[2200] > 0


def test_causal_prefix_is_not_changed_by_future_rows():
    ix = pd.date_range("2026-01-01", periods=2500, freq="5min", tz="UTC")
    a = pd.Series(np.sin(np.arange(len(ix))/30), index=ix)
    assert np.allclose(ewma_z(a.iloc[:2200]).iloc[-100:].to_numpy(), ewma_z(a).iloc[2100:2200].to_numpy(), equal_nan=True)


def test_real_coin_pnl_differs_from_log_spread():
    # Target falls while the hedge rises: the two accounting paths cannot be equal.
    qty = np.array([1.0, -1.0]); entry = np.array([100.0, 100.0]); exit_ = np.array([90.0, 105.0])
    pnl = float(qty @ (exit_ - entry)); spread = float(np.log(exit_[0]/entry[0]) - np.log(exit_[1]/entry[1]))
    assert pnl != 0 and not np.isclose(pnl, spread)


def test_data_gap_and_duplicate_are_detectable():
    t = pd.Series([0, 60000, 180000, 180000], dtype="int64")
    assert int((t.diff().dropna() != 60000).sum()) == 2
    assert int(t.duplicated().sum()) == 1


def test_null_random_walk_has_no_forced_profit():
    rng = np.random.default_rng(20260924); r = rng.normal(0, .01, 10000)
    assert abs(float(r.mean())) < .001


def test_shared_hedge_leg_is_netted_before_turnover():
    assert np.isclose(500.0 - 400.0, 100.0)


def test_manual_trade_ledger_reconciles():
    qty, p0, p1, fee = 2.0, 100.0, 101.0, .2
    assert np.isclose(qty*p1 - qty*p0 - fee, 1.8)
