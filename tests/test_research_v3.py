import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import research_v2
from research_v2 import Feature, SYMBOLS
from research_v3 import ar1_feature, configs, entry_mask, median_peer_feature, score
from summarize_v3 import period_stats


def test_v3_score_is_causal():
    ix = pd.date_range("2026-01-01", periods=1000, freq="15min", tz="UTC")
    a = pd.DataFrame({"x": np.sin(np.arange(1000) / 7) + np.arange(1000) / 10000}, index=ix)
    shocked = a.copy(); shocked.iloc[800:] += 100.0
    first = score(a.iloc[:800], 15, 4, 7)
    second = score(shocked, 15, 4, 7)
    assert np.isfinite(first.iloc[700:800].to_numpy()).any()
    assert np.allclose(first.iloc[700:800].to_numpy(), second.iloc[700:800].to_numpy(), equal_nan=True)


def test_entry_filters_only_use_prior_bars():
    ix = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    z = pd.DataFrame({"x": [0.0, 2.1, 2.2, -2.1, -2.3]}, index=ix)
    mask = entry_mask(z, "confirm2")
    assert not bool(mask.iloc[1, 0])
    assert bool(mask.iloc[2, 0])
    assert not bool(mask.iloc[3, 0])
    assert bool(mask.iloc[4, 0])
    assert entry_mask(z, "wide_exit") is None
    assert entry_mask(z, "zero_cross") is None


def test_configs_have_callable_entry_rules_and_unique_parameters():
    cfgs = configs()
    fields = ("model", "frequency_min", "center_hours", "scale_days",
              "center_kind", "logic", "entry_sigma", "exit_sigma",
              "hold_hours", "exit_rule")
    keys = [tuple(c[k] for k in fields) for c in cfgs]
    assert len(keys) == len(set(keys))
    ix = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    z = pd.DataFrame(np.arange(100).reshape(5, 20) / 10.0, index=ix, columns=SYMBOLS)
    for cfg in cfgs:
        mask = entry_mask(z, cfg["logic"], cfg["entry_sigma"])
        assert mask is None or mask.shape == z.shape


def test_median_feature_is_invariant_to_price_units():
    rng = np.random.default_rng(7)
    ix = pd.date_range("2026-01-01", periods=900, freq="15min", tz="UTC")
    returns = rng.normal(0.0, 0.002, size=(len(ix), len(SYMBOLS)))
    prices = np.exp(np.cumsum(returns, axis=0))
    scaled = prices * np.linspace(0.5, 2.0, len(SYMBOLS))
    a = pd.DataFrame(prices, index=ix, columns=SYMBOLS)
    b = pd.DataFrame(scaled, index=ix, columns=SYMBOLS)
    fa, fb = median_peer_feature(a, 15), median_peer_feature(b, 15)
    assert np.allclose(fa.level.to_numpy(), fb.level.to_numpy(), equal_nan=True)
    za, zb = score(fa.level, 15, 4, 7), score(fb.level, 15, 4, 7)
    assert np.allclose(za.to_numpy(), zb.to_numpy(), equal_nan=True)


def test_ar1_entries_do_not_neutralize_against_peers(monkeypatch):
    calls = []
    original = research_v2.signed_weights

    def spy(z, hedge, neutralize=True):
        calls.append(neutralize)
        return original(z, hedge, neutralize)

    monkeypatch.setattr(research_v2, "signed_weights", spy)
    starts = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
    minutes = pd.date_range(starts[0], periods=1900, freq="1min", tz="UTC")
    close = pd.DataFrame(100.0, index=starts, columns=SYMBOLS)
    prices = pd.DataFrame(100.0, index=minutes, columns=SYMBOLS)
    level = pd.DataFrame(0.0, index=starts, columns=SYMBOLS)
    level.iloc[0, :3] = 3.0
    feature = ar1_feature(close, 15)
    feature.level = level
    z = level.copy()
    research_v2.backtest(feature, z, prices, prices, 2.0, 0.5, 4, 0.0,
                         flat_boundaries=(), funding_events=[])
    assert calls and not any(calls)


def test_boundary_close_fee_belongs_to_prior_period():
    starts = pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC")
    minutes = pd.date_range(starts[0], periods=40, freq="1min", tz="UTC")
    close = pd.DataFrame(100.0, index=starts, columns=SYMBOLS)
    prices = pd.DataFrame(100.0, index=minutes, columns=SYMBOLS)
    z = pd.DataFrame(0.0, index=starts, columns=SYMBOLS)
    z.iloc[:4, 0] = 3.0
    hedge = {d: np.eye(len(SYMBOLS)) for d in starts.normalize().unique()}
    feature = Feature("B0", 5, close, close, hedge, False)
    boundary = starts[4]
    cut = boundary - pd.Timedelta(minutes=1)
    eq, trades, _ = research_v2.backtest(
        feature, z, prices, prices, 2.0, 0.5, 4, 2.0,
        flat_boundaries=(boundary,), fee_mode="gross")
    assert len(trades) == 1 and trades.iloc[0].reason == "period_boundary"
    assert pd.Timestamp(trades.iloc[0].exit_time) == cut
    assert np.isclose(trades.iloc[0].fee, 4.0)
    assert np.isclose(eq.loc[cut], 100000.0 - trades.iloc[0].fee)
    assert eq[eq.index < boundary].iloc[-1] == eq.loc[boundary]


def test_period_stats_includes_first_of_62_holdout_days():
    start = pd.Timestamp("2026-07-01", tz="UTC")
    end = pd.Timestamp("2026-09-01", tz="UTC")
    daily = np.full(62, 0.001); daily[0] = 0.01
    days = pd.date_range(start, end, freq="1D", inclusive="left")
    marks = days + pd.Timedelta(hours=23, minutes=59)
    eq = pd.Series(100000.0 * np.cumprod(1.0 + daily), index=marks)
    eq.loc[start - pd.Timedelta(minutes=1)] = 100000.0
    eq = eq.sort_index()
    base, finish, actual = period_stats(eq, start, end)
    assert len(actual) == 62
    assert np.isclose(actual[0], 0.01)
    assert np.allclose(actual, daily)
    assert np.isclose(np.prod(1.0 + actual) - 1.0, finish / base - 1.0)
