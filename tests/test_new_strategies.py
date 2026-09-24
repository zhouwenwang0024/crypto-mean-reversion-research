import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from research_new_strategies import (  # noqa: E402
    HORIZONS,
    _period_return,
    funding_reversal,
    reversal_events,
    summarize,
)
from research_v2 import SYMBOLS  # noqa: E402


def test_period_return_uses_half_open_interval():
    t = pd.to_datetime(["2026-05-01", "2026-06-30", "2026-07-01"], utc=True)
    x = pd.Series([0.1, 0.2, 0.3])
    assert np.isclose(_period_return(x, t, "2026-05-01", "2026-07-01"), 0.32)


def test_summary_cost_is_monotone_and_pair_budget_is_split():
    t = pd.date_range("2026-05-01", periods=2, freq="h", tz="UTC")
    trades = pd.DataFrame({"gross_return": [0.01, -0.005], "entry_time": t, "exit_time": t + pd.Timedelta(minutes=1), "pair_id": [0, 1]})
    a = summarize(trades, "pair", 0.5)
    b = summarize(trades, "pair", 1.0)
    assert a["allocation_per_pair"] == 0.5
    assert a["return"] > b["return"]


def test_sign_reversal_samples_each_horizon_without_overlap():
    ix15 = pd.date_range("2026-03-01", periods=32, freq="15min", tz="UTC")
    mins = pd.date_range("2026-03-01", periods=8 * 60, freq="min", tz="UTC")
    close15 = pd.DataFrame(100.0, index=ix15, columns=SYMBOLS)
    open1 = pd.DataFrame(100.0, index=mins, columns=SYMBOLS)
    open1.loc[ix15[1]:ix15[1] + pd.Timedelta(minutes=14), SYMBOLS[0]] = 99.0
    close1 = open1.copy()
    out = reversal_events(open1, close1, close15)
    assert out.mean_abs_signal.max() > 0.0
    for h in HORIZONS:
        s = pd.to_datetime(out.loc[out.horizon_min == h, "signal_time"], utc=True)
        assert len(s) > 0
        assert s.diff().dropna().eq(pd.Timedelta(minutes=h)).all()


def test_funding_signal_uses_previous_settlement_rate():
    mins = pd.date_range("2026-03-01", periods=24 * 60, freq="min", tz="UTC")
    open1 = pd.DataFrame(100.0, index=mins, columns=SYMBOLS)
    close1 = open1.copy()
    r0 = np.zeros(20); r0[0] = -0.01; r0[1] = 0.01
    r1 = np.zeros(20); r1[0] = 0.10; r1[1] = -0.10
    r2 = np.zeros(20); r2[0] = 0.01; r2[1] = -0.01
    marks = np.full(20, 100.0)
    events = [(pd.Timestamp("2026-03-01 00:00", tz="UTC"), r0, marks), (pd.Timestamp("2026-03-01 08:00", tz="UTC"), r1, marks), (pd.Timestamp("2026-03-01 16:00", tz="UTC"), r2, marks)]
    out = funding_reversal(open1, close1, events, q=1)
    assert len(out) == 1
    assert out.iloc[0].entry_time == pd.Timestamp("2026-03-01 08:15", tz="UTC")
    assert out.iloc[0].funding_return < 0
