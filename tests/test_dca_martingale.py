import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dca_martingale import _simulate


def test_long_only_dca_adds_and_closes_a_basket():
    open_ = np.array([100, 100, 96, 94, 92, 95, 98], dtype=float)
    high = open_ + 1
    low = open_ - 1
    close = open_.copy()
    ma = np.full(len(close), 99.0)
    rates = np.zeros(len(close))
    marks = np.full(len(close), 100.0)
    equity, trades, count = _simulate(open_, high, low, close, ma, rates, marks,
                                      .02, .02, 1.5, .02, .20, 4, .0005, .05, .75)
    assert count == 1
    assert trades[0, 2] >= 2
    assert trades[0, 7] > 0
    assert equity[-1] > 1.0


def test_stop_loss_wins_when_stop_and_target_share_a_bar():
    open_ = np.array([100, 97, 97, 97], dtype=float)
    high = np.array([101, 98, 99, 98], dtype=float)
    low = np.array([99, 96, 93, 96], dtype=float)
    close = open_.copy()
    ma = np.full(len(close), 99.0)
    rates = np.zeros(len(close))
    marks = np.full(len(close), 100.0)
    _, trades, count = _simulate(open_, high, low, close, ma, rates, marks,
                                  .02, .01, 1.5, .02, .03, 4, .0005, .05, .75)
    assert count == 1
    assert trades[0, 8] == 1.0
