import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from adaptive_residual_grid import (execute, formation_rows, elapsed_alpha,
                                    entry_budget_room, risk_reduction_factor, run)  # noqa: E402


def test_formation_rows_accepts_timezone_aware_timestamps():
    selected = pd.DataFrame([{
        "model": "johansen", "model_end_time": "2025-01-01 00:00:00+00:00",
        "a": 1, "b": 2, "alpha": 0.1, "beta": 0.8,
        "cal_mean": 0.0, "cal_scale": 0.01,
    }])
    rows = formation_rows(selected)
    assert list(rows) == ["2025-01"]
    assert rows["2025-01"][0]["key"] == ("2025-01", 1, 2)


def test_execute_charges_fixed_one_way_fee_on_each_leg():
    cash, fee = execute(1.0, np.zeros(2), np.array([1.0, -2.0]), np.array([0.5, 0.25]), 5 / 10_000)
    assert fee == 0.0005
    assert cash == 0.9995


def test_ewma_uses_elapsed_signal_interval():
    one_hour = elapsed_alpha(168, 1)
    four_hours = elapsed_alpha(168, 4)
    assert np.isclose(one_hour, 1 - np.exp(-np.log(2) / 168))
    assert four_hours > one_hour * 3


def test_entry_room_includes_fee_in_gross_cap():
    room = entry_budget_room(1.0, 0.5, 0.9, 5 / 10_000)
    assert 0 < room < 0.4
    assert 0.5 + room <= 0.9 * (1.0 - room * 5 / 10_000) + 1e-12


def test_risk_reduction_factor_pays_its_fee_before_cap_check():
    factor = risk_reduction_factor(1.0, 1.1, 0.9, 5 / 10_000)
    gross_after = factor * 1.1
    fee = (1.0 - factor) * 1.1 * 5 / 10_000
    assert gross_after <= 0.9 * (1.0 - fee) + 1e-12


def _synthetic_inputs(trend=False, future_shift=0.0):
    n = 2500
    index = pd.date_range("2024-09-01", periods=n, freq="5min", tz="UTC")
    op = np.ones((n, 20)); close = op.copy(); volume = np.ones_like(op)
    rates = np.zeros_like(op); marks = op.copy(); spread = np.zeros(n)
    if trend:
        spread[2020:2040] = -0.012
        spread[2040:] = -0.04
    else:
        spread[2020:2040] = -0.012
        spread[2040:2100] = np.linspace(-0.012, 0.0, 60)
    spread[2200:] += future_shift
    close[:, 1] = np.exp(spread); op[:, 1] = close[:, 1]
    selected = pd.DataFrame([{"model": "johansen", "model_end_time": "2024-09-01 00:00:00+00:00",
                              "a": 1, "b": 0, "alpha": 0.0, "beta": 1.0,
                              "cal_mean": 0.0, "cal_scale": 0.01}])
    cfg = {"name": "synthetic", "observe_hours": 1, "center_hours": 168, "scale_hours": 168}
    return index, op, close, volume, rates, marks, selected, cfg


def test_synthetic_reversion_covers_fee_and_trend_stops():
    args = _synthetic_inputs()
    result, _, orders = run(*args, 0, len(args[0]))
    assert result["stop_count"] == 0
    assert result["return"] > 0
    assert "exit_mean" in set(orders.kind)

    args = _synthetic_inputs(trend=True)
    result, _, orders = run(*args, 0, len(args[0]))
    assert result["stop_count"] == 1
    assert result["return"] < 0
    assert result["replay_error"] < 1e-12


def test_future_perturbation_cannot_change_prior_orders():
    base = _synthetic_inputs()
    future = _synthetic_inputs(future_shift=0.08)
    _, _, _, _, _, _, _, _ = base
    base_result, _, base_orders = run(*base, 0, len(base[0]))
    future_result, _, future_orders = run(*future, 0, len(future[0]))
    cutoff = base[0][2200]
    cols = ["time", "kind", "symbol", "quantity_change", "price"]
    left = base_orders[base_orders.time < str(cutoff)][cols].reset_index(drop=True)
    right = future_orders[future_orders.time < str(cutoff)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert base_result["replay_error"] < 1e-12 and future_result["replay_error"] < 1e-12
