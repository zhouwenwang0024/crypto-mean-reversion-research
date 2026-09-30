import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "extended_data"))
from adaptive_residual_grid import execute, formation_rows  # noqa: E402


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
