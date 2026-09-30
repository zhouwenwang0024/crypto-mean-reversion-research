"""Run the frozen long-horizon candidate from flat at each reporting segment."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402

PERIODS = {
    "development": ("2024-06-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}


def main() -> None:
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    rows = []
    for name, (a, b) in PERIODS.items():
        start = int(index.searchsorted(pd.Timestamp(a, tz="UTC")))
        end = int(index.searchsorted(pd.Timestamp(b, tz="UTC")))
        bars, trades, orders, s = bt.run_model(
            index, op, cl, rates, marks, selected, "johansen", start, end,
            fee_bp=5.0, use_funding=True, hold_bars=56 * 288,
            entry_z=3.0, exit_z=0.5, rearm_z=1.0,
            force_month_boundary=False,
        )
        s.update({"period": name, "entry_z": 3.0, "exit_z": 0.5,
                  "hold_days": 56, "force_month_boundary": False,
                  "independent_replay_error": bt.replay_orders(bars, trades, orders) - bars.equity.iloc[-1]})
        rows.append(s)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_long_segments.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
