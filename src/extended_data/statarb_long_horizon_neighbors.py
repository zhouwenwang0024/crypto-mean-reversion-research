"""Neighbor check around the predeclared long-horizon Johansen mechanism."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402
from statarb_long_horizon import PERIODS, period_return  # noqa: E402

COMBOS = tuple((56, entry, 0.50) for entry in (2.50, 2.75, 3.00, 3.25))


def main() -> None:
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows = []
    for hold, entry, exit_z in COMBOS:
        bars, trades, orders, summary = bt.run_model(
            index, op, cl, rates, marks, selected, "johansen", start, end,
            fee_bp=5.0, use_funding=True, hold_bars=hold * 288,
            entry_z=entry, exit_z=exit_z, rearm_z=1.0,
            force_month_boundary=False,
        )
        t = pd.to_datetime(bars.time, utc=True)
        event = (t >= pd.Timestamp("2025-10-11", tz="UTC")) & (t < pd.Timestamp("2025-10-12", tz="UTC"))
        changes = bars.equity.pct_change().fillna(0.0)
        row = {
            "model": "johansen", "hold_days": hold, "entry_z": entry,
            "exit_z": exit_z, "return": summary["return"], "mdd": summary["mdd"],
            "trades": summary["trades"], "fees": summary["fees"],
            "funding": summary["funding"], "gross_pnl": summary["gross_pnl"],
            "turnover": summary["turnover"], "max_gross": summary["max_gross"],
            "max_open_positions": summary["max_open_positions"],
            "independent_replay_error": float(
                bt.replay_orders(bars, trades, orders, use_funding=True) - bars.equity.iloc[-1]
            ),
            "return_excluding_2025_10_11_utc": float(np.prod(1.0 + changes.where(~event, 0.0)) - 1.0),
            "force_month_boundary": False,
        }
        row.update({f"{name}_return": period_return(bars, *bounds)
                    for name, bounds in PERIODS.items()})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_long_horizon_neighbors.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
