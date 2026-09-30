"""Predeclared long-horizon frozen-spread sensitivity.

Unlike the monthly-reset baseline, an open lot may cross a formation month;
its alpha, beta, mean and scale remain frozen until mean reversion or timeout.
No result is used to choose another row in this grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402

MODELS = ("ols_log", "johansen")
COMBOS = tuple((hold, entry, exit_z)
               for hold in (28, 56)
               for entry in (2.50, 3.00)
               for exit_z in (0.50, 0.75))
PERIODS = {
    "development": ("2024-06-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}


def period_return(bars: pd.DataFrame, start: str, end: str) -> float:
    t = pd.to_datetime(bars.time, utc=True)
    a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    mask = (t >= a) & (t < b)
    if not mask.any():
        return np.nan
    prior = bars.loc[t < a, "equity"]
    return float(bars.loc[mask, "equity"].iloc[-1] / (prior.iloc[-1] if len(prior) else 1.0) - 1)


def main() -> None:
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows = []
    for model in MODELS:
        for hold, entry, exit_z in COMBOS:
            bars, _, _, summary = bt.run_model(
                index, op, cl, rates, marks, selected, model, start, end,
                fee_bp=5.0, use_funding=True, hold_bars=hold * 288,
                entry_z=entry, exit_z=exit_z, rearm_z=1.0,
                force_month_boundary=False,
            )
            t = pd.to_datetime(bars.time, utc=True)
            event = (t >= pd.Timestamp("2025-10-11", tz="UTC")) & (t < pd.Timestamp("2025-10-12", tz="UTC"))
            changes = bars.equity.pct_change().fillna(bars.equity.iloc[0] - 1.0)
            no_event = float(np.prod(1.0 + changes.where(~event, 0.0)) - 1.0)
            row = {"model": model, "hold_days": hold, "entry_z": entry,
                   "exit_z": exit_z, "return": summary["return"],
                   "mdd": summary["mdd"], "trades": summary["trades"],
                   "fees": summary["fees"], "funding": summary["funding"],
                   "return_excluding_2025_10_11_utc": no_event,
                   "force_month_boundary": False}
            row.update({f"{name}_return": period_return(bars, *bounds)
                        for name, bounds in PERIODS.items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_long_horizon_grid.csv", index=False)
    print(out.sort_values("return", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
