"""Historical holding/threshold sensitivity for frozen pair formations.

This is a disclosed exploration grid on already studied data; no row is
presented as unseen validation.
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
# A small Latin grid keeps the historical exploration practical while covering
# the requested neighboring values in each axis.
COMBOS = tuple(
    (hold, entry, exit_z)
    for hold in (14, 28, 56)
    for entry, exit_z in ((2.50, 0.25), (2.75, 0.50), (3.00, 0.75), (3.25, 0.50))
)
PERIODS = {
    "development": ("2024-06-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}


def _period_return(bars: pd.DataFrame, start: str, end: str) -> float:
    t = pd.to_datetime(bars.time, utc=True)
    m = (t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))
    if not m.any():
        return np.nan
    eq = bars.loc[m, "equity"].to_numpy(float)
    before = float(bars.loc[~(t >= pd.Timestamp(start, tz="UTC")), "equity"].iloc[-1]) if (t < pd.Timestamp(start, tz="UTC")).any() else 1.0
    return float(eq[-1] / before - 1.0)


def _without_event(bars: pd.DataFrame) -> float:
    t = pd.to_datetime(bars.time, utc=True)
    r = bars.equity.astype(float).pct_change().fillna(0.0).to_numpy()
    event = (t >= pd.Timestamp("2025-10-11", tz="UTC")) & (t < pd.Timestamp("2025-10-12", tz="UTC"))
    r[event.to_numpy()] = 0.0
    return float(np.prod(1.0 + r) - 1.0)


def main() -> None:
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows: list[dict] = []
    for model in MODELS:
        for hold, entry, exit_z in COMBOS:
            bars, _, _, summary = bt.run_model(
                index, op, cl, rates, marks, selected, model, start, end,
                fee_bp=5.0, use_funding=True, hold_bars=hold * 288,
                entry_z=entry, exit_z=exit_z,
            )
            row = {
                "model": model, "hold_days": hold, "entry_z": entry,
                "exit_z": exit_z, "return": summary["return"],
                "mdd": summary["mdd"], "trades": summary["trades"],
                "fees": summary["fees"], "funding": summary["funding"],
                "reconciliation_error": summary["reconciliation_error"],
                "return_excluding_2025_10_11_utc": _without_event(bars),
            }
            for name, (a, b) in PERIODS.items():
                row[name + "_return"] = _period_return(bars, a, b)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_sensitivity_grid.csv", index=False)
    print(out.to_string(index=False))
    print("saved", len(out), "rows")


if __name__ == "__main__":
    main()
