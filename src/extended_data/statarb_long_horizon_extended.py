"""Predeclared 56/84/112-day frozen-spread study at fixed 5 bp fees.

Formation rows are immutable.  No row is selected after looking at the new
grid.  The primary grid uses the already audited Johansen formation and allows
lots to cross months; two new-order gross-cap reruns are recorded for the
112-day 3-sigma candidate as a risk-control diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402

MODEL = "johansen"
FEE_BP = 5.0
PERIODS = {"development": ("2024-06-01", "2025-03-01"),
           "validation": ("2025-03-01", "2025-09-01"),
           "historical_holdout": ("2025-09-01", "2026-03-01"),
           "extension": ("2026-03-01", "2026-09-01")}
GRID = [(hold, entry, exit_z) for hold in (56, 84, 112)
        for entry in (2.5, 3.0, 3.5) for exit_z in (0.5, 0.75)]


def period_return(bars: pd.DataFrame, start: str, end: str) -> float:
    t = pd.to_datetime(bars.time, utc=True)
    mask = (t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))
    if not mask.any(): return np.nan
    prior = bars.loc[t < pd.Timestamp(start, tz="UTC"), "equity"]
    return float(bars.loc[mask, "equity"].iloc[-1] / (prior.iloc[-1] if len(prior) else 1.0) - 1.0)


def run_one(index, op, cl, rates, marks, selected, hold_days, entry, exit_z, gross_limit=None):
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    bars, trades, orders, summary = bt.run_model(
        index, op, cl, rates, marks, selected, MODEL, start, end,
        fee_bp=FEE_BP, use_funding=True, hold_bars=hold_days * 288,
        entry_z=entry, exit_z=exit_z, rearm_z=1.0,
        force_month_boundary=False, gross_limit=gross_limit)
    replay_error = float(bt.replay_orders(bars, trades, orders, use_funding=True) - bars.equity.iloc[-1])
    if abs(replay_error) > 1e-9 or abs(summary["reconciliation_error"]) > 1e-9:
        raise AssertionError({"summary": summary, "replay_error": replay_error})
    row = {"model": MODEL, "hold_days": hold_days, "entry_z": entry, "exit_z": exit_z,
           "gross_limit": gross_limit, "fee_bp_one_way": FEE_BP, "return": summary["return"],
           "mdd": summary["mdd"], "trades": summary["trades"], "fees": summary["fees"],
           "funding": summary["funding"], "turnover": summary["turnover"],
           "max_gross": summary["max_gross"], "max_open_positions": summary["max_open_positions"],
           "replay_error": replay_error}
    row.update({f"{name}_return": period_return(bars, *bounds) for name, bounds in PERIODS.items()})
    return row, bars, trades, orders


def main() -> None:
    index, op, cl, _ = bt.load_prices(); rates, marks, funding_events = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    rows = []
    for hold_days, entry, exit_z in GRID:
        row, _, _, _ = run_one(index, op, cl, rates, marks, selected, hold_days, entry, exit_z)
        rows.append(row)
    # The cap reruns are fixed before inspecting their output and only diagnose
    # new-order limits; the engine does not cap mark-to-market drift in an open lot.
    for cap in (0.8, 0.9):
        row, bars, trades, orders = run_one(index, op, cl, rates, marks, selected, 112, 3.0, 0.5, cap)
        rows.append(row)
        tag = f"112d_e3_x05_cap{int(cap * 100)}"
        trades.to_csv(bt.RESULTS / f"statarb_long_extended_{tag}_trades.csv", index=False)
        orders.to_csv(bt.RESULTS / f"statarb_long_extended_{tag}_orders.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_long_horizon_extended.csv", index=False)
    manifest = {"model": MODEL, "fee_bp_one_way": FEE_BP, "funding_events": funding_events,
                "formation": "results/statarb_formation_selected.csv (frozen)",
                "grid": {"hold_days": [56, 84, 112], "entry_z": [2.5, 3.0, 3.5],
                         "exit_z": [0.5, 0.75], "force_month_boundary": False},
                "cap_diagnostics": [0.8, 0.9], "selection": "all grid rows retained; no tail-based selection"}
    (bt.RESULTS / "statarb_long_horizon_extended_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(out.sort_values("return", ascending=False).to_string(index=False))


if __name__ == "__main__": main()
