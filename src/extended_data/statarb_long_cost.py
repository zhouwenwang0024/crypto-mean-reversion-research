"""Cost/funding check for the predeclared long-horizon Johansen rule."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402


def main() -> None:
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(bt.RESULTS / "statarb_formation_selected.csv")
    start = int(index.searchsorted(pd.Timestamp("2024-06-01", tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp("2026-09-01", tz="UTC")))
    rows = []
    for bp in (0.0, 2.0, 5.0, 10.0):
        for funding in (True, False):
            _, _, _, s = bt.run_model(
                index, op, cl, rates, marks, selected, "johansen", start, end,
                fee_bp=bp, use_funding=funding, hold_bars=56 * 288,
                entry_z=3.0, exit_z=0.5, rearm_z=1.0,
                force_month_boundary=False,
            )
            rows.append(s)
    out = pd.DataFrame(rows)
    out.to_csv(bt.RESULTS / "statarb_long_cost_sensitivity.csv", index=False)
    print(out[["fee_bp_one_way", "funding_enabled", "return", "mdd", "trades", "fees", "funding"]].to_string(index=False))


if __name__ == "__main__":
    main()
