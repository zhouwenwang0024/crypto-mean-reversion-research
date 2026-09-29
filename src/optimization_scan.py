"""Small, predeclared Ridge shrinkage sensitivity scan.

This is deliberately separate from the v3 grid.  It keeps one signal and one
execution rule fixed (15-minute Ridge, 4-hour SMA, 7-day scale, 2-sigma entry,
0.5-sigma exit, four-hour timeout) and changes only the standardized Ridge
penalty.  The penalty list was fixed before reading the July--August rows.
Outputs are compact tables derived from a fresh feature/backtest run; no old
PnL is re-used as a substitute for an account replay.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import (END, RESULTS, START, TRAIN_END, VALID_END,
                         backtest, load_funding_events, load_prices, make_bars,
                         residual_feature)
from research_v3 import score


PENALTIES = (50.0, 100.0, 200.0)
COSTS_BP = (0.0, 2.0, 5.0, 10.0)
FREQUENCY = 15
CENTER_HOURS = 4
SCALE_DAYS = 7


def hedge_diagnostics(feature) -> dict[str, float]:
    """Report coefficient movement and concentration without selecting on it."""
    mats = [feature.hedge_by_day[d] for d in sorted(feature.hedge_by_day)]
    if not mats:
        return {"days": 0, "mean_day_l1": np.nan, "median_day_l1": np.nan,
                "median_max_abs_peer": np.nan}
    moves = []
    for a, b in zip(mats, mats[1:]):
        # Rows contain target=1 and peer hedge coefficients.  Scale by the
        # previous row norm so movement is comparable across targets.
        den = np.abs(a).sum(axis=1)
        moves.append(float(np.nanmean(np.abs(b - a).sum(axis=1) / np.maximum(den, 1e-12))))
    cube = np.asarray(mats)
    peer = cube.copy()
    idx = np.arange(peer.shape[1])
    peer[:, idx, idx] = np.nan
    return {
        "days": len(mats),
        "mean_day_l1": float(np.nanmean(moves)) if moves else 0.0,
        "median_day_l1": float(np.nanmedian(moves)) if moves else 0.0,
        "median_max_abs_peer": float(np.nanmedian(np.nanmax(np.abs(peer), axis=2))),
    }


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    opens, closes, volumes = load_prices()
    bars = make_bars(closes, volumes, FREQUENCY)
    funding = load_funding_events()
    rows: list[dict] = []
    diagnostics: list[dict] = []
    assumptions = {
        "purpose": "Ridge shrinkage sensitivity; predeclared before holdout inspection",
        "penalties": list(PENALTIES),
        "penalty_scale": "standardized 28-day hourly returns; lambda=100 is audited baseline",
        "frequency_min": FREQUENCY, "center_hours": CENTER_HOURS,
        "scale_days": SCALE_DAYS, "entry_sigma": 2.0, "exit_sigma": 0.5,
        "hold_hours": 4, "cost_bp": list(COSTS_BP),
        "fee_mode": "gross aggregate order turnover",
        "funding": "local cached Binance funding events; cashflow reported",
        "flat_boundaries": [str(TRAIN_END), str(VALID_END)],
        "allocation": 0.10, "max_positions": 3,
    }
    for penalty in PENALTIES:
        feature = residual_feature(bars, FREQUENCY, "RIDGE", ridge_lambda=penalty)
        z = score(feature.level, FREQUENCY, CENTER_HOURS, SCALE_DAYS, "sma")
        diagnostics.append({"ridge_lambda": penalty, **hedge_diagnostics(feature)})
        for bp in COSTS_BP:
            eq, trades, meta = backtest(
                feature, z, opens, closes, 2.0, 0.5, 4, bp,
                flat_boundaries=(TRAIN_END, VALID_END),
                funding_events=funding, fee_mode="gross")
            for period, a, b in (("validation", TRAIN_END, VALID_END),
                                 ("holdout", VALID_END, END),
                                 ("all", START, END)):
                m = metrics_safe(eq, trades, a, b, meta["funding_rows"])
                rows.append({"ridge_lambda": penalty, "cost_bp": bp,
                             "period": period, **m})
        # Keep the per-lambda replay available for later audit without storing
        # all minute-by-minute account rows in the repository.
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "ridge_lambda_scan.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULTS / "ridge_lambda_stability.csv", index=False)
    (RESULTS / "ridge_lambda_scan.json").write_text(
        json.dumps({"assumptions": assumptions,
                    "rows": len(table), "funding_events": len(funding)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(pd.DataFrame(diagnostics).to_string(index=False))


def metrics_safe(eq: pd.Series, trades: pd.DataFrame, start: pd.Timestamp,
                 end: pd.Timestamp, funding_rows: list[dict]) -> dict:
    """Call the audited metrics function while keeping this script import-light."""
    from research_v2 import metrics
    return metrics(eq, trades, start, end, funding_rows)


if __name__ == "__main__":
    run()
