"""Fair fixed-rule comparison of Ridge and PCA residual models.

PCA2 is the only factor count absent from the existing v3 matrix.  This
script reruns all four models at the common 15-minute/4-hour/7-day rule so
PCA2 is evaluated by exactly the same account ledger and cost schedule.
"""
from __future__ import annotations

import json

import pandas as pd

from research_v2 import (END, RESULTS, START, TRAIN_END, VALID_END, backtest,
                         load_funding_events, load_prices, make_bars,
                         metrics, residual_feature)
from research_v3 import score
from optimization_scan import hedge_diagnostics


MODELS = ("RIDGE", "PCA2", "PCA3", "PCA5")
COSTS_BP = (0.0, 2.0, 5.0, 10.0)


def run() -> None:
    opens, closes, volumes = load_prices()
    bars = make_bars(closes, volumes, 15)
    funding = load_funding_events()
    rows, stability = [], []
    for model in MODELS:
        feature = residual_feature(bars, 15, model)
        z = score(feature.level, 15, 4, 7, "sma")
        stability.append({"model": model, **hedge_diagnostics(feature)})
        for bp in COSTS_BP:
            eq, trades, meta = backtest(
                feature, z, opens, closes, 2.0, 0.5, 4, bp,
                flat_boundaries=(TRAIN_END, VALID_END),
                funding_events=funding, fee_mode="gross")
            for period, a, b in (("validation", TRAIN_END, VALID_END),
                                 ("holdout", VALID_END, END),
                                 ("all", START, END)):
                rows.append({"model": model, "cost_bp": bp, "period": period,
                             **metrics(eq, trades, a, b, meta["funding_rows"])})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "pca_ridge_fixed_scan.csv", index=False)
    pd.DataFrame(stability).to_csv(RESULTS / "pca_ridge_fixed_stability.csv", index=False)
    (RESULTS / "pca_ridge_fixed_scan.json").write_text(json.dumps({
        "models": list(MODELS), "pca2_added": True, "frequency_min": 15,
        "center_hours": 4, "scale_days": 7, "entry_sigma": 2.0,
        "exit_sigma": 0.5, "hold_hours": 4, "cost_bp": list(COSTS_BP),
        "fee_mode": "gross aggregate order turnover",
        "funding": "local cached Binance funding events",
        "selection": "fixed rule; no model selected from holdout",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(pd.DataFrame(stability).to_string(index=False))


if __name__ == "__main__":
    run()
