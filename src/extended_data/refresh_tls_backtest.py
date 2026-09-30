"""Refresh only TLS rows after correcting the TLS estimator, then merge outputs."""
from __future__ import annotations

import json
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
    start = int(index.searchsorted(bt._ts("2024-06-01")))
    end = int(index.searchsorted(bt._ts("2026-09-01")))
    bars, trades, orders, summary = bt.run_model(index, op, cl, rates, marks, selected, "tls_log", start, end)
    summary["independent_replay_error"] = bt.replay_orders(bars, trades, orders) - bars.equity.iloc[-1]
    if abs(summary["reconciliation_error"]) > 1e-9 or abs(summary["independent_replay_error"]) > 1e-9:
        raise AssertionError(summary)
    prefix = bt.RESULTS / "statarb_5m_tls_log"
    bars.to_csv(str(prefix) + "_equity.csv", index=False)
    trades.to_csv(str(prefix) + "_trades.csv", index=False)
    orders.to_csv(str(prefix) + "_orders.csv", index=False)
    base = pd.read_csv(bt.RESULTS / "statarb_5m_summary.csv")
    base = pd.concat([base[base.model.ne("tls_log")], pd.DataFrame([summary])], ignore_index=True)
    base.to_csv(bt.RESULTS / "statarb_5m_summary.csv", index=False)

    costs = []
    for bp in (0.0, 2.0, 5.0, 10.0):
        for funding in (True, False):
            _, _, _, s = bt.run_model(index, op, cl, rates, marks, selected, "tls_log", start, end,
                                      fee_bp=bp, use_funding=funding)
            costs.append(s)
    old_cost = pd.read_csv(bt.RESULTS / "statarb_5m_cost_sensitivity.csv")
    old_cost = pd.concat([old_cost[old_cost.model.ne("tls_log")], pd.DataFrame(costs)], ignore_index=True)
    old_cost.to_csv(bt.RESULTS / "statarb_5m_cost_sensitivity.csv", index=False)
    _, _, _, nofund = bt.run_model(index, op, cl, rates, marks, selected, "tls_log", start, end,
                                   fee_bp=bt.FEE_BP, use_funding=False)
    funding = pd.read_csv(bt.RESULTS / "statarb_5m_funding_sensitivity.csv")
    funding = pd.concat([funding[funding.model.ne("tls_log")], pd.DataFrame([nofund])], ignore_index=True)
    funding.to_csv(bt.RESULTS / "statarb_5m_funding_sensitivity.csv", index=False)
    mutant = bt.run_model(index, op, cl, rates, marks, selected, "tls_log", start, end, direction_mutant=True)[3]
    checks = json.loads((bt.RESULTS / "statarb_5m_checks.json").read_text(encoding="utf-8"))
    checks["models"] = sorted(base.model.unique())
    checks["tls_refresh_replay_error"] = float(summary["independent_replay_error"])
    checks["cost_sensitivity_rows"] = int(len(old_cost))
    checks["funding_sensitivity_rows"] = int(len(funding))
    checks["direction_mutant_rows"] = 5
    checks["direction_mutant_returns"]["tls_log"] = mutant["return"]
    (bt.RESULTS / "statarb_5m_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    events = bt.attribute(bars, trades, "Asia/Shanghai", "tls_log")
    events2 = bt.attribute(bars, trades, "UTC", "tls_log")
    old_events = pd.read_csv(bt.RESULTS / "statarb_5m_event_attribution.csv")
    old_events = pd.concat([old_events[old_events.model.ne("tls_log")], pd.DataFrame([events, events2])], ignore_index=True)
    old_events.to_csv(bt.RESULTS / "statarb_5m_event_attribution.csv", index=False)
    print(base.to_string(index=False))
    print(pd.DataFrame(costs)[["model", "fee_bp_one_way", "funding_enabled", "return"]].to_string(index=False))


if __name__ == "__main__":
    main()
