"""Independent checks of saved v3 accounts and period returns."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v2 import ROOT, RESULTS, START, TRAIN_END, VALID_END, END, INITIAL
from summarize_v3 import load_equity, period_stats


def verify():
    configs = json.loads((RESULTS / "v3_selected_configs.json").read_text(encoding="utf-8"))
    costs = pd.read_csv(RESULTS / "v3_selected_costs.csv")
    checks, expected = [], set()
    for rank, cfg in enumerate(configs, 1):
        stem = f"v3_selected_{rank}_{cfg['model']}_{int(cfg['frequency_min'])}m_2bp"
        expected.add(f"{stem}_equity.csv")
        eq = load_equity(RESULTS / f"{stem}_equity.csv")
        trades = pd.read_csv(RESULTS / f"{stem}_trades.csv")
        ledger = json.loads((RESULTS / f"{stem}_ledger.json").read_text(encoding="utf-8"))
        error = float(trades.net_pnl_including_funding.sum() - (eq.iloc[-1] - INITIAL))
        assert abs(error) < 1e-7, (rank, error)
        month_returns = []
        for month in range(3, 9):
            a = pd.Timestamp(f"2026-{month:02d}-01", tz="UTC")
            b = pd.Timestamp(f"2026-{month+1:02d}-01", tz="UTC")
            base, finish, _ = period_stats(eq, a, b)
            month_returns.append(finish / base - 1)
        assert np.isclose(np.prod(1 + np.asarray(month_returns)), eq.iloc[-1] / INITIAL)
        for period, start, end in (("valid", TRAIN_END, VALID_END), ("holdout", VALID_END, END)):
            base, finish, daily = period_stats(eq, start, end)
            assert len(daily) == (end - start).days
            row = costs[(costs.selection_rank == rank) & (costs.cost_bp == 2) & (costs.period == period)].iloc[0]
            assert np.isclose(row["return"], finish / base - 1)
            assert np.isclose(row.net_pnl_including_funding, finish - base, atol=1e-7)
            fee = sum(e.get("fee", 0) for e in ledger if start <= pd.Timestamp(e["time"]) < end)
            assert np.isclose(fee, row.fees)
            assert np.isclose(fee, row.period_gross_fees)
        checks.append({"rank": rank, "model": cfg["model"], "reconciliation_error": error})
    actual = {p.name for p in RESULTS.glob("v3_selected_*_2bp_equity.csv")}
    assert actual == expected, (actual - expected, expected - actual)
    grid = pd.read_csv(RESULTS / "v3_grid_2bp.csv")
    fields = ["model", "frequency_min", "center_hours", "scale_days", "center_kind",
              "logic", "entry_sigma", "exit_sigma", "hold_hours", "exit_rule"]
    assert not grid.duplicated(fields).any()
    assert len(grid) == 67 and len(costs) == 8 * len(configs)
    monotone = []
    for (rank, period), g in costs.groupby(["selection_rank", "period"]):
        monotone.append(bool((g.sort_values("cost_bp")["return"].diff().dropna() <= 1e-12).all()))
    report = {"grid_rows": len(grid), "selected_accounts": len(configs),
              "cost_rows": len(costs), "daily_period_days": {"valid": 61, "holdout": 62},
              "accounts": checks, "all_cost_curves_nonincreasing": all(monotone),
              "confirmation_status": "reused evaluation sample; no untouched holdout claim"}
    paths = [ROOT / "src" / n for n in ["research_v2.py", "research_v3.py", "summarize_v3.py", "verify_v3.py"]]
    paths += [RESULTS / "v3_experiment_manifest.json", RESULTS / "v3_grid_2bp.csv", RESULTS / "v3_selected_costs.csv"]
    report["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (RESULTS / "v3_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["grid_rows", "selected_accounts", "cost_rows", "all_cost_curves_nonincreasing"]}))


if __name__ == "__main__":
    verify()
