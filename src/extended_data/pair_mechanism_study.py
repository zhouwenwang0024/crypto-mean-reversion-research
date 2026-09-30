"""Predeclared robustness study for the existing Johansen frozen-spread candidate.

This is a continuation study on the already-seen history.  It does not select
coins or parameters using the resulting returns.  Every run uses the audited
pair engine and its independent order replay; only the baseline writes the
full order, trade and daily ledgers.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_statarb_backtest as bt  # noqa: E402

START = "2024-06-01"
END = "2026-09-01"
FEE_BP = 5.0
GROSS_LIMIT = 0.90
REARM_Z = 1.0
PERIODS = {
    "development": ("2024-06-01", "2025-03-01"),
    "validation": ("2025-03-01", "2025-09-01"),
    "historical_holdout": ("2025-09-01", "2026-03-01"),
    "extension": ("2026-03-01", "2026-09-01"),
}

# Declared before any result is read or returned.  The sixth run uses only a
# prior formation diagnostic (beta half-sample gap), never a later return.
EXPERIMENTS = (
    ("baseline", 56, 3.00, 0.50, "all_selected"),
    ("entry_275", 56, 2.75, 0.50, "all_selected"),
    ("entry_325", 56, 3.25, 0.50, "all_selected"),
    ("hold_42d", 42, 3.00, 0.50, "all_selected"),
    ("hold_70d", 70, 3.00, 0.50, "all_selected"),
    ("stable_gap015", 56, 3.00, 0.50, "beta_relative_gap_le_015"),
)


def _stamp() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _period_return(bars: pd.DataFrame, start: str, end: str) -> float:
    t = pd.to_datetime(bars.time, utc=True)
    a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    mask = (t >= a) & (t < b)
    if not mask.any():
        return np.nan
    prior = bars.loc[t < a, "equity"]
    return float(bars.loc[mask, "equity"].iloc[-1] / (prior.iloc[-1] if len(prior) else 1.0) - 1.0)


def _event_row(bars: pd.DataFrame, trades: pd.DataFrame, zone: str) -> dict:
    a, b = bt._event_window(zone)
    t = pd.to_datetime(bars.time, utc=True)
    mask = (t >= a) & (t < b)
    ret = bars.equity.pct_change().fillna(bars.equity.iloc[0] - 1.0)
    zeroed = float((1.0 + ret.where(~mask, 0.0)).prod() - 1.0)
    crossing = pd.DataFrame()
    if len(trades):
        et = pd.to_datetime(trades.entry_time, utc=True)
        xt = pd.to_datetime(trades.exit_time, utc=True)
        crossing = trades.loc[(et < b) & (xt >= a)]
    ids = np.flatnonzero(mask.to_numpy())
    before = float(bars.equity.iloc[ids[0] - 1]) if len(ids) and ids[0] else 1.0
    after = float(bars.equity.iloc[ids[-1]]) if len(ids) else before
    return {
        "event_timezone": zone,
        "event_start_utc": str(a),
        "event_end_utc_exclusive": str(b),
        "event_bars": int(mask.sum()),
        "event_equity_change": after - before,
        "event_return_zeroed": zeroed,
        "crossing_trades": int(len(crossing)),
        "crossing_trade_net_pnl": float(crossing.net_pnl.sum()) if len(crossing) else 0.0,
        "crossing_trade_gross_pnl": float(crossing.gross_pnl.sum()) if len(crossing) else 0.0,
    }


def _daily(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x["time"] = pd.to_datetime(x.time, utc=True)
    x["date_utc"] = x.time.dt.floor("D")
    return x.groupby("date_utc", as_index=False).agg(
        equity=("equity", "last"), cash=("cash", "last"), fees=("fees", "last"),
        funding=("funding", "sum"), gross_exposure=("gross_exposure", "max"),
        net_exposure=("net_exposure", "last"), open_positions=("open_positions", "max"),
    )


def _run(index, op, cl, rates, marks, selected, name, hold_days, entry_z, exit_z, selector, fee_bp=FEE_BP):
    if selector == "beta_relative_gap_le_015":
        selected = selected.loc[(selected.model == "johansen") & (selected.beta_relative_gap <= 0.15)].copy()
        selected["model"] = "johansen_stable_gap015"
        model = "johansen_stable_gap015"
    else:
        model = "johansen"
    start = int(index.searchsorted(pd.Timestamp(START, tz="UTC")))
    end = int(index.searchsorted(pd.Timestamp(END, tz="UTC")))
    bars, trades, orders, summary = bt.run_model(
        index, op, cl, rates, marks, selected, model, start, end,
        fee_bp=fee_bp, use_funding=True, hold_bars=hold_days * 288,
        entry_z=entry_z, exit_z=exit_z, rearm_z=REARM_Z,
        force_month_boundary=False, gross_limit=GROSS_LIMIT,
    )
    replay = bt.replay_orders(bars, trades, orders, use_funding=True)
    model_rows = selected.loc[selected.model == model]
    summary.update({
        "experiment": name, "selector": selector,
        "independent_replay_error": float(replay - bars.equity.iloc[-1]),
        "selected_rows": int(len(model_rows)),
    })
    if abs(summary["reconciliation_error"]) > 1e-9 or abs(summary["independent_replay_error"]) > 1e-9:
        raise AssertionError(summary)
    for label, bounds in PERIODS.items():
        summary[f"{label}_return"] = _period_return(bars, *bounds)
    for zone in ("UTC", "Asia/Shanghai"):
        summary.update({f"{zone}_{k}": v for k, v in _event_row(bars, trades, zone).items()
                        if k not in ("event_timezone",)})
    return bars, trades, orders, summary


def main() -> None:
    manifest_path = RESULTS / "pair_mechanism_manifest.json"
    manifest = {
        "status": "running", "started_at_utc": _stamp(),
        "source_runner_sha256": _sha256(Path(bt.__file__)),
        "source_formation": "results/statarb_formation_selected.csv",
        "start": START, "end": END, "fee_bp": FEE_BP, "gross_limit": GROSS_LIMIT,
        "funding": True, "experiments": [dict(name=n, hold_days=h, entry_z=e, exit_z=x, selector=s)
                                             for n, h, e, x, s in EXPERIMENTS],
        "known_history_continuation": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index, op, cl, _ = bt.load_prices()
    rates, marks, _ = bt.load_funding(index)
    selected = pd.read_csv(RESULTS / "statarb_formation_selected.csv")
    rows, events = [], []
    for name, hold, entry, exit_z, selector in EXPERIMENTS:
        bars, trades, orders, row = _run(index, op, cl, rates, marks, selected, name, hold, entry, exit_z, selector)
        rows.append(row)
        for zone in ("UTC", "Asia/Shanghai"):
            er = _event_row(bars, trades, zone); er["experiment"] = name; events.append(er)
        if name == "baseline":
            prefix = RESULTS / "pair_mechanism_baseline"
            trades.to_csv(str(prefix) + "_trades.csv", index=False)
            orders.to_csv(str(prefix) + "_orders.csv", index=False)
            _daily(bars).to_csv(str(prefix) + "_daily.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "pair_mechanism_summary.csv", index=False)
    pd.DataFrame(events).to_csv(RESULTS / "pair_mechanism_events.csv", index=False)

    costs = []
    for bp in (0.0, 2.0, 5.0, 10.0):
        _, _, _, row = _run(index, op, cl, rates, marks, selected, "baseline", 56, 3.0, 0.5,
                            "all_selected", fee_bp=bp)
        row["fee_sensitivity_bp"] = bp; costs.append(row)
    pd.DataFrame(costs).to_csv(RESULTS / "pair_mechanism_costs.csv", index=False)
    manifest.update({"status": "complete", "finished_at_utc": _stamp(), "summary_rows": len(rows),
                     "cost_rows": len(costs), "event_rows": len(events),
                     "max_replay_error": float(max(abs(x["independent_replay_error"]) for x in rows))})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["experiment", "return", "mdd", "trades", "fees", "funding", "max_gross", "independent_replay_error"]].to_string(index=False))
    print(pd.DataFrame(costs)[["fee_sensitivity_bp", "return", "fees", "funding", "trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
