"""Reproduce the handoff strategy on the repository's local minute data.

The account engine is the handoff engine with the inclusive 3% stop boundary
fix.  ``--raw`` runs the preserved pre-fix engine so the two result sets stay
separate.  No network or environment-variable mutation is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data6 import END, N, RESULTS, load, minute
    from .models import SPECS, load_feat, make
    from .engine import simulate6
    from . import engine_original
except ImportError:
    from data6 import END, N, RESULTS, load, minute
    from models import SPECS, load_feat, make
    from engine import simulate6
    import engine_original


def summarize(a, start: int, end: int) -> dict:
    eq, tv, gross, net, trades, _ = a
    pnl = float(trades[:, 7].sum())
    fees = float(trades[:, 8].sum())
    m = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return {
        "return": float(eq[-1] - 1),
        "mdd": float(np.min(eq / m - 1)),
        "trades": int(len(trades)),
        "price_pnl": pnl,
        "fees": fees,
        "turnover": float(tv.sum()),
        "mean_gross": float(gross.mean()),
        "max_gross": float(gross.max()),
        "mean_hold": float(np.mean(trades[:, 1] - trades[:, 0])) if len(trades) else 0.0,
        "stop_count": int(np.sum(trades[:, 10] == 1)),
        "timeout_count": int(np.sum(trades[:, 10] == 2)),
        "ref_exit_count": int(np.sum(trades[:, 10] == 0)),
        "reconciliation_error": float(eq[-1] - 1 - (pnl - fees + float(trades[:, 9].sum()))),
        "start": str(pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(minutes=int(start))),
        "end": str(pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(minutes=int(end))),
    }


def run_once(start: str, end: str, cost_bp: float, *, raw: bool = False, phase: int = 0,
             model: str = "ridge_sma3", policy: str = "live_gap25", record: bool = True):
    op, cl, lc, vol, _ = load()
    spec = next(s for s in SPECS if s["id"] == model)
    make(spec, force=True)
    f = load_feat(model)
    ends = np.arange(15 + phase, N, 5, dtype=np.int64)
    v = pd.DataFrame(vol).rolling(5, min_periods=5).sum().to_numpy()
    valid = np.isfinite(v[ends - 1]).all(1) & (v[ends - 1] > 0).all(1)
    fn = engine_original.simulate6 if raw else simulate6
    a = fn(op, cl, lc, np.zeros((1, 1)), ends, np.ascontiguousarray(f["wi"][ends]),
           np.ascontiguousarray(f["W"]), np.ascontiguousarray(f["dev"][ends]),
           np.ascontiguousarray(f["center"][ends]), np.ascontiguousarray(f["scale"][ends]),
           valid, minute(start), minute(end), entry_z=2.0, gap_min=0.015,
           structure=0, exit_type=8, repair=0.5, hold_min=240, cost_bp=cost_bp,
           stop_fraction=0.03, entry_delay=1, record=record,
           live_dev=f["dev"], live_scale=f["scale"])
    return summarize(a, minute(start), minute(end)), a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="preserve the pre-tolerance stop result")
    ap.add_argument("--out", default="rolling_ridge_replay.csv")
    args = ap.parse_args()
    rows = []
    for start, end in (("2026-05-01", "2026-07-01"), ("2026-07-01", "2026-09-01")):
        for bp in (0.0, 2.0, 5.0, 10.0):
            row, a = run_once(start, end, bp, raw=args.raw)
            row.update({"model": "ridge_sma3", "policy": "live_gap25", "cost_bp": bp, "raw_stop": args.raw})
            rows.append(row)
            if bp == 5.0:
                stem = "raw" if args.raw else "corrected"
                Path(RESULTS / f"rolling_ridge_{stem}_{start}_5bp_trades.csv").write_text(
                    pd.DataFrame(a[4]).to_csv(index=False), encoding="utf-8")
    out = RESULTS / args.out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    (RESULTS / "rolling_ridge_replay_manifest.json").write_text(json.dumps({
        "strategy": "ridge_sma3/live_gap25", "data": "local data/klines; 20 symbols; 2026-03-01/2026-09-01",
        "costs_bp": [0, 2, 5, 10], "run_mode": "raw" if args.raw else "corrected",
        "stop_fix": "corrected engine pfrac <= -0.03 + 1e-12; raw engine retained separately",
        "feature_cache": "spec-tagged and force rebuilt for each run"
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
