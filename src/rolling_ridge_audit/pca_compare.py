"""Strict handoff-engine PCA/Ridge comparison.

All models use the handoff clock and ledger: 1-minute monitoring, 5-minute
signals, close[t-1] sizing, open[t+1] fill, live-gap25 exit, 30% single-lot
and 90% aggregate budget, net turnover fees, and no funding in the primary
run.  Model features are built once per spec with a spec-tagged cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data6 import END, N, RESULTS, load, minute
    from .engine import simulate6
    from .models import SPECS, load_feat, make
    from .replay import summarize
except ImportError:
    from data6 import END, N, RESULTS, load, minute
    from engine import simulate6
    from models import SPECS, load_feat, make
    from replay import summarize


MODELS = ("ridge_sma3", "pca2_sma3", "pca3_sma3", "pca5_sma3")
COSTS_BP = (0.0, 2.0, 5.0, 10.0)
PERIODS = (("validation", "2026-05-01", "2026-07-01"),
           ("holdout", "2026-07-01", "2026-09-01"))


def run_once(op, cl, lc, vol, feat, start, end, cost_bp, record=False):
    ends = np.arange(15, N, 5, dtype=np.int64)
    v = pd.DataFrame(vol).rolling(5, min_periods=5).sum().to_numpy()
    valid = np.isfinite(v[ends - 1]).all(1) & (v[ends - 1] > 0).all(1)
    return simulate6(
        op, cl, lc, np.zeros((1, 1)), ends,
        np.ascontiguousarray(feat["wi"][ends]),
        np.ascontiguousarray(feat["W"]),
        np.ascontiguousarray(feat["dev"][ends]),
        np.ascontiguousarray(feat["center"][ends]),
        np.ascontiguousarray(feat["scale"][ends]), valid,
        minute(start), minute(end), entry_z=2.0, gap_min=0.015,
        structure=0, exit_type=8, repair=0.5, hold_min=240,
        cost_bp=cost_bp, stop_fraction=0.03, entry_delay=1,
        record=record, live_dev=feat["dev"], live_scale=feat["scale"])


def stability(feat):
    w = np.asarray(feat["W"], float)
    if len(w) < 2:
        return {"fit_days": len(w), "mean_day_l1": np.nan,
                "median_day_l1": np.nan, "median_max_abs_peer": np.nan}
    den = np.abs(w[:-1]).sum(axis=2)
    move = np.abs(w[1:] - w[:-1]).sum(axis=2) / np.maximum(den, 1e-12)
    off = w.copy(); idx = np.arange(off.shape[1]); off[:, idx, idx] = np.nan
    return {"fit_days": len(w), "mean_day_l1": float(np.nanmean(move)),
            "median_day_l1": float(np.nanmedian(move)),
            "median_max_abs_peer": float(np.nanmedian(np.nanmax(np.abs(off), axis=2)))}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    op, cl, lc, vol, _ = load()
    rows, stable = [], []
    for model in MODELS:
        spec = next(s for s in SPECS if s["id"] == model)
        make(spec, force=True)
        feat = load_feat(model)
        stable.append({"model": model, **stability(feat)})
        for period, start, end in PERIODS:
            for bp in COSTS_BP:
                a = run_once(op, cl, lc, vol, feat, start, end, bp,
                             record=(bp == 5.0))
                row = summarize(a, minute(start), minute(end))
                row.update({"model": model, "period": period,
                            "cost_bp": bp, "funding": False})
                rows.append(row)
                if bp == 5.0:
                    name = f"pca_ridge_{model}_{period}_5bp_trades.csv"
                    pd.DataFrame(a[4]).to_csv(RESULTS / name, index=False)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "pca_ridge_package_compare.csv", index=False)
    pd.DataFrame(stable).to_csv(RESULTS / "pca_ridge_package_stability.csv", index=False)
    manifest = {
        "models": list(MODELS), "periods": [x[0] for x in PERIODS],
        "cost_bp": list(COSTS_BP), "engine": "handoff engine corrected stop",
        "clock": "1m monitor, 5m signals, close[t-1] sizing, open[t+1] fill",
        "entry": "|z|>=2 and abs(expm1 residual)>=1.5%, target-only",
        "exit": "live_gap25, max 240m, inclusive 3% stop",
        "budget": "30% per lot, 90% aggregate, max 3",
        "fees": "net aggregate order turnover; funding excluded",
        "pca_fit": "other 19 standardized hourly returns, 28d, normalized penalty 0.1",
    }
    (RESULTS / "pca_ridge_package_compare.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(pd.DataFrame(stable).to_string(index=False))


if __name__ == "__main__":
    main()
