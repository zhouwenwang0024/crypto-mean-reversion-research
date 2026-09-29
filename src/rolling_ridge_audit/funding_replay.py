"""Add the locally cached funding settlements to fixed, already chosen runs."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

try:
    from .data6 import END, N, RESULTS, SYMBOLS, START, load, minute
    from .engine import simulate6
    from .models import SPECS, load_feat, make
    from .replay import summarize
except ImportError:
    from data6 import END, N, RESULTS, SYMBOLS, START, load, minute
    from engine import simulate6
    from models import SPECS, load_feat, make
    from replay import summarize


def funding_array(cl: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros((N + 1, len(SYMBOLS))); rows = 0
    for j, symbol in enumerate(SYMBOLS):
        d = pd.read_parquet(RESULTS / "funding_api" / f"{symbol}.parquet")
        for r in d.itertuples(index=False):
            t = int((int(r.funding_time_utc_ms) - START.value // 10**6) // 60000)
            if 0 <= t < N:
                out[t, j] = -float(r.funding_rate) * float(r.mark_price); rows += 1
    if rows == 0 or not np.isfinite(out).all(): raise ValueError("funding cache is missing or non-finite")
    return out, rows


def run(model, start, end, cost, fund, feat):
    op, cl, lc, vol, _ = load(); ends = np.arange(15, N, 5, dtype=np.int64)
    vv = pd.DataFrame(vol).rolling(5, min_periods=5).sum().to_numpy()
    valid = np.isfinite(vv[ends - 1]).all(1) & (vv[ends - 1] > 0).all(1)
    a = simulate6(op, cl, lc, np.ascontiguousarray(fund), ends, np.ascontiguousarray(feat["wi"][ends]), np.ascontiguousarray(feat["W"]), np.ascontiguousarray(feat["dev"][ends]), np.ascontiguousarray(feat["center"][ends]), np.ascontiguousarray(feat["scale"][ends]), valid, minute(start), minute(end), entry_z=2., gap_min=.015, structure=0, exit_type=8, repair=.5, hold_min=240, cost_bp=cost, include_funding=True, stop_fraction=.03, entry_delay=1, record=False, live_dev=feat["dev"], live_scale=feat["scale"])
    row = summarize(a, minute(start), minute(end)); row.update({"model": model, "start": start, "end": end, "cost_bp": cost, "funding_rows": int(np.count_nonzero(fund))})
    row["funding_cash"] = float(a[4][:, 9].sum())
    return row


def main():
    fund, rows = funding_array(load()[1]); out = []
    for model in ("ridge_sma3", "pca5_sma3"):
        spec = next(s for s in SPECS if s["id"] == model); make(spec, force=True); feat = load_feat(model)
        for start, end in (("2026-05-01", "2026-07-01"), ("2026-07-01", "2026-09-01")):
            row = run(model, start, end, 5., fund, feat); row["funding_event_rows"] = rows; out.append(row)
    pd.DataFrame(out).to_csv(RESULTS / "funding_replay.csv", index=False)
    (RESULTS / "funding_replay_manifest.json").write_text(json.dumps({"source": "results/funding_api/*.parquet", "rows": rows, "mark_price": True, "cashflow": "-side*quantity*mark_price*funding_rate", "missing_is_error": True}, indent=2), encoding="utf-8")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__": main()
