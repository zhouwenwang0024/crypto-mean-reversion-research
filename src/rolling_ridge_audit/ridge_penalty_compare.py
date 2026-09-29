"""Strict-engine neighboring Ridge penalty check."""
import json
import numpy as np
import pandas as pd

try:
    from .data6 import N, RESULTS, load, minute
    from .engine import simulate6
    from .models import SPECS, load_feat, make
    from .replay import summarize
except ImportError:
    from data6 import N, RESULTS, load, minute
    from engine import simulate6
    from models import SPECS, load_feat, make
    from replay import summarize


def main():
    op, cl, lc, vol, _ = load(); ends = np.arange(15, N, 5, dtype=np.int64)
    rolling = pd.DataFrame(vol).rolling(5, min_periods=5).sum().to_numpy(); valid = np.isfinite(rolling[ends - 1]).all(1) & (rolling[ends - 1] > 0).all(1)
    rows = []
    for name in ("ridge_p05_sma3", "ridge_sma3", "ridge_p20_sma3"):
        spec = next(s for s in SPECS if s["id"] == name); make(spec, force=True); f = load_feat(name)
        for period, start, end in (("validation", "2026-05-01", "2026-07-01"), ("holdout", "2026-07-01", "2026-09-01")):
            for bp in (0., 2., 5., 10.):
                a = simulate6(op, cl, lc, np.zeros((1, 1)), ends, np.ascontiguousarray(f["wi"][ends]), np.ascontiguousarray(f["W"]), np.ascontiguousarray(f["dev"][ends]), np.ascontiguousarray(f["center"][ends]), np.ascontiguousarray(f["scale"][ends]), valid, minute(start), minute(end), entry_z=2., gap_min=.015, structure=0, exit_type=8, repair=.5, hold_min=240, cost_bp=bp, stop_fraction=.03, entry_delay=1, record=False, live_dev=f["dev"], live_scale=f["scale"])
                row = summarize(a, minute(start), minute(end)); row.update({"model": name, "period": period, "cost_bp": bp, "penalty": spec.get("penalty", .1)}); rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "ridge_penalty_package_compare.csv", index=False)
    (RESULTS / "ridge_penalty_package_compare.json").write_text(json.dumps({"models": ["ridge_p05_sma3", "ridge_sma3", "ridge_p20_sma3"], "penalty": [0.05, 0.1, 0.2], "same_engine": True, "cost_bp": [0, 2, 5, 10]}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
