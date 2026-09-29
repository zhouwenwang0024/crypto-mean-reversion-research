"""Recreate the one floating-point stop-boundary difference."""
import json
import numpy as np
import pandas as pd

try:
    from .data6 import RESULTS, SYMBOLS, load, minute
    from .replay import run_once
except ImportError:
    from data6 import RESULTS, SYMBOLS, load, minute
    from replay import run_once


def main():
    op, cl, *_ = load(); rows = []
    for start, end in (("2026-05-01", "2026-07-01"), ("2026-07-01", "2026-09-01")):
        raw, ra = run_once(start, end, 5., raw=True); fixed, fa = run_once(start, end, 5., raw=False)
        pairs = lambda tr: sorted((int(x[14]), int(x[2]), int(x[0]), int(x[1])) for x in tr)
        changed = [(x, y) for x, y in zip(pairs(ra[4]), pairs(fa[4])) if x != y]
        rows.append({"start": start, "raw_return": raw["return"], "corrected_return": fixed["return"], "difference_pp": (fixed["return"] - raw["return"]) * 100, "raw_trades": raw["trades"], "corrected_trades": fixed["trades"], "changed_trade_count": len(changed)})
        if changed:
            (sig, target, entry, raw_exit), fixed_key = changed[0]; fixed_exit = fixed_key[3]
            rr = next(x for x in ra[4] if int(x[14]) == sig and int(x[2]) == target and int(x[0]) == entry)
            checks = []
            for t in range(raw_exit - 2, raw_exit + 1):
                frac = float(rr[18 + target] * (cl[t - 1, target] - op[entry, target]) / rr[6])
                checks.append({"time": str(pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(minutes=t)), "fraction": repr(frac), "last_close": float(cl[t - 1, target])})
            rows[-1]["evidence"] = {"symbol": SYMBOLS[target], "signal_minute": sig, "entry_minute": entry, "raw_exit_minute": raw_exit, "corrected_exit_minute": fixed_exit, "checks": checks}
    pd.DataFrame([{k: v for k, v in r.items() if k != "evidence"} for r in rows]).to_csv(RESULTS / "stop_replay_local.csv", index=False)
    (RESULTS / "stop_boundary_local.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
