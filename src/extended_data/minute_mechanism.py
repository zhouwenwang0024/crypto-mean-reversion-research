"""Frozen native-minute mechanism sensitivities for the Ridge baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from extended_data.minute_validation import PERIODS, features, load_data, run  # noqa: E402


def main() -> None:
    index, op, cl, vol = load_data(); feat = features(np.log(cl), "ridge_sma3")
    rows = []
    for period, (a, b) in PERIODS.items():
        start, end = int(index.searchsorted(pd.Timestamp(a, tz="UTC"))), int(index.searchsorted(pd.Timestamp(b, tz="UTC")))
        for structure, name in ((0, "target"), (2, "half_hedge"), (3, "full_hedge")):
            account = run(op, cl, vol, feat, index, start, end, 5.0, structure=structure)
            rows.append({"period": period, "experiment": name, "structure": structure, **{
                "return": float(account[0][-1] - 1), "mdd": float((account[0] / np.maximum.accumulate(account[0]) - 1).min()),
                "trades": int(len(account[4])), "turnover": float(account[1].sum()),
                "max_gross_fraction": float(account[2].max()), "max_abs_net_fraction": float(abs(account[3]).max())}})
        if period == "holdout":
            for delay in (2, 3):
                account = run(op, cl, vol, feat, index, start, end, 5.0, entry_delay=delay)
                rows.append({"period": period, "experiment": f"entry_delay_{delay}m", "structure": 0, **{
                    "return": float(account[0][-1] - 1), "mdd": float((account[0] / np.maximum.accumulate(account[0]) - 1).min()),
                    "trades": int(len(account[4])), "turnover": float(account[1].sum()),
                    "max_gross_fraction": float(account[2].max()), "max_abs_net_fraction": float(abs(account[3]).max())}})
            account = run(op, cl, vol, feat, index, start, end, 5.0, signal_stride=3)
            rows.append({"period": period, "experiment": "signal_stride_15m", "structure": 0, **{
                "return": float(account[0][-1] - 1), "mdd": float((account[0] / np.maximum.accumulate(account[0]) - 1).min()),
                "trades": int(len(account[4])), "turnover": float(account[1].sum()),
                "max_gross_fraction": float(account[2].max()), "max_abs_net_fraction": float(abs(account[3]).max())}})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "extended_minute_mechanism.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__": main()
